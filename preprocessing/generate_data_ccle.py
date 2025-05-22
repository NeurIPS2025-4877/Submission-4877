import numpy as np
from scipy.special import softmax
import itertools
from typing import List, Tuple, Union
import argparse
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from typing import List, Optional
from torch_geometric.data.batch import Batch
import pickle
from pathlib import Path
from data.dataset import create_pt_geometric_dataset, TestUnits, TestUnit, Dataset
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from collections import defaultdict
from sklearn.model_selection import train_test_split

def convert_edge(adj_list, make_undirected=True):
    edge_list = []
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            edge_list.append((i, j))
            if make_undirected:
                edge_list.append((j, i))
    return list(set(edge_list))

def load_graph_dict(path='dataset/CCLE'):
    pubchemid2smile = pickle.load(open(os.path.join(path, 'pubchem_smiles.pkl'), 'rb'))
    molecules_dict = dict()
    
    bond_type_to_int = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    treatment_ids = sorted(pubchemid2smile.keys())  # 전체 treatment ID 목록
    id_to_index = {tid: i for i, tid in enumerate(treatment_ids)}  # one-hot 인코딩용

    for each in pubchemid2smile.keys():
        smile = pubchemid2smile[each]
        mol = Chem.MolFromSmiles(smile)
        featurizer = dc.feat.ConvMolFeaturizer()
        mol_object = featurizer.featurize(smile)

        node_features = mol_object[0].atom_features
        mol_data = dict()
        mol_data['node_features'] = node_features
        mol_data['edges'] = convert_edge(mol_object[0].canon_adj_list)
        mol_data['c_size'] = len(node_features)  
        one_hot = np.zeros(len(treatment_ids), dtype=np.float32)
        one_hot[id_to_index[each]] = 1.0
        mol_data['one_hot_encoding'] = one_hot

        # # edge types
        # edge_type_dict = dict()
        # for bond in mol.GetBonds():
        #     a1 = bond.GetBeginAtomIdx()
        #     a2 = bond.GetEndAtomIdx()
        #     bond_type = bond.GetBondType()
        #     etype = bond_type_to_int.get(bond_type, 0)
        #     edge_type_dict[(a1, a2)] = etype
        #     edge_type_dict[(a2, a1)] = etype

        # edge_list = mol_data['edges']
        # edge_types = [edge_type_dict.get((src, tgt), 0) for (src, tgt) in edge_list]
        # mol_data['edge_types'] = edge_types

        molecules_dict[each] = mol_data

    return molecules_dict

# ----------------------------------------
# ----------------------------------------
    
    
def create_selection_bias_probs(y: np.ndarray, bias: float = 1.0) -> np.ndarray:
    std = np.std(y)
    rho = bias / 100 * (std + 1e-8)
    return softmax(rho * y)


def split_unit_indices(N: int, train_ratio: float = 0.66, seed: int = 42):
    np.random.seed(seed)
    indices = np.arange(N)
    np.random.shuffle(indices)
    split_point = int(train_ratio * N)
    return indices[:split_point], indices[split_point:]


class DatasetBuilder:
    def __init__(
        self,
        covariate,  # DataFrame (index: unit_id)
        response,   # DataFrame (index: unit_id, columns: drug_id)
        id_to_graph_dict: dict,
        train_frac=0.5,
        bias=1.0,
        seed=42
    ):
        self.covariate = covariate
        self.response = response
        self.id_to_graph_dict = id_to_graph_dict
        self.train_frac = train_frac
        self.bias = bias
        self.seed = seed
        self.used_pairs = None
        self.max_test_assignments = 10 
        self.min_test_assignments = 2

        self.unit_indices = np.arange(len(response))
        np.random.seed(seed)
        np.random.shuffle(self.unit_indices)

        split_point = int(train_frac * len(self.unit_indices))
        self.train_indices = self.unit_indices[:split_point]
        self.test_out_indices = self.unit_indices[split_point:]
        self.test_in_indices = self.train_indices  # same units as train

    def create_training_dataset(self):
        units, treatment_ids, outcomes = [], [], []
        self.train_pairs = set()

        for i in self.train_indices:
            unit_id = self.response.index[i]
            try:
                unit_feat = self.covariate.loc[unit_id].values
            except:
                continue

            observed_drugs = self.response.columns[~self.response.loc[unit_id].isna()].tolist()            
            observed_outcomes = self.response.loc[unit_id, observed_drugs].values
            probs = create_selection_bias_probs(observed_outcomes, self.bias)
            
            assigned_treatment = np.random.choice(
                a=observed_drugs, p=probs
            )
        
            units.append(unit_feat)
            treatment_ids.append(assigned_treatment)
            outcomes.append(self.response.loc[unit_id, assigned_treatment])
            self.train_pairs.add((unit_id, assigned_treatment))

        data_dict = {
            "units": np.array(units),
            "dim_covariates": self.covariate.shape[1],
            "all_treatments": list(self.id_to_graph_dict.keys()),
            "id_to_graph_dict": self.id_to_graph_dict,
        }

        self.set_train_ids = set(treatment_ids)
        dataset = Dataset(data_dict)
        dataset.add_assigned_treatments(treatment_ids)
        dataset.add_outcomes(np.array(outcomes))
        return dataset

    def create_testunits(self):
        test_units_dict = {"in_sample": [], "out_sample": []}
        set_test_ids = set()

        for label, indices in [("in_sample", self.test_in_indices), ("out_sample", self.test_out_indices)]:
            for i in indices:
                unit_id = self.response.index[i]
                try:
                    unit_feat = self.covariate.loc[unit_id].values
                except:
                    continue

                observed_drugs = self.response.columns[~self.response.loc[unit_id].isna()].tolist()
                if label == "in_sample":
                    observed_drugs = [t for t in observed_drugs if (unit_id, t) not in self.train_pairs]

                outcomes = self.response.loc[unit_id, observed_drugs].values
                if len(outcomes) < self.max_test_assignments:
                    continue
                
                probs = create_selection_bias_probs(outcomes, self.bias)
                test_unit = TestUnit(
                    covariates=unit_feat,
                    treatment_ids=observed_drugs,
                    treatment_propensities=probs,
                    true_outcomes=outcomes,
                )
                test_units_dict[label].append(test_unit)
                set_test_ids.update(observed_drugs)

        unseen_treatments = set_test_ids - self.set_train_ids
        return TestUnits(test_units_dict, self.id_to_graph_dict, list(unseen_treatments))

    def creating_training_test_sets(self, verbose=True):
        train_dataset = self.create_training_dataset()
        test_units = self.create_testunits()

        if verbose:
            print(f"[Train] Units: {len(train_dataset.get_units())}, Treatments: {len(train_dataset.get_unique_treatment_ids())}")
            print(f"[Test-In] {len(test_units.get_test_units(in_sample=True))} units")
            print(f"[Test-Out] {len(test_units.get_test_units(in_sample=False))} units")

        return train_dataset, test_units

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="GDSC", choices=["CCLE", "GDSC"])
    args.add_argument("--bias", type=float, default=0.1)
    args.add_argument("--seed", type=int, default=7)
    args.add_argument(
        "--data_path",
        type=str,
        default="../generated_data/",
        help="Path to save/load generated data",
    )
    params = args.parse_args()
    
    path=f'dataset/{params.data}'

    id_to_graph_dict = load_graph_dict(path)
    covariates = pd.read_csv(os.path.join(path, 'cellline_pcor_ess_genes.csv'), index_col=0)
    reponses = pd.read_csv(os.path.join(path, 'all_abs_ic50_bayesian_sigmoid.csv'), index_col=0)
    reponses = reponses.drop(list(set(reponses.columns) - set(id_to_graph_dict.keys())), axis=1)
    reponses = reponses.drop(list(set(reponses.index) - set(covariates.index)), axis=0)

    builder = DatasetBuilder(covariate=covariates, response=reponses, id_to_graph_dict=id_to_graph_dict, bias=params.bias)
    train_data, test_data = builder.creating_training_test_sets()


    def pickle_dump(file_name: str, content: object) -> None:
        with open(file_name, "wb") as out_file:
            pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)

    def save_dataset(
        in_sample_dataset: Dataset, test_units: TestUnits) -> None:
        file_path = params.data_path + f"{params.data.lower()}/seed-{params.seed}/bias-{params.bias}/"
        Path(file_path).mkdir(parents=True, exist_ok=True)
        pickle_dump(file_name=file_path + "in_sample.p", content=in_sample_dataset)
        pickle_dump(file_name=file_path + "test.p", content=test_units)
        print("Saved training and test dataset successfully.")
        
        
    save_dataset(train_data, test_data)
