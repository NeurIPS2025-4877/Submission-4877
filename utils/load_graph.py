import pickle
from operator import itemgetter
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List
from preprocessing.data.dataset import Dataset, create_pt_geometric_dataset

def load_graph_dataset(config):
    data_info = config["data_loader"]
    
    train_set, valid_set = get_train_and_val_dataset(data_info)
    test_set = pickle.load(open(data_info['path_to_test'], "rb"))

    n_test = len(test_set.get_test_units(in_sample=True))
    
    data_info['n_samples'] = len(train_set) + len(valid_set) + n_test
    data_info['n_train'] = len(train_set)
    data_info['n_valid'] = len(valid_set)
    data_info['n_test'] = n_test
    config['hyper_params']['x_input_dim'] = train_set[0].covariates.shape[1]
    
    return config, train_set, valid_set, test_set


def get_train_and_val_dataset(data_info):
    path_to_train = data_info['path_to_train']
    in_sample_data = pickle.load(open(path_to_train, "rb"))
    units = (
        in_sample_data.get_units()["features"]
        if "tcga" in path_to_train.lower()
        else in_sample_data.get_units()
    )
    graphs = in_sample_data.get_treatment_graphs()
    outcomes = in_sample_data.get_outcomes()
    train_data, val_data = split_train_val(
        units=units, graphs=graphs, outcomes=outcomes, val_size=data_info["valid_ratio"]
    )

    train_data_pt = load_pt_dataset(train_data)
    val_data_pt = load_pt_dataset(val_data)
    
    assert train_data is not None
    assert val_data_pt is not None
    return train_data_pt, val_data_pt

def load_pt_dataset(data):
    return create_pt_geometric_dataset(
            units=data["units"],
            treatment_graphs=data["graphs"],
            outcomes=data["outcomes"],
        )

def split_train_val(units, graphs, outcomes, val_size=0.3):
    np.random.seed(1234)
    num_train = len(units)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_data, val_data = {}, {}
    train_data["units"], train_data["graphs"], train_data["outcomes"] = (
        units[train_idx],
        itemgetter(*train_idx)(graphs),
        outcomes[train_idx],
    )
    val_data["units"], val_data["graphs"], val_data["outcomes"] = (
        units[valid_idx],
        itemgetter(*valid_idx)(graphs),
        outcomes[valid_idx],
    )
    return train_data, val_data

