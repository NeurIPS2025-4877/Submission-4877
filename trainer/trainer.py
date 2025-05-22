import numpy as np
import torch
from .base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import pandas as pd
from torch_geometric.loader import DataLoader
from model.utils.util import padding
from torch_geometric.data.batch import Batch
from typing import List
from data.utils import get_treatment_graphs
from data.dataset import create_pt_geometric_dataset, TestUnit, TestUnits #, TestUnit_NonSynth
from collections import defaultdict

class Trainer(BaseTrainer):
    def __init__(self, model, 
                      metric_ftns,
                      config,
                      train_set,
                      valid_set,
                      test_set
                      ):
        super().__init__(model, metric_ftns, config)
        self.config = config
        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set

        self.batch_size = config['data_loader']['batch_size']
        self.n_train = self.config['data_loader']['n_train']
        self.x_input_dim = self.model.x_input_dim
        
        self.data = config['data_loader']['data']
        self.hparam = config['hyper_params']
        self.data_type = self.config['data_loader']['data_type']
        if self.data_type == 'tabular':
            self.drug_lst = np.unique(batch_to_arr(self.train_set, attr_name='d'))
        
        self.do_validation = self.valid_set is not None
        self.log_step = 16 
        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        
    def _train_epoch(self, epoch):
        self.model.train()
        loss = 0
        y_outs = torch.tensor([]).to(self.device)
        for index, batch in enumerate(DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)):
            batch = self.to_tensor(batch)
            loss_dict, y_pred = self.model.update(batch)
            if index % self.log_step == 0:
                loss_formatted = ", ".join([f"{k}: {v.item():.3f}" for k, v in loss_dict.items()])
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(index)} Loss: {loss_formatted}')
            y_outs = torch.cat([y_outs, y_pred], dim=0)
            loss += loss_dict['total_loss'].item()
        loss /= (index+1)
        
        log = self._metrics_graph(
            self.train_set, y_outs, loss
            )
        
        if self.do_validation:
            val_log = self._infer(self.valid_set)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

        
    def _infer(self, data_set):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            y_outs = torch.tensor([]).to(self.device)
            for index, batch in enumerate(DataLoader(data_set, batch_size=self.batch_size, shuffle=False)):
                batch = self.to_tensor(batch)
                loss_dict, y_pred = self.model.predict(batch)
                y_outs = torch.cat([y_outs, y_pred], dim=0)
                loss += loss_dict['total_loss'].item()       
        loss /= (index+1)
        
        log = self._metrics(
            data_set, y_outs, loss
            )
        
        return log
        

    def _test_epoch(self):
        if self.save_model:
            self.model.load_state_dict(torch.load(str(self.checkpoint_dir / 'model_best.pth'))['state_dict'])
        else:
            self.model.load_state_dict(self.model_best_params)
        self.logger.info('Loaded best parameters...')        
        log = {}
        for phase, dataset in [('train', self.train_set),
                                ('val', self.valid_set),
                                ('test', self.test_set)]:
            if phase=='test':
                sub_log = self.infer_graph_test(dataset)
            else:
                sub_log = self._infer(dataset)
            log.update({f'{phase}_{k}': v for k, v in sub_log.items()})
        
        self.logger.info('='*100)
        self.logger.info('Inference Completed')
        for key, value in log.items():
            if 'test_6' in key:
                self.logger.info(f'{key:20s}: {value}')
        self.logger.info('='*100)
        return log
    
    def _metrics(self, data_set, y_outs, loss):
        y_outs = y_outs.detach().cpu().numpy().flatten()
        y_trgs = batch_to_arr(data_set, attr_name='y')
        
        self.metrics.reset()
        self.metrics.update('loss', loss)
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(y_trgs, y_outs))

        return self.metrics.result()

    def infer_graph_test(self, data_set):
        log = {}
        log_in_sample = self._infer_graph_test(data_set, in_sample=True)
        log_out_sample = self._infer_graph_test(data_set, in_sample=False)
        
        log.update({f'{k}_in_sample': v for k, v in log_in_sample.items()})
        log.update({f'{k}_out_sample': v for k, v in log_out_sample.items()})

        return log


    def _infer_graph_test(self, data_set, in_sample):
        self.model.eval()
        id_to_graph_dict = data_set.get_id_to_graph_dict()    
        test_units = data_set.get_test_units(in_sample=in_sample)
        for i, test_unit in enumerate(test_units):
            treatment_ids = test_unit.get_treatment_ids()
            treatment_graphs = get_treatment_graphs(
                treatment_ids=treatment_ids, id_to_graph_dict=id_to_graph_dict
            )
            unit = test_unit.get_covariates()
            units = np.repeat(np.expand_dims(unit, axis=0), len(treatment_ids), axis=0)
            test_unit_pt_dataset = create_pt_geometric_dataset(
                units=units, treatment_graphs=treatment_graphs
            )
            with torch.no_grad():
                batch = Batch.from_data_list(test_unit_pt_dataset).to(self.device)
                predicted_outcomes = self.model.test_predict(batch).cpu().numpy()
            predicted_outcomes_dict = dict(zip(treatment_ids, predicted_outcomes))
            test_unit.set_predicted_outcomes(predicted_outcomes=predicted_outcomes_dict)
            
        log = {}
        for k in range(self.hparam['min_test_assignments'], self.hparam['max_test_assignments'] + 1):
            unweighted_errors_lst, weighted_errors_lst, ATE_errors_lst = [], [], []
            predicted_effect_dict = defaultdict(list)
            true_effect_dict = defaultdict(list)
            for i, test_unit in enumerate(data_set.get_test_units(in_sample=in_sample)):
                unweighted_error, weighted_error = test_unit.evaluate_predictions(k=k)
                unweighted_errors_lst.append(unweighted_error)
                weighted_errors_lst.append(weighted_error)
                
                for combination in test_unit.get_treatment_combinations(k):
                    treatment_1_id, treatment_2_id = combination[0], combination[1]
                    outcome_1, outcome_2 = (
                        test_unit.data_dict["predicted_outcomes"][treatment_1_id],
                        test_unit.data_dict["predicted_outcomes"][treatment_2_id],
                    )
                    predicted_effect_dict[f'{treatment_1_id}_{treatment_2_id}'].append(outcome_1 - outcome_2)
    
                    outcome_1, outcome_2 = (
                        test_unit.data_dict["true_outcomes"][treatment_1_id],
                        test_unit.data_dict["true_outcomes"][treatment_2_id],
                    )
                    true_effect_dict[f'{treatment_1_id}_{treatment_2_id}'].append(outcome_1 - outcome_2)

            ate_errors_lst = []
            for key in predicted_effect_dict.keys():
                predicted_ate = np.mean(predicted_effect_dict[key])
                true_ate = np.mean(true_effect_dict[key])
                ate_errors_lst.append(np.abs(true_ate - predicted_ate))
                
            log[f"{k}_UPEHE"] = np.mean(unweighted_errors_lst)
            log[f"{k}_WPEHE"] = np.mean(weighted_errors_lst)
            log[f"{k}_ATE"] = np.mean(ate_errors_lst)
                
        return log
    
    
    def to_tensor(self, batch: Batch):
        if isinstance(batch.x, torch.Tensor):
            return batch.to(self.device)
        batch.covariates = torch.Tensor(batch.covariates).to(self.device)
        batch.x = torch.Tensor(np.array(batch.x)).to(self.device)
        batch.y = torch.Tensor(batch.y).to(self.device)
        batch.t = torch.Tensor(batch.t).to(self.device)
        return batch
    
    def _progress(self, batch_idx):
        current = batch_idx * self.batch_size
        return f"[{current}/{self.n_train} ({100.0 * current / self.n_train:.0f}%)]"
                    
                    
def batch_to_arr(data_set, attr_name='y'):
    return np.array([getattr(data, attr_name).cpu().numpy() for data in data_set]).flatten()
    


            
