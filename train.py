import argparse
import os
import numpy as np
import pandas as pd
import model.metric as module_metric
import model as model_arch
from utils.parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import load_graph_dataset, exp_type
from model.HyperTE import HyperTE
import time 
import torch
import warnings
from itertools import product
warnings.filterwarnings("ignore")



def main(params, config, iter):
    config, train_set, valid_set, test_set = load_graph_dataset(config)
    
    model = HyperTE(config, params)
  
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", num_params)

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = Trainer(model, 
                      metrics,
                      config,
                      train_set,
                      valid_set,
                      test_set)

    return trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', type=str, default='config/SW.json',
                      help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data', type=str, default='TCGA')
    args.add_argument('--drug_n_dims', type=int, default=100)
    args.add_argument('--drug_n_layers', type=int, default=2)
    args.add_argument('--feat_n_dims', type=int, default=100)
    args.add_argument('--feat_n_layers', type=int, default=2)
    args.add_argument('--pred_n_dims', type=int, default=100)
    args.add_argument('--pred_n_layers', type=int, default=2)
    params = args.parse_args()


    exper_name = '{}/{}_{}_{}_{}_{}_{}'.format(params.data, params.drug_n_dims, params.drug_n_layers, params.feat_n_dims, params.feat_n_layers, params.pred_n_dims, params.pred_n_layers)
    config = ConfigParser.from_args(args, exper_name)

    config, train_set, valid_set, test_set = load_graph_dataset(config)
    
    model = HyperTE(config, params)
  
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", num_params)

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = Trainer(model, 
                      metrics,
                      config,
                      train_set,
                      valid_set,
                      test_set)
    
    trainer.train()

                        
