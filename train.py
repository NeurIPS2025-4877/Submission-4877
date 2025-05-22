import argparse
import model.metric as module_metric
from utils.parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import load_graph_dataset
from model.HyperTE import HyperTE
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config/', type=str,
                      help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data', type=str, default='TCGA')

    params = args.parse_args()

    config = ConfigParser.from_args(args)
    config, train_set, valid_set, test_set = load_graph_dataset(config)
    config['data_loader']['data'] = params.data
    
    print(config)
    model = HyperTE(config)
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

                        
