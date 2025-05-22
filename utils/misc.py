import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Callable, Iterator, Optional, Union
from torch_geometric.nn import GATConv, GCNConv, GraphConv, RGCNConv
from argparse import Namespace
# import model.models as models
# import model.single_models as single_models


# def get_model(model, config, params) -> th.nn.Module:
#     str_to_model_dict = {
#         "gnn": GNNRegressionModel,
#         "graphite": GraphITE,
#         "zero": ZeroBaseline,
#         "sin": SIN,
#         "cat": CategoricalTreatmentRegressionModel,
#         "transtee": TransTEE
#     }
#     model = str_to_model_dict[model](args=args).to(device)

#     return model


def get_active_function(name: Optional[str] = None, leaky_relu: Optional[float] = 0.5):
    functions = {
        None: lambda x: x,
        'leaky_relu': nn.LeakyReLU(leaky_relu),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(),
    }
    name = name.lower() if isinstance(name, str) else None  
    if name in functions:
        return functions[name]
    raise ValueError("output_func must be 'leaky_relu','relu', 'sigmoid', 'softmax', or None")

 

def get_gnn_conv(name: str) -> Union[GCNConv, GATConv, GraphConv, RGCNConv]:
    if name == "gcn":
        return GCNConv
    elif name == "gat":
        return GATConv
    elif name == "graph_conv":
        return GraphConv
    elif name == "rcgn":
        return RGCNConv
    else:
        raise Exception("Unknown GNN layer")


def get_initialiser(name: str) -> Callable:
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")



def exp_type(value):
    if value.lower() == 'all':
        return 'all'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("exp should be 'all' or an integer.")
    
    
'''
def gnn_model(name: str) -> Union[GCNConv, GATConv, GraphConv, RGCNConv]:
    if name == "gcn":
        return GCNConv
    elif name == "gat":
        return GATConv
    elif name == "graph_conv":
        return GraphConv
    elif name == "rcgn":
        return RGCNConv
    else:
        raise Exception("Unknown GNN layer")


def get_initialiser(name: str) -> Callable:
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")
'''


