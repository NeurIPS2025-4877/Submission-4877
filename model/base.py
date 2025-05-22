import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Optional
from utils.misc import get_active_function, get_initialiser, get_gnn_conv
from model.utils.gnn import GNN
from torch_geometric.data.batch import Batch
from copy import deepcopy


class TreatmentFeatureExtractor(nn.Module):
    def __init__(self, 
                 config, 
                 ):
        super(TreatmentFeatureExtractor, self).__init__()
        hparam = config['hyper_params']
        d_input_dim = hparam['d_input_dim']
        self.is_graph = (config['data_loader']['data_type'] == 'graph')
        is_hyperTE = hparam['model'].startswith('HyperTE')
        
        if not self.is_graph:
            self.treatment_net = LinearNet(d_input_dim, hparam['drug_n_layers'], hparam['drug_n_dims'], 
                                           out_dim=hparam['drug_n_dims'], output_func=hparam['activation'])
        else:
            self.treatment_net = GNN( 
                gnn_conv=hparam['gnn_conv'],
                dim_input=d_input_dim,
                dim_hidden=hparam['dim_hidden_treatment'],
                dim_output=hparam['drug_n_dims'],
                num_layers=hparam['num_treatment_layer'],
                batch_norm=hparam['gnn_batch_norm'],
                initialiser=hparam['initialiser'],
                dropout=hparam['gnn_dropout'],
                activation=hparam['activation'],
                leaky_relu=hparam['leaky_relu'],
                is_output_activation=hparam['activation'],
                num_relations=hparam['gnn_num_relations'],
                num_bases=hparam['gnn_num_bases'],
                is_multi_relational=hparam['gnn_multirelational'],
            )
            self.is_multi_relational = hparam['gnn_multirelational']
            
    def forward_nongraph(self, input):        
        return self.treatment_net(input)
    
    def forward_graph(self, batch: Batch):
        treatment_node_features, treatment_edges, batch_assignments = (
            batch.x,
            batch.edge_index,
            batch.batch,
        )
        treatment_edge_types = batch.edge_types if self.is_multi_relational else None
        treatment_features = self.treatment_net(
            treatment_node_features,
            treatment_edges,
            treatment_edge_types,
            batch_assignments,
        )
        return treatment_features
    
    def forward(self, batch: Batch):
        if self.is_graph:
            return self.forward_graph(batch)
        else:
            return self.forward_nongraph(batch.x)                  

class LinearNet(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 n_layers: int, 
                 hidden_dim: int, 
                 out_dim: Optional[int] = None, 
                 active_func: Optional[str] = 'relu',
                 output_func: Optional[str] = None, 
                 leaky_relu: Optional[float] = 0.0, 
                 initializer: Optional[str] ='xavier',
                 bias: Optional[bool] = True, 
                 ):
        
        super(LinearNet, self).__init__()
        dim_lst = [input_dim] + [hidden_dim] * n_layers + ([out_dim] * bool(out_dim))
        self.out_dim = dim_lst[-1] 
        active_func = get_active_function(active_func, leaky_relu)
        self.output_func = get_active_function(output_func, leaky_relu)

        layers = []
        for i in range(len(dim_lst) - 2):
            layers += [nn.Linear(dim_lst[i], dim_lst[i + 1], bias=bias), active_func]
        layers += [nn.Linear(dim_lst[-2], dim_lst[-1], bias=bias)]
        self.layers = nn.Sequential(*nn.ModuleList(layers)) 
        # self._init_weights(get_initialiser(initializer))
    
    def forward(self, input):
        return self.output_func(self.layers(input))
    
    # def _init_weights(self, initializer):
    #     for layer in self.layers:
    #         if isinstance(layer, nn.Linear):
    #             initializer(layer.weight)
    #             if layer.bias is not None: 
    #                 nn.init.zeros_(layer.bias)


      
