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
                 params, 
                 ):
        super(TreatmentFeatureExtractor, self).__init__()
        hparam = config['hyper_params']
        d_input_dim = hparam['d_input_dim']
        self.is_graph = (config['data_loader']['data_type'] == 'graph')
        is_hyperTE = hparam['model'].startswith('HyperTE')
        
        if not self.is_graph:
            self.treatment_net = LinearNet(d_input_dim, params.drug_n_layers, params.drug_n_dims, 
                                           out_dim=params.drug_n_dims, output_func=hparam['activation'])
        else:
            self.treatment_net = GNN( 
                gnn_conv=hparam['gnn_conv'],
                dim_input=d_input_dim,
                dim_hidden=hparam['dim_hidden_treatment'],
                dim_output=params.drug_n_dims,
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


      
class BaseModel_NonTreatment(nn.Module):
    def __init__(self, config, params, outcome_in_dims=None):
        super(BaseModel_NonTreatment, self).__init__()
        hparam = config['hyper_params']
        self.x_input_dim = hparam['x_input_dim']
        self.d_input_dim = hparam['d_input_dim']
        self.is_single_outcome = ('lincs' in params.data.lower())

        self.feature_net = LinearNet(self.x_input_dim, params.feat_n_layers, params.feat_n_dims, output_func='relu')
        
        if not outcome_in_dims:
            outcome_in_dims = params.feat_n_dims
        outcome_net = LinearNet(outcome_in_dims, params.pred_n_layers, params.pred_n_dims, out_dim=1)
        if self.is_single_outcome:
            self.outcome_net = nn.ModuleList([deepcopy(outcome_net)])
        else:
            self.outcome_net = nn.ModuleList([deepcopy(outcome_net), deepcopy(outcome_net)])
        del outcome_net
        self.propensity = LinearNet(outcome_in_dims, params.pred_n_layers, params.pred_n_dims, 
                                    out_dim=1, output_func='sigmoid')

    def baseloss(self, batch, y_pred, t_pred, alpha=1.0):
        loss_t = torch.sum(F.binary_cross_entropy(t_pred, batch.t))
        if self.is_single_outcome:
            loss_y = torch.sum(torch.square(batch.y - y_pred))
        else:
            y0_pred, y1_pred = y_pred[:,0], y_pred[:,1]
            loss_y = torch.sum((1. - batch.t) * torch.square(batch.y - y0_pred)) +\
                torch.sum(batch.t * torch.square(batch.y - y1_pred)) 
        return loss_y + alpha * loss_t
    
    def loss_func(self, batch, y_pred, t_pred):
        return self.baseloss(batch, y_pred, t_pred) 
    
    def predict(self, batch: Batch):
        y_pred, t_pred = self.forward(batch)
        loss = self.loss_func(batch, y_pred, t_pred)
        
        loss_dict = {
            'total_loss': loss,
        }
        return loss_dict, y_pred

    def update(self, batch: Batch):
        loss, y_pred = self.predict(batch)
        self.optimizer.zero_grad()
        loss['total_loss'].backward()
        self.optimizer.step()
        return loss, y_pred
    
    def forward_outcome(self, input):
        y_preds = [net(input) for net in self.outcome_net]
        return torch.cat(y_preds, dim=1)
    
    def _last_sequence(self, input, lengths):
        return input[np.arange(len(input)), lengths-1]
    
# class HyperNet(nn.Module):
#     def __init__(self, input_dim, n_layers, hidden_dim, *dims):
#         super(HyperNet, self).__init__()
#         self.dim_sizes = [sum([dims[i][j] * dims[i][j + 1] for j in range(len(dims[i]) - 1)]) for i in range(len(dims))]       
#         self.weights = LinearNet(input_dim, n_layers, hidden_dim, out_dim=sum(self.dim_sizes))
        
#     def forward(self, input):
#         weights = self.weights(input)
#         split_weights = []
#         start = 0
#         for dim_size in self.dim_sizes:
#             split_weights.append(weights[:, start:start + dim_size])
#             start += dim_size
#         return split_weights
    
