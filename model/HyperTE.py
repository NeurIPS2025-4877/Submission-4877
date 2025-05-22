import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model.base import TreatmentFeatureExtractor, LinearNet
from torch_geometric.data.batch import Batch
from typing import Optional
from utils.misc import get_active_function, get_initialiser
from torch_scatter import scatter_mean


class HyperBaseNet(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, hidden_dim: int, out_dim: Optional[int] = None, output_func: Optional[str] = None):
        super(HyperBaseNet, self).__init__()
        self.input_dim = input_dim
        self.dim_lst = [input_dim] + [hidden_dim] * (n_layers - bool(out_dim)) + ([out_dim] * bool(out_dim))
        self.out_dim = self.dim_lst[-1]
        self.n_layers = len(self.dim_lst) - 1
        self.output_func = get_active_function(output_func)
        self.params_list = self.calc_n_params()
        self.n_params = sum(self.params_list)
        # print('dim_lst:', self.dim_lst)
        
    def calc_n_params(self):
        return [
            self.dim_lst[i] * self.dim_lst[i-1] + self.dim_lst[i]
            for i in range(1, len(self.dim_lst))
        ]
    
    def shapeWeights(self, weights):
        batch_size = weights.shape[0] 
        w, t = [], 0
        for i in range(1, len(self.dim_lst)):
            size = self.params_list[i-1]
            wt = weights[:, t:t+size]
            weight_matrix = wt[:, :-self.dim_lst[i]].reshape(batch_size, self.dim_lst[i-1], self.dim_lst[i])
            bias_vector = wt[:, -self.dim_lst[i]:]  
            w.append([weight_matrix, bias_vector])
            t += size
        return w
    
    def forward(self, x, weights):
        w = self.shapeWeights(weights)
        x = x.unsqueeze(1)
        for i in range(len(w) - 1):
            x = torch.bmm(x, w[i][0]) + w[i][1].unsqueeze(1)  
            x = F.relu(x)
        x = torch.bmm(x, w[-1][0]) + w[-1][1].unsqueeze(1)
        return self.output_func(x).squeeze(1)

    
class HyperTE(nn.Module):
    def __init__(self, config):
        super(HyperTE, self).__init__()
        hparam = config['hyper_params']
        self.x_input_dim = hparam['x_input_dim']
        self.alpha = 0.1
        self.treatment_net = TreatmentFeatureExtractor(
            config, 
        )  
        self.mu_layer = nn.Linear(hparam['drug_n_dims'], hparam['drug_n_dims'])
        self.logvar_layer = nn.Linear(hparam['drug_n_dims'], hparam['drug_n_dims'])    
                
        self.feature_net = HyperBaseNet(
            self.x_input_dim, hparam['feat_n_layers'], hparam['feat_n_dims'], output_func='relu'
            )

        self.hyper_net = LinearNet(hparam['drug_n_dims'], 
                                    hparam['drug_n_layers'], 
                                    hparam['drug_n_dims'], 
                                    out_dim = self.feature_net.n_params
                                    )
        
        self.attn_net = nn.Sequential(
                nn.Linear(self.feature_net.dim_lst[-1]+self.x_input_dim, 1), nn.Sigmoid(),
                )        

        self.outcome_net_m = LinearNet(self.feature_net.dim_lst[-1], 
                            hparam['pred_n_layers'], 
                            hparam['pred_n_dims'], 
                            out_dim = 1,
                            output_func=None
                            )                    

        self.outcome_net_x = LinearNet(self.x_input_dim, 
                            hparam['pred_n_layers'], 
                            hparam['pred_n_dims'], 
                            out_dim = 1,
                            output_func=None
                            )   

        self._init_weights_vae(get_initialiser('xavier'))


        params_x = (
            list(self.outcome_net_m.parameters()) +
            list(self.outcome_net_x.parameters()) +
            list(self.attn_net.parameters()) 
        )
        
        self.optimizer_x = config.init_obj('optimizer', torch.optim, params_x)
        
        params_m = (
            list(self.treatment_net.parameters()) +
            list(self.mu_layer.parameters()) +
            list(self.logvar_layer.parameters()) +
            list(self.hyper_net.parameters())
        )
        self.optimizer_m = config.init_obj('optimizer', torch.optim, params_m)

    def _init_weights_vae(self, initializer):
        initializer(self.mu_layer.weight)
        initializer(self.logvar_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.constant_(self.logvar_layer.bias, -1.0)
        
    def loss_func(self, y_pred: Tensor, batch: Batch):
        return F.mse_loss(y_pred, batch.y)

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def reparameterization(self, mu, logvar, is_inference):
        if is_inference:
            return mu
        std = torch.exp(0.5 * logvar)       
        z = mu + std * torch.randn_like(std).to(mu.device)                        
        return z

    def encode(self, batch: Batch):
        z = self.treatment_net(batch)
        mu, logvar = self.mu_layer(z), self.logvar_layer(z)  
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar 

    def generate_weights(self, batch: Batch, is_inference: Optional[bool]):
        mu, logvar = self.encode(batch)
        z = self.reparameterization(mu, logvar, is_inference)
        weights = self.hyper_net(z)
        kl_loss = self.kl_divergence(mu, logvar)
        return weights, kl_loss
    
    def forward(self, batch: Batch, is_inference: Optional[bool] = False):
        x = batch.covariates
        weights, kl_loss = self.generate_weights(batch, is_inference)
        m = self.feature_net(x, weights)
        alpha = self.attn_net(torch.cat([m, x], dim=1))
        y_m, y_x = self.outcome_net_m(m), self.outcome_net_x(x)
        
        loss_m = self.loss_func(y_m, batch)
        loss_x = self.loss_func(y_x, batch)

        y_pred = alpha * y_m + (1 - alpha) * y_x    
        attn_loss = self.loss_func(y_pred, batch)  
        return y_pred, kl_loss, loss_m, loss_x, attn_loss

    def predict(self, batch: Batch):
        y_pred, kl_loss, loss_m, loss_x, attn_loss = self.forward(batch)
        loss = self.loss_func(y_pred, batch)

        loss_dict = {
            'total_loss': loss_m + loss_x + attn_loss,
            'm_pred_loss': loss + self.alpha * kl_loss,
            'x_pred_loss': loss_m + loss_x + attn_loss,
            'kl_loss': kl_loss,
        }
        return loss_dict, y_pred

    def update(self, batch: Batch):
        for _ in range(10):
            loss, y_pred = self.predict(batch)
            self.optimizer_x.zero_grad()
            loss['x_pred_loss'].backward()
            self.optimizer_x.step()
            
        for _ in range(1):
            loss, y_pred = self.predict(batch)
            self.optimizer_m.zero_grad()
            loss['m_pred_loss'].backward()
            self.optimizer_m.step()

        return loss, y_pred
    
    def test_predict(self, batch: Batch):
        x = batch.covariates
        weights, kl_loss = self.generate_weights(batch, is_inference=True)
        m = self.feature_net(x, weights)
        alpha = self.attn_net(torch.cat([m, x], dim=1))
        y_m, y_x = self.outcome_net_m(m), self.outcome_net_x(x)
        y_pred = alpha * y_m + (1 - alpha) * y_x 
        return y_pred.view(-1)
    
 