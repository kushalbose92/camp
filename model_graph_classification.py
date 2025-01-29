import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx
import sys

from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
# from ginconv import GINConv
from torch_geometric.utils import degree, to_networkx
# from torch.nn.parameter import Parameter
import torch_geometric.transforms as T 
from torch_geometric.data import Data

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp

# torch.autograd.set_detect_anomaly(True)


class RGATConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GATConv(in_features, out_features))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

# defining gnn models
class MolGNN(nn.Module):

    def __init__(self, dataset, model, num_layers, mlp_layers, input_dim, hidden_dim, dropout, batch_size, batch_frac, centrality, device):
        super(MolGNN, self).__init__()

        self.num_layers = num_layers
        self.dataset = dataset
        self.model = model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.gnn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self.batch_frac = batch_frac
        self.centrality = centrality
        
        # classifier layer
        self.cls = nn.Linear(self.hidden_dim, self.dataset.num_classes)

        # init layer
        self.init = nn.Linear(self.input_dim, self.hidden_dim)
        
        if model == 'gcn':
            # print("Using GCN model...")
            for i in range(self.num_layers):
                if i == self.num_layers - 1:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                else:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                    self.lns.append(nn.LayerNorm(hidden_dim))
        elif model == 'gin':
            # print("Using GIN model...")
            for i in range(num_layers):
                nn_obj = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.BatchNorm1d(self.hidden_dim),   
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim))
                self.gnn_convs.append(GINConv(nn_obj, eps=0.1, train_eps=False).to(self.device))
                self.lns.append(nn.LayerNorm(hidden_dim))
        else:
            print("Invalid model name...")

        
    def forward(self, x, edge_index, batch, ptr, centrality):
        x = x.to(self.device)
        x = self.init(x)
        num_nodes = x.shape[0]
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        edge_batch = batch[edge_index[0]]
        # print(batch)
        _, node_counts = torch.unique(batch, return_counts = True)
        # print(node_counts)
        # print(x.shape, "   ", edge_index.shape)

        # generation of batch of masks for the graphs in the batch --- only for RAMP
        # batch_mask = self.mask_generation_ramp(node_counts, self.batch_frac)
        # print("before ", batch_mask)

        prev_embed = x
        for i in range(self.num_layers):
            x = self.gnn_convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)

            if self.batch_frac != 0:
                batch_mask = self.mask_generation_centrality(node_counts, ptr, centrality, i)
                # batch_mask = self.mask_generation_ramp(node_counts, self.batch_frac)
                x, prev_embed = self.async_update(batch_mask, x, prev_embed)

        # applying mean pooling
        x = global_mean_pool(x, batch)

        # x = F.dropout(x, p=self.dropout, training=True)
        x = self.cls(x)

        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x

    '''
    Randomized Asynchronous Message Passing --- common function for both 
    RAMP and CAMP
    '''
    def rand_async_msg_passing(self, x, prev_embed, batch_mask):
        update_mask = torch.from_numpy(batch_mask).to(self.device)
        update_mask = update_mask.unsqueeze(1)
        x = torch.where(update_mask == 1, x, prev_embed)
        prev_embed = x
        return x, prev_embed

     
    # For RAMP -------------

    # shuffling node indices
    # def batch_shuffle(self, batch_mask):
    #     if isinstance(batch_mask, np.ndarray):
    #         num_graphs = batch_mask.shape[0]
    #     else:
    #         num_graphs = len(batch_mask)
    #     for g in range(num_graphs):
    #         np.random.shuffle(batch_mask[g])
    #     return batch_mask

    # # generating mask for the selecting nodes
    # def mask_generation_ramp(self, node_counts, batch_frac):
    #     num_graphs = node_counts.shape[0]
    #     batch_mask = []
    #     for g in range(num_graphs):
    #         num_target_nodes = math.floor(node_counts[g] * batch_frac)
    #         # print(num_target_nodes)
    #         ones_ = np.ones(num_target_nodes)
    #         zeros_ = np.zeros(node_counts[g] - num_target_nodes)
    #         node_idx_g = np.concatenate([ones_, zeros_], axis = 0)
    #         # np.random.shuffle(node_idx_g)
    #         batch_mask.append(node_idx_g)
    #     return batch_mask

    # def async_update(self, batch_mask, x, prev_embed):
    #     batch_mask = self.batch_shuffle(batch_mask)
    #     # print("before ", batch_mask)
    #     index_mask = np.hstack(batch_mask)
    #     # print("after ", index_mask)
    #     x, prev_embed = self.rand_async_msg_passing(x, prev_embed, index_mask)
    #     return x, prev_embed
    
    
    #  For CAMP ----
    
    def async_update(self, batch_mask, x, prev_embed):
        index_mask = np.hstack(batch_mask)
        x, prev_embed = self.rand_async_msg_passing(x, prev_embed, index_mask)
        return x, prev_embed
    
    # generate mask based on centrality for a single graph 
    def mask_based_on_centrality(self, centrality, num_nodes, layer):
        sorted_indices = centrality
        chunk_size = math.floor(num_nodes / self.num_layers)
        if layer == self.num_layers - 1:
            candidate_nodes = sorted_indices[layer*chunk_size:num_nodes]
        else:
            candidate_nodes = sorted_indices[layer*chunk_size:(layer+1)*chunk_size]
        sample_size = min(math.floor(self.batch_frac*len(candidate_nodes)), len(candidate_nodes))
        rand_indices = torch.randperm(candidate_nodes.shape[0])
        selected_nodes = candidate_nodes[rand_indices[:sample_size]]
        node_idx = torch.zeros(num_nodes).to(self.device)
        node_idx[selected_nodes] = 1
        node_idx = node_idx.detach().cpu().numpy()
        return node_idx

    def mask_generation_centrality(self, node_counts, ptr, centrality, layer):
        batch_mask = []
        num_graphs = node_counts.shape[0]
        for g in range(num_graphs):
            start, end = int(ptr[g]), int(ptr[g+1])
            centrality_values = centrality[start:end]
            graph_wise_mask = self.mask_based_on_centrality(centrality_values, node_counts[g], layer)
            batch_mask.append(graph_wise_mask)
        return batch_mask
            

'''
centrality --- node degree / betweeness / closeness / pagrank / load
nodes with higher degree are highly likely to cause oversquashing. 
nodes with higher centrality should communicate in the earlier layers than the nodes with lower centrality 
'''