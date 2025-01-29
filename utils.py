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

from torch_geometric.utils import sort_edge_index, degree
from torch_geometric.data import Data

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# visualize node embeddings
def visualize(feat_map, color, data_name, num_layers, id):
    z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set1")
    if id != -1:
        plt.savefig(os.getcwd() + "/visuals/" + data_name + "_embedding_" + str(num_layers) + "_file_" + str(id) + "_.png")
    else: 
        plt.savefig(os.getcwd() + "/visuals/" + data_name + "_embedding_" + str(num_layers) + "_.png")
    plt.clf()


# loss function
def loss_fn(pred, label):
    return F.nll_loss(pred, label)
    # return F.binary_cross_entropy_with_logits(pred, label.squeeze(0))


def mask_generation(index, num_nodes):
    # mask = torch.zeros(num_nodes, dtype = torch.bool)
    # mask[index] = 1
    return index


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.unsqueeze(1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    # print(y_true.shape, "   ", y_pred.shape)

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    rocauc_list = []
    y_true = y_true.unsqueeze(1)
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


# function for visualizing rewired graph
def visualize_rewired_graphs(edge_index, edge_weights, num_nodes, data_name, num_layers, id):
    
    num_edges = edge_index.shape[1]
    G = nx.Graph()
    node_list = [i for i in range(num_nodes)]
    edge_list = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(num_edges)]
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=num_nodes)

    # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list, width=edge_weights, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=3, font_family="sans-serif")
    # edge weight labels
    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    
    if id != -1:
        plt.savefig(os.getcwd() + "/rewired/" + data_name + "_rewired_graph_" + str(num_layers) + "_file_" + str(id) + "_.png")
    else: 
        plt.savefig(os.getcwd() + "/rewired/" + data_name + "_rewired_graph_" + str(num_layers) + "_.png")
    plt.clf()
    
    plt.show()

    
def feature_class_relation(edge_index, node_labels, feature_matrix):
    edge_index = edge_index.detach().cpu()
    node_labels = node_labels.detach().cpu()
    feature_matrix = feature_matrix.detach().cpu()

    row, col = edge_index
    row_feats = feature_matrix[row]
    col_feats = feature_matrix[col]
    
    row_labels = node_labels[row]
    col_labels = node_labels[col]

    homo_edge_count = (row_labels == col_labels).sum().int()
    hetero_edge_count = (row_labels != col_labels).sum().int()
    
    # edge_mask = torch.zeros(len(row))
    # edge_mask[row_labels == col_labels] = 1
    # edge_mask[row_labels != col_labels] = 1
   
    similarity_scores = (row_feats * col_feats).sum(dim = 1)
    # print(similarity_scores.device, "    ", edge_mask.device)
    # similarity_scores = torch.pow((row_feats - col_feats), 2).sum(dim = 1)
    # similarity_scores *= edge_mask
    total_scores = similarity_scores.sum()

    return (total_scores / (homo_edge_count + hetero_edge_count))


def degree_distribution(edge_index, num_nodes, dataname):
    node_degrees = degree(edge_index[0], num_nodes = num_nodes)
    print("Isolated nodes ", torch.sum(torch.eq(node_degrees, 0)).item())
    node_degrees = node_degrees.numpy().astype(int)
    plt.hist(node_degrees, bins=range(min(node_degrees), max(node_degrees)), alpha=0.75, color='skyblue', edgecolor='black')
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(os.getcwd() + "/plots/" + "degree_dist_" + dataname + "_.png")
    plt.close()


def plot_details(dataname):
    prob_list = np.load(os.getcwd() + "/plots/" + dataname + "_prob_list.npy")
    de_list = np.load(os.getcwd() + "/plots/" + dataname + "_de_list.npy")
    edge_ratio_list = np.load(os.getcwd() + "/plots/" + dataname + "_edge_ratio_list.npy")

    fig, ax = plt.subplots(3, 1)
    train_iter = len(prob_list)

    ax[0].plot([i for i in range(train_iter)], prob_list, color='red', label='edge threshold')
    ax[0].set_xlabel('Epochs', fontsize=10)
    # plt.ylabel('')
    # plt.legend()
    ax[0].grid(True)
    ax[0].set_title('Edge threshold', fontsize=10)
    # plt.savefig(os.getcwd() + "/plots/" + "prob_list_" + dataname + "_.png")
    # plt.close()

    ax[1].plot([i for i in range(train_iter)], de_list, color='blue', label='Dirichlet energy')
    ax[1].set_xlabel('Epochs', fontsize=10)
    # plt.legend()
    ax[1].grid(True)
    ax[1].set_title('Dirichlet energy', fontsize=10)
    # plt.savefig(os.getcwd() + "/plots/" + "de_list_" + dataname + "_.png")
    # plt.close()

    ax[2].plot([i for i in range(train_iter)], edge_ratio_list, color='green', label='edge ratio')
    ax[2].set_xlabel('Epochs', fontsize=10)
    # plt.legend()
    ax[2].grid(True)
    ax[2].set_title('Edge ratio', fontsize=10)


    plt.savefig(os.getcwd() + "/plots/" + "rewiring_" + dataname + "_.png")
    plt.close()


# to remove homophilous or heterophilous edges in the graph
def edge_filtering(edge_index, y, homophily):
    
    src_labels = y[edge_index[0]]
    tgt_labels = y[edge_index[1]]
    edge_mask = torch.zeros(len(edge_index[0]), dtype=torch.bool).to(edge_index.device)
    if homophily == 'True':
        edge_mask[src_labels == tgt_labels] = 1
        print("homophily")
    else:
        edge_mask[src_labels != tgt_labels] = 1
        print("heterophily")
        
    edge_index = edge_index[:, edge_mask] 
    return edge_index


def find_edge_labels(edge_index, node_labels):
    source_labels = node_labels[edge_index[0]]
    target_labels = node_labels[edge_index[1]]
    comparison = (source_labels == target_labels)
    edge_labels = comparison.int()
    return edge_labels
    
    
def dirichlet_energy(X, edge_index):
    # computes Dirichlet energy of a vector field X with respect to a graph with a given edge index
    n = X.shape[0]
    m = len(edge_index[0])
    l = X.shape[1]
    degrees = np.zeros(n)
    for I in range(m):
        u = edge_index[0][I]
        degrees[u] += 1
    y = np.linalg.norm(X.flatten()) ** 2
    for I in range(m):
        for i in range(l):
            u = edge_index[0][I]
            v = edge_index[1][I]
            y -= X[u][i] * X[v][i] / (degrees[u] * degrees[v]) ** 0.5
    return y

def dirichlet_normalized(X, edge_index):
    energy = dirichlet_energy(X, edge_index)
    norm_squared = sum(sum(X ** 2))
    return energy / norm_squared



# plot for signal propagation
def smooth_plot(x, y=None, ax=None, label='', halflife=10):

    """ 
    Function to plot smoothed x VS y graphs.

    Parameters:	
    ----------

    x - x-axis data.
    y - y-axis data.
    label - Legend for the current graph.
    halflife - Smoothing level.

    ----------

    Yields:	x VS y graphs.

    ----------
    """

    if y is None:
      y_int = x
    else:
      y_int = y
    
    x_ewm = pd.Series(y_int).ewm(halflife=halflife)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    if y is None:
      ax.plot(x_ewm.mean(), label=label, color=color)
      ax.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.15)
    else:
      ax.plot(x, x_ewm.mean(), label=label, color=color)
      ax.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.15)


def get_resistances(graph, reference_node):
    resistances = {}
    
    for node in graph.nodes:
        if node != reference_node:
            print("node ", node)
            resistances[node] = round(nx.resistance_distance(graph, reference_node, node),3)
        else:
            resistances[node] = 0
    return resistances