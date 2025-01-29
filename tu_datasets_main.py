import torch
import torch.nn as nn 
import numpy as np
import os
import sys
import random 
from tqdm import tqdm

from torch_geometric.utils import degree, remove_self_loops, to_dense_adj, to_dense_batch, dense_to_sparse

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.nn import Parameter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from statistics import mean
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tu_datasets_loader import *
from model_graph_classification import *
from utils import *
from custom_parser import argument_parser

    
# torch.autograd.set_detect_anomaly(True)

# train function
def train(train_loader, val_loader, model, opti_train, scheduler, train_iter, model_path, device):

    best_val_acc = 0.0
    de_list = []
    pbar = tqdm(total = train_iter)
    # pbar.set_description("Training")
    for i in range(train_iter):
        total_train_loss = 0.0
        total_val_loss = 0.0
        counter = 0
        for _, data in enumerate(train_loader):
            # print("Data ", data)
            model.train()
            opti_train.zero_grad()
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            emb, pred = model(data.x, data.edge_index, data.batch, data.ptr, data.centrality)
            data.y = data.y.to(device)
            loss_train = loss_fn(pred, data.y) 
            total_train_loss += loss_train
            loss_train.backward()
            opti_train.step()
            counter += 1
        
        scheduler.step(total_train_loss)
        avg_train_loss = total_train_loss / counter
        val_acc = test(model, val_loader)
        # print(f'Iteration: {i:02d} || Train loss: {avg_train_loss.item(): .4f} || Valid Acc: {val_acc: .4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
        pbar.update(1)
        pbar.set_postfix({"step": i, "train loss": avg_train_loss.item(), "valid acc": val_acc})
    pbar.close()


# test function
@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        emb, pred = model(data.x, data.edge_index, data.batch, data.ptr, data.centrality)
        pred = pred.argmax(dim = 1)
        data.y = data.y.to(pred.device)
        correct += int((pred == data.y).sum())
    acc = correct / len(loader.dataset)
    return acc


# choice of centrality
def centrality_measure(edge_index, num_nodes, centrality):
    pyg_g = Data(edge_index = edge_index, num_nodes = num_nodes)
    nx_G = to_networkx(pyg_g)
    if centrality == 'degree':
        node_degrees = degree(edge_index[0], num_nodes = num_nodes)
        sorted_node_degree, sorted_indices = torch.sort(node_degrees, descending = True)
        return sorted_indices
    elif centrality == 'betweenness':
        centrality = nx.betweenness_centrality(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    elif centrality == 'pagerank':
        centrality = nx.pagerank(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    elif centrality == 'load':
        centrality = nx.load_centrality(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    elif centrality == 'closeness':
        centrality = nx.closeness_centrality(nx_G)
        centrality = torch.tensor(list(centrality.values()))
        centrality = torch.argsort(centrality, descending = True)
        return centrality
    else:
        print("Incorrect centrality...")



parsed_args = argument_parser().parse_args()

print(parsed_args)

dataname = parsed_args.dataset
train_lr = parsed_args.train_lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
mlp_layers = parsed_args.mlp_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
dropout = parsed_args.dropout
train_weight_decay = parsed_args.train_w_decay
num_splits = parsed_args.num_splits
batch_size = parsed_args.batch_size
batch_frac = parsed_args.batch_frac
model_name = parsed_args.model 
centrality = parsed_args.centrality
device = parsed_args.device


# setting seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print("Fraction of Nodes ", batch_frac)

train_split = 0.80
val_split = 0.10
test_split = 0.10
graph_obj = TUDatasetLoader(dataname)
num_graphs = len(graph_obj.graphs)
train_graphs = int(num_graphs * train_split)
valid_graphs = int(num_graphs * val_split)
test_graphs = num_graphs - (train_graphs + valid_graphs)



# adding centrality
for graph in graph_obj.graphs:
    graph.centrality = centrality_measure(graph.edge_index, graph.num_nodes, centrality=centrality)
  
test_acc_list = []
for fid in range(num_splits):
    print("----------------- Split " + str(fid) + " -------------------------")
    
    model_path = 'best_gnn_model_' + dataname + '_.pt'

    # load split, get indices, select graphs
    # path = os.getcwd() + "/splits/" + dataname + "/" + dataname + "_" + str(fid) + ".npz"
    # f = np.load(path)
    # train_indices, val_indices, test_indices = f['arr1'], f['arr2'], f['arr3']
    # train_graphs = torch.utils.data.Subset(graph_obj.graphs, train_indices)
    # val_graphs = torch.utils.data.Subset(graph_obj.graphs, val_indices)
    # test_graphs = torch.utils.data.Subset(graph_obj.graphs, test_indices)
    
    train_graphs, val_graphs, test_graphs = graph_obj.load_random_splits(graph_obj.graphs, train_split, val_split)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle = True)

    # for i, data in enumerate(train_loader):
    #     print(data)
    #     break
   
    model = MolGNN(graph_obj, model_name, num_layers, mlp_layers, graph_obj.num_features, hidden_dim, dropout, batch_size, batch_frac, centrality, device)
    model = model.to(device)

    opti_train = torch.optim.Adam(model.parameters(), lr=train_lr, weight_decay=train_weight_decay)
    scheduler = ReduceLROnPlateau(opti_train)

    train(train_loader, val_loader, model, opti_train, scheduler, train_iter, model_path, device)

    print("=" * 30)
    print("Model Testing....")

    avg_test_acc = 0.0
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # print("Learned threshold: ", model.prob)

    for _ in range(test_iter):
        test_acc = test(model, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")
        avg_test_acc += test_acc

    avg_test_acc = avg_test_acc / test_iter 
    print(f"Average Test Accuracy: {(avg_test_acc * 100):.4f}")

    test_acc_list.append(avg_test_acc)
    # emb, pred, dir_energy,  _, _ = model(data.node_features, adj_matrix)
    # visualize(emb, data.node_labels.detach().cpu().numpy(), dataset, num_layers, fid+1)
    print("---------------------------------------------\n")
    # break
    
test_acc_list = torch.tensor(test_acc_list)
print(f'Final Test Statistics: {100 * test_acc_list.mean():.2f} +- {100 * test_acc_list.std():.2f}')

