import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import random
from torch_geometric.data import Data


'''
COLLAB, IMDB-BINARY, REDDIT-BINARY do not have node features
'''

class TUDatasetLoader():
    def __init__(self, dataname):
        
        '''
        self.graphs contains the list of all input graphs
        '''
        self.graphs = TUDataset(root='TUDatasets/', name = dataname, transform = None, pre_transform = None, pre_filter = None, use_node_attr = True, use_edge_attr = False, cleaned = False)
        # print(type(self.graphs))
        self.num_features = self.graphs.num_features
        self.num_classes = self.graphs.num_classes
        self.graphs = list(self.graphs)
        
        if dataname == 'COLLAB' or dataname == 'IMDB-BINARY' or dataname == 'REDDIT-BINARY':
            n_feats = 1
            self.num_features = n_feats
            self.pre_process(self.graphs, n_feats=n_feats)
            print("Feature pre-processing done.....")
           
        
    def pre_process(self, graphs, n_feats):
        for g in graphs:
            # print(g)
            n = g.num_nodes
            node_degrees = degree(g.edge_index[0], num_nodes=g.num_nodes)
            g.x = torch.ones((n, n_feats))
            # print(node_degrees.shape)
            # g.x = node_degrees.unsqueeze(1)
            
    def load_random_splits(self, graphs, train_split, val_split):

        num_graphs = len(graphs)
        indices = [i for i in range(num_graphs)]
        random.shuffle(indices)
        # print(indices)
        num_train = int(train_split*num_graphs)
        num_val = int(val_split*num_graphs)
        num_test = num_graphs - (num_train + num_val)
        train_indices = indices[:num_train]
        val_indices = indices[num_train : num_train+num_val]
        test_indices = indices[num_train+num_val:]

        train_graphs = torch.utils.data.Subset(graphs, train_indices)
        val_graphs = torch.utils.data.Subset(graphs, val_indices)
        test_graphs = torch.utils.data.Subset(graphs, test_indices)
        
        # print(test_indices)
        # train_graphs = self.graphs[train_indices]
        # val_graphs = self.graphs[val_indices]
        # test_graphs = self.graphs[test_indices]
        
        return train_graphs, val_graphs, test_graphs
        
        
        
# datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}
# #datasets = {"proteins": proteins, "collab": collab}
# for key in datasets:
#     if key in ["reddit", "imdb", "collab"]:
#         for graph in datasets[key]:
#             n = graph.num_nodes
#             graph.x = torch.ones((n,1))


# MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY


# dataname = 'COLLAB'
# print(dataname)
# graph_obj = TUDatasetLoader(dataname)




# total_nodes = 0
# total_edges = 0
# count = 0
# for i, g in enumerate(graph_obj.graphs):
#     print(i, g)
#     if dataname == 'COLLAB' or 'IMDB-BINARY' or 'REDDIT-BINARY':
#         total_nodes += g.num_nodes
#     else:
#         total_nodes += g.x.shape[0]
#     total_edges += g.edge_index.shape[1]
#     count += 1
# avg_nodes = total_nodes / count 
# avg_edges = total_edges / count
# print(f'Avg #nodes: {avg_nodes} || Avg #edges : {avg_edges}')
    
# train_graphs, val_graphs, test_graphs = graph_obj.load_random_splits(0.80, 0.10)
# print(len(train_graphs), "  ", len(val_graphs), "  ", len(test_graphs))

# for i in range(len(test_graphs)):
#     print(i, "  ", test_graphs[i])

