import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from scipy.sparse import coo_matrix
import numpy as np

from model import *


dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# graph_path = './embeddings/mitstates-graph.t7'
# graph_path = './embeddings/graph_cw.pt'
graph_path = './embeddings/graph_op.pt'
weighted_graph_path = './embeddings/graph_op_weighted.pt'

class GraphModel(nn.Module):
    def __init__(self, dset, train_only=True, static_emb=True, weighted_graph=False):
        super(GraphModel, self).__init__()
        self.nattrs, self.nobjs = len(dset.attrs), len(dset.objs)
        self.train_only = train_only
        if self.train_only:
          train_idx = []
          for current in dset.train_pairs:
            train_idx.append(dset.all_pair2idx[current]+self.nattrs+self.nobjs)
          self.train_idx = torch.LongTensor(train_idx).to(dev)

        if weighted_graph and dset.open_world:
          graph = torch.load(weighted_graph_path)
        else:
          graph = torch.load(graph_path)
        
        self.nodes = graph["embeddings"].to(dev)
        self.node_dim = self.nodes.size(1)
        if not static_emb:
          self.nodes = nn.Parameter(self.nodes)
        adj = graph["adj"]
        row_idx, col_idx = adj.row, adj.col
        self.edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long).to(dev)
        
        self.shared_emb_dim = 512
        
        hidden_layer_sizes = [2048, 2048]
        hidden_layers = []
        in_features = self.node_dim
        for hidden_size in hidden_layer_sizes:
          layer = [
            (gnn.GCNConv(in_features, hidden_size), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Dropout()
          ]
          hidden_layers.extend(layer)
          in_features = hidden_size
        hidden_layers.append((gnn.GCNConv(in_features, self.shared_emb_dim), 'x, edge_index -> x'))
        self.gcn = gnn.Sequential('x, edge_index', hidden_layers)

    def forward(self, x):
      img = x[4].to(dev)
      img_feats = (img)

      current_embeddings = self.gcn(self.nodes, self.edge_index)

      if self.train_only and self.training:
        pair_embed = current_embeddings[self.train_idx]
      else:
        pair_embed = current_embeddings[self.nattrs+self.nobjs:,:]

      pair_pred = torch.matmul(img_feats, pair_embed.T)
      return pair_pred
    
class GraphMLP(nn.Module):
    def __init__(self, dset, static_emb=True, weighted_graph=False):
        super(GraphMLP, self).__init__()
        self.nattrs, self.nobjs = len(dset.attrs), len(dset.objs)

        if weighted_graph and dset.open_world:
          graph = torch.load(weighted_graph_path)
        else:
          graph = torch.load(graph_path)
        
        self.nodes = graph["embeddings"][:self.nattrs+self.nobjs].to(dev)
        if not static_emb:
          self.nodes = nn.Parameter(self.nodes)
        adj = graph["adj"].todense()
        adj = adj[:self.nattrs+self.nobjs, :self.nattrs+self.nobjs]
        adj = coo_matrix(adj)
        row_idx, col_idx = adj.row, adj.col
        self.edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long).to(dev)
        
        self.node_dim = 512
        
        hidden_layer_sizes = [3000]
        hidden_layers = []
        in_features = self.nodes.size(1)
        for hidden_size in hidden_layer_sizes:
          layer = [
            (gnn.SAGEConv(in_features, hidden_size), 'x, edge_index -> x'),
            nn.ReLU(),
            nn.Dropout()
          ]
          hidden_layers.extend(layer)
          in_features = hidden_size
        hidden_layers.append((gnn.SAGEConv(in_features, self.node_dim), 'x, edge_index -> x'))
        self.gcn = gnn.Sequential('x, edge_index', hidden_layers)
          
        self.img_feat_dim = dset.feat_dim
        self.shared_emb_dim = 800
        self.img_fc = ParametricMLP(self.img_feat_dim, self.shared_emb_dim, [768, 1000], norm_output=True)
        self.pair_fc = ParametricMLP(self.node_dim*2, self.shared_emb_dim, [1000], norm_output=True)
        
    def get_all_pairs(self, nodes):
      attr_nodes = nodes[:self.nattrs]
      obj_nodes = nodes[self.nattrs:]
      all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.node_dim)
      all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      return all_pairs

    def forward(self, x):
      img = x[4].to(dev)
      img_feats = self.img_fc(img)
      nodes = self.gcn(self.nodes, self.edge_index)
      all_pair_nodes = self.get_all_pairs(nodes)
      all_pairs = self.pair_fc(all_pair_nodes)
      pair_pred = torch.matmul(img_feats, all_pairs.T)
      return pair_pred