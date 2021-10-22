import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from scipy.sparse import coo_matrix
import numpy as np

from model import *
from gcn import CGE

dev = 'cuda' if torch.cuda.is_available() else 'cpu'



class GraphModelBase(nn.Module):
  def __init__(self, hparam, dset, graph_path, train_only=False, resnet_name=None, static_inp=True):
    super(GraphModelBase, self).__init__()
    self.hparam = hparam
    self.nattrs, self.nobjs = len(dset.attrs), len(dset.objs)
    self.img_feat_dim = dset.feat_dim

    if resnet_name:    
      import torchvision.models as models
      self.resnet = models.resnet18(pretrained=True).to(dev)
      if static_inp:
        self.resnet = frozen(self.resnet)
      self.img_feat_dim = self.resnet.fc.in_features
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
        
    self.train_only = train_only
    if self.train_only:
      train_idx = []
      for pair in dset.train_pairs:
        train_idx.append(dset.all_pair2idx[pair])
      self.train_idx = torch.LongTensor(train_idx).to(dev)

    graph = torch.load(graph_path)
    self.nodes = graph["embeddings"].to(dev)
    adj = graph["adj"]
    self.edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long).to(dev) # np.array purely for suppressing pytorch conversion warning



class GraphEncoder(nn.Module):
    def __init__(self, conv_layer, in_features, out_features, hidden_layer_sizes=[]):
      super(GraphEncoder, self).__init__()
      hidden_layers = []
      for hidden_size in hidden_layer_sizes:
        layer = [
          (conv_layer(in_features, hidden_size), 'x, edge_index -> x'),
          nn.ReLU(),
          nn.Dropout()
        ]
        hidden_layers.extend(layer)
        in_features = hidden_size
      hidden_layers.append((conv_layer(in_features, out_features), 'x, edge_index -> x'))

      self.encoder = gnn.Sequential('x, edge_index', hidden_layers)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)
      
      

class GraphModel(GraphModelBase):
  def __init__(self, dset, graph_path, train_only=True):
    super(GraphModel, self).__init__(dset, graph_path, train_only=train_only)

    self.shared_emb_dim = 512
    hidden_layer_sizes = [2048, 2048]
    self.gcn = GraphEncoder(gnn.GCNConv, self.nodes.size(1), self.shared_emb_dim, hidden_layer_sizes = [2048, 2048])
    
  def forward_cross_entropy(self, x):
    img = x[4].to(dev)
    img_feats = F.normalize(img)

    current_embeddings = F.normalize(self.gcn(self.nodes, self.edge_index))

    if self.train_only and self.training:
      pair_embed = current_embeddings[self.train_idx]
    else:
      pair_embed = current_embeddings[self.nattrs+self.nobjs:,:]

    pair_pred = torch.matmul(img_feats, pair_embed.T)
    return pair_pred

  def forward_triplet4(self, x):
    img = x[4].to(dev)
    pair_id = x[3]
    attr_id, obj_id = x[1], np.array(x[2])
    nsample = len(img)

    img_feats = F.normalize(img)

    current_embeddings = F.normalize(self.gcn(self.nodes, self.edge_index))
    pairs = current_embeddings[self.nattrs+self.nobjs:,:]

    dot_mat = img_feats @ pairs.T
    s_it = dot_mat[range(nsample), pair_id] # dot products of correct pairs of (img, label)
    negative = dot_mat[:, pair_id]
    s_nit = negative - s_it.view(1, -1)
    s_int = negative - s_it.view(-1, 1)

    return s_it, s_nit, s_int

  def forward(self, x):
    return self.forward_cross_entropy(x)
    if self.training:
      return self.forward_triplet4(x)
    else:
      return self.forward_cross_entropy(x)



class GraphMLP(GraphModelBase):
  def __init__(self, hparam, dset, graph_path=None, resnet_name=None, static_inp=True):
    super(GraphMLP, self).__init__(hparam, dset, graph_path, resnet_name=resnet_name, static_inp=static_inp)
    
    self.hparam.add_dict({'graph_encoder_layers': [2048], 'node_dim': 800})
    self.gcn = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.hparam.node_dim, hidden_layer_sizes = self.hparam.graph_encoder_layers)

    
    self.hparam.add('shared_emb_dim', 800)
    self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                          'pair_fc_layers': [1000], 'pair_fc_norm': True})
    self.img_fc = ParametricMLP(self.img_feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers,
                                norm_output=self.hparam.img_fc_norm)
    self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, 
                                 self.hparam.pair_fc_layers, norm_output=self.hparam.pair_fc_norm)

  def get_all_pairs(self, nodes):
    attr_nodes = nodes[:self.nattrs]
    obj_nodes = nodes[self.nattrs:]
    all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.hparam.node_dim)
    all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
    all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
    return all_pairs

  def forward_cross_entropy(self, x):
    if self.resnet:
      img = self.resnet(x[0].to(dev))
    else:
      img = x[4].to(dev)
    img_feats = self.img_fc(img)
    nodes = self.gcn(self.nodes, self.edge_index)
    all_pair_nodes = self.get_all_pairs(nodes)
    all_pairs = self.pair_fc(all_pair_nodes)
    pair_pred = torch.matmul(img_feats, all_pairs.T)
    
    return pair_pred

  def forward_metric_learning(self, x):
    if self.resnet:
      img = self.resnet(x[0].to(dev))
    else:
      img = x[4].to(dev)
    img_feats = self.img_fc(img)
    pair_id = x[3]
    nodes = self.gcn(self.nodes, self.edge_index)
    all_pair_nodes = self.get_all_pairs(nodes)
    all_pairs_nodes = self.pair_fc(all_pair_nodes)
    pair_pred = torch.matmul(img_feats, all_pairs_nodes.T)
    attr_pred = self.attr_classifier(img_feats)
    obj_pred = self.obj_classifier(img_feats)
    
    if self.training:
      return attr_pred, obj_pred, pair_pred, img_feats, all_pair_nodes[pair_id]
    else:
      return pair_pred
      
  def forward(self, x):
    return self.forward_cross_entropy(x)


def recon_loss(model, z):
  """Only consider valid edges (attr-obj) when calculating reconstruction loss"""
  from torch_geometric.utils import remove_self_loops, add_self_loops
  from utils import gae_negative_sampling as negative_sampling
  
  pos_edge_index = model.train_pair_edges
  pos_loss = -torch.log(
      model.gae.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

  # Do not include self-loops in negative samples

  pos_edge_index, _ = remove_self_loops(pos_edge_index)
  pos_edge_index, _ = add_self_loops(pos_edge_index)
  neg_edge_index = negative_sampling(pos_edge_index, model.nattrs, model.nobjs)
  neg_loss = -torch.log(1 -
                        model.gae.decoder(z, neg_edge_index, sigmoid=True) +
                        1e-15).mean()

  return pos_loss + neg_loss

  
class GAE(GraphModelBase):
  def __init__(self, hparam, dset, graph_path=None, train_only=False, resnet_name=None, pretrained_gae=None, pretrained_mlp=None):
    super(GAE, self).__init__(hparam, dset, graph_path, train_only=train_only, resnet_name=resnet_name)

    self.train_pair_edges = torch.zeros((2, len(dset.train_pairs)), dtype=torch.long).to(dev)
    for i, (attr, obj) in enumerate(dset.train_pairs):
      self.train_pair_edges[0, i] = dset.attr2idx[attr]
      self.train_pair_edges[1, i] = dset.obj2idx[obj]

    self.hparam.add_dict({'graph_encoder_layers': [2048], 'node_dim': 512})
    self.encoder = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.hparam.node_dim, self.hparam.graph_encoder_layers)
    self.gae = gnn.GAE(self.encoder)
    self.gae.recon_loss = recon_loss
    self.hparam.add('shared_emb_dim', 800)
    self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                          'pair_fc_layers': [1000], 'pair_fc_norm': True})
    self.img_fc = ParametricMLP(self.img_feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers,
                                norm_output=self.hparam.img_fc_norm)
    self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, 
                                 self.hparam.pair_fc_layers, norm_output=self.hparam.pair_fc_norm)

    self.hparam.add_dict({'attr_cls_layers': [1500], 'obj_cls_layers': [1500]})
    self.attr_classifier = ParametricMLP(self.hparam.shared_emb_dim, self.nattrs, self.hparam.attr_cls_layers)
    self.obj_classifier = ParametricMLP(self.hparam.shared_emb_dim, self.nobjs, self.hparam.obj_cls_layers)

    self.dset = dset

  def get_all_pairs(self, nodes):
    attr_nodes = nodes[:self.nattrs]
    obj_nodes = nodes[self.nattrs:]
    if self.dset.open_world:
      all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.hparam.node_dim)
      all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
    else:
      pairs = self.dset.pairs
      all_pair_attr_ids = [self.dset.attr2idx[attr] for attr, obj in pairs]
      all_pair_obj_ids = [self.dset.obj2idx[obj] for attr, obj in pairs]
      all_pair_attrs = attr_nodes[all_pair_attr_ids]
      all_pair_objs = obj_nodes[all_pair_obj_ids]

    all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
    if self.train_only and self.training:
      all_pairs = all_pairs[self.train_idx]
    return all_pairs

  def forward(self, x):
    if self.resnet:
      img = self.resnet(x[4].to(dev))
    else:
      img = x[4].to(dev)
    pair_id = x[3]
    img_feats = self.img_fc(img)
    nodes = self.gae.encode(self.nodes, self.train_pair_edges)
    all_pair_nodes = self.pair_fc(self.get_all_pairs(nodes))

    pair_pred = torch.matmul(img_feats, all_pair_nodes.T)

#     attr_pred = self.attr_classifier(img_feats)
#     obj_pred = self.obj_classifier(img_feats)
#     return pair_pred, attr_pred, obj_pred, img_feats, all_pair_nodes[pair_id], nodes, self
    return pair_pred, nodes, self