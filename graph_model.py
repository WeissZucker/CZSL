import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from scipy.sparse import coo_matrix
import numpy as np

from model import *

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphModelBase(nn.Module):
  def __init__(self, hparam, dset, graph_path, train_only=False, static_inp=True, resnet_name=None):
    super(GraphModelBase, self).__init__()
    self.hparam = hparam
    self.nattrs, self.nobjs = len(dset.attrs), len(dset.objs)
    self.img_feat_dim = dset.feat_dim

    if resnet_name:    
      import torchvision.models as models
      self.resnet = models.resnet18(pretrained=True).to(dev)
      self.img_feat_dim = self.resnet.fc.in_features
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
        
    self.train_only = train_only
    if self.train_only:
      train_idx = []
      for current in dset.train_pairs:
        train_idx.append(dset.all_pair2idx[current]+self.nattrs+self.nobjs)
      self.train_idx = torch.LongTensor(train_idx).to(dev)

    graph = torch.load(graph_path)
    self.nodes = graph["embeddings"].to(dev)
    adj = graph["adj"]
    row_idx, col_idx = adj.row, adj.col
    self.edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long).to(dev)
    
    if static_inp and self.resnet:
      self.resnet = frozen(self.resnet)
    else:
      #self.nodes = nn.Parameter(self.nodes)
      pass



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
  def __init__(self, dset, graph_path, train_only=True, static_inp=True):
    super(GraphModel, self).__init__(dset, graph_path, train_only=train_only, static_inp=static_inp)

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
  def __init__(self, dset, graph_path=None, static_inp=True, resnet_name=None):
    super(GraphMLP, self).__init__(dset, graph_path, resnet_name=resnet_name)

    self.node_dim = 512
    self.gcn = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.node_dim, hidden_layer_sizes = [2048])

    self.shared_emb_dim = 800
    self.img_fc = ParametricMLP(self.img_feat_dim, self.shared_emb_dim, [768, 1000], norm_output=True, dropout=0.5)
    self.pair_fc = ParametricMLP(self.node_dim*2, self.shared_emb_dim, [1000], norm_output=True, dropout=0.5)
    
    self.attr_classifier = ParametricMLP(self.shared_emb_dim, self.nattrs, [1500], norm_output=False, dropout=0.5)
    self.obj_classifier = ParametricMLP(self.shared_emb_dim, self.nobjs, [1500], norm_output=False, dropout=0.5)

  def get_all_pairs(self, nodes):
    attr_nodes = nodes[:self.nattrs]
    obj_nodes = nodes[self.nattrs:]
    all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.node_dim)
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
    return self.forward_metric_learning(x)


class ReciprocalClassifierGraph(GraphModelBase):
  def __init__(self, dset, graph_path, projector_mlp_layer_sizes, resnet_name=None):
    super(ReciprocalClassifierGraph, self).__init__(dset, graph_path, resnet_name=None)

    self.node_dim = 512

    self.gcn = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.node_dim, hidden_layer_sizes=[4096])
    
    self.obj_emb_dim = 800 
    self.attr_emb_dim = 800
      
    self.obj_project = ParametricMLP(self.img_feat_dim+self.node_dim, self.obj_emb_dim, projector_mlp_layer_sizes, norm_output=True)
    self.attr_project = ParametricMLP(self.img_feat_dim+self.node_dim, self.attr_emb_dim, projector_mlp_layer_sizes, norm_output=True)
                                               
    self.obj_to_logits = ParametricMLP(self.obj_emb_dim, OBJ_CLASS, [])
    self.attr_to_logits = ParametricMLP(self.attr_emb_dim, ATTR_CLASS, [])


  def forward(self, sample):
    if self.resnet:
      img_features = self.resnet(sample[0].to(dev))
    else:
      img_features = sample[4].to(dev)
      
    obj_pre_emb = self.obj_project(torch.cat((img_features, torch.zeros((len(img_features), self.node_dim)).to(dev)), dim=1))
    attr_pre_emb = self.attr_project(torch.cat((img_features, torch.zeros((len(img_features), self.node_dim)).to(dev)), dim=1))
    
    t = 0.5
    obj_pre_pred = F.softmax(1/t * self.obj_to_logits(obj_pre_emb), dim=1)
    attr_pre_pred = F.softmax(1/t * self.attr_to_logits(attr_pre_emb), dim=1)
    
    nodes = self.gcn(self.nodes, self.edge_index)
    all_attr_nodes = nodes[:self.nattrs]
    all_obj_nodes = nodes[self.nattrs:]
    
    attr_semantic = torch.matmul(attr_pre_pred, all_attr_nodes) 
    obj_semantic = torch.matmul(obj_pre_pred, all_obj_nodes)
    
    obj_emb = self.obj_project(torch.cat((img_features, attr_semantic), dim=1))
    attr_emb = self.attr_project(torch.cat((img_features, obj_semantic), dim=1))
    
    obj_pred = self.obj_to_logits(obj_emb)
    attr_pred = self.attr_to_logits(attr_emb)
    return attr_pred, obj_pred, attr_pre_pred, obj_pre_pred

  
class GAE(GraphModelBase):
    def __init__(self, dset, graph_path, static_inp=True, resnet_name=None, pretrained_gae=None):
      super(GAE, self).__init__(dset, graph_path, static_inp=static_inp, resnet_name=resnet_name)

      self.train_pair_edges = torch.zeros((2, len(dset.train_pairs)), dtype=torch.long).to(dev)
      for i, (attr, obj) in enumerate(dset.train_pairs):
        self.train_pair_edges[0, i] = dset.attr2idx[attr]
        self.train_pair_edges[1, i] = dset.obj2idx[obj]

      self.node_dim = 512

      self.encoder = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.node_dim, [2048])
      self.gae = gnn.GAE(self.encoder)

      if pretrained_gae:
        pretrained = torch.load(pretrained_gae)
        self.gae.load_state_dict(pretrained['model_state_dict'])
        del pretrained

      self.shared_emb_dim = 800
      self.img_fc = ParametricMLP(self.img_feat_dim, self.shared_emb_dim, [800, 1000], norm_output=True, dropout=0.5)
      self.pair_fc = ParametricMLP(self.node_dim*2, self.shared_emb_dim, [1000], norm_output=True, dropout=0.5)

      
    def get_all_pairs(self, nodes):
      attr_nodes = nodes[:self.nattrs]
      obj_nodes = nodes[self.nattrs:]
      all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.node_dim)
      all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      return all_pairs
    
    def forward(self, x):
      if self.resnet:
        img = self.resnet(x[0].to(dev))
      else:
        img = x[4].to(dev)

      img_feats = self.img_fc(img)
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)
      all_pair_nodes = self.get_all_pairs(nodes)
      all_pairs = self.pair_fc(all_pair_nodes)
      pair_pred = torch.matmul(img_feats, all_pairs.T)
      if self.training:
        return pair_pred, nodes, self
      else:
        return pair_pred


class GAEStage3(GraphModelBase):
    def __init__(self, hparam, dset, graph_path=None, static_inp=True, resnet_name=None, pretrained_gae=None, pretrained_mlp=None):
        super(GAEStage3, self).__init__(hparam, dset, graph_path, static_inp=static_inp, resnet_name=resnet_name)
        
        self.train_pair_edges = torch.zeros((2, len(dset.train_pairs)), dtype=torch.long).to(dev)
        for i, (attr, obj) in enumerate(dset.train_pairs):
          self.train_pair_edges[0, i] = dset.attr2idx[attr]
          self.train_pair_edges[1, i] = dset.obj2idx[obj]
        
        self.hparam.add_dict({'graph_encoder_layers': [2048], 'node_dim': 512})
        self.encoder = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.hparam.node_dim, self.hparam.graph_encoder_layers)
        self.gae = gnn.GAE(self.encoder)
          
        self.hparam.add('shared_emb_dim', 800)
        self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                              'pair_fc_layers': [1000], 'pair_fc_norm': True})
        self.img_fc = ParametricMLP(self.img_feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers,
                                    norm_output=self.hparam.img_fc_norm)
        self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, self.hparam.pair_fc_layers,
                                     norm_output=self.hparam.pair_fc_norm)
        
        self.hparam.add_dict({'attr_cls_layers': [1500], 'obj_cls_layers': [1500]})
        self.attr_classifier = ParametricMLP(self.hparam.shared_emb_dim, self.nattrs, self.hparam.attr_cls_layers)
        self.obj_classifier = ParametricMLP(self.hparam.shared_emb_dim, self.nobjs, self.hparam.obj_cls_layers)
      
    def get_all_pairs(self, nodes):
      attr_nodes = nodes[:self.nattrs]
      obj_nodes = nodes[self.nattrs:]
      all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.hparam.node_dim)
      all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      return all_pairs
    
    def forward(self, x):
      if self.resnet:
        img = self.resnet(x[0].to(dev))
      else:
        img = x[4].to(dev)
      pair_id = x[3]
      img_feats = self.img_fc(img)
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)
      all_pair_nodes = self.pair_fc(self.get_all_pairs(nodes))
      
      attr_pred = self.attr_classifier(img_feats)
      obj_pred = self.obj_classifier(img_feats)

      pair_pred = torch.matmul(img_feats, all_pair_nodes.T)
      
      if self.training:
        return attr_pred, obj_pred, pair_pred, img_feats, all_pair_nodes[pair_id], nodes, self
      else:
        return pair_pred

      
class GAEBiD(GraphModelBase):
    def __init__(self, hparam, dset, graph_path=None, static_inp=True, resnet_name=None, pretrained_gae=None, pretrained_mlp=None, fscore_path=None):
        super(GAEBiD, self).__init__(hparam, dset, graph_path, static_inp=static_inp, resnet_name=resnet_name)
        
        self.train_pair_edges = torch.zeros((2, len(dset.train_pairs)), dtype=torch.long).to(dev)
        for i, (attr, obj) in enumerate(dset.train_pairs):
          self.train_pair_edges[0, i] = dset.attr2idx[attr]
          self.train_pair_edges[1, i] = dset.obj2idx[obj]
          
        self.hparam.add('graph_encoder_layers', [2048])
        
        self.encoder = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.node_dim, self.hparam.graph_encoder_layers)
        self.gae = gnn.GAE(self.encoder)
          
        self.hparam.add('shared_emb_dim', 800)
        self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                              'pair_fc_layers': [1000], 'pair_fc_norm': True})
        self.img_fc = ParametricMLP(self.img_feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers,
                                    norm_output=self.hparam.img_fc_norm)
        self.pair_fc = ParametricMLP(self.node_dim*2, self.hparam.shared_emb_dim, self.hparam.pair_fc_layers,
                                     norm_output=self.hparam.pair_fc_norm)
        
        self.hparam.add_dict({'attr_fc_layers': [1500], 'attr_fc_norm': False,
                              'obj_fc_layers': [1500], 'obj_fc_norm': False})
        self.attr_fc = ParametricMLP(self.img_feat_dim, self.node_dim, self.hparam.attr_fc_layers,
                                     norm_output=self.hparam.attr_fc_norm)
        self.obj_fc = ParametricMLP(self.img_feat_dim, self.node_dim, self.hparam.obj_fc_layers,
                                    norm_output=self.hparam.obj_fc_norm)
      
    def get_all_pairs(self, nodes):
      attr_nodes = nodes[:self.nattrs]
      obj_nodes = nodes[self.nattrs:]
      all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.node_dim)
      all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      return all_pairs
    
    def forward(self, x):
      if self.resnet:
        img = self.resnet(x[0].to(dev))
      else:
        img = x[4].to(dev)
      pair_id = x[3]
      img_feats = self.img_fc(img)
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)
      all_attr_nodes = nodes[:self.nattrs]
      all_obj_nodes = nodes[self.nattrs:]
      all_pair_nodes = self.pair_fc(self.get_all_pairs(nodes))
      
      attr_feats = self.attr_fc(img)
      obj_feats = self.obj_fc(img)
      
      attr_pred = attr_feats @ all_attr_nodes.T
      obj_pred = obj_feats @ all_obj_nodes.T
      pair_pred = img_feats @ all_pair_nodes.T
      
      if self.training:
        return attr_pred, obj_pred, pair_pred, img_feats, all_pair_nodes[pair_id], nodes, self
      else:
        return pair_pred
      
      
class GAEStage3ED(GraphModelBase):
    def __init__(self, dset, graph_path=None, static_inp=True, resnet_name=None, pretrained_gae=None, pretrained_mlp=None, fscore_path=None):
        super(GAEStage3ED, self).__init__(dset, graph_path, static_inp=static_inp, resnet_name=resnet_name)
        
        self.train_pair_edges = torch.zeros((2, len(dset.train_pairs)), dtype=torch.long).to(dev)
        for i, (attr, obj) in enumerate(dset.train_pairs):
          self.train_pair_edges[0, i] = dset.attr2idx[attr]
          self.train_pair_edges[1, i] = dset.obj2idx[obj]
        
        self.encoder = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.node_dim, [2048])
        self.gae = gnn.GAE(self.encoder)
          
        self.shared_emb_dim = 800
        self.img_fc = ParametricMLP(self.img_feat_dim, self.shared_emb_dim, [800, 1000], norm_output=True, dropout=0.5)
        self.pair_fc = ParametricMLP(self.node_dim*2, self.shared_emb_dim, [1000], norm_output=True, dropout=0.5)
        
        self.attr_classifier = ParametricMLP(self.shared_emb_dim, self.nattrs, [1500], norm_output=False, dropout=0.5)
        self.obj_classifier = ParametricMLP(self.shared_emb_dim, self.nobjs, [1500], norm_output=False, dropout=0.5)
        
        if fscore_path:
          self.fscore = torch.load(fscore_path)
        else:
          self.fscore = None
        
        if pretrained_gae:
          pretrained = torch.load(pretrained_gae)
          self.gae.load_state_dict(pretrained['model_state_dict'])
          del pretrained
        
        if pretrained_mlp:
          pretrained = torch.load(pretrained_mlp)
          self.img_fc.load_state_dict(pretrained['model_state_dict'])
          del pretrained
      
    def get_all_pairs(self, nodes):
      attr_nodes = nodes[:self.nattrs]
      obj_nodes = nodes[self.nattrs:]
      all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.node_dim)
      all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      return all_pairs
    
    def forward(self, x):
      if self.resnet:
        img = self.resnet(x[0].to(dev))
      else:
        img = x[4].to(dev)
      pair_id = x[3]
      img_feats = self.img_fc(img)
      
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)
      all_pair_nodes = self.pair_fc(self.get_all_pairs(nodes))
      
      attr_pred = self.attr_classifier(img_feats)
      obj_pred = self.obj_classifier(img_feats)

#       pair_pred = torch.matmul(img_feats, all_pair_nodes.T)
      
      if self.training:
        return attr_pred, obj_pred, img_feats, all_pair_nodes[pair_id], nodes, self
      else:
        distance = []
        npairs = len(all_pair_nodes)
        for img in img_feats:
          dist = F.pairwise_distance(all_pair_nodes, img.unsqueeze(0).repeat(npairs, 1))
          distance.append(dist)
        distance = torch.cat(distance)
        scores = torch.exp(-distance/10)
        return scores