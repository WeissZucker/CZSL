import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from scipy.sparse import coo_matrix
import numpy as np

from model import *

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphModelBase(nn.Module):
  def __init__(self, hparam, dset, graph_path, train_only=False, resnet_name=None):
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
  def __init__(self, hparam, dset, graph_path=None, resnet_name=None):
    super(GraphMLP, self).__init__(hparam, dset, graph_path, resnet_name=resnet_name)
    
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
    def __init__(self, hparam, dset, graph_path=None, train_only=False, resnet_name=None, pretrained_gae=None, pretrained_mlp=None):
        super(GAE, self).__init__(hparam, dset, graph_path, train_only=train_only, resnet_name=resnet_name)
        
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
      
#       attr_pred = self.attr_classifier(img_feats)
#       obj_pred = self.obj_classifier(img_feats)
#       return pair_pred, attr_pred, obj_pred, img_feats, all_pair_nodes[pair_id], nodes, self
      return pair_pred, nodes, self



class GAE_IR(GraphModelBase):
    def __init__(self, hparam, dset, graph_path=None, train_only=False, resnet_name=None, pretrained_gae=None):
        super(GAE_IR, self).__init__(hparam, dset, graph_path, train_only=train_only, resnet_name=resnet_name)
        
        if pretrained_gae:
          checkpoint = torch.load(pretrained_gae)
          hparam.freeze() # parameters that have already been set won't be updated 
          hparam.add_dict(checkpoint['hparam_dict'])
          del checkpoint
        
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
        self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, 
                                     self.hparam.pair_fc_layers, batch_norm=False, norm_output=self.hparam.pair_fc_norm)
        
        self.hparam.add_dict({'compo_fc_layers': [800, 1000], 'compo_fc_norm': True})
        self.compo_fc = ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                      self.hparam.compo_fc_layers, batch_norm=False, norm_output=self.hparam.compo_fc_norm)
        
        self.dset = dset
 
    def get_pair(self, attr_id, obj_id, nodes):
      attr_node = nodes[attr_id]
      obj_node = nodes[self.nattrs+obj_id]
      pair = torch.cat((attr_node, obj_node), dim=1)
      return self.pair_fc(pair)
    
    def get_all_neg_pairs(self, attr_id, obj_id, nodes):
      """all pairs with the same obj but different attr"""
      
      attr_nodes = nodes[:self.nattrs]
      obj_node = nodes[self.nattrs+obj_id].unsqueeze(0)
      if self.dset.open_world and not self.train_only:
        all_pair_attrs = torch.cat((attr_nodes[:attr_id], attr_nodes[attr_id+1:]))
      else:
        pairs = self.dset.pairs
        obj = self.dset.objs[obj_id]
        attr = self.dset.attrs[attr_id]
        all_pair_attr_ids = [self.dset.attr2idx[_attr] for _attr, _obj in pairs if obj==_obj and attr!=_attr]
        all_pair_attrs = attr_nodes[all_pair_attr_ids]
        
      all_pair_objs = obj_node.repeat(all_pair_attrs.size(0), 1)
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      return self.pair_fc(all_pairs)
    
    def train_simple_forward(self, x):
      if self.resnet:
        s_img = self.resnet(x[4].to(dev))
        t_img = self.resnet(x[9].to(dev)) # img with the same obj but different attr
      else:
        s_img = x[4].to(dev)
        t_img = x[9].to(dev)
      t_attr_id, obj_id = x[6], x[2]
      
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)
      t_pair_feats = self.get_pair(t_attr_id, obj_id, nodes)
      
      # s_img_feats or s_img
      theta = self.compo_fc(torch.cat((s_img, t_pair_feats), dim=1))
      target = self.img_fc(t_img)
      return theta, target
    
    def train_object_aware_forward(self, x):
      if self.resnet:
        s_img = self.resnet(x[4].to(dev))
        t_img = self.resnet(x[9].to(dev)) # img with the same obj but different attr
      else:
        s_img = x[4].to(dev)
        t_img = x[9].to(dev)
      s_attr_ids, t_attr_ids, obj_ids = x[1], x[6], x[2]
      
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)
      t_pair_feats = self.get_pair(t_attr_ids, obj_ids, nodes)
      
      # s_img_feats or s_img
      theta = self.compo_fc(torch.cat((s_img, t_pair_feats), dim=1))
      targets = self.img_fc(t_img)
      neg_pairs = []
      for i, (attr_id, obj_id) in enumerate(zip(s_attr_ids, obj_ids)):
        neg_pair = self.get_all_neg_pairs(attr_id, obj_id, nodes)
        neg_pair = self.compo_fc(torch.cat((s_img[i:i+1].repeat(len(neg_pair),1), neg_pair), dim=1))
        neg_pairs.append(neg_pair)
      return theta, targets, neg_pairs
    
    def generate_test_queries(self, attr_id, obj_id):
      attr = self.dset.attrs[attr_id]
      obj = self.dset.objs[obj_id]
      t_attrs = set([_attr for _attr, _obj in self.dset.test_pairs if obj==_obj and attr!=_attr])
      t_pairs = torch.tensor([self.dset.pair2idx[(_attr, obj)] for _attr in t_attrs])
      t_attrs = torch.tensor([self.dset.attr2idx[_attr] for _attr in t_attrs])
      return t_attrs, t_pairs
    
    def val_forward(self, x):
      if self.resnet:
        s_img = self.resnet(x[4].to(dev))
      else:
        s_img = x[4].to(dev)
 
      s_attr_id, obj_id, s_pair_id = x[1:4]
      t_attr_id, t_pair_id = self.generate_test_queries(s_attr_id, obj_id)
      ntarget = len(t_pair_id)
      s_attr_id = s_attr_id.repeat(ntarget)
      s_pair_id = s_pair_id.repeat(ntarget)
      obj_id = obj_id.repeat(ntarget)
      s_img = s_img.repeat(ntarget, 1)
      
      s_img_feats = self.img_fc(s_img)
      
      nodes = self.gae.encode(self.nodes, self.train_pair_edges)

      t_pair_feats = self.get_pair(t_attr_id, obj_id, nodes)

      theta = self.compo_fc(torch.cat((s_img, t_pair_feats), dim=1))

      return theta, t_pair_id, s_img_feats[0].view(1,-1), s_pair_id[0].view(1,-1)
    
    def forward(self, x):
      if self.training:
        return self.train_object_aware_forward(x)
      else:
        return self.val_forward(x)
      