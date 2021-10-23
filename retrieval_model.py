import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from scipy.sparse import coo_matrix
import numpy as np

from model import *
from graph_model import *
from gcn import CGE

dev = 'cuda' if torch.cuda.is_available() else 'cpu'



class ImageRetrievalModel(nn.Module):
  def get_pair(self, attr_id, obj_id, nodes):
    """Primitive embeddings to pair embeddings"""
    pass

  def get_nodes(self):
    """Get embeddings for all primitives"""
    pass

  def generate_theta_normal(self, img, obj_id, t_attr_id, nodes):
    img_feat = self.img_fc(img)
    t_pair_feat = self.get_pair(t_attr_id, obj_id, nodes)
    theta = self.compo_fc(torch.cat((img, t_pair_feat), dim=1))
    return theta, img_feat
  
  def generate_theta_bert(self, img, obj_id, t_attr_id, nodes):
    img_feat = self.img_fc(img)
    t_pair_feat = self.get_pair(t_attr_id, obj_id, nodes)
    t_caption_feat = self.caption_feats[t_attr_id]
    theta = self.compo_fc(torch.cat((img, t_caption_feat), dim=1))
    return theta, img_feat
  
  def generate_theta_gated(self, img, obj_id, t_attr_id, nodes):
    img_feat = self.img_fc(img)
    t_pair_feat = self.get_pair(t_attr_id, obj_id, nodes)
    t_caption_feat = self.caption_feats[t_attr_id]
    residual_feat = self.compo_fc(torch.cat((img, t_caption_feat), dim=1))
    gate = self.compo_gate(torch.cat((img, t_pair_feat), dim=1))
    theta = self.compo_weight[0]*residual_feat + self.compo_weight[1]*gate
    return theta, img_feat

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
    t_attr_id, obj_id = x[6].to(dev), x[2].to(dev)

    nodes = self.get_nodes()

    theta, _ = self.generate_theta(s_img, obj_id, t_attr_id, nodes)
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

    nodes = self.get_nodes()
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
  
  def generate_test_queries(self, attr_id, obj_id, maxlen=10):
    attr = self.dset.attrs[attr_id]
    obj = self.dset.objs[obj_id]
    t_attrs = list(set([_attr for _attr, _obj in self.dset.test_pairs if obj==_obj and attr!=_attr]))[:maxlen]
    t_pair_ids = torch.tensor([self.dset.all_pair2idx[(_attr, obj)] for _attr in t_attrs]).to(dev)
    t_attr_ids = torch.tensor([self.dset.attr2idx[_attr] for _attr in t_attrs]).to(dev)
    return t_attr_ids, t_pair_ids

  def val_forward_mitstates(self, x):
    if self.resnet:
      s_img = self.resnet(x[4].to(dev))
    else:
      s_img = x[4].to(dev)

    s_attr_ids, obj_ids, s_pair_ids = [item.to(dev) for item in x[1:4]]
    thetas, t_pair_ids, img_feats = [], [], []
    nodes = self.get_nodes()

    for s_attr_id, s_pair_id, img, obj_id in zip(s_attr_ids, s_pair_ids, s_img, obj_ids):
      # Find target samples for each source sample
      t_attr_id, t_pair_id = self.generate_test_queries(s_attr_id, obj_id)
      ntarget = len(t_pair_id)
      s_attr_id = s_attr_id.repeat(ntarget)
      s_pair_id = s_pair_id.repeat(ntarget)
      obj_id = obj_id.repeat(ntarget)
      img = img.unsqueeze(0).repeat(ntarget, 1)
      theta, img_feat = self.generate_theta(img, obj_id, t_attr_id, nodes)
      thetas.append(theta)
      t_pair_ids.append(t_pair_id)
      img_feats.append(img_feat[0].unsqueeze(0))

    thetas = torch.cat(thetas).to(dev)
    t_pair_ids = torch.cat(t_pair_ids).to(dev)
    img_feats = torch.cat(img_feats).to(dev)

    return thetas, t_pair_ids, img_feats, s_pair_ids
  
  def val_forward_fashion(self, x):
    if self.resnet:
      s_img = self.resnet(x[4].to(dev))
    else:
      s_img = x[4].to(dev)

    obj_id= x[2]
    t_attr_id = x[6]
    s_img_idx = x[0]
    t_caption = x[5]

    nodes = self.get_nodes()
    theta, _ = self.generate_theta(s_img, obj_id, t_attr_id, nodes)
    return theta, t_attr_id, s_img_idx, t_caption
  
  def forward(self, x):
    if self.training:
      return self.train_simple_forward(x)
    else:
      if self.dset.name == 'Fashion200k':
        return self.val_forward_fashion(x)
      else:
        return self.val_forward_mitstates(x)

      
      
class GAEIR(GraphModelBase, ImageRetrievalModel):
  def __init__(self, hparam, dset, graph_path=None, train_only=False, resnet_name=None, static_inp=True, pretrained_gae=None):
    super(GAEIR, self).__init__(hparam, dset, graph_path, train_only=train_only, resnet_name=resnet_name, static_inp=True)

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

    self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                          'pair_fc_layers': [1000], 'pair_fc_norm': True})
    self.hparam.add('shared_emb_dim', 800)
    self.img_fc = ParametricMLP(self.img_feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers,
                                norm_output=self.hparam.img_fc_norm)

    self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, 
                                 self.hparam.pair_fc_layers, batch_norm=True, norm_output=self.hparam.pair_fc_norm)

    self.hparam.add_dict({'compo_fc_layers': [1000, 1200], 'compo_fc_norm': True})
    self.compo_fc = ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.compo_fc_layers, batch_norm=True, norm_output=self.hparam.compo_fc_norm)

    self.dset = dset
    self.generate_theta = self.generate_theta_normal
 
  def get_pair(self, attr_id, obj_id, nodes):
    attr_node = nodes[attr_id]
    obj_node = nodes[self.nattrs+obj_id]
    pair = torch.cat((attr_node, obj_node), dim=1)
    return self.pair_fc(pair)
  
  def get_nodes(self):
    return self.gae.encode(self.nodes, self.train_pair_edges)
  
  
  
class GAEIRBert(GAEIR):
  def __init__(self, hparam, dset, graph_path=None, train_only=False, resnet_name=None, static_inp=True, pretrained_gae=None):
    super(GAEIRBert, self).__init__(hparam, dset, graph_path, train_only=train_only, resnet_name=resnet_name, static_inp=True)

    self.train_pair_edges = torch.zeros((2, len(dset.train_pairs)), dtype=torch.long).to(dev)
    for i, (attr, obj) in enumerate(dset.train_pairs):
      self.train_pair_edges[0, i] = dset.attr2idx[attr]
      self.train_pair_edges[1, i] = dset.obj2idx[obj]

    self.hparam.add_dict({'graph_encoder_layers': [2048], 'node_dim': 512})
    self.encoder = GraphEncoder(gnn.SAGEConv, self.nodes.size(1), self.hparam.node_dim, self.hparam.graph_encoder_layers)
    self.gae = gnn.GAE(self.encoder)

    self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                          'pair_fc_layers': [1000], 'pair_fc_norm': True})

    self.img_fc = nn.Identity()
    self.hparam.add('shared_emb_dim', self.img_feat_dim)
    self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, 
                                 self.hparam.pair_fc_layers, batch_norm=True, norm_output=self.hparam.pair_fc_norm)

    self.hparam.add_dict({'compo_fc_layers': [1000, 1200], 'compo_fc_norm': True})
    self.compo_fc = ParametricMLP(self.img_feat_dim+self.nodes.size(1), self.hparam.shared_emb_dim,
                                  self.hparam.compo_fc_layers, norm_output=self.hparam.compo_fc_norm)
    self.hparam.add_dict({'gate_fc_layers': [1000], 'gate_fc_norm': False})
    self.compo_gate = nn.Sequential(ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.gate_fc_layers, norm_output=self.hparam.gate_fc_norm),
                                     nn.Sigmoid())
    self.compo_weight = nn.Parameter(torch.tensor([10.0, 1.0]))

    self.dset = dset
    self.generate_theta = self.generate_theta_gated
    self.caption_feats = self.nodes[:self.nattrs]

  
  
class CGEIR(ImageRetrievalModel, CGE):
  def __init__(self, hparam, dset, train_only=True, static_inp=True, graph_path=None):
    super(CGEIR, self).__init__(hparam, dset, train_only=train_only, static_inp=static_inp, graph_path=graph_path)
    self.resnet=None
    self.img_feat_dim = dset.feat_dim
    
    self.img_fc = nn.Identity()
    self.hparam.add('node_dim', self.embeddings.size(1))

    self.hparam.add_dict({'compo_fc_layers': [1000, 1200], 'compo_fc_norm': True})
    self.compo_fc = ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.compo_fc_layers, batch_norm=True, norm_output=self.hparam.compo_fc_norm)
    self.generate_theta = self.generate_theta_normal
    
  def get_pair(self, attr_id, obj_id, nodes):
    attrs = [self.dset.attrs[idx] for idx in attr_id]
    objs = [self.dset.objs[idx] for idx in obj_id]
    pair_ids = torch.tensor([self.dset.all_pair2idx[(attr, obj)] for attr, obj in zip(attrs, objs)])
    pair_ids += self.nattrs + self.nobjs
    return nodes[pair_ids]
  
  def get_nodes(self):
    return self.gcn(self.embeddings)
  
  
  
class CGEIRBert(CGEIR):
  def __init__(self, hparam, dset, train_only=True, static_inp=True, graph_path=None):
    super(CGEIRBert, self).__init__(hparam, dset, train_only=train_only, static_inp=static_inp, graph_path=graph_path)
    self.resnet=None
    self.img_feat_dim = dset.feat_dim
    
    self.img_fc = nn.Identity();
    self.hparam.add('node_dim', self.embeddings.size(1))

    self.hparam.add_dict({'compo_fc_layers': [1000, 1200], 'compo_fc_norm': True})
    self.compo_fc = ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.compo_fc_layers, batch_norm=True, norm_output=self.hparam.compo_fc_norm)
    self.hparam.add_dict({'gate_fc_layers': [1000], 'gate_fc_norm': False})
    self.compo_gate = nn.Sequential(ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.gate_fc_layers, norm_output=self.hparam.gate_fc_norm),
                                     nn.Sigmoid())
    self.compo_weight = nn.Parameter(torch.tensor([10.0, 1.0]))
    self.generate_theta = self.generate_theta_bert

  
  
from compcos import CompCos
class CompcosIR(CompCos, ImageRetrievalModel):
  def __init__(self, hparam, dset, attr_emb_path, obj_emb_path, train_only=True, static_inp=True, graph_path=None, resnet_name=None):
    super(CompcosIR, self).__init__(hparam, dset, attr_emb_path, obj_emb_path, resnet_name=resnet_name)
    self.img_feat_dim = dset.feat_dim
    
    self.img_fc = self.image_embedder

    self.hparam.add_dict({'compo_fc_layers': [1000, 1200], 'compo_fc_norm': True})
    self.compo_fc = ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.compo_fc_layers, batch_norm=True, norm_output=self.hparam.compo_fc_norm)
    self.generate_theta = self.generate_theta_normal
    
  def get_pair(self, attr_id, obj_id, nodes):
    return self.compose(attr_id, obj_id)
  
  def get_nodes(self):
    return None
  


class CompcosIRBert(CompcosIR):
  def __init__(self, hparam, dset, attr_emb_path, obj_emb_path, train_only=True, static_inp=True, graph_path=None, resnet_name=None):
    super(CompcosIRBert, self).__init__(hparam, dset, attr_emb_path, obj_emb_path, resnet_name=resnet_name)
    self.img_feat_dim = dset.feat_dim
    self.generate_theta = self.generate_theta_gated
    self.caption_feats = torch.load(attr_emb_path).to(dev)
    

    self.hparam.add_dict({'compo_fc_layers': [1000, 1200], 'compo_fc_norm': True})
    self.compo_fc = ParametricMLP(self.img_feat_dim+self.caption_feats.size(1), self.hparam.shared_emb_dim,
                                  self.hparam.compo_fc_layers, batch_norm=True, norm_output=self.hparam.compo_fc_norm)
    
    self.hparam.add_dict({'gate_fc_layers': [1000], 'gate_fc_norm': False})
    self.compo_gate = nn.Sequential(ParametricMLP(self.img_feat_dim+self.hparam.shared_emb_dim, self.hparam.shared_emb_dim,
                                  self.hparam.gate_fc_layers, norm_output=self.hparam.gate_fc_norm),
                                     nn.Sigmoid())
    self.compo_weight = nn.Parameter(torch.tensor([10.0, 1.0]))