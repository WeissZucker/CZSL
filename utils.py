import dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class DummyLogger():
  log_dir = None
  filename_suffix = None
  def add_scalar(self, a, b, c):
    pass
  
  def add_hparams(self, a, b):
    pass
  
  def add_text(self, a, b):
    pass
  
  def flush(self):
    pass
  
  def close(self):
    pass
  
class HParam():
  def __init__(self):
    self.hparam_dict = dict() # For tensorboard logger
    self.freeze = False
    
  def add(self, name, value):
    if self.freeze and name in self.hparam_dict:
      return
    setattr(self, name, value)
    if isinstance(value, list) or isinstance(value, tuple):
      self.hparam_dict[name] = str(value)
      
  def add_dict(self, d):
    for name, value in d.items():
      self.add(name, value)
      if isinstance(value, list) or isinstance(value, tuple):
        d[name] = str(value)
    if self.freeze:
      self.hparam_dict = d | self.hparam_dict
    else:
      self.hparam_dict |= d
      
  def get(self, name):
    if name not in self.hparam_dict:
      raise Exception(f"HParam: {name} doesn't exist!")
    return getattr(self, name)
  
  def __repr__(self):
    return repr(self.hparam_dict)
  
cross_entropy_loss = nn.CrossEntropyLoss()

class _Loss():
  def __call__(self, model_output, sample):
    return self.loss(model_output, sample)
  
  def hparam_dict(self):
    return dict()
  
  
class EuclideanLoss(_Loss):
  def loss(self, model_output, sample):
    x, y = model_output # [batch_size, npairs]
    loss = F.pairwise_distance(x, y).mean()
    loss_dict = {'ed_loss': loss}
    return loss, loss_dict
  
euclidean_dist_loss = EuclideanLoss()
  
class PrimitiveCE(_Loss):  
  def loss(self, model_output, sample):
    attr_scores, obj_scores = model_output
    attr_labels = sample[1].to(dev)
    obj_labels = sample[2].to(dev)
    attr_loss = cross_entropy_loss(attr_scores, attr_labels)
    obj_loss = cross_entropy_loss(obj_scores, obj_labels)
    loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss}
    total_loss = attr_loss + obj_loss
    return total_loss, loss_dict
  
primitive_cross_entropy_loss = PrimitiveCE()

class PairCE(_Loss):
  def loss(self, model_output, sample):
    compo_score = model_output # [batch_size, npairs]
    pair_labels = sample[3].to(dev)
    loss = cross_entropy_loss(compo_score, pair_labels)
    loss_dict = {'contra_loss': loss}
    return loss, loss_dict
  
pair_cross_entropy_loss = PairCE()

class GAELoss(_Loss):
  def __init__(self, recon_loss_ratio):
    self.recon_loss_ratio = recon_loss_ratio
    
  def loss(self, model_output, sample):
    if not isinstance(model_output, tuple): # during evaluation
      return pair_cross_entropy_loss(model_output, sample)
    pair_pred, nodes, model = model_output
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    ce_loss, ce_loss_dict = pair_cross_entropy_loss(pair_pred, sample)
    total_loss = (1-self.recon_loss_ratio) * ce_loss + self.recon_loss_ratio * recon_loss
    loss_dict = {'recon_loss':recon_loss} | ce_loss_dict
    return total_loss, loss_dict
  
  def hparam_dict(self):
    return {'recon_loss_ratio': self.recon_loss_ratio}
  
  
class MetricLearningLoss(_Loss):
  def __init__(self, ml_loss, loss_weights=[1, 1], miner=None):
    self.ml_loss = ml_loss
    self.miner = miner
    self.img_loss_weight, self.text_loss_weight = loss_weights
    
  def loss(self, model_output, sample):
    img_feats, pair_emb = model_output
    pair_id = sample[3]
    img_miner_output = self.miner(img_feats, pair_id) if self.miner else None
    img_loss = self.ml_loss(img_feats, pair_id, indices_tuple=img_miner_output)
    text_miner_output = self.miner(pair_emb, pair_id) if self.miner else None
    text_loss = self.ml_loss(pair_emb, pair_id, indices_tuple=text_miner_output)
    total_loss = self.img_loss_weight * img_loss + self.text_loss_weight * text_loss
    loss_dict = {'img_loss': img_loss, 'text_loss': text_loss}
    return total_loss, loss_dict
  
  def hparam_dict(self):
    return {'loss_ratio': [self.img_loss_weight, self.text_loss_weight],
           'ml_loss_func': self.ml_loss.__class__.__name__,
           'ml_loss_miner': self.miner.__class__.__name__ if self.miner else 'None'}


class GAE3MetricLearningLoss(_Loss):
  def __init__(self, loss_func, loss_weights, miner=None):
    self.ml_weights = loss_weights[:2]
    self.primitive_weight, self.ce_weight, self.recon_weight = loss_weights[2:]
    self.ml_loss = MetricLearningLoss(loss_func, self.ml_weights, miner)
    
  def loss(self, model_output, sample):
    if not isinstance(model_output, tuple):
      return pair_cross_entropy_loss(model_output, sample)
    attr_pred, obj_pred, pair_pred, img_feats, pair_emb, nodes, model = model_output
    primitive_loss, primitive_loss_dict = primitive_cross_entropy_loss((attr_pred, obj_pred), sample)
    ce_loss, ce_loss_dict = pair_cross_entropy_loss(pair_pred, sample)
    ml_total_loss, ml_loss_dict = self.ml_loss((img_feats, pair_emb), sample)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = ml_total_loss + self.primitive_weight * primitive_loss + self.ce_weight * ce_loss + self.recon_weight * recon_loss
    loss_dict = {'recon_loss': recon_loss} | primitive_loss_dict | ce_loss_dict | ml_loss_dict 
    return total_loss, loss_dict

  def hparam_dict(self):
    return self.ml_loss.hparam_dict() | {'loss_ratio': self.ml_weights + [self.primitive_weight, self.ce_weight, self.recon_weight]}
  

class GAE3MetricLearningEDLoss(_Loss):
  def __init__(self, loss_func, loss_weights, miner=None):
    self.ml_weights = loss_weights[:2]
    self.primitive_weight, self.pair_weight, self.recon_weight = loss_weights[2:]
    self.ml_loss = MetricLearningLoss(loss_func, ml_weights, miner)
    
  def loss(self, model_output, sample):
    if not isinstance(model_output, tuple):
      return pair_cross_entropy_loss(model_output, sample)
    attr_pred, obj_pred, pair_pred, img_feats, pair_emb, nodes, model = model_output
    primitive_loss, primitive_loss_dict = primitive_cross_entropy_loss((attr_pred, obj_pred), sample)
    ed_loss, ed_loss_dict = euclidean_dist_loss((img_feat, pair_emb), sample)
    ml_total_loss, ml_loss_dict = self.ml_loss((img_feats, pair_emb), sample)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = self.ml_total_loss + self.primitive_weight * primitive_loss + self.pair_weight * ed_loss + self.recon_weight * recon_loss
    loss_dict = {'recon_loss': recon_loss} | primitive_loss_dict | ed_loss_dict | ml_loss_dict 
    return total_loss, loss_dict

  def hparam_dict(self):
    return self.ml_loss.hparam_dict() | {'loss_ratio': self.ml_weights + [self.primitive_weight, self.pair_weight, self.recon_weight]}


class ReciprocalLoss(_Loss):
  def __init__(pre_loss_scale=1, adaptive_scale=False, total_epochs=None):
    self.pre_loss_scale = pre_loss_scale
    assert(not adaptive_scale or total_epochs)
    self.epoch = 0
    self.total_epochs = total_epochs
    self.adaptive_scale = adaptive_scale

  def loss(self, model_output, sample):
    attr_scores, obj_scores, attr_pre_scores, obj_pre_scores = model_output
    attr_labels = sample[1].to(dev)
    obj_labels = sample[2].to(dev)
    attr_loss = cross_entropy_loss(attr_scores, attr_labels)
    attr_pre_loss = cross_entropy_loss(attr_pre_scores, attr_labels)
    obj_loss = cross_entropy_loss(obj_scores, obj_labels)
    obj_pre_loss = cross_entropy_loss(obj_pre_scores, obj_labels)
    loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss, 'attr_pre_loss': attr_pre_loss, 'obj_pre_loss': obj_pre_loss}
    if self.adaptive_scale:
      # reduce pre_loss_scale linearly to 0 in the middle of training
      pre_loss_scale = max(0, 1-self.epoch/(self.total_epochs/2))
    total_loss = (1-self.pre_loss_scale) * (attr_loss + obj_loss) + self.pre_loss_scale * (attr_pre_loss + obj_pre_loss)
    self.epoch += 1
    return total_loss, loss_dict

  def hparam_dict(self):
    return {'pre_loss_ratio': self.pre_loss_scale, 'adaptive_scale': self.adaptive_scale}


def op_graph_emb_init(nattrs, nobjs, pairs):
  from scipy.sparse import coo_matrix
  w2v_attrs = torch.load('./embeddings/w2v_attrs.pt')
  w2v_objs = torch.load('./embeddings/w2v_objs.pt')
  ind = []
  values = []
  adj_dim = nattrs + nobjs + len(pairs)
  for i in range(nattrs):
    for j in range(nobjs):
#       if (i, j) in pairs:
#         ind.append([i, nattrs + j]) # attr -> obj
#         ind.append([i, dset.pair2idx[(attrs[i], objs[j])]+nattrs+nobjs]) # attr -> compo
#         ind.append([nattrs+j, dset.pair2idx[(attrs[i], objs[j])]+nattrs+nobjs]) # obj -> compo

      ind.append([i, nattrs + j]) # attr -> obj
      ind.append([i, i*nattrs+j+nattrs+nobjs]) # attr -> compo
      ind.append([nattrs+j, i*nattrs+j+nattrs+nobjs]) # obj -> compo
      
  back_ward = [(j, i) for i, j in ind]
  ind += back_ward
  for i in range(adj_dim):
    ind.append((i,i))

  ind = torch.tensor(ind)    
  values = torch.tensor([1] * len(ind), dtype=torch.float)

  adj = torch.sparse_coo_tensor(ind.T, values, (adj_dim, adj_dim)) # for pytorch-geometric version only this
  adj = coo_matrix(adj.to_dense())
  
  embs = []
  for i in range(nattrs):
    embs.append(w2v_attrs[i])
  for i in range(nobjs):
    embs.append(w2v_objs[i])
  for i, j in pairs:
    compo_emb = (w2v_attrs[i] + w2v_objs[j]) / 2
    embs.append(compo_emb)

  embs = torch.vstack(embs)
  
  return {"adj": adj, "embeddings": embs}