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
  
  def flush(self):
    pass
  
cross_entropy_loss = nn.CrossEntropyLoss()

def gae_loss(recon_loss_ratio=0.1):
  def loss(model_output, sample):
    if not isinstance(model_output, tuple): # during evaluation
      return contrastive_cross_entropy_loss(model_output, sample)
    pair_pred, nodes, model = model_output
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    pair_labels = sample[3].to(dev)
    ce_loss = cross_entropy_loss(pair_pred, pair_labels)
    total_loss = (1-recon_loss_ratio) * ce_loss + recon_loss_ratio * recon_loss
    loss_dict = {'contra_loss': ce_loss, 'recon_loss':recon_loss}
    return total_loss, loss_dict
  return loss

def gae_stage_3_loss(loss_weights):
  attr_wt, obj_wt, pair_wt, recon_wt = loss_weights
  def loss(model_output, sample):
    if len(model_output) != 5: # during evaluation
#       return primitive_cross_entropy_loss(model_output, sample)
      return contrastive_cross_entropy_loss(model_output, sample)
    attr_label, obj_label, pair_label = sample[1].to(dev), sample[2].to(dev), sample[3].to(dev)
    attr_pred, obj_pred, pair_pred, nodes, model = model_output
    attr_loss = cross_entropy_loss(attr_pred, attr_label)
    obj_loss = cross_entropy_loss(obj_pred, obj_label)
    pair_loss = cross_entropy_loss(pair_pred, pair_label)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = attr_wt * attr_loss + obj_wt * obj_loss + pair_wt * pair_loss + recon_wt * recon_loss
    loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss, 'contra_loss': pair_loss, 'recon_loss':recon_loss}
    return total_loss, loss_dict
  return loss


def metric_learning_loss(loss_func, loss_weights=[1, 1], miner=None):
  img_loss_weight, text_loss_weight = loss_weights
  def loss(model_output, sample):
    img_feats, pair_emb = model_output
    pair_id = sample[3]
    img_miner_output = miner(img_feats, pair_id) if miner else None
    img_loss = loss_func(img_feats, pair_id, indices_tuple=img_miner_output)
    text_miner_output = miner(pair_emb, pair_id) if miner else None
    text_loss = loss_func(pair_emb, pair_id, indices_tuple=text_miner_output)
    total_loss = img_loss_weight * img_loss +  text_loss_weight * text_loss
    loss_dict = {'img_loss': img_loss, 'text_loss': text_loss}
    return total_loss, loss_dict
  return loss


def gae_stage_3_metric_learning_loss(loss_func, loss_weights, miner=None):
  ml_weights = loss_weights[:2]
  primitive_weight, ce_weight, recon_weight = loss_weights[2:]
  ml_loss = metric_learning_loss(loss_func, ml_weights, miner)
  def loss(model_output, sample):
    if not isinstance(model_output, tuple):
      return contrastive_cross_entropy_loss(model_output, sample)
    attr_pred, obj_pred, pair_pred, img_feats, pair_emb, nodes, model = model_output
    primitive_loss, primitive_loss_dict = primitive_cross_entropy_loss((attr_pred, obj_pred), sample)
    ce_loss, ce_loss_dict = contrastive_cross_entropy_loss(pair_pred, sample)
    ml_total_loss, ml_loss_dict = ml_loss((img_feats, pair_emb), sample)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = ml_total_loss + primitive_weight * primitive_loss + ce_weight * ce_loss + recon_weight * recon_loss
    loss_dict = {'recon_loss': recon_loss} | primitive_loss_dict | ce_loss_dict | ml_loss_dict 
    return total_loss, loss_dict
  return loss
  

def primitive_cross_entropy_loss(model_output, sample):
  attr_scores, obj_scores = model_output
  attr_labels = sample[1].to(dev)
  obj_labels = sample[2].to(dev)
  attr_loss = cross_entropy_loss(attr_scores, attr_labels)
  obj_loss = cross_entropy_loss(obj_scores, obj_labels)
  loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss}
  total_loss = attr_loss + obj_loss
  return total_loss, loss_dict

def reciprocal_cross_entropy_loss(pre_loss_scale=1, adaptive_scale=False, total_epochs=None):
  assert(0<=pre_loss_scale<=1)
  assert(not adaptive_scale or total_epochs)
  epoch = 0
  def loss(model_output, sample):
    nonlocal epoch, pre_loss_scale
    attr_scores, obj_scores, attr_pre_scores, obj_pre_scores = model_output
    attr_labels = sample[1].to(dev)
    obj_labels = sample[2].to(dev)
    attr_loss = cross_entropy_loss(attr_scores, attr_labels)
    attr_pre_loss = cross_entropy_loss(attr_pre_scores, attr_labels)
    obj_loss = cross_entropy_loss(obj_scores, obj_labels)
    obj_pre_loss = cross_entropy_loss(obj_pre_scores, obj_labels)
    loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss, 'attr_pre_loss': attr_pre_loss, 'obj_pre_loss': obj_pre_loss}
    if adaptive_scale:
      # reduce pre_loss_scale linearly to 0 in the middle of training
      pre_loss_scale = max(0, 1-epoch/(total_epochs/2))
    total_loss = (1-pre_loss_scale) * (attr_loss + obj_loss) + pre_loss_scale * (attr_pre_loss + obj_pre_loss)
    epoch += 1
    return total_loss, loss_dict
  return loss


def contrastive_cross_entropy_loss(model_output, sample):
  compo_score = model_output # [batch_size, npairs]
  pair_labels = sample[3].to(dev)
  loss = cross_entropy_loss(compo_score, pair_labels)
  loss_dict = {'contra_loss': loss}
  return loss, loss_dict


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
