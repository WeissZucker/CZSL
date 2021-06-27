from symnet.utils import dataset as dataset
import os
import torch
import torch.nn as nn


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class DummyLogger():
  log_dir = None
  def add_scalar(self, a, b, c):
    pass
  
  def flush(self):
    pass
  
cross_entropy_loss = nn.CrossEntropyLoss()

def primitive_cross_entropy_loss(model_output, sample):
  attr_scores, obj_scores = model_output
  attr_labels = sample[1].to(dev)
  obj_labels = sample[2].to(dev)
  attr_loss = cross_entropy_loss(attr_scores, attr_labels)
  obj_loss = cross_entropy_loss(obj_scores, obj_labels)
  loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss}
  total_loss = attr_loss + obj_loss
  return total_loss, loss_dict

def reciprocal_cross_entropy_loss(model_output, sample):
  attr_scores, obj_scores, attr_pre_scores, obj_pre_scores = model_output
  attr_labels = sample[1].to(dev)
  obj_labels = sample[2].to(dev)
  attr_loss = cross_entropy_loss(attr_scores, attr_labels)
  attr_pre_loss = cross_entropy_loss(attr_pre_scores, attr_labels)
  obj_loss = cross_entropy_loss(obj_scores, obj_labels)
  obj_pre_loss = cross_entropy_loss(obj_pre_scores, obj_labels)
  loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss, 'attr_pre_loss': attr_pre_loss, 'obj_pre_loss': obj_pre_loss}
  pre_loss_scale = 1
  total_loss = attr_loss + obj_loss + pre_loss_scale * (attr_pre_loss + obj_pre_loss)
  return total_loss, loss_dict

def contrastive_cross_entropy_loss(model_output, sample):
  compo_score = model_output # [batch_size, npairs]
  pair_labels = sample[3].to(dev)
  loss = cross_entropy_loss(compo_score, pair_labels)
  loss_dict = {'contra_loss': loss}
  return loss, loss_dict

def contrastive_hinge_loss(model_output, sample, margin=0.1):
  compo_score = model_output # [batch_size, npairs]
  pair_labels = sample[3].to(dev)
  target_score = compo_score[range(len(compo_score)), pair_labels].unsqueeze(-1)
  loss = compo_score - target_score + margin
  loss = torch.max(loss, torch.zeros_like(loss))
  loss = torch.mean(torch.mean(loss, dim=-1))
  loss_dict = {'contra_loss': loss}
  return loss, loss_dict