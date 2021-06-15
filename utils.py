from symnet.utils import dataset as symnet_dataset
import os
import torch.nn as nn

class DummyLogger():
  def add_scalar(self, a, b, c):
    pass
  
  def flush(self):
    pass
  
cross_entropy_loss = nn.CrossEntropyLoss()
  
def primitive_scores_criterion(model_output, sample):
  attr_scores, obj_scores = model_output
  attr_labels = sample[1].to(dev)
  obj_labels = sample[2].to(dev)
  attr_loss = cross_entropy_loss(attr_scores, attr_labels)
  obj_loss = cross_entropy_loss(obj_scores, obj_labels)
  loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss}
  total_loss = attr_loss + obj_loss
  return total_loss, loss_dict

def compo_scores_criterion(model_output, sample):
  return None, None
'''
  compo_score = model_output
  attr_labels = sample[1].to(dev)
  obj_labels = sample[2].to(dev)
  attr_loss = cross_entropy_loss(attr_score, attr_labels)
  obj_loss = cross_entropy_loss(obj_score, obj_labels)
  loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss}
  total_loss = attr_loss + obj_loss
  return total_loss, loss_dict
'''