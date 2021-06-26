import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from symnet.utils import dataset
from itertools import product
from gensim.models import KeyedVectors

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

dset = dataset.get_dataloader('MIT', 'train', batchsize=64, with_image=False).dataset
OBJ_CLASS = len(dset.objs)
ATTR_CLASS = len(dset.attrs)
del dset


class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

  
class HalvingMLP(nn.Module):
  '''Output size of each layer except the last one is half of the input size'''
  def __init__(self, in_features, out_features, num_layers=None, norm_output=False):
    super(HalvingMLP, self).__init__()
    layers = []
    for i in range(num_layers):
      layer = nn.Sequential(
        nn.Linear(in_features, in_features//2),
        nn.BatchNorm1d(in_features//2),
        nn.ReLU(),
        nn.Dropout())
      layers.append(layer)
      in_features //= 2
    layers.append(nn.Linear(in_features, out_features))
    if norm_output:
      layers.append(nn.LayerNorm(out_features, elementwise_affine=False))
    self.mlp = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.mlp(x)
  
class ParametricMLP(nn.Module):
  '''Output size of each layer specified by [layer_sizes]'''
  def __init__(self, in_features, out_features, layer_sizes, norm_output=False):
    super(ParametricMLP, self).__init__()
    layers = []
    for layer_size in layer_sizes:
      layer = nn.Sequential(
        nn.Linear(in_features, layer_size),
        nn.BatchNorm1d(layer_size),
        nn.ReLU(),
        nn.Dropout())
      layers.append(layer)
      in_features = layer_size
    layers.append(nn.Linear(in_features, out_features))
    if norm_output:
      layers.append(nn.LayerNorm(out_features, elementwise_affine=False))
    self.mlp = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.mlp(x)

def frozen(model):
  for param in model.parameters():
    param.requires_grad = False
  return model


class DoubleClassifier(nn.Module):
  def __init__(self, resnet_name, mlp_layer_sizes=None, num_mlp_layers=1):
    super(DoubleClassifier, self).__init__()
    resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
    in_features = resnet.fc.in_features # 2048 for resnet101
    resnet.fc = Identity()
    self.resnet = resnet
    
    self.img_emb_size = 800
    self.obj_classifier_input_size = 400
    assert self.img_emb_size > self.obj_classifier_input_size
    
    if mlp_layer_sizes is not None:
      assert isinstance(mlp_layer_sizes, list)
      self.fc = ParametricMLP(in_features, self.img_emb_size, mlp_layer_sizes)
    else:
      self.fc = HalvingMLP(in_features, self.img_emb_size, num_layers=num_mlp_layers)            
    self.obj_fc = HalvingMLP(self.obj_classifier_input_size, OBJ_CLASS, 1)
    self.attr_fc = HalvingMLP(self.img_emb_size-self.obj_classifier_input_size, ATTR_CLASS, 1)

  def forward(self, sample):
    imgs= sample[4].to(dev)
    img_features = self.fc(imgs)
    obj_pred = self.obj_fc(img_features[:, :self.obj_classifier_input_size])
    attr_pred = self.attr_fc(img_features[:, self.obj_classifier_input_size:])
    return attr_pred, obj_pred


class Contrastive(nn.Module):
  def __init__(self, dataloader, mlp_layer_sizes=[], num_mlp_layers=1, resnet_name=None):
    super(Contrastive, self).__init__()
    if resnet_name:
      self.resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
      self.img_emb_dim = self.resnet.fc.in_features # 2048 for resnet101
      self.resnet.fc = Identity()
    else:
      self.resnet = None
      sample = next(iter(dataloader))
      self.img_emb_dim = sample[4].size(-1)
      
    word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    vectors = torch.tensor(word2vec.vectors)
    w2v_emb = nn.Embedding.from_pretrained(vectors, freeze=True)
    w2v_idx_dict = word2vec.key_to_index
    self.word_emb_dim = vectors.shape[-1]
    self.init_pair_embs(dataloader, w2v_emb, w2v_idx_dict)
    
    self.compo_dim = 800
    
    if mlp_layer_sizes is not None:
      assert isinstance(mlp_layer_sizes, list)
      self.img_fc = ParametricMLP(self.img_emb_dim, self.compo_dim, mlp_layer_sizes)
    else:
      self.img_fc = HalvingMLP(self.img_emb_dim, self.compo_dim, num_layers=num_mlp_layers)            

    self.pair_fc = HalvingMLP(self.word_emb_dim*2, self.compo_dim, num_layers=1)
    
  def init_pair_embs(self, dataloader, w2v_emb, w2v_idx_dict):
      attr_labels = dataloader.dataset.attrs
      obj_labels = dataloader.dataset.objs
      all_pairs = product(attr_labels, obj_labels)
      all_pairs_attr, all_pairs_obj = list(zip(*all_pairs))
      all_pairs_attr = torch.tensor([w2v_idx_dict[label] for label in all_pairs_attr])
      all_pairs_obj = torch.tensor([w2v_idx_dict[label] for label in all_pairs_obj])
      all_pairs_attr_embs = w2v_emb(all_pairs_attr)
      all_pairs_obj_embs = w2v_emb(all_pairs_obj)
      self.all_pairs_emb = torch.cat((all_pairs_attr_embs, all_pairs_obj_embs), dim=-1).to(dev)

  def forward(self, sample, similarity_score_scale=1):
    if self.resnet:
      imgs = self.resnet(sample[0].to(dev))
    else:
      imgs = sample[4].to(dev)
    img_features = F.normalize(self.img_fc(imgs), dim=1)
    all_pair_features = F.normalize(self.pair_fc(self.all_pairs_emb), dim=1)
  
    compo_scores = 20*torch.matmul(img_features, all_pair_features.T)
    return compo_scores
