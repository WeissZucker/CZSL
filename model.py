import torch
import torch.nn as nn

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
  def __init__(self, in_features, out_features, num_layers=None):
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
    self.mlp = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.mlp(x)

def frozen(model):
  for param in model.parameters():
    param.requires_grad = False
  return model


class CompoResnet(nn.Module):
  def __init__(self, resnet_name, num_mlp_layers):
    super(CompoResnet, self).__init__()
    resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
    in_features = resnet.fc.in_features # 2048 for resnet101
    resnet.fc = Identity()
    self.resnet = resnet
    
    self.fc = HalvingMLP(in_features, 800, num_layers=num_mlp_layers)            
    self.obj_fc = HalvingMLP(400, OBJ_CLASS, 1)
    self.attr_fc = HalvingMLP(400, ATTR_CLASS, 1)

  def forward(self, sample):
    imgs= sample[4].to(dev)
    img_features = self.fc(imgs)
    obj_pred = self.obj_fc(img_features[:, :400])
    attr_pred = self.attr_fc(img_features[:, 400:])
    return obj_pred, attr_pred

class Contrastive(nn.Module):
  def __init__(self, resnet_name, num_mlp_layers, dataloader):
    super(Contrastive, self).__init__()
    resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
    in_features = resnet.fc.in_features # 2048 for resnet101
    self.init_word_emb()
    self.img_fc = HalvingMLP(in_features, 800, num_layers=num_mlp_layers)            
    self.pair_fc = HalvingMLP(self.word_emb_dim*2, 800, num_layers=num_mlp_layers)
    attr_ids = range(len(dataloader.dataset.attrs))
    obj_ids = range(len(dataloader.dataset.objs))
    all_pairs = product(attr_ids, obj_ids)
    all_pair_attrs, all_pair_objs = list(zip(*all_pairs))
    self.all_pair_attrs = torch.tensor(all_pair_attrs).to(dev)
    self.all_pair_objs = torch.tensor(all_pair_objs).to(dev)
    
  def init_word_emb(self):
    word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    vectors = torch.tensor(word2vec.vectors)
    self.w2v_emb = nn.Embedding.from_pretrained(vectors, freeze=True)
    self.w2v_idx_dict = word2vec.key_to_index
    self.word_emb_dim = vectors.shape[-1]
    
  def get_pair_features(self, attr_ids, obj_ids):
    attr_embs = self.w2v_emb(attr_ids)
    obj_embs = self.w2v_emb(obj_ids)
    pair_embs = torch.cat((attr_embs, obj_embs), dim=-1)
    return self.pair_fc(pair_embs)

  def _forward(self, imgs, attr_ids, obj_ids):
    img_features = self.img_fc(imgs)
    pair_features = self.get_pair_features(attr_ids, obj_ids)
    return img_features, pair_features
  
  def forward(self, sample):
    imgs = sample[4].to(dev)
    img_features = self.img_fc(imgs)
    all_pair_features = self.get_pair_features(self.all_pair_attrs, self.all_pair_objs)
    return torch.matmul(img_features, all_pair_features.T)

