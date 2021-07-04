import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from symnet.utils import dataset
from itertools import product
from gensim.models import KeyedVectors
import numpy as np
import fasttext

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

dset = dataset.get_dataloader('MIT', 'train', batchsize=64, with_image=False).dataset
OBJ_CLASS = len(dset.objs)
ATTR_CLASS = len(dset.attrs)
del dset

word2vec_path = './word_embedding/GoogleNews-vectors-negative300.bin'
fasttext_path = './word_embedding/fasttext.cc.en.300.bin'

def frozen(model):
  for param in model.parameters():
    param.requires_grad = False
  return model


def layer_norm(layer, output_size, affine=False):
    layernorm = nn.LayerNorm(output_size, elementwise_affine=affine)
    if isinstance(layer, nn.Sequential):
      layer.add_module(layernorm)
    else:
      return nn.Sequential(layer, layernorm)
    
def get_word2vec_embs(tokens, word2vec):
  embs = []
  for token in tokens:
    embs.append(torch.tensor(word2vec.get_vector(token)).unsqueeze(0))
  embs = torch.cat(embs)
  return embs
  
def get_fasttext_embs(tokens, ft):
  embs = []
  for token in tokens:
    embs.append(torch.tensor(ft.get_word_vector(token)).unsqueeze(0))
  embs = torch.cat(embs)
  return embs

class GBU(nn.Module):
  '''Gated Bimodal Unit for '''
  def __init__(self, dim):
    super(GBU, self).__init__()
    self.dim = dim
    self.gate_fc = nn.Linear(dim*2, dim)
    
  def forward(self, fa, fb):
    f = torch.cat((fa, fb), dim=1)
    gate = F.sigmoid(self.gate_fc(f))
    f = F.tanh(f)
    return gate*f[:, :self.dim] + (1-gate)*f[:, self.dim:], gate
  
class HalvingMLP(nn.Module):
  '''Output size of each layer except the last one is half of the input size'''
  def __init__(self, in_features, out_features, num_layers, norm_output=False):
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
  
def LinearShrinkMLP(in_features, out_features, num_layers, norm_output=False):
    layer_sizes = np.linspace(in_features, out_features, num_layers+2)
    layer_sizes = [int(s) for s in layer_sizes[1:-1]]
    return ParametricMLP(in_features, out_features, layer_sizes, norm_output)
  
class ParametricMLP(nn.Module):
  '''Output size of each inner layer specified by [layer_sizes]'''
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

class ResnetDoubleHead(nn.Module):
  def __init__(self, resnet_name, mlp_layer_sizes=None, num_mlp_layers=1):
    super(ResnetDoubleHead, self).__init__()
    resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
    in_features = resnet.fc.in_features # 2048 for resnet101
    self.resnet_head = resnet.fc
    resnet.fc = nn.Identity()
    self.resnet = resnet
    
    self.img_emb_size = 800
    
    if mlp_layer_sizes is not None:
      assert isinstance(mlp_layer_sizes, list)
      self.fc = ParametricMLP(in_features, self.img_emb_size, mlp_layer_sizes)
    else:
      self.fc = HalvingMLP(in_features, self.img_emb_size, num_layers=num_mlp_layers)            
    self.obj_fc = ParametricMLP(in_features, OBJ_CLASS, [1000, 800, 600])
    self.attr_fc = ParametricMLP(in_features, ATTR_CLASS, [1000, 800, 600])

  def forward(self, sample):
#     imgs= sample[4].to(dev)
#     img_features = self.fc(imgs)
    img_features = sample[4].to(dev)
    obj_pred = self.obj_fc(img_features)
    attr_pred = self.attr_fc(img_features)
    return attr_pred, obj_pred
  
  
class ReciprocalClassifier(nn.Module):
  def __init__(self, resnet_name, img_mlp_layer_sizes=None, projector_mlp_layer_sizes=None, num_mlp_layers=1):
    super(ReciprocalClassifier, self).__init__()
    resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
    in_features = resnet.fc.in_features # 2048 for resnet101
    self.resnet_head = resnet.fc
    resnet.fc = nn.Identity()
    self.resnet = resnet

#     self.img_emb_size = 1200
    self.img_emb_size = in_features
    self.obj_emb_size = 800
    self.attr_emb_size = 800
    
    if img_mlp_layer_sizes is not None:
      assert isinstance(img_mlp_layer_sizes, list)
      self.img_fc = ParametricMLP(in_features, self.img_emb_size, img_mlp_layer_sizes, norm_output=True)
    else:
      self.img_fc = HalvingMLP(in_features, self.img_emb_size, num_mlp_layers, norm_output=True)            
      
    self.obj_projector = layer_norm(nn.Linear(self.img_emb_size, self.obj_emb_size), self.obj_emb_size)
    self.obj_project_knowing_attr = ParametricMLP(self.img_emb_size+ATTR_CLASS, self.obj_emb_size, projector_mlp_layer_sizes, norm_output=True)
    
    self.attr_projector = layer_norm(nn.Linear(self.img_emb_size, self.attr_emb_size), self.attr_emb_size)
    self.attr_project_knowing_obj = ParametricMLP(self.img_emb_size+OBJ_CLASS, self.attr_emb_size, projector_mlp_layer_sizes, norm_output=True)
                                               
    self.obj_to_logits = ParametricMLP(self.obj_emb_size, OBJ_CLASS, [])
    self.attr_to_logits = ParametricMLP(self.attr_emb_size, ATTR_CLASS, [])

  def forward(self, sample):
    img_features = sample[4].to(dev)
#     img_features = self.img_fc(imgs)
    obj_pre_emb = self.obj_project_knowing_attr(torch.cat((img_features, torch.zeros((len(img_features), ATTR_CLASS)).to(dev)), dim=1))
    attr_pre_emb = self.attr_project_knowing_obj(torch.cat((img_features, torch.zeros((len(img_features), OBJ_CLASS)).to(dev)), dim=1))
    obj_pre_pred = F.normalize(self.obj_to_logits(obj_pre_emb), dim=1)
    attr_pre_pred = F.normalize(self.attr_to_logits(attr_pre_emb), dim=1)
    obj_emb = self.obj_project_knowing_attr(torch.cat((img_features, attr_pre_pred), dim=1))
    attr_emb = self.attr_project_knowing_obj(torch.cat((img_features, obj_pre_pred), dim=1))
    obj_pred = self.obj_to_logits(obj_emb)
    attr_pred = self.attr_to_logits(attr_emb)
    return attr_pred, obj_pred, attr_pre_pred, obj_pre_pred
  
class SemanticReciprocalClassifier(nn.Module):
  def __init__(self, dataloader, projector_mlp_layer_sizes, resnet_name=None):
    super(SemanticReciprocalClassifier, self).__init__()
    if resnet_name:
      self.resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
      self.img_feature_size = self.resnet.fc.in_features # 2048 for resnet101
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
      sample = next(iter(dataloader))
      self.img_feature_size = sample[4].size(-1)
    
    self.init_primitive_embs()

    self.obj_emb_size = 800
    self.attr_emb_size = 800        
      
    self.obj_project = ParametricMLP(self.img_feature_size+self.word_emb_size, self.obj_emb_size, projector_mlp_layer_sizes, norm_output=True)
    self.attr_project = ParametricMLP(self.img_feature_size+self.word_emb_size, self.attr_emb_size, projector_mlp_layer_sizes, norm_output=True)
                                               
    self.obj_to_logits = ParametricMLP(self.obj_emb_size, OBJ_CLASS, [])
    self.attr_to_logits = ParametricMLP(self.attr_emb_size, ATTR_CLASS, [])
    
  def init_primitive_embs(self):
    w2v_attrs = torch.load('./word_embedding/w2v_attrs.pt')
    w2v_objs = torch.load('./word_embedding/w2v_objs.pt')
    ft_attrs = torch.load('./word_embedding/ft_attrs.pt')
    ft_objs = torch.load('./word_embedding/ft_objs.pt')
#     self.all_attr_embs = torch.cat((w2v_attrs, ft_attrs), dim=1).to(dev)
#     self.all_obj_embs = torch.cat((w2v_objs, ft_objs), dim=1).to(dev)
    self.all_attr_embs = ft_attrs.to(dev)
    self.all_obj_embs = ft_objs.to(dev)
    self.word_emb_size = self.all_attr_embs.shape[-1]

  def forward(self, sample):
    img_features = sample[4].to(dev)
    obj_pre_emb = self.obj_project(torch.cat((img_features, torch.zeros((len(img_features), self.word_emb_size)).to(dev)), dim=1))
    attr_pre_emb = self.attr_project(torch.cat((img_features, torch.zeros((len(img_features), self.word_emb_size)).to(dev)), dim=1))
    softmax_scale = 2
    obj_pre_pred = F.softmax(softmax_scale*self.obj_to_logits(obj_pre_emb), dim=1)
    attr_pre_pred = F.softmax(softmax_scale*self.attr_to_logits(attr_pre_emb), dim=1)
    attr_semantic = torch.matmul(attr_pre_pred, self.all_attr_embs) 
    obj_semantic = torch.matmul(obj_pre_pred, self.all_obj_embs)
    obj_emb = self.obj_project(torch.cat((img_features, attr_semantic), dim=1))
    attr_emb = self.attr_project(torch.cat((img_features, obj_semantic), dim=1))
    obj_pred = self.obj_to_logits(obj_emb)
    attr_pred = self.attr_to_logits(attr_emb)
    return attr_pred, obj_pred, attr_pre_pred, obj_pre_pred
  
class SemanticReciprocalClassifierGBU(nn.Module):
  def __init__(self, dataloader, projector_mlp_layer_sizes, resnet_name=None):
    super(SemanticReciprocalClassifierGBU, self).__init__()
    if resnet_name:
      self.resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
      self.img_feature_size = self.resnet.fc.in_features # 2048 for resnet101
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
      sample = next(iter(dataloader))
      self.img_feature_size = sample[4].size(-1)
    
    self.init_primitive_embs()

    self.primitive_emb_size = 600
      
    self.obj_projector = ParametricMLP(self.img_feature_size, self.primitive_emb_size, projector_mlp_layer_sizes, norm_output=True)
    self.attr_projector = ParametricMLP(self.img_feature_size, self.primitive_emb_size, projector_mlp_layer_sizes, norm_output=True)
    self.semantic_projector = ParametricMLP(self.word_emb_size, self.primitive_emb_size, [], norm_output=True)
                                               
    self.obj_to_logits = ParametricMLP(self.primitive_emb_size, OBJ_CLASS, [500])
    self.attr_to_logits = ParametricMLP(self.primitive_emb_size, ATTR_CLASS, [500])
    
    self.attr_gbu = GBU(self.primitive_emb_size)
    self.obj_gbu = GBU(self.primitive_emb_size)
    
  def init_primitive_embs(self):
    w2v_attrs = torch.load('./word_embedding/w2v_attrs.pt')
    w2v_objs = torch.load('./word_embedding/w2v_objs.pt')
    ft_attrs = torch.load('./word_embedding/ft_attrs.pt')
    ft_objs = torch.load('./word_embedding/ft_objs.pt')
    self.all_attr_embs = torch.cat((w2v_attrs, ft_attrs), dim=1).to(dev)
    self.all_obj_embs = torch.cat((w2v_objs, ft_objs), dim=1).to(dev)
#     self.all_attr_embs = w2v_attrs.to(dev)
#     self.all_obj_embs = w2v_objs.to(dev)
    self.word_emb_size = self.all_attr_embs.shape[-1]

  def forward(self, sample):
    img_features = sample[4].to(dev)
    obj_visual_emb = self.obj_projector(img_features)
    attr_visual_emb = self.attr_projector(img_features)
    
    softmax_scale = 2
    obj_pre_pred = F.softmax(softmax_scale*self.obj_to_logits(obj_visual_emb), dim=1)
    attr_pre_pred = F.softmax(softmax_scale*self.attr_to_logits(attr_visual_emb), dim=1)
    
    attr_semantic = self.semantic_projector(torch.matmul(attr_pre_pred, self.all_attr_embs))
    obj_semantic = self.semantic_projector(torch.matmul(obj_pre_pred, self.all_obj_embs))
    

    attr_emb, _ = self.attr_gbu(attr_visual_emb, attr_semantic)
    obj_emb, _ = self.obj_gbu(obj_visual_emb, obj_semantic)
    
    obj_pred = self.obj_to_logits(obj_emb)
    attr_pred = self.attr_to_logits(attr_emb)
    return attr_pred, obj_pred, attr_pre_pred, obj_pre_pred


class Contrastive(nn.Module):
  def __init__(self, dataloader, mlp_layer_sizes=[], num_mlp_layers=1, resnet_name=None):
    super(Contrastive, self).__init__()
    if resnet_name:
      self.resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
      self.img_emb_dim = self.resnet.fc.in_features # 2048 for resnet101
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
      sample = next(iter(dataloader))
      self.img_emb_dim = sample[4].size(-1)
      
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
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
  
class PrimitiveContrastive(nn.Module):
  def __init__(self, dataloader, resnet_name=None):
    super(PrimitiveContrastive, self).__init__()
    if resnet_name:
      self.resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
      self.img_feature_size = self.resnet.fc.in_features # 2048 for resnet101
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
      sample = next(iter(dataloader))
      self.img_feature_size = sample[4].size(-1)
      
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    vectors = torch.tensor(word2vec.vectors)
    w2v_emb = nn.Embedding.from_pretrained(vectors, freeze=True)
    w2v_idx_dict = word2vec.key_to_index
    self.word_emb_size = vectors.shape[-1]
    self.init_primitive_embs(dataloader, w2v_emb, w2v_idx_dict)
    
    self.shared_embedding_size = 400
    self.visual_obj_mlp = ParametricMLP(self.img_feature_size, self.shared_embedding_size, [1000, 800, 600], norm_output=True)
    self.visual_attr_mlp = ParametricMLP(self.img_feature_size, self.shared_embedding_size, [1000, 800, 600], norm_output=True)
    self.semantic_obj_mlp = ParametricMLP(self.word_emb_size, self.shared_embedding_size, [500, 450], norm_output=True)
    self.semantic_attr_mlp = ParametricMLP(self.word_emb_size, self.shared_embedding_size, [500, 450], norm_output=True)
    
  def init_primitive_embs(self, dataloader, w2v_emb, w2v_idx_dict):
    all_attr_idx = torch.tensor([w2v_idx_dict[attr] for attr in dataloader.dataset.attrs])
    all_obj_idx = torch.tensor([w2v_idx_dict[obj] for obj in dataloader.dataset.objs])
    self.all_attr_embs = w2v_emb(all_attr_idx).to(dev)
    self.all_obj_embs = w2v_emb(all_obj_idx).to(dev)


  def forward(self, sample):
    img_features = sample[4].to(dev)
#     img_features = self.fc(imgs)
    visual_attr_emb = self.visual_attr_mlp(img_features)
    visual_obj_emb = self.visual_obj_mlp(img_features)
    semantic_attr_emb = self.semantic_attr_mlp(self.all_attr_embs)
    semantic_obj_emb = self.semantic_obj_mlp(self.all_obj_embs)
    attr_scores = torch.matmul(visual_attr_emb, semantic_attr_emb.T)
    obj_scores = torch.matmul(visual_obj_emb, semantic_obj_emb.T)

    return attr_scores, obj_scores