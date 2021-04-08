import torch
import torch.nn as nn

from functools import partial
from symnet.utils import dataset

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
    
    MLP = partial(HalvingMLP, num_layers=num_mlp_layers)
    self.obj_fc = MLP(in_features, OBJ_CLASS)
    self.attr_fc = MLP(in_features, ATTR_CLASS)

  def forward(self, x):
    img_features = self.resnet(x)
    obj_pred = self.obj_fc(img_features)
    attr_pred = self.attr_fc(img_features)
    return obj_pred, attr_pred