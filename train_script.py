import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

import dataset

from train import *
from evaluator import *
from model import *
from utils import *
from graph_model import *
from gcn import *

from pytorch_metric_learning import losses, miners

torch.backends.cudnn.enabled = True
torch.manual_seed(12345)
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

  

model_name = "cge_ut"

feat_file = 'compcos.t7'
resnet_name = None #'resnet18'
with_image = resnet_name is not None

# for graph model
train_only = False
static_inp = True

take_compo_scores = True
open_world = True
lr = 5e-5
num_epochs = 200
batch_size = 128

hparam = HParam()
hparam.add_dict({'lr': lr, 'batchsize': batch_size})

# =======   Dataset & Evaluator  =======
dataset_name = 'UTg'
data_folder = dataset_name if dataset_name[-1] != 'g' else dataset_name[:-1]

train_dataloader = dataset.get_dataloader(dataset_name, 'train', feature_file=feat_file, batchsize=batch_size, with_image=with_image, open_world=open_world, 
                                          train_only=train_only, shuffle=True)
val_dataloader = dataset.get_dataloader(dataset_name, 'test', feature_file=feat_file, batchsize=batch_size, with_image=with_image, open_world=open_world)
dset = train_dataloader.dataset
nbias = 20
val_evaluator = Evaluator(val_dataloader, nbias, take_compo_scores=take_compo_scores)

# ======  Load HParam from checkpoint =======
try:
  model_name
except NameError:
  model_name = 'tmp'
model_dir = './models/'
model_path = os.path.join(model_dir, model_name+'.pt')

if model_name == 'tmp' and os.path.isfile(model_path):
  os.remove(model_path)
try:
  checkpoint = torch.load(model_path)
except FileNotFoundError:
  checkpoint = None

if checkpoint and 'hparam_dict' in checkpoint:
  hparam.add_dict(checkpoint['hparam_dict'])
  hparam.freeze = True

# ====     Model & Loss    ========
graph_name = 'graph_primitive.pt'
graph_path = os.path.join('./embeddings', data_folder, graph_name)

# model = ResnetDoubleHead(resnet_name, [768,1024,1200]).to(dev)
# model = Contrastive(train_dataloader, num_mlp_layers=1).to(dev)
# model = Contrastive(train_dataloader, mlp_layer_sizes=[768,1024,1200], train_only=train_only).to(dev)
# model = ReciprocalClassifier(resnet_name, img_mlp_layer_sizes=[1000], projector_mlp_layer_sizes=[1200,1150, 1000]).to(dev)
# model = PrimitiveContrastive(train_dataloader).to(dev)
# model = SemanticReciprocalClassifier(train_dataloader, [1000, 1300, 1500], resnet_name = resnet_name).to(dev)
# model = SemanticReciprocalClassifierOracle(train_dataloader, [1000, 1300, 1500]).to(dev)
# model = SemanticReciprocalClassifierGBU(train_dataloader, [700, 800, 900]).to(dev)
# model = SemanticReciprocalClassifier2CompoOutput(train_dataloader, [768,1024]).to(dev)
# model = GraphModel(dset, './embeddings/graph_primitive.pt', train_only=train_only, static_inp=static_inp).to(dev)
# model = UnimodalContrastive(train_dataloader).to(dev)
# model = GraphMLP(hparam, dset, graph_path=graph_path, static_inp=static_inp, resnet_name=resnet_name).to(dev)
model = CGE(dset, train_only=train_only, static_inp=static_inp, graph_path='./embeddings/graph_op.pt').to(dev)
# model = ReciprocalClassifierGraph(dset, './embeddings/graph_primitive.pt', [1000, 1300, 1500], resnet_name = resnet_name).to(dev)
# model = ReciprocalClassifierAttn(dset, [700, 800, 900], graph_path='./embeddings/graph_op.pt', resnet_name = resnet_name).to(dev)
# model = GAE(dset, graph_path='./embeddings/graph_primitive.pt', static_inp=static_inp, resnet_name=resnet_name, pretrained_gae=None).to(dev)
model = GAEStage3(hparam, dset, graph_path=graph_path, static_inp=static_inp, train_only=train_only, resnet_name=resnet_name, pretrained_gae=None, pretrained_mlp=None).to(dev)
# model = GAEBiD(hparam, dset, graph_path=graph_path, static_inp=static_inp, resnet_name=resnet_name, pretrained_gae=None, pretrained_mlp=None).to(dev)
# model = GAEStage3ED(hparam, dset, graph_path=graph_path, static_inp=static_inp, resnet_name=resnet_name, pretrained_gae=None, pretrained_mlp=None).to(dev)

optimizer = torch.optim.Adam(model.parameters(), lr=hparam.lr)

# criterion = contrastive_hinge_loss
# criterion = pair_cross_entropy_loss
# criterion = contrastive_triplet_loss4
# criterion = contrastive_triplet_loss
# criterion = primitive_cross_entropy_loss
# criterion = reciprocal_cross_entropy_loss(pre_loss_scale=0, adaptive_scale=False, total_epochs=200)
# criterion = unimodal_contrastive_loss(adaptive_scale=True, warm_up_steps=10)
# criterion = three_loss
# criterion = gae_loss(recon_loss_ratio=0.4)
# criterion = gae_stage_3_loss([0.4, 0.4, 1, 0.2])
# criterion = gae_stage_3_npair_loss([0.4, 0.4, 0, 0.2])
# criterion = triplet_loss_x(10)
# criterion = gae_stage_3_triplet_loss([0.4, 0.4, 0, 0.2], 20)

# hparam.add('margin', 0.1)
# criterion = EuclidNpairLoss(hparam.margin)

# criterion = npair_loss
ml_loss = losses.NPairsLoss()
miner = miners.BatchHardMiner()
hparam.add('loss_weights', [0.8, 0.8, 0.4, 1, 0.2])
# criterion = GAENPLoss(hparam.loss_weights)
criterion = GAEMLLoss(ml_loss, loss_weights=hparam.loss_weights, miner=miner)
# criterion = GAEEDLoss(ml_loss, loss_weights=hparam.loss_weights, miner=miner)

hparam.add_dict(criterion.hparam_dict)

# === Restore model and logger from Checkpoint ===
curr_epoch = 0
best = {'OpAUC':-1, 'best_epoch':-1}
log_dir = None

if checkpoint:
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if 'best_auc' in checkpoint:
    best['OpAUC'] = checkpoint['best_auc']
  if 'epoch' in checkpoint:
    curr_epoch = checkpoint['epoch']
  if 'log_dir' in checkpoint:
    log_dir = checkpoint['log_dir']
  if 'log_name_suffix' in checkpoint:
    filename_suffix = checkpoint['log_name_suffix']
  del checkpoint
  print('Model loaded.')

if log_dir:
  logger = SummaryWriter(log_dir, filename_suffix=model_name+filename_suffix)
else:
  if model_name == 'tmp':
    logger = DummyLogger()
  else:
    datetime_str = datetime.datetime.today().strftime("%d_%m-%H-%M-%S")
    logger = SummaryWriter("runs/"+model_name+' '+datetime_str)
print(f"Logging to: {logger.log_dir}")

try:
  train(model, hparam, optimizer, criterion, num_epochs, batch_size, train_dataloader, val_dataloader, logger, val_evaluator,
        curr_epoch=curr_epoch, best=best, save_path=model_path, open_world=open_world)
except KeyboardInterrupt:
  print("Training stopped.")
finally:
  logger.add_text(f'hparam/{model_name}', repr(hparam.hparam_dict | best))
  logger.flush()
  logger.close()
  print('Best auc:', best['OpAUC'])