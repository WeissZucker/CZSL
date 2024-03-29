import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os, sys, argparse
import datetime

import dataset

from train import *
from evaluator import *
from model import *
from graph_model import *
from retrieval_model import *
from utils import *

from gcn import *

from pytorch_metric_learning import losses, miners

torch.manual_seed(12345)

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
  
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='only evaluation, no training.')
parser.add_argument('--dry', action='store_true', help='dry run, set model_name to "tmp".')
args = parser.parse_args()



model_name = "gae_cgqa_cw"

dataset_name = 'CGQAg'
open_world = False
train_only = True
cpu_eval = True
take_compo_scores = True

feat_file = 'features.t7'#'resnet50.pt'
resnet_name = None #'resnet18'
static_inp = True
resnet_lr = 5e-6
with_image = resnet_name is not None

lr = 5e-5
weight_decay = 0
num_epochs = 180
batch_size = 128

eval_every = 3

hparam = HParam()
hparam.add_dict({'lr': lr, 'batchsize': batch_size, 'wd': weight_decay,
                'open_world': open_world, 'train_only': train_only})
if resnet_name and not static_inp:
  hparam.add_dict({'resnet': resnet_name, 'resnet_lr': resnet_lr})

# =======   Dataset & Evaluator  =======
data_folder = dataset_name if dataset_name[-1] != 'g' else dataset_name[:-1]
rand_sampling=False
ignore_objs = []

# ignore_objs = [
#             'armor', 'bracelet', 'bush', 'camera', 'candy', 'castle',
#             'ceramic', 'cheese', 'clock', 'clothes', 'coffee', 'fan', 'fig',
#             'fish', 'foam', 'forest', 'fruit', 'furniture', 'garden', 'gate',
#             'glass', 'horse', 'island', 'laptop', 'lead', 'lightning',
#             'mirror', 'orange', 'paint', 'persimmon', 'plastic', 'plate',
#             'potato', 'road', 'rubber', 'sand', 'shell', 'sky', 'smoke',
#             'steel', 'stream', 'table', 'tea', 'tomato', 'vacuum', 'wax',
#             'wheel', 'window', 'wool'
#             ]

# ignore_objs = [
#             'Shoes.Boat.Shoes',
#             'Boots.Knee.High'
#             ]

# rand_sampling = True

train_dataloader = dataset.get_dataloader(dataset_name, 'train', feature_file=feat_file, batchsize=batch_size, with_image=with_image, open_world=open_world, 
                                          train_only=train_only, shuffle=True, random_sampling=rand_sampling, ignore_objs=ignore_objs)
val_set = 'test' if args.eval else 'val'
val_dataloader = dataset.get_dataloader(dataset_name, val_set, feature_file=feat_file, batchsize=batch_size, with_image=with_image,
                                        open_world=open_world, random_sampling=rand_sampling, ignore_objs=ignore_objs)
dset = train_dataloader.dataset
nbias = 20


# ======  Load HParam from checkpoint =======
if args.dry:
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
  hparam.freeze()

# hparam.add_dict({'graph_encoder_layers': [1024]})

# ====     Model & Loss    ========
graph_name = 'graph_primitive.pt'
graph_path = os.path.join('./embeddings', data_folder, graph_name)

# model = ResnetDoubleHead(resnet_name, [768,1024,1200]).to(dev)
# model = Contrastive(train_dataloader, num_mlp_layers=1).to(dev)
# model = Contrastive(train_dataloader, mlp_layer_sizes=[768,1024,1200], train_only=train_only).to(dev)
# model = ReciprocalClassifier(resnet_name, img_mlp_layer_sizes=[1000], projector_mlp_layer_sizes=[1200,1150, 1000]).to(dev)
# model = PrimitiveContrastive(train_dataloader).to(dev)
# model = SemanticReciprocalClassifier(train_dataloader, [1000, 1300, 1500], resnet_name = resnet_name).to(dev)
# model = GraphModel(dset, './embeddings/graph_primitive.pt', train_only=train_only).to(dev)
# model = GraphMLP(hparam, dset, graph_path=graph_path, resnet_name=resnet_name).to(dev)
# model = CGE(hparam, dset, train_only=train_only, graph_path=graph_path).to(dev)
# model = ReciprocalClassifierGraph(dset, './embeddings/graph_primitive.pt', [1000, 1300, 1500], resnet_name = resnet_name).to(dev)
model = GAE(hparam, dset, graph_path=graph_path, train_only=train_only, resnet_name=resnet_name).to(dev)
# model = GAEIR(hparam, dset, graph_path=graph_path, train_only=train_only, resnet_name=resnet_name, static_inp=static_inp).to(dev)
# model = CGEIR(hparam, dset, train_only=train_only, graph_path=graph_path).to(dev)
# model = GAE_IR_Bert(hparam, dset, graph_path=graph_path, train_only=train_only, resnet_name=resnet_name, static_inp=static_inp).to(dev)
# attr_emb_path = os.path.join('./embeddings/', data_folder, 'w2v_attrs.pt')
# obj_emb_path = os.path.join('./embeddings/', data_folder, 'w2v_objs.pt')
# model = CompcosIR(hparam, dset, attr_emb_path, obj_emb_path, train_only=train_only, static_inp=static_inp, graph_path=graph_path, resnet_name=resnet_name).to(dev)


model_params, resnet_params = [], []
for name, param in model.named_parameters():
  if name.split('.')[0] == 'resnet':
    resnet_params.append(param)
  else:
    model_params.append(param)
params = [{'params': model_params}]
if not static_inp:
  params.append({'params': resnet_params, 'lr': resnet_lr})
optimizer = torch.optim.Adam(params, lr=hparam.lr, weight_decay=hparam.wd)

# criterion = pair_cross_entropy_loss
# criterion = primitive_cross_entropy_loss
# criterion = reciprocal_cross_entropy_loss(pre_loss_scale=0, adaptive_scale=False, total_epochs=200)
# criterion = gae_stage_3_loss([0.4, 0.4, 1, 0.2])
# criterion = gae_stage_3_npair_loss([0.4, 0.4, 0, 0.2])
# criterion = gae_stage_3_triplet_loss([0.4, 0.4, 0, 0.2], 20)
# criterion = npair_loss
# criterion = batch_ce_loss
# criterion = obj_aware_loss

hparam.add('recon_loss_ratio', 0.1)
criterion = GAELoss(recon_loss_ratio=hparam.recon_loss_ratio)


# hparam.add('margin', 0.1)
# criterion = EuclidNpairLoss(hparam.margin)

# ml_loss = losses.NPairsLoss()
# miner = miners.BatchHardMiner()
# hparam.add('loss_weights', [0.8, 0.8, 0.4, 1, 0.2])
# criterion = GAENPLoss(hparam.loss_weights)
# criterion = GAEMLLoss(ml_loss, loss_weights=hparam.loss_weights, miner=miner)
# criterion = GAEEDLoss(ml_loss, loss_weights=hparam.loss_weights, miner=miner)

# criterion = euclidean_dist_loss

hparam.add_dict(criterion.hparam_dict)

val_criterion = criterion
# val_criterion = dummy_loss
# hparam.add_dict(val_criterion.hparam_dict)


val_evaluator = Evaluator(val_dataloader, nbias, cpu_eval, take_compo_scores=take_compo_scores)
target_metric = 'OpAUC'
# target_metric = 'OpUnseen'
# val_evaluator = IREvaluator(cpu_eval)
# val_evaluator = IREvaluatorFashion(cpu_eval, val_dataloader.dataset, model)
# target_metric = 'IR_Rec/top1'

# === Restore model and logger from Checkpoint ===
curr_epoch = 0
best = {target_metric:-1, 'best_epoch':-1}
log_dir = None

if checkpoint:
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if target_metric in checkpoint:
    best[target_metric] = checkpoint[target_metric]
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


# =====   Evaluation  ======
if args.eval:
  summary, _ = evaluate(model, val_criterion, val_dataloader, val_evaluator, open_world, cpu_eval)
  for key, value in summary.items():
    print(f'{key}:{value:.4f}|', end='')
  print()
  sys.exit(0)

# ====     Train    ========
try:
  train(model, hparam, optimizer, criterion, val_criterion, num_epochs, batch_size, train_dataloader, val_dataloader, logger, val_evaluator, target_metric,
        curr_epoch=curr_epoch, best=best, save_path=model_path, open_world=open_world, eval_every=eval_every, cpu_eval=cpu_eval)
except KeyboardInterrupt:
  print("Training stopped.")
finally:
  logger.add_text(f'hparam/{model_name}', repr(hparam.hparam_dict | best))
  logger.flush()
  logger.close()
  print(f'Best {target_metric}: {best[target_metric]}')