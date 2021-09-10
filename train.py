from typing import *
import os
import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import dataset
from evaluator import Evaluator

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
  


def train_with_config(config: dict, num_epochs: int = 1, checkpoint_dir: str = None) -> None:
  """
  Hyperparameter tuning with Ray[tune]. Train the model with config.
  config: dict with keys {'lr', 'resnet_name, num_mlp_layers'}  
  """
  lr = config['lr']
  resnet_name = config['resnet']
  num_mlp_layers = config['num_mlp_layers']
  mlp = partial(HalvingMLP, num_layers=num_mlp_layers)
  batch_size = 64

  resnet = frozen(torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=True))
  compoResnet = CompoResnet(resnet, mlp).to(dev)
  obj_loss_history = [[],[]]
  attr_loss_history = [[],[]]
  optimizer = optim.Adam(compoResnet.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  if checkpoint_dir:
    model_state, optimizer_state, obj_loss_history, attr_loss_history = torch.load(
        os.path.join(checkpoint_dir, "checkpoint"))
    compoResnet.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  dset = dataset.get_dataloader('MIT', 'train', with_image=True).dataset
  train_with_val(compoResnet, optimizer, criterion, num_epochs, obj_loss_history, attr_loss_history, batch_size, dset, use_tune=True)

  
def train_with_val(net, optimizer, criterion, num_epochs, obj_loss_history: List[List], attr_loss_history: List[List], batch_size, dataset, curr_epoch=0, use_tune=False, model_dir: str = None) -> None:
  """
  Train the model with validation set.
  Parameters:
    [obj/attr]_loss_history: nested list of length 2. history[0] the training loss history and history[1] the validation loss history.
    curr_epoch: the epoch number the model already been trained for.
    model_dir: directory to save model states.
  """
  
  test_abs = int(len(dataset) * 0.8)
  train_subset, val_subset = random_split(
        dataset, [test_abs, len(dataset) - test_abs], generator=torch.Generator().manual_seed(42))
  train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

  for epoch in range(curr_epoch, curr_epoch+num_epochs):
    epoch_steps = 0
    obj_running_loss = 0.0
    attr_running_loss = 0.0
    net.train()
    # ==== Training ====
    for i, batch in tqdm.tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        disable=use_tune,
        position=0,
        leave=True,
        postfix='Train: epoch %d/%d'%(epoch, curr_epoch+num_epochs)):
      optimizer.zero_grad()
      img, attr_id, obj_id = batch[:3]
      if len(img) == 1:
        # Batchnorm doesn't accept batch with size 1
        continue
      obj_pred, attr_pred = net(img.to(dev))
      obj_loss = criterion(obj_pred, obj_id.to(dev))
      attr_loss = criterion(attr_pred, attr_id.to(dev))
      loss = obj_loss + attr_loss
      loss.backward()
      optimizer.step()

      obj_running_loss += obj_loss.item()
      attr_running_loss += attr_loss.item()
      epoch_steps += 1
      if i % 100 == 99:
          print("[%d, %5d] obj_loss: %.3f, attr_loss: %.3f" % (epoch+1, i + 1,
                                          obj_running_loss / epoch_steps, attr_running_loss / epoch_steps))
          obj_loss_history[0].append(obj_running_loss/epoch_steps)
          attr_loss_history[0].append(attr_running_loss/epoch_steps)
          running_loss = 0.0

    # ==== Validation ====
    obj_val_loss = 0.0
    attr_val_loss = 0.0
    val_steps = 0
    
    net.eval()
    for i, batch in tqdm.tqdm(
          enumerate(val_dataloader),
          total=len(val_dataloader),
          disable=use_tune,
          position=0,
          leave=True):
        with torch.no_grad():
            img, attr_id, obj_id = batch[:3]
            obj_pred, attr_pred = net(img.to(dev))
            obj_loss = criterion(obj_pred, obj_id.to(dev))
            attr_loss = criterion(attr_pred, attr_id.to(dev))
            obj_val_loss += obj_loss.item()
            attr_val_loss += attr_loss.item()
            val_steps += 1
    
    obj_val_loss /= val_steps
    attr_val_loss /= val_steps
    print("[%d] obj_val_loss: %.3f, attr_val_loss: %.3f" % (epoch+1, obj_val_loss, attr_val_loss ))
    obj_loss_history[1].append(obj_val_loss)
    attr_loss_history[1].append(attr_val_loss)
    
    # ==== Save model, report to tune ====
    if use_tune:
      with tune.checkpoint_dir(epoch) as checkpoint_dir: 
          path = os.path.join(checkpoint_dir, "checkpoint")
          torch.save({
                      'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'obj_loss': obj_loss_history,
                      'attr_loss': attr_loss_history,
                      }, path)
      acc = calc_acc(net, val_dataloader, use_tune)
      tune.report(loss=(obj_val_loss+attr_val_loss), accuracy=acc)
      print("accuracy: ", acc)
    else:
      if model_dir:
        model_path = os.path.join(model_dir, f"model_{epoch}.pt")
        torch.save({
                      'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'obj_loss': obj_loss_history,
                      'attr_loss': attr_loss_history,
                      }, model_path)
        old_model = os.path.join(model_dir, f"model_{epoch-1}.pt")
        if os.path.isfile(old_model):
          os.remove(old_model)
    print("Finished training.")
    
def tqdm_iter(curr_epoch, total_epoch, dataloader):
  postfix = f'Train: epoch {curr_epoch}/{total_epoch}'
  return tqdm.tqdm(enumerate(dataloader), 
                 total=len(dataloader), position=0, leave=True, postfix=postfix)

def log_summary(summary, logger, epoch):
  for key, value in summary.items():
    if 'Op' in key:
      logger.add_scalar(key[2:]+'/op', value, epoch)
    elif 'Cw' in key:
      logger.add_scalar(key[2:]+'/cw', value, epoch)
    else:
      logger.add_scalar('Acc/'+key, value, epoch)
      
def cw_output_converter(output, dataloader, cpu_eval=False):
  val_dev = 'cpu' if cpu_eval else dev
  dataset = dataloader.dataset
  if isinstance(output[0], tuple):
    output = list(zip(*output))[0]
  output = torch.cat(output).to(val_dev)
  batch_size = output.size(0)
  nattr, nobj = len(dataset.attrs), len(dataset.objs)
  new_output = torch.ones((batch_size, nattr*nobj)).to(val_dev) * 1e-10
  op_idx = [dataset.op_pair2idx[pair] for pair in dataset.pairs]
  new_output[:, op_idx] = output
  return new_output


def train(net, hparam, optimizer, criterion, val_criterion, num_epochs, batch_size, train_dataloader, val_dataloader, logger,
          evaluator, target_metric, curr_epoch=0, best=None, save_path=None, open_world=True, eval_every=1, cpu_eval=False) -> None:
  """
  Train the model.
  """
  scheduler = None
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=0.0001, last_epoch=-1)
  iters = len(train_dataloader)
  if not best:
    best = defaultdict(lambda: -1)
  val_dev = 'cpu' if cpu_eval else dev
  
  for epoch in range(curr_epoch, curr_epoch+num_epochs):
    
    # ==== Training ====
    running_loss = defaultdict(lambda : 0)
    
    net.train()
    for i, sample in tqdm_iter(epoch, curr_epoch+num_epochs, train_dataloader):
      optimizer.zero_grad()
      if len(sample[0]) == 1:
        # Batchnorm doesn't accept batch with size 1
        continue
      output = net(sample)
      total_loss, loss_dict = criterion(output, sample)
      total_loss.backward()
      optimizer.step()
      if scheduler:
        scheduler.step(epoch + i / iters)
      for key, loss in loss_dict.items():
        running_loss[key] += loss.item()
      if i % 100 == 99:
        for key, loss in running_loss.items():
          logger.add_scalar(f'{key}/train', loss/i, epoch*len(train_dataloader)//100+i//100)

    # ==== Validation ====
    if epoch % eval_every != 0:
      continue
    test_loss = defaultdict(lambda: 0)
    outputs = []
    attr_labels, obj_labels = [], []
    net.eval()
    with torch.no_grad():
      for i, sample in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        output = net(sample)
        total_loss, loss_dict = val_criterion(output, sample)
        for key, loss in loss_dict.items():
          test_loss[key] += loss.item()
        if isinstance(output, tuple):
          output = tuple([x.to(val_dev) if isinstance(x, torch.Tensor) else x for x in output])
        else:
          output = output.to(val_dev)
        outputs.append(output)
        attr_labels.append(sample[1])
        obj_labels.append(sample[2])

    attr_labels = torch.cat(attr_labels).to(val_dev)
    obj_labels = torch.cat(obj_labels).to(val_dev)
    if not open_world:
      outputs = cw_output_converter(outputs, val_dataloader, cpu_eval)
    
    summary = evaluator.eval_output(outputs, attr_labels, obj_labels)

    # ==== Logging ====
    log_summary(summary, logger, epoch)
    print("Train: ", end='')
    for key, loss in running_loss.items():
      print(f"{key}: {loss/len(train_dataloader)}", end=' - ')
    print()
    
    print("Test: ", end='')
    for key, loss in test_loss.items():
      loss /= len(val_dataloader) 
      logger.add_scalar(f'{key}/test', loss, epoch)
      print(f"{key}: {loss}", end=' - ')
    print()
    
    for key, value in summary.items():
      print(f'{key}:{value:.4f}|', end='')
    print()
  
    if summary[target_metric] > best[target_metric]:
      best[target_metric] = summary[target_metric]
      best['best_epoch'] = epoch
      if save_path:
        torch.save({
                      'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      target_metric: summary[target_metric],
                      'epoch': epoch+1,
                      'log_dir': logger.log_dir,
                      'log_name_suffix': logger.filename_suffix,
                      'model_str': repr(net),
                      'hparam_dict': hparam.hparam_dict
                      }, save_path)
        
  print("Finished training.")