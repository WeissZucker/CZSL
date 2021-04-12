import torch
import torch.nn.functional as F
import numpy as np
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
  
symnet_biaslist = [-5.1663e-01, -4.7130e-02, -1.9583e-02, -6.9781e-03, -2.0409e-03, 0,
         2.7767e-04,  2.1317e-03,  3.9892e-03,  6.5810e-03,  9.4104e-03,
         1.2964e-02,  1.7363e-02,  2.2133e-02,  2.8242e-02,  3.7643e-02,
         5.0190e-02,  6.7753e-02,  9.4308e-02,  1.3616e-01,  2.0667e-01,
         4.5399e-01]


def getCompoMask(dataloader):
  """Mask of (attr x obj) matrix with compositions appeared in the dataset being marked as 1."""
  obj2idx, attr2idx = dataloader.dataset.obj2idx, dataloader.dataset.attr2idx
  attr_class, obj_class = len(dataloader.dataset.attrs), len(dataloader.dataset.objs)
  if dataloader.dataset.phase == 'train':
    pairs = dataloader.dataset.train_pairs
  else:
    pairs = dataloader.dataset.test_pairs
  pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in pairs])
  mask = torch.zeros((attr_class, obj_class))
  mask[(pair_idx[:, 0], pair_idx[:, 1])] = 1
  return mask
 
  
def match(labels, preds):
  preds = torch.argmax(preds, axis=-1)
  return torch.sum(preds == labels)


def compo_match(obj_labels, obj_preds, attr_labels, attr_preds, seen_mask, test_compo_mask=None):
  """Calculate match count lists for each bias term for seen and unseen.
  Return: [2 x biaslist_size] with first row for seen and second row for unseen.
  """
  def _compo_match(comp_preds, obj_labels, attr_labels):
    """comp_preds: [batch, attr_class, obj_class]
    Return the count of correct composition predictions.
    """
    comp_preds_ncol = comp_preds.shape[-1]
    masked_preds = torch.argmax(comp_preds.view(len(comp_preds), -1), dim=-1)
    obj_preds = masked_preds % comp_preds_ncol
    attr_preds = masked_preds // comp_preds_ncol
    comp_match = (obj_labels == obj_preds) * (attr_labels == attr_preds)
    return comp_match
  
  def softmax_compo_preds(comp_preds):
    comp_preds_shape = comp_preds.shape
    comp_preds = F.softmax(comp_preds.reshape(comp_preds_shape[0], -1), -1)
    return comp_preds.reshape(comp_preds_shape)

  obj_preds = torch.softmax(obj_preds, dim=-1)
  attr_preds = torch.softmax(attr_preds, dim=-1)
  compo_preds_original = torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
  
  if test_compo_mask is not None: # For closed world, only keep compositions appeared in the test set.
    compo_preds_original *= test_compo_mask
  compo_preds_original = softmax_compo_preds(compo_preds_original)
    
  results = torch.zeros((2, len(symnet_biaslist)))
  for i, bias in enumerate(symnet_biaslist):
    compo_preds = compo_preds_original.clone()
    compo_preds += seen_mask * bias # add bias term to seen composition
    compo_preds = softmax_compo_preds(compo_preds)
    matches = _compo_match(compo_preds, obj_labels, attr_labels)
    
    for match, obj_label, attr_label in zip(matches, obj_labels, attr_labels):
      if seen_mask[attr_label, obj_label] == 1:
        results[0, i] += match
      else:
        results[1, i] += match
    
  return results


def analyse_acc_report(acc_table):
  """acc_table: [2 x biaslist_size] with first row for seen and second row for unseen.
  Return: best_seen, best_unseen, best_harmonic, auc
  """
  seen, unseen = acc_table[0], acc_table[1]
  best_seen = torch.max(seen)
  best_unseen = torch.max(unseen)
  best_geometric = torch.max((seen * unseen) ** (1/2))
  best_harmonic = torch.max(2/(1/seen + 1/unseen))
  auc = np.trapz(unseen, seen)
  return best_seen, best_unseen, best_geometric, best_harmonic, auc
  
  
def eval_model(model, test_dataloader, train_dataloader):
  """Return: Tuple of (closed_world_report, open_word_report).
  report: best_seen, best_unseen, best_harmonic, auc
  """
  obj_match, attr_match = 0, 0
  test_compo_mask = getCompoMask(test_dataloader) # 2d (attr x obj) matrix, with compositions appeared in the test dataset being marked as 1.
  seen_compo_mask = getCompoMask(train_dataloader) # mask of compositions seen during training
#   unseen_compo_OW_mask = 1 - seen_compo_mask # mask of compositions not seen during training in the open world setting
#   unseen_compo_CW_mask = test_compo_mask * unseen_compo_OW_mask # mask of compositions not seen during training in the closed world setting
  matches_cw = torch.zeros((2, len(symnet_biaslist)))
  matches_ow = torch.zeros((2, len(symnet_biaslist)))
  with torch.no_grad():
    model.eval()
    for i, batch in tqdm.tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        position=0,
        leave=True):
      img, attr_id, obj_id = batch[:3]
      obj_preds, attr_preds = model(img.to(dev))
      obj_preds, attr_preds = obj_preds.cpu(), attr_preds.cpu()
      obj_match += match(obj_id, obj_preds)
      attr_match += match(attr_id, attr_preds)
      matches_cw += compo_match(obj_id, obj_preds, attr_id, attr_preds, seen_compo_mask, test_compo_mask)
      matches_ow += compo_match(obj_id, obj_preds, attr_id, attr_preds, seen_compo_mask)
  obj_acc, attr_acc = obj_match / len(test_dataloader.dataset), attr_match / len(test_dataloader.dataset)
  report_cw = analyse_acc_report(matches_cw / len(test_dataloader.dataset))
  report_ow = analyse_acc_report(matches_ow / len(test_dataloader.dataset))
  return obj_acc, attr_acc, report_cw, report_ow