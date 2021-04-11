import torch
import torch.nn.functional as F
import numpy as np
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
  
symnet_biaslist = [-1e3, -5.1663e-01, -4.7130e-02, -1.9583e-02, -6.9781e-03, -2.0409e-03, 0,
         2.7767e-04,  2.1317e-03,  3.9892e-03,  6.5810e-03,  9.4104e-03,
         1.2964e-02,  1.7363e-02,  2.2133e-02,  2.8242e-02,  3.7643e-02,
         5.0190e-02,  6.7753e-02,  9.4308e-02,  1.3616e-01,  2.0667e-01,
         4.5399e-01, 1e3]


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


def compo_match(obj_labels, obj_preds, attr_labels, attr_preds, seen_mask, unseen_mask, test_compo_mask=None):
  """Calculate match count lists for each bias term with corresponding masks been applied."""
  def _compo_match(comp_preds, obj_labels, attr_labels):
    """comp_preds: [batch, attr_class, obj_class]
    Return the count of correct composition predictions.
    """
    comp_preds_ncol = comp_preds.shape[-1]
    masked_preds = torch.argmax(comp_preds.view(len(comp_preds), -1), dim=-1)
    obj_preds = masked_preds % comp_preds_ncol
    attr_preds = masked_preds // comp_preds_ncol
    comp_match = (obj_labels == obj_preds) * (attr_labels == attr_preds)
    return torch.sum(comp_match)

  obj_preds = torch.softmax(obj_preds, dim=-1)
  attr_preds = torch.softmax(attr_preds, dim=-1)
  comp_preds_original = torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
  if test_compo_mask is not None: # For closed world, only keep compositions appeared in the test set.
    comp_preds_original *= test_compo_mask
  matches = torch.zeros((2, len(symnet_biaslist)))
  for i, bias in enumerate(symnet_biaslist):
    comp_preds = comp_preds_original.clone()
    comp_preds += seen_mask * bias # add bias term to seen composition
    F.softmax(comp_preds, dim=-1)
    seen_preds = comp_preds * seen_mask
    unseen_preds = comp_preds * unseen_mask
    matches[0,i] = _compo_match(seen_preds, obj_labels, attr_labels)
    matches[1,i] = _compo_match(unseen_preds, obj_labels, attr_labels)
  return matches


def analyse_acc_report(acc_table):
  """acc_table: [2 x biaslist_size] with first row for seen and second rwo for unseen.
  Return: best_seen, best_unseen, best_harmonic, auc
  """
  best_seen = torch.max(acc_table[0])
  best_unseen = torch.max(acc_table[1])
  best_harmonic = torch.max(((1/acc_table[0] + 1/acc_table[1]) / 2) ** -1)
  auc_curve_y = acc_table[1] / (acc_table[0]+1e-6)
  auc_curve_x = acc_table[0]
  auc = np.trapz(auc_curve_y, auc_curve_x)
  return best_seen, best_unseen, best_harmonic, auc
  
  
def eval_model(model, test_dataloader, training_dataloader):
  """Return: Tuple of (closed_world_report, open_word_report).
  report: best_seen, best_unseen, best_harmonic, auc
  """
  obj_match, attr_match, comp_match, comp_closed_match = 0, 0, 0, 0
  test_compo_mask = getCompoMask(test_dataloader) # 2d (attr x obj) matrix, with compositions appeared in the test dataset being marked as 1.
  seen_compo_mask = getCompoMask(train_dataloader) # mask of compositions seen during training
  unseen_compo_OW_mask = 1 - seen_compo_mask # mask of compositions not seen during training in the open world setting
  unseen_compo_CW_mask = test_compo_mask * unseen_compo_OW_mask # mask of compositions not seen during training in the closed world setting
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
      matches_cw += compo_match(obj_id, obj_preds, attr_id, attr_preds, seen_compo_mask, unseen_compo_CW_mask, test_compo_mask)
      matches_ow += compo_match(obj_id, obj_preds, attr_id, attr_preds, seen_compo_mask, unseen_compo_OW_mask)
  report_cw = analyse_acc_report(matches_cw / len(test_dataloader.dataset))
  report_ow = analyse_acc_report(matches_ow / len(test_dataloader.dataset))
  return report_cw, report_ow