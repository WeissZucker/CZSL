import torch
import torch.nn.functional as F
import numpy as np
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class Evaluator():
  def __init__(self, test_dataloader, num_bias):
    self.num_bias = num_bias
    self.test_dataloader = test_dataloader
    self.attr_class = len(test_dataloader.dataset.attrs)
    self.obj_class = len(test_dataloader.dataset.objs)
    self.attr_labels, self.obj_labels = self.getLabels(dataloader)
    
    self.test_mask, self.seen_mask = self.getCompoMask(test_dataloader) # (attr x obj) matrices
    self.close_mask = self.test_mask + self.seen_mask
    self.unseen_mask_ow = ~self.seen_mask # mask of compositions not seen during training in the open world setting
    self.unseen_mask_cw = self.close_mask * self.unseen_mask_ow # mask of compositions not seen during training in the closed world setting
  
  def getLabels(self, dataloader):
    obj_labels, attr_labels = [], []
    for batch in dataloader:
      img, attr_id, obj_id = batch[:3]
      obj_labels.append(obj_id.to(dev))
      attr_labels.append(attr_id.to(dev))
    obj_labels = torch.cat(obj_labels)
    attr_labels = torch.cat(attr_labels)
    return attr_labels, obj_labels
  
  def getCompoMask(self, dataloader):
    """Mask of (attr x obj) matrix with compositions appeared in the dataset being marked as 1."""
    obj2idx, attr2idx = dataloader.dataset.obj2idx, dataloader.dataset.attr2idx
    attr_class, obj_class = len(dataloader.dataset.attrs), len(dataloader.dataset.objs)
    train_pairs, test_pairs = dataloader.dataset.train_pairs, dataloader.dataset.test_pairs
    
    train_pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_pairs])
    test_pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in test_pairs])
    closed_mask = torch.zeros((attr_class, obj_class), dtype=torch.bool).to(dev)
    seen_mask = torch.zeros_like(closed_mask)
    closed_mask[(test_pair_idx[:, 0], test_pair_idx[:, 1])] = True
    seen_mask[(train_pair_idx[:, 0], train_pair_idx[:, 1])] = True
    return closed_mask, seen_mask


  def acc(self, labels, preds):
    preds = torch.argmax(preds, axis=-1)
    return torch.sum(preds == labels) / len(preds)


  def get_biaslist(self, compo_preds, obj_labels, attr_labels):
    nsample = len(compo_preds)
    preds_correct_label = compo_preds[range(nsample), attr_labels, obj_labels]
    seen_preds = (compo_preds * self.seen_mask).reshape(nsample, -1)
    max_seen_preds, _ = torch.max(seen_preds, 1)
    score_diff = max_seen_preds - preds_correct_label - 1e-4

    # only take samples with prediction being correct and target being unseen labels
    correct_prediction_mask = (torch.argmax(F.softmax(compo_preds.reshape(nsample, -1), -1),-1)
                               == attr_labels*self.obj_class + obj_labels)
    target_label_unseen_mask = self.unseen_mask_cw[attr_labels, obj_labels]
    score_diff, _ = torch.sort(score_diff[correct_prediction_mask * target_label_unseen_mask])
    bias_skip = max(len(score_diff) // self.num_bias, 1)
    biaslist = score_diff[::bias_skip]
    
    return biaslist


  def compo_acc(self, obj_preds, attr_preds, open_world=False):
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

    obj_preds = torch.softmax(obj_preds, dim=-1)
    attr_preds = torch.softmax(attr_preds, dim=-1)
    compo_preds_original = torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
    biaslist = self.get_biaslist(compo_preds_original, obj_labels, attr_labels)
    if not open_world: # For closed world, only keep compositions appeared in the test set.
      compo_preds_original[:,~self.test_mask] = -1e10

    results = torch.zeros((2, len(biaslist))).to(dev)
    target_label_seen_mask = self.seen_mask[self.attr_labels, self.obj_labels]
    for i, bias in enumerate(biaslist):
      if open_world:
        compo_preds = compo_preds_original + self.unseen_mask_ow * bias # add bias term to unseen composition
      else:
        compo_preds = compo_preds_original + self.unseen_mask_cw * bias # add bias term to unseen composition
      matches = _compo_match(compo_preds, self.obj_labels, self.attr_labels)
      results[0, i] = torch.sum(matches[target_label_seen_mask])
      results[1, i] = torch.sum(matches) - results[0, i]
      
    results[0] /= torch.sum(target_label_seen_mask)
    results[1] /= torch.sum(~target_label_seen_mask)
    return results


  def analyse_acc_report(self, acc_table):
    """acc_table: [2 x biaslist_size] with first row for seen and second row for unseen.
    Return: best_seen, best_unseen, best_harmonic, auc
    """
    seen, unseen = acc_table[0].cpu(), acc_table[1].cpu()
    best_seen = torch.max(seen)
    best_unseen = torch.max(unseen)
    best_harmonic = torch.max((seen * unseen) ** (1/2))
    auc = np.trapz(seen, unseen)
    return best_seen, best_unseen, best_harmonic, auc


  def eval_model(self, net):
    """Return: Tuple of (closed_world_report, open_word_report).
    report: best_seen, best_unseen, best_harmonic, auc
    """
    obj_preds, attr_preds = [], [], [], []
    with torch.no_grad():
      net.eval()
      for i, batch in tqdm.tqdm(
          enumerate(self.test_dataloader),
          total=len(self.test_dataloader),
          position=0,
          leave=True):
        img, attr_id, obj_id = batch[:3]
        preds = net(img.to(dev))
        obj_preds.append(preds[0])
        attr_preds.append(preds[1])
        obj_labels.append(obj_id.to(dev))
        attr_labels.append(attr_id.to(dev))

    obj_preds = torch.cat(obj_preds)
    attr_preds = torch.cat(attr_preds)

    obj_acc = self.acc(obj_labels, obj_preds)
    attr_acc = self.acc(attr_labels, attr_preds)
    acc_cw = self.compo_acc(obj_preds, attr_preds)
    acc_ow = self.compo_acc(obj_preds, attr_preds, open_world=True)
    report_cw = self.analyse_acc_report(acc_cw)
    report_ow = self.analyse_acc_report(acc_ow)

    return obj_acc, attr_acc, report_cw, report_ow