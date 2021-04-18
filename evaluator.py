import torch
import torch.nn.functional as F
import numpy as np
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class BaseEvaluator():
  def __init__(self, test_dataloader, num_bias):
    self.num_bias = num_bias
    self.test_dataloader = test_dataloader
    self.attr_class = len(test_dataloader.dataset.attrs)
    self.obj_class = len(test_dataloader.dataset.objs)
    self.attr_labels, self.obj_labels = self.getLabels(test_dataloader)
    
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


  def acc(self, preds, labels):
    preds = torch.argmax(preds, axis=-1)
    return torch.sum(preds == labels) / len(preds)


  def get_biaslist(self, compo_preds):
    nsample = len(compo_preds)
    preds_correct_label = compo_preds[range(nsample), self.attr_labels, self.obj_labels]
    seen_preds = (compo_preds.reshape(nsample, -1)[:,self.seen_mask.reshape(-1)]).reshape(nsample, -1)
    max_seen_preds, _ = torch.max(seen_preds, 1)
    score_diff = max_seen_preds - preds_correct_label - 1e-4
    
    _compo_preds = compo_preds.clone()
    _compo_preds[:,~self.close_mask] = -1e10
    _compo_preds += self.unseen_mask_cw * 1e3
    
    # only take samples with prediction being correct and target being unseen labels
    correct_prediction_mask = (torch.argmax(_compo_preds.reshape(nsample, -1),-1)
                               == self.attr_labels * self.obj_class + self.obj_labels)
    target_label_unseen_mask = self.unseen_mask_cw[self.attr_labels, self.obj_labels]
    score_diff, _ = torch.sort(score_diff[correct_prediction_mask * target_label_unseen_mask])

    bias_skip = max(len(score_diff) // self.num_bias, 1)
    biaslist = score_diff[::bias_skip]
    return biaslist.cpu()


  def compo_acc(self, compo_scores, open_world=False):
    """Calculate match count lists for each bias term for seen and unseen.
    Return: [2 x biaslist_size] with first row for seen and second row for unseen.
    """
    def _compo_match(compo_scores, obj_labels, attr_labels):
      """compo_scores: [batch, attr_class, obj_class]
      Return the count of correct composition predictions.
      """
      compo_scores_ncol = compo_scores.shape[-1]
      compo_scores = torch.argmax(compo_scores.view(len(compo_scores), -1), dim=-1)
      obj_preds = compo_scores % compo_scores_ncol
      attr_preds = compo_scores // compo_scores_ncol
      compo_match = (obj_labels == obj_preds) * (attr_labels == attr_preds)
      return compo_match
    
    compo_scores_original = compo_scores.clone()
    biaslist = self.get_biaslist(compo_scores_original)
    
    if not open_world:
      compo_scores_original[:,~self.close_mask] = -1e10

    results = torch.zeros((2, len(biaslist))).to(dev)
    target_label_seen_mask = self.seen_mask[self.attr_labels, self.obj_labels]
    for i, bias in enumerate(biaslist):
      if open_world:
        compo_scores = compo_scores_original + self.unseen_mask_ow * bias
      else:
        compo_scores = compo_scores_original + self.unseen_mask_cw * bias
      matches = _compo_match(compo_scores, self.obj_labels, self.attr_labels)
      results[0, i] = torch.sum(matches[target_label_seen_mask])
      results[1, i] = torch.sum(matches[~target_label_seen_mask])
      
    results[0] /= torch.sum(target_label_seen_mask)
    results[1] /= torch.sum(~target_label_seen_mask)
    results = [result.cpu() for result in results]
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


class CompoResnetEvaluator(BaseEvaluator):
  def get_composcores(self, obj_scores, attr_scores):
    obj_preds = torch.softmax(obj_scores, dim=-1)
    attr_preds = torch.softmax(attr_scores, dim=-1)
    return torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
  
  def eval_model(self, net):
    """Return: Tuple of (closed_world_report, open_word_report).
    report: best_seen, best_unseen, best_harmonic, auc
    """
    obj_scores, attr_scores = [], []
    with torch.no_grad():
      net.eval()
      for i, batch in tqdm.tqdm(
          enumerate(self.test_dataloader),
          total=len(self.test_dataloader),
          position=0,
          leave=True):
        img, attr_id, obj_id = batch[:3]
        scores = net(img.to(dev))
        obj_scores.append(scores[0])
        attr_scores.append(scores[1])

    obj_scores = torch.cat(obj_scores)
    attr_scores = torch.cat(attr_scores)
    compo_scores = self.get_composcores(obj_scores, attr_scores)

    obj_acc = self.acc(obj_scores, self.obj_labels)
    attr_acc = self.acc(attr_scores, self.attr_labels)
    acc_cw = self.compo_acc(compo_scores)
    acc_ow = self.compo_acc(compo_scores, open_world=True)
    report_cw = self.analyse_acc_report(acc_cw)
    report_ow = self.analyse_acc_report(acc_ow)

    return obj_acc, attr_acc, report_cw, report_ow
