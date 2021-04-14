import torch
import torch.nn.functional as F
import numpy as np
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class Evaluator():
  def __init__(self, train_dataloader, test_dataloader, num_bias):
    self.num_bias = num_bias
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader
    self.attr_class = len(train_dataloader.dataset.attrs)
    self.obj_class = len(train_dataloader.dataset.objs)
    
    self.test_mask = self.getCompoMask(test_dataloader) # 2d (attr x obj) matrix, with compositions appeared in the test dataset being marked as 1.
    self.seen_mask = self.getCompoMask(train_dataloader) # mask of compositions seen during training
    self.unseen_mask_ow = 1 - self.seen_mask # mask of compositions not seen during training in the open world setting
    self.unseen_mask_cw = self.test_mask * self.unseen_mask_ow # mask of compositions not seen during training in the closed world setting
  
  def getCompoMask(self, dataloader):
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


  def acc(self, labels, preds):
    preds = torch.argmax(preds, axis=-1)
    return torch.sum(preds == labels) / len(preds)


  def get_biaslist(self, compo_preds, obj_labels, attr_labels):
    nsample = len(compo_preds)
    preds_correct_label = compo_preds[range(nsample), attr_labels, obj_labels]
    seen_preds = (compo_preds * self.seen_mask).reshape(nsample, -1)
    max_seen_preds, _ = torch.max(seen_preds, 1)
    score_diff = preds_correct_label - max_seen_preds
    
    # only take samples with prediction being correct and target being unseen labels
    correct_prediction_mask = torch.argmax(F.softmax(compo_preds.reshape(nsample, -1), -1),-1) == attr_labels*self.obj_class + obj_labels
    target_label_unseen_mask = self.unseen_mask_cw[attr_labels, obj_labels]
    score_diff, _ = torch.sort(score_diff[correct_prediction_mask * target_label_unseen_mask])
    bias_skip = max(len(score_diff) // self.num_bias, 1)
    biaslist = score_diff[::bias_skip]
    return biaslist


  def compo_acc(self, obj_labels, obj_preds, attr_labels, attr_preds, open_world=False):
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
    biaslist = self.get_biaslist(compo_preds_original, obj_labels, attr_labels)

    if not open_world: # For closed world, only keep compositions appeared in the test set.
      compo_preds_original *= self.test_mask

    results = torch.zeros((2, len(biaslist)))
    for i, bias in enumerate(biaslist):
      compo_preds = compo_preds_original.clone()
      compo_preds += self.seen_mask * bias # add bias term to seen composition
      compo_preds = softmax_compo_preds(compo_preds)
      matches = _compo_match(compo_preds, obj_labels, attr_labels)

      for match, obj_label, attr_label in zip(matches, obj_labels, attr_labels):
        if self.seen_mask[attr_label, obj_label] == 1:
          results[0, i] += match
        else:
          results[1, i] += match

    return results / len(obj_labels)


  def analyse_acc_report(self, acc_table):
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


  def eval_model(self, net):
    """Return: Tuple of (closed_world_report, open_word_report).
    report: best_seen, best_unseen, best_harmonic, auc
    """
    obj_preds, obj_labels, attr_preds, attr_labels = [], [], [], []
    with torch.no_grad():
      net.eval()
      for i, batch in tqdm.tqdm(
          enumerate(self.test_dataloader),
          total=len(self.test_dataloader),
          position=0,
          leave=True):
        img, attr_id, obj_id = batch[:3]
        preds = net(img.to(dev))
        obj_preds.append(preds[0].cpu())
        attr_preds.append(preds[1].cpu())
        obj_labels.append(obj_id)
        attr_labels.append(attr_id)

    obj_preds = torch.cat(obj_preds)
    attr_preds = torch.cat(attr_preds)
    obj_labels = torch.cat(obj_labels)
    attr_labels = torch.cat(attr_labels)

    obj_acc = self.acc(obj_labels, obj_preds)
    attr_acc = self.acc(attr_labels, attr_preds)
    acc_cw = self.compo_acc(obj_labels, obj_preds, attr_labels, attr_preds)
    acc_ow = self.compo_acc(obj_labels, obj_preds, attr_labels, attr_preds)
    report_cw = self.analyse_acc_report(acc_cw)
    report_ow = self.analyse_acc_report(acc_ow)

    return obj_acc, attr_acc, report_cw, report_ow