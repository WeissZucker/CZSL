import torch
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class BaseEvaluator():
  def __init__(self, test_dataloader, num_bias):
    self.num_bias = num_bias
    self.test_dataloader = test_dataloader
    self.attrs, self.objs = np.array(test_dataloader.dataset.attrs), np.array(test_dataloader.dataset.objs)
    self.attr_class, self.obj_class = len(self.attrs), len(self.objs)
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
    dset = dataloader.dataset
    obj2idx, attr2idx = dset.obj2idx, dset.attr2idx
    attr_class, obj_class = len(dset.attrs), len(dset.objs)

    train_pairs = dset.train_pairs
    if dset.phase == 'test':
      test_pairs = dset.test_pairs
    elif dset.phase == 'val':
      test_pairs = dset.val_pairs

    train_pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_pairs])
    test_pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in test_pairs])
    test_mask = torch.zeros((attr_class, obj_class), dtype=torch.bool).to(dev)
    seen_mask = torch.zeros_like(test_mask)
    test_mask[(test_pair_idx[:, 0], test_pair_idx[:, 1])] = True
    seen_mask[(train_pair_idx[:, 0], train_pair_idx[:, 1])] = True
    return test_mask, seen_mask


  def acc(self, preds, labels):
    """Calculate top-k accuracy"""
    if len(preds.shape) == 1:
      preds = preds.unsqueeze(-1)
    if len(labels.shape) == 1:
      labels = labels.unsqueeze(-1)
    match = torch.any(preds == labels, dim=1).float()
    return torch.mean(match)

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

    if len(biaslist) == 0:
      biaslist = [0]
    return list(biaslist)

  def compo_acc(self, compo_scores, topk=1, open_world=False):
    """Calculate match count lists for each bias term for seen and unseen.
    Return: [2 x biaslist_size] with first row for seen and second row for unseen.
    """
    def _compo_match(compo_scores, obj_labels, attr_labels, topk=1):
      """compo_scores: [batch, attr_class, obj_class]
      Return the count of correct composition predictions.
      """
      ncol = compo_scores.shape[-1]
      _, topk_preds = torch.topk(compo_scores.view(len(compo_scores), -1), topk, dim=-1) # [batch, k]
      topk_obj_preds = topk_preds % ncol
      topk_attr_preds = topk_preds// ncol
      compo_match = (obj_labels.unsqueeze(1) == topk_obj_preds) * (attr_labels.unsqueeze(1) == topk_attr_preds)
      compo_match = torch.any(compo_match, dim=-1)
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
      matches = _compo_match(compo_scores, self.obj_labels, self.attr_labels, topk)
      results[0, i] = torch.sum(matches[target_label_seen_mask])
      results[1, i] = torch.sum(matches[~target_label_seen_mask])
    acc = torch.max(results[0] + results[1]) / len(compo_scores_original)
   
    results[0] /= torch.sum(target_label_seen_mask)
    results[1] /= torch.sum(~target_label_seen_mask)
    results = [result.cpu() for result in results]
    return acc, results

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


class Evaluator(BaseEvaluator):
  def __init__(self, test_dataloader, num_bias, take_compo_scores=True):
    super().__init__(test_dataloader, num_bias)
    self.take_compo_scores = take_compo_scores
    
  def get_composcores(self, attr_scores, obj_scores):
    obj_preds = torch.softmax(obj_scores, dim=-1)
    attr_preds = torch.softmax(attr_scores, dim=-1)
    return torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
  
  def get_primitive_preds(self, compo_scores, topk):
    ncol = compo_scores.shape[-1]
    _, topk_preds = torch.topk(compo_scores.view(len(compo_scores), -1), topk, dim=-1) # [batch, k]
    topk_obj_preds = topk_preds % ncol
    topk_attr_preds = topk_preds // ncol
    return topk_obj_preds, topk_attr_preds
  
  def format_summary(self, attr_acc, obj_acc, report_cw, report_op):
    summary = dict()
    summary['A'] = attr_acc
    summary['O'] = obj_acc
    bias_evals = ['Seen', 'Unseen', 'HM', 'AUC']
    for eval, x in zip(bias_evals, report_cw): summary['Cw'+eval] = x
    for eval, x in zip(bias_evals, report_op): summary['Op'+eval] = x
    return summary
  
  def evaluate(self, attr_preds, obj_preds, compo_scores, topk):
    attr_acc = self.acc(attr_preds, self.attr_labels)
    obj_acc = self.acc(obj_preds, self.obj_labels)
    acc_cw, acc_cw_biased = self.compo_acc(compo_scores, topk)
    acc_ow, acc_ow_biased = self.compo_acc(compo_scores, topk, open_world=True)
    report_cw = self.analyse_acc_report(acc_cw_biased)
    report_ow = self.analyse_acc_report(acc_ow_biased)

    return self.format_summary(attr_acc, obj_acc, report_cw, report_ow)
  
  def eval_primitive_scores(self, attr_scores, obj_scores, topk=1):
    """Return: Tuple of (closed_world_report, open_word_report).
    report: best_seen, best_unseen, best_harmonic, auc
    """
    compo_scores = self.get_composcores(attr_scores, obj_scores)
    _, obj_preds = torch.topk(obj_scores, topk, axis=-1)
    _, attr_preds = torch.topk(attr_scores, topk, axis=-1)
    return self.evaluate(attr_preds, obj_preds, compo_scores, topk)
    
  def eval_compo_scores(self, compo_scores, topk=1):
    obj_preds, attr_preds = self.get_primitive_preds(compo_scores, topk)
    _, obj_preds = torch.topk(obj_preds, topk, axis=-1)
    _, attr_preds = torch.topk(attr_preds, topk, axis=-1)
    return self.evaluate(attr_preds, obj_preds, compo_scores, topk)
  
  def eval_output(self, output, topk=1):
    if self.take_compo_scores:
      compo_scores = output
      if isinstance(compo_scores, list):
        compo_scores = torch.cat(compo_scores)
      compo_scores = compo_scores.reshape(-1, self.attr_class, self.obj_class)
      return self.eval_compo_scores(compo_scores, topk=topk)
    else:
      if isinstance(output, list):
        attr_scores, obj_scores = zip(*output)
        attr_scores = torch.cat(attr_scores)
        obj_scores = torch.cat(obj_scores)
      else:
        attr_scores, obj_scores = output
      return self.eval_primitive_scores(attr_scores, obj_scores, topk=topk)

    
  
class EvaluatorWithFscore(Evaluator):
  def __init__(self, test_dataloader, num_bias, take_compo_scores, fscore_threshold, word2vec_path, fscore_path=None):
    super().__init__(test_dataloader, num_bias, take_compo_scores)
    self.word2vec_path = word2vec_path
    if fscore_path:
      self.fscore = torch.load(fscore_path)
    else:
      self.fscore = self.calc_fscore()
    self.update_threshold(fscore_threshold)
    
  def update_threshold(self, fscore_threshold):
    self.fscore_mask = self.fscore < fscore_threshold
    
  def calc_fscore(self):
    def cos_sim(x, y):
      x, y = torch.tensor(x).to(dev), torch.tensor(y).unsqueeze(0).to(dev)
      return cos(x, y)
    fscore = torch.ones_like(self.seen_mask, dtype=torch.float)
    word2vec = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
    cos = torch.nn.CosineSimilarity(dim=1)
    for i in tqdm.tqdm(range(self.attr_class * self.obj_class), total=self.attr_class*self.obj_class):
      attr_id = i // self.obj_class
      obj_id = i % self.obj_class
      if self.unseen_mask_ow[attr_id, obj_id]:
        attr, obj = self.attrs[attr_id], self.objs[obj_id]
        paired_attrs_with_obj = self.attrs[self.seen_mask[:, obj_id].cpu()]
        paired_objs_with_attr = self.objs[self.seen_mask[attr_id, :].cpu()]
        obj_score = max(cos_sim(word2vec[paired_objs_with_attr], word2vec[obj]))
        attr_score = max(cos_sim(word2vec[paired_attrs_with_obj], word2vec[attr]))
        fscore[attr_id, obj_id] = (obj_score + attr_score) / 2
    return fscore
  
  def evaluate(self, attr_preds, obj_preds, compo_scores, topk):
    obj_acc = self.acc(obj_preds, self.obj_labels)
    attr_acc = self.acc(attr_preds, self.attr_labels)
    acc_cw, acc_cw_biased = self.compo_acc(compo_scores, topk)
    compo_scores[:, self.fscore_mask] = -1e10
    acc_ow, acc_ow_biased = self.compo_acc(compo_scores, topk, open_world=True)
    report_cw = self.analyse_acc_report(acc_cw_biased)
    report_ow = self.analyse_acc_report(acc_ow_biased)

    return self.format_summary(attr_acc, obj_acc, report_cw, report_ow)