import torch
import numpy as np
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

def calc_acc(model, test_dataloader, close_setting=True, use_tune=False):
  def getClosedWorldMask(test_dataloader):
    """Only keep compositions that appear in the test dataset."""
    obj2idx, attr2idx = test_dataloader.dataset.obj2idx, test_dataloader.dataset.attr2idx
    attr_class, obj_class = len(test_dataloader.dataset.attrs), len(test_dataloader.dataset.objs)
    pairs = test_dataloader.dataset.test_pairs
    pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in pairs])
    mask = torch.zeros((attr_class, obj_class))
    mask[(pair_idx[:, 0], pair_idx[:, 1])] = 1
    return mask
  
  def match(labels, preds):
    preds = torch.argmax(preds, axis=-1)
    return torch.sum(preds == labels)

  def compoMatch(obj_labels, obj_preds, attr_labels, attr_preds):
    obj_preds = torch.argmax(obj_preds, axis=-1)
    attr_preds = torch.argmax(attr_preds, axis=-1)
    comp_match = (obj_labels == obj_preds) * (attr_labels == attr_preds)
    return torch.sum(comp_match)
  
  def compoMatchClosed(obj_labels, obj_preds, attr_labels, attr_preds, mask):
    obj_preds = torch.softmax(obj_preds, dim=-1)
    attr_preds = torch.softmax(attr_preds, dim=-1)
    comp_preds = torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
    comp_preds *= mask
    comp_preds_ncol = comp_preds.shape[-1]
    masked_preds = torch.argmax(comp_preds.view(len(comp_preds), -1), dim=-1)
    obj_preds = masked_preds % comp_preds_ncol
    attr_preds = masked_preds // comp_preds_ncol
    comp_match = (obj_labels == obj_preds) * (attr_labels == attr_preds)
    return torch.sum(comp_match)

  obj_match, attr_match, comp_match, comp_closed_match = 0, 0, 0, 0
  mask = getClosedWorldMask(test_dataloader)
  with torch.no_grad():
    model.eval()
    for i, batch in tqdm.tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        disable=use_tune,
        position=0,
        leave=True):
      img, attr_id, obj_id = batch[:3]
      obj_preds, attr_preds = model(img.to(dev))
      obj_preds, attr_preds = obj_preds.to('cpu'), attr_preds.to('cpu')
      obj_match += match(obj_id, obj_preds)
      attr_match += match(attr_id, attr_preds)
      comp_match += compoMatch(obj_id, obj_preds, attr_id, attr_preds)
      comp_closed_match += compoMatchClosed(obj_id, obj_preds, attr_id, attr_preds, mask)
  return np.array([obj_match, attr_match, comp_match, comp_closed_match]) / len(test_dataloader.dataset)