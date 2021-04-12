from symnet.utils import dataset as symnet_dataset
import os

def get_dataset(dataset_name, phase, with_image=True, shuffle=False):
  # Symnet output format
  # train [img, attr_id, obj_id, pair_id, img_feature, img, attr_id, obj_id, pair_id, img_feature, aff_mask]
  # test [img, attr_id, obj_id, pair_id, img_feature, aff_mask]
  
  # Target output format
  # unique img id, img file path, label pair text, label pair id, label att id, label obj id
  dataset = symnet_dataset.get_dataloader(dataset_name, phase, with_image=with_image, shuffle=shuffle).dataset
  pairs = dataset.pairs
  root = dataset.root
  reform_output_func = lambda x: (hash(x[0]), os.path.join(root, x[0]), pairs[x[3]], x[3], x[1], x[2])
  return map(reform_output_func, dataset)
  