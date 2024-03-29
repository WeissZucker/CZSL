from . import CZSL_dataset, GCZSL_dataset, fashion200k_dataset
from torch.utils.data import DataLoader
import numpy as np

DATA_ROOT_DIR = './data'

CZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-original',
    'UT':  DATA_ROOT_DIR+'/ut-zap50k-original',
}

GCZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-natural',
    'UT':  DATA_ROOT_DIR+'/ut-zap50k-natural',
    'CGQA': DATA_ROOT_DIR+'/cgqa-natural',
    'Fashion200k': DATA_ROOT_DIR+'/fashion200k'
}

def get_dataloader(dataset_name, phase, feature_file="features.t7", batchsize=1, with_image=False, num_workers=0, open_world=True, train_only=False, 
                   random_sampling=False,ignore_attrs=[], ignore_objs=[], shuffle=None, **kwargs):
    if dataset_name=='Fashion200k':
      dataset = fashion200k_dataset.Fashion200k(GCZSL_DS_ROOT[dataset_name], phase, None if with_image else feature_file)
    elif dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
        dataset =  GCZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = GCZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            with_image=with_image,
            open_world = open_world,
            train_only = train_only,
            random_sampling = random_sampling,
            ignore_attrs=ignore_attrs,
            ignore_objs=ignore_objs,
            **kwargs)
    else:
        dataset =  CZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = CZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            **kwargs)

    if shuffle is None:
        shuffle = (phase=='train')
    
    return DataLoader(dataset, batchsize, shuffle, num_workers=num_workers,
#       collate_fn = lambda data: [np.stack(d, axis=0) for d in zip(*data)]
    )


    

