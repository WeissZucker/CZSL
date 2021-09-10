from . import CZSL_dataset, GCZSL_dataset
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
}

def get_dataloader(dataset_name, phase, feature_file="features.t7", batchsize=1, num_workers=0, open_world=True, train_only=False, 
                   random_sample_size=1,ignore_attrs=[], ignore_objs=[], shuffle=None, **kwargs):
    if dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
        dataset =  GCZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = GCZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            open_world = open_world,
            train_only = train_only,
            random_sample_size = random_sample_size,
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


    

