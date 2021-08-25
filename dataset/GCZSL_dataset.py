'''Symnet Dataset: https://arxiv.org/abs/2004.00587'''

import numpy as np
import torch, torchvision
import os, logging, pickle, sys
import tqdm
from itertools import product
import os.path as osp

from . import data_utils


class CompositionDatasetActivations(torch.utils.data.Dataset):

    def __init__(self, name, root, phase, feat_file, split='compositional-split', with_image=False, transform_type='normal', 
                 open_world=True, train_only=False, neg_sample_size=3, ignore_attrs=None, ignore_objs=None):
        self.root = root
        self.phase = phase
        self.split = split
        self.with_image = with_image
        self.neg_sample_size = neg_sample_size

        self.feat_dim = None
        self.transform = data_utils.imagenet_transform(phase, transform_type)
        self.loader = data_utils.ImageLoader(self.root+'/images/')
        
        self.ignore_objs = ignore_objs
        self.ignore_attrs = ignore_attrs
        self.ignore_mode = ignore_objs or ignore_attrs

        if feat_file is not None:
          feat_file = os.path.join(root, feat_file)
          activation_data = torch.load(feat_file)
          activation_data['files'] = ['_'.join(file.split()) for file in activation_data['files']]
          self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
          self.feat_dim = activation_data['features'].size(1)
          print ('%d activations loaded'%(len(self.activation_dict)))

        # pair = (attr, obj)
        (self.attrs, self.objs, self.pairs, 
        self.train_pairs, self.val_pairs, self.test_pairs) = self.parse_split()

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        
        self.op_pair2idx = dict()
        for i, attr in enumerate(self.attrs):
          for j, obj in enumerate(self.objs):
            self.op_pair2idx[(attr, obj)] = i * len(self.objs) + j
        
        self.open_world = open_world
        if open_world:
          self.all_pair2idx = self.op_pair2idx
        else:
          self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        
        if train_only:
          self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        else:
          self.pair2idx = self.all_pair2idx


        self.train_data, self.val_data, self.test_data = self.get_split_info()
        
        if self.phase=='train':
            self.data = self.train_data
        elif self.phase=='val':
            self.data = self.val_data
        elif self.phase=='test':
            self.data = self.test_data

        
        # list of [img_name, attr, obj, attr_id, obj_id, feat]
        print ('#images = %d'%len(self.data))
        
        
        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance_mask = []
        for _obj in self.objs:
            candidates = [x[1] for x in self.train_data+self.test_data if x[2]==_obj]
            # x = (_,attr,obj,_,_,_)
            affordance = set(candidates)
            mask = [1 if x in affordance else 0   for x in self.attrs]
            self.obj_affordance_mask.append(mask)
        

        # negative image pool
        samples_grouped_by_obj = [[] for _ in range(len(self.objs))]
        for i,x in enumerate(self.train_data):
            samples_grouped_by_obj[x[4]].append(i)

        self.neg_pool = []  # [obj_id][attr_id] => list of sample id
        for obj_id in range(len(self.objs)):
            self.neg_pool.append([])
            for attr_id in range(len(self.attrs)):
                self.neg_pool[obj_id].append(
                    [i for i in samples_grouped_by_obj[obj_id] if 
                        self.train_data[i][3] != attr_id ]
                )
        
        self.comp_gamma = {'a':1, 'b':1}
        self.attr_gamma = {'a':1, 'b':1}
        
    def ignored(self, attr_id, obj_id):
        return ((self.ignore_attrs is not None and attr_id in self.ignore_attrs) or
                (self.ignore_objs is not None and obj_id in self.ignore_objs))
        
    def get_split_info(self):
        data = torch.load(self.root+'/metadata.t7')
        train_pair_set = set(self.train_pairs)
        test_pair_set = set(self.test_pairs)
        train_data, val_data, test_data = [], [], []


        print("natural split "+self.phase)
        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']

            if attr=='NA' or (attr, obj) not in self.pairs or settype=='NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
            attr_id, obj_id = self.attr2idx[attr], self.obj2idx[obj]  
            data_i = [image, attr, obj, attr_id, obj_id, self.activation_dict[image]]
            
            if settype == 'train':
                if not self.ignore_mode or not self.ignored(attr_id, obj_id):
                    train_data.append(data_i)
                elif self.ignore_mode and self.ignored(attr_id, obj_id):
                    test_data.append(data_i)
            elif settype == 'val':
                if not self.ignore_mode or self.ignored(attr_id, obj_id):
                  val_data.append(data_i)
                elif self.ignore_mode and not self.ignored(attr_id, obj_id):
                  train_data.append(data_i)
            elif settype == 'test':
                if not self.ignore_mode or self.ignored(attr_id, obj_id):
                  test_data.append(data_i)
                elif self.ignore_mode and not self.ignored(attr_id, obj_id):
                  train_data.append(data_i)
            else:
                raise NotImplementedError(settype)

        return train_data, val_data, test_data


    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list,'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt'%(self.root, self.split))
        val_attrs, val_objs, val_pairs = parse_pairs('%s/%s/val_pairs.txt'%(self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt'%(self.root, self.split))

        all_attrs =  sorted(list(set(tr_attrs + val_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + val_objs + ts_objs)))    
        all_pairs = sorted(list(set(tr_pairs + val_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, val_pairs, ts_pairs



    def sample_negative(self, attr_id, obj_id):
        return np.random.choice(self.neg_pool[obj_id][attr_id], self.neg_sample_size)

    def __getitem__(self, index):
        def get_sample(i):
            image, attr, obj, attr_id, obj_id, feat = self.data[i]
              
            if self.with_image:
                img = self.loader(image)
                img = self.transform(img)
            else:
                img = image
            return [img, attr_id, obj_id, self.pair2idx[(attr, obj)], feat, image]
          
        def get_batch_sample(sample_ids):
          samples = [get_sample(i) for i in sample_ids]
          samples = list(zip(*samples))
          return samples
        
        image_name = self.data[index][0]
        pos = get_sample(index)

        mask = np.array(self.obj_affordance_mask[pos[2]], dtype=np.float32)

        if self.phase=='train':
            if self.ignore_mode:
              negid = [0] * self.neg_sample_size
            else:
              negid = self.sample_negative(pos[1], pos[2]) # negative example
            if self.neg_sample_size > 1:
              neg = get_batch_sample(negid)
            else:
              neg = get_sample(negid[0])
            data = pos + neg + [mask]
        else:
            data = pos + [mask]

        # train [img, attr_id, obj_id, pair_id, img_feature, img_path, img, attr_id, obj_id, pair_id, img_feature, img_path, aff_mask]
        # test [img, attr_id, obj_id, pair_id, img_feature, img_path, aff_mask]

        return data

    def __len__(self):
        return len(self.data)


class CompositionDatasetActivationsGenerator(CompositionDatasetActivations):

    def __init__(self, root, feat_file, split='compositional-split', feat_extractor=None, transform_type='normal'):
        super(CompositionDatasetActivationsGenerator, self).__init__(root, 'train', None, split, transform_type=transform_type)

        assert os.path.exists(root)
        with torch.no_grad():
            self.generate_features(feat_file, feat_extractor, transform_type)
        print('Features generated.')

    def get_split_info(self):
        data = torch.load(self.root+'/metadata.t7')
        train_pair_set = set(self.train_pairs)
        test_pair_set = set(self.test_pairs)
        train_data, val_data, test_data = [], [], []

        print("natural split")
        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']

            if attr=='NA' or (attr, obj) not in self.pairs or settype=='NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
                
            data_i = [image, attr, obj, self.attr2idx[attr], self.obj2idx[obj], None]

            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            elif settype == 'test':
                test_data.append(data_i)
            else:
                raise NotImplementedError(settype)

        return train_data, val_data, test_data
        

    def generate_features(self, out_file, feat_extractor, transform_type):

        data = self.train_data+self.val_data+self.test_data
        transform = data_utils.imagenet_transform('test', transform_type)

        if feat_extractor is None:
            feat_extractor = torchvision.models.resnet18(pretrained=True)
            feat_extractor.fc = torch.nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(data_utils.chunks(data, 512), total=len(data)//512):
            files = zip(*chunk)[0]
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print ('features for %d images generated'%(len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)





if __name__=='__main__':
    """example code for generating new features for MIT states and UT Zappos
    CompositionDatasetActivationsGenerator(
        root = 'data-dir', 
        feat_file = 'filename-to-save', 
        feat_extractor = torchvision.models.resnet18(pretrained=True),
    )
    """

    if sys.argv[1]=="MIT":
        name = "mit-states"
    elif sys.argv[1]=="UT":
        name = "ut-zap50k"
    

    CompositionDatasetActivationsGenerator(
        root = 'data/%s-natural'%name, 
        feat_file = 'data/%s-natural/features.t7'%name,
    )