import numpy as np
import PIL
import skimage
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
import os


class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.imgs = []
        self.test_queries = []

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError

class Fashion200k(BaseDataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', feat_file=None):
        super().__init__()
        
        self.name = 'Fashion200k'
        self.split = split
        self.img_path = path + '/'
        self.open_world = False
        
        if feat_file:
          self.train_feat, self.test_feat = torch.load(os.path.join(path,feat_file))
          self.feat_dim = self.train_feat.size(1)
        else:
          self.train_feat, self.test_feat = None, None
          self.feat_dim = None

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        train_label_files, test_label_files = [], []
        for f in label_files:
            if 'train' in f:
                train_label_files.append(f)
            else:
                test_label_files.append(f)

        # read image info from label files
        self.train_imgs, self.test_imgs = [], []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')
        
        self.attrs, self.objs = set(), set()
        self.train_pairs, self.test_pairs = set(), set()
        
        for i, label_files in enumerate([train_label_files, test_label_files]):
            for filename in label_files:
                with open(label_path + '/' + filename) as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.split('	')
                    obj = line[0].split('/')[1]
                    caption = caption_post_process(line[2])
                    self.objs.add(obj)
                    for attr in caption.split():
                      self.attrs.add(attr)
                    img = {
                        'file_path': line[0],
                        'detection_score': line[1],
                        'captions': [caption],
                        'split': split,
                        'modifiable': False,
                        'obj': obj
                    }
                    if i==0:
                        self.train_imgs += [img]
                        for attr in caption.split():
                          self.train_pairs.add((attr, obj))
                    else:
                        self.test_imgs += [img]
                        for attr in caption.split():
                          self.test_pairs.add((attr, obj))
        
        self.attrs = sorted(list(self.attrs))
        self.objs = sorted(list(self.objs))
        self.pairs = sorted(list(self.train_pairs | self.test_pairs))
        self.train_pairs = sorted(list(self.train_pairs))
        self.test_pairs = sorted(list(self.test_pairs))
        self.attr2idx = {attr:i for i, attr in enumerate(self.attrs)}
        self.obj2idx = {obj:i for i, obj in enumerate(self.objs)}
        self.pair2idx = {pair:i for i, pair in enumerate(self.pairs)}

        # generate query for training and testing
        
        self.caption_index_init_()
        self.generate_test_queries_()
        
        if self.split=='train':
          print('Fashion200k - train:', len(self.train_imgs), 'images')
          self.test_imgs = None
          self.test_feat = None
        else:
          print('Fashion200k - test:', len(self.test_imgs), 'images')
          self.train_imgs = None
          self.train_feat = None

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.test_imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.test_imgs[idx]['captions'][0]
            target_caption = self.test_imgs[target_idx]['captions'][0]
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'obj': self.test_imgs[idx]['obj']
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.train_imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique captions')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.train_imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.train_imgs[imgid]['modifiable'] = True
                        self.train_imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.train_imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.train_imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.train_imgs))

        # find random target image (same parent)
        img = self.train_imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        return idx, target_idx

    def get_all_texts(self, split):
        texts = []
        imgs = self.train_imgs if split=='train' else self.test_imgs
        for img in imgs:
            for c in img['captions']:
                texts.append(c)
        return texts
        
    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str
        
    # [img_name, attr_id, obj_id, pair_id, img_feature, img_name, attr_id, obj_id, pair_id, img_feature]
    def getitem_train(self, idx):
        def get_sample(idx, attr):
            img = self.train_imgs[idx]
            attr_id = self.attr2idx[attr]
            obj_id = self.obj2idx[img['obj']]
            pair_id = self.pair2idx[(attr, img['obj'])]
            return [idx, attr_id, obj_id, pair_id, self.get_img(idx)]
          
        idx, target_idx = self.caption_index_sample_(
            idx)
        s_caption = self.train_imgs[idx]['captions'][0]
        t_caption = self.train_imgs[target_idx]['captions'][0]
        s_attr, t_attr, _ = self.get_different_word(s_caption, t_caption)
        sample = get_sample(idx, s_attr)
        sample += get_sample(target_idx, t_attr)
        return sample
      
    def getitem_test(self, query_idx):
        '''
        'source_img_id': idx,
        'source_caption': source_caption,
        'target_caption': target_caption,
        'obj': self.imgs[idx]['obj']
        '''
        query = self.test_queries[query_idx]
        img_idx  = query['source_img_id']
        s_caption, t_caption = query['source_caption'], query['target_caption']
        s_attr, t_attr, _ = self.get_different_word(s_caption, t_caption)
        attr_id = self.attr2idx[s_attr]
        obj_id = self.obj2idx[query['obj']]
        pair_id = self.pair2idx[(s_attr, query['obj'])]
        t_attr_id = self.attr2idx[t_attr]
        t_pair_id = self.pair2idx[(t_attr, query['obj'])]
        sample = [img_idx, attr_id, obj_id, pair_id, self.get_img(img_idx), t_caption, t_attr_id, obj_id, t_pair_id]
        return sample
      
    def __getitem__(self, idx):
        if self.split == 'train':
          return self.getitem_train(idx)
        else:
          return self.getitem_test(idx)
        
    def __len__(self):
        if self.split == 'train':
          return len(self.train_imgs)
        else:
          return len(self.test_queries)

    def get_img(self, idx, raw_img=False):
        feat_file = self.train_feat if self.split=='train' else self.test_feat
        if feat_file is not None:
          return feat_file[idx]
        
        imgs = self.train_imgs if self.split=='train' else self.test_imgs
        img_path = self.img_path + imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img

        img = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])(img)
        return img