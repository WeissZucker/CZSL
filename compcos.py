import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

from itertools import product

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to('cpu')

class CompCos(nn.Module):
    def __init__(self, hparam, dset, resnet_name):
        super(CompCos, self).__init__()
        self.hparam = hparam
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(dev)
            objs = torch.LongTensor(objs).to(dev)
            pairs = torch.LongTensor(pairs).to(dev)
            return attrs, objs, pairs
          
        if resnet_name:    
          import torchvision.models as models
          self.resnet = models.resnet18(pretrained=True).to(dev)
          if static_inp:
            self.resnet = frozen(self.resnet)
          self.img_feat_dim = self.resnet.fc.in_features
          self.resnet.fc = nn.Identity()
        else:
          self.resnet = None

        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(dev), \
                                          torch.arange(len(self.dset.objs)).long().to(dev)
        self.factor = 2
        self.scale = 20


        # Precompute training compositions
        if self.hparam.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        self.hparam.add_dict({'img_fc_layers': [768, 1024, 1200],
                              'shared_emb_dim': 600})
          
        self.image_embedder = ParametricMLP(dset.feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers)

        # Fixed
        self.composition = 'mlp_add'
        
        attrs_init = torch.load('./embeddings/MIT/w2v_ft_attrs.pt').to(dev)
        objs_init = torch.load('./embeddings/MIT/w2v_ft_objs.pt').to(dev)
        init_dim = attrs_init.size(-1)
        assert init_dim == self.hparam.shared_emb_dim, "The primitive emb dimension doesn't match the shared_emb_dim"
        self.attr_embedder = nn.Embedding(len(dset.attrs), self.hparam.shared_emb_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), self.hparam.shared_emb_dim)

        self.attr_embedder.weight.data.copy_(attrs_init)
        self.obj_embedder.weight.data.copy_(objs_init)
 
        # Composition MLP
        self.projection = nn.Linear(self.hparam.shared_emb_dim * 2, self.hparam.shared_emb_dim)


    def freeze_representations(self):
        print('Freezing representations')
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False


    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output = F.normalize(output, dim=1)
        return output


    def compute_feasibility(self):
        obj_embeddings = self.obj_embedder(torch.arange(len(self.objs)).long().to('cuda'))
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                           return_dict=True)
        attr_embeddings = self.attr_embedder(torch.arange(len(self.attrs)).long().to('cuda'))
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                            return_dict=True)

        feasibility_scores = self.seen_mask.clone().float()
        for a in self.attrs:
            for o in self.objs:
                if (a, o) not in self.known_pairs:
                    idx = self.dset.all_pair2idx[(a, o)]
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    feasibility_scores[idx] = score

        self.feasibility_scores = feasibility_scores

        return feasibility_scores * (1 - self.seen_mask.float())


    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o!=obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj,o)]
                if temp_score>score:
                    score=temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score

    def update_feasibility(self,epoch):
        self.activated = True
        feasibility_scores = self.compute_feasibility()
        self.feasibility_margin = min(1.,epoch/self.epoch_max_margin) * \
                                  (self.cosine_margin_factor*feasibility_scores.float().to(dev))

'''
    def val_forward(self, x):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
        score = torch.matmul(img_feats_normed, pair_embeds)

        return None, score


    def val_forward_with_threshold(self, x, th=0.):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
        score = torch.matmul(img_feats_normed, pair_embeds)

        # Note: Pairs are already aligned here
        mask = (self.feasibility_scores>=th).float()
        score = score*mask + (1.-mask)*(-1.)

        return None, score


    def train_forward_open(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_feats = self.image_embedder(img)

        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
        img_feats_normed = F.normalize(img_feats, dim=1)

        pair_pred = torch.matmul(img_feats_normed, pair_embed)

        if self.activated:
            pair_pred += (1 - self.seen_mask) * self.feasibility_margin
            loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
        else:
            pair_pred = pair_pred * self.seen_mask + (1 - self.seen_mask) * (-10)
            loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)

        return loss_cos.mean(), None


    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_feats = self.image_embedder(img)

        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
        img_feats_normed = F.normalize(img_feats, dim=1)

        pair_pred = torch.matmul(img_feats_normed, pair_embed)

        loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)

        return loss_cos.mean(), None


    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)

        return loss, pred
'''