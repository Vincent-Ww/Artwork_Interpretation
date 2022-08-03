import random
from data import ImageDetectionsField, TextField, RawField
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset, SequentialSampler, RandomSampler, DataLoader as TorchDataLoader
from data.dataset import AP_Dataset0, AP_Dataset, APeval_Dataset
import torch
import pickle
import numpy as np
import pandas as pd
import six
import h5py
import json
from collections import Counter

from data.utils import get_tokenizer



class SA_Dataset(TorchDataset):
    def __init__(self, img_caption_pairs, roi_feats, text_field, max_detections, lower=False, remove_punctuation=False,
                 tokenize=(lambda s: s.split()), nopoints=True):
        self.img_caption_pairs = img_caption_pairs  # csv file
        self.roi_feats = roi_feats  # np.array
        self.text_field = text_field
        # roi parameters
        self.max_detections = max_detections
        # caption parameters
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.tokenize = get_tokenizer(tokenize)
        self.punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                             ".", "?", "!", ",", ":", "-", "--", "...", ";"]
        if nopoints:
            self.punctuations.append("..")

    def __len__(self):
        return self.img_caption_pairs.shape[0]

    def __getitem__(self, item):
        eg = self.img_caption_pairs.iloc[item]
        img_name = eg['img_name'][2:-6]
        cap = eg['caption']
        print('---------------------')
        print(item, img_name, cap)
        roi_feat = self.roi_feats[img_name]

        delta = self.max_detections - len(roi_feat)
        if delta > 0:
            precomp_data = np.concatenate([roi_feat, np.zeros((delta, len(roi_feat[0])))], axis=0)
        else:
            precomp_data = roi_feat[:self.max_detections]

        if self.lower:
            cap = six.text_type.lower(cap)
        tokens = self.tokenize(cap.rstrip('\n'))
        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in self.punctuations]

        return {
            "roi_feat": precomp_data.astype(np.float32),
            "cap": tokens
        }



# 构建evaluatioin的dataloader,返回原始的text caption
class SAeval_Dataset(TorchDataset):
    def __init__(self, img_caption_pairs, img_names, img_caps_map, roi_feats, text_field, max_detections, lower=False, remove_punctuation=False, tokenize=(lambda s: s.split()), nopoints=True):
        self.img_caption_pairs = img_caption_pairs  # csv file
        self.img_names = img_names                  # list of image names
        self.img_caps_map = img_caps_map              # number of captions of each image
        self.roi_feats = roi_feats                  # hdf5 file
        self.text_field = text_field
        # roi parameters
        self.max_detections = max_detections
        # caption parameters
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.tokenize = get_tokenizer(tokenize)
        self.punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                             ".", "?", "!", ",", ":", "-", "--", "...", ";"]
        if nopoints:
            self.punctuations.append("..")

    def __len__(self):
        return self.img_caption_pairs.shape[0]

    def __getitem__(self, item):
        img_name = self.img_names[item][2:-6]
        print("img_name: ", img_name)
        cap = self.img_caption_pairs.iloc[self.img_caps_map[img_name]]['caption'].to_numpy()
#         print('---------------------')
#         print(item, img_name, cap)
        roi_feat = self.roi_feats[img_name]

        delta = self.max_detections - len(roi_feat)
        if delta > 0:
            precomp_data = np.concatenate([roi_feat, np.zeros((delta, len(roi_feat[0])))], axis=0)
        else:
            precomp_data = roi_feat[:self.max_detections]
        
        return {
            "roi_feat": precomp_data.astype(np.float32),
            "cap": cap
        }




random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

# Pipeline for text
# text_field = TextField(init_token='<bos>', eos_token='<eos>', fix_length=20, lower=True, tokenize='spacy',
#                        remove_punctuation=True, nopoints=False)
text_field = TextField(init_token='<bos>', eos_token='<eos>', fix_length=40, lower=True, tokenize='spacy',
                       remove_punctuation=True, nopoints=False)

text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))



sa_test_csv = pd.read_csv("../Dataset/SemArt/prediction_csvs/semart_test_prediction.csv")
sa_test_csv = sa_test_csv[sa_test_csv['predictioin']==0]

roi_feats = h5py.File("../Dataset/SemArt/sa_test_roi.hdf5", "r")

test_img_names = np.unique(sa_test_csv['img_name'].to_numpy())
test_img_caps_map = json.load(open('../Dataset/SemArt/test_img_caps_map.json'))
print("loading files: done!!!")

sa_ds = SA_Dataset(sa_test_csv, roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True,
                   tokenize='spacy')

dict_sa_ds = SAeval_Dataset(sa_test_csv, test_img_names, test_img_caps_map, roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True, tokenize='spacy')


def text_progress(minibatch, text_field, device=None):
    # print(f"text_progress {0}", minibatch)
    batch_tokens = [batch['cap'] for batch in minibatch]
    # print(f"text_progress {1}", batch_tokens)
    padded_tokens = text_field.pad(batch_tokens)
    # print(f"text_progress {2}", padded_tokens)
    token_ids = text_field.numericalize(padded_tokens, device=device)
    # print(f"text_progress {3}", token_ids)
    return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
            "cap": token_ids}

def text_progress2(minibatch):
#         print(f"text_progress {0}", minibatch)
    batch_tokens = [batch['cap'] for batch in minibatch]
    return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
            "cap": batch_tokens}

device = torch.device('cuda')

data_loader = TorchDataLoader(sa_ds, batch_size=1, collate_fn=lambda x: text_progress(x, sa_ds.text_field, device=device))

dict_data_loader = TorchDataLoader(dict_sa_ds, batch_size=1, collate_fn=lambda x: text_progress2(x))


for id, batch in enumerate(data_loader):
    print("==============")
    print(batch['roi_feat'].shape)
    print(batch['cap'].shape)
    print(batch['roi_feat'][:, 0])
    print(batch['cap'])
    if id > 3:
        break


print("\n" * 10)
for id, batch in enumerate(dict_data_loader):
    print("==============")
    print(batch['roi_feat'].shape)
    print(batch['cap'])
    print(batch['roi_feat'][:, 0])
    if id > 3:
        break

