import random
from data import ImageDetectionsField, TextField, RawField
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset, SequentialSampler, RandomSampler, DataLoader as TorchDataLoader
import torch
import pickle
import numpy as np
import six
import h5py

from data.utils import get_tokenizer

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


# Pipeline for text
# text_field = TextField(init_token='<bos>', eos_token='<eos>', fix_length=20, lower=True, tokenize='spacy',
#                        remove_punctuation=True, nopoints=False)
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                       remove_punctuation=True, nopoints=False)

text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))


class AP_Dataset(TorchDataset):
    def __init__(self, ds_hdf5, my_idx, text_field, max_detections, lower=False, remove_punctuation=False,
                 tokenize=(lambda s: s.split()), nopoints=True):
        self.ds_hdf5 = ds_hdf5  # hdf5 object
        self.my_idx = my_idx  # np.array
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
        return len(self.my_idx)

    def __getitem__(self, item):
        print(item, my_idx[item])
        roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_features'][()]
        cap = self.ds_hdf5[f'{self.my_idx[item]}_cap'][()]

        delta = self.max_detections - len(roi_feat)
        if delta > 0:
            precomp_data = np.concatenate([roi_feat, np.zeros((delta, len(roi_feat[0])))], axis=0)
        else:
            precomp_data = roi_feat[:self.max_detections]

        print(1, cap)
        if self.lower:
            cap = six.text_type.lower(cap)
        print(2, cap)
        tokens = self.tokenize(cap.rstrip('\n'))
        print(3, tokens)
        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in self.punctuations]
        print(4, tokens)

        # padded = self.text_field.pad([tokens])
        # print(5, padded)
        # cap_ids = self.text_field.numericalize(padded)
        # print(6, cap_ids)
        return {
            "roi_feat": precomp_data.astype(np.float32),
            # "cap": cap_ids[0]
            "cap": tokens
        }


ap_roi_cap = h5py.File("../Dataset/artpedia/ap_roi_cap.hdf5", "r")
my_idx = np.load('../Dataset/artpedia/my_idx.npy')
print("loading data: done!!!")
ap_ds = AP_Dataset(ap_roi_cap, my_idx, text_field, max_detections=50, lower=True, remove_punctuation=True,
                   tokenize='spacy')


def text_progress(minibatch, text_field, device=None):
    print(f"text_progress {0}", minibatch)
    batch_tokens = [batch['cap'] for batch in minibatch]
    print(f"text_progress {1}", batch_tokens)
    padded_tokens = text_field.pad(batch_tokens)
    print(f"text_progress {2}", padded_tokens)
    token_ids = text_field.numericalize(padded_tokens, device=device)
    print(f"text_progress {3}", token_ids)
    return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
            "cap": token_ids}


device = torch.device('cuda')
data_loader = TorchDataLoader(ap_ds, batch_size=2, collate_fn=lambda x: text_progress(x, ap_ds.text_field, device=device))
# data_loader = DataLoader(ap_ds, batch_size=2, )

for b in data_loader:
    print("-----")
    print(b)
    print(b['cap'].shape)
    print(b['roi_feat'].shape)
    break



