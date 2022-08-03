import os
import numpy as np
import itertools
import collections
import torch
from torch.utils.data import Dataset as TorchDataset
from .example import Example
from .utils import nostdout, get_tokenizer
from pycocotools.coco import COCO as pyCOCO
import six 



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
        self.img_caps_map = img_caps_map            # number of captions of each image
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
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item][2:-6]
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




    
    

# each image has multiple captions
class AP_Dataset(TorchDataset):
    def __init__(self, ds_hdf5, item2imgCapIdx, text_field, max_detections, feature_type, lower=False, remove_punctuation=False,
                 tokenize=(lambda s: s.split()), nopoints=True):
        self.ds_hdf5 = ds_hdf5  # hdf5 object
        self.item2imgCapIdx = item2imgCapIdx  # np.array
        self.text_field = text_field
        # roi parameters
        self.max_detections = max_detections
        self.feature_type = feature_type
        # caption parameters
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.tokenize = get_tokenizer(tokenize)
        self.punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                             ".", "?", "!", ",", ":", "-", "--", "...", ";"]
        if nopoints:
            self.punctuations.append("..")

    def __len__(self):
        return len(self.item2imgCapIdx)

    def __getitem__(self, item):
        img_id, cap_cnt = self.item2imgCapIdx[item]
        if self.feature_type == "grid":
            roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_grids'][()]
        elif self.feature_type == "detector":
            roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_features'][()]
        cap = self.ds_hdf5[f'{img_id}_cap'][()]
        cap = cap[cap_cnt]

        delta = self.max_detections - len(roi_feat)
        if delta > 0:
            precomp_data = np.concatenate([roi_feat, np.zeros((delta, len(roi_feat[0])))], axis=0)
        else:
            precomp_data = roi_feat[:self.max_detections]

        # print(1, cap)
        if self.lower:
            cap = six.text_type.lower(cap)
        # print(2, cap)
        tokens = self.tokenize(cap.rstrip('\n'))
        # print(3, tokens)
        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in self.punctuations]
#         print(4, tokens)

        # padded = self.text_field.pad([tokens])
        # print(5, padded)
        # cap_ids = self.text_field.numericalize(padded)
        # print(6, cap_ids)

        return {
            "roi_feat": precomp_data.astype(np.float32),
            # "cap": cap_ids[0]
            "cap": tokens
        }

# each image has just one caption
class AP_Dataset0(TorchDataset):
    def __init__(self, ds_hdf5, my_idx, text_field, max_detections, feature_type, lower=False, remove_punctuation=False,
                 tokenize=(lambda s: s.split()), nopoints=True):
        self.ds_hdf5 = ds_hdf5  # hdf5 object
        self.my_idx = my_idx  # np.array
        self.text_field = text_field
        # roi parameters
        self.max_detections = max_detections
        self.feature_type = feature_type
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
#         print(item, self.my_idx[item])
#         print("getitem0", item)
#         print("getitem0", self.my_idx[item])
        if self.feature_type == "grid":
            roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_grids'][()]
        elif self.feature_type == "detector":
            roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_features'][()]
        cap = self.ds_hdf5[f'{self.my_idx[item]}_cap'][()]
#         print("getitem1", cap)
        cap = cap[0]
#         print("getitem2", cap)
        
        delta = self.max_detections - len(roi_feat)
        if delta > 0:
            precomp_data = np.concatenate([roi_feat, np.zeros((delta, len(roi_feat[0])))], axis=0)
        else:
            precomp_data = roi_feat[:self.max_detections]
        
        if self.lower:
            cap = six.text_type.lower(cap)
#         print(2, cap)
        tokens = self.tokenize(cap.rstrip('\n'))
#         print(3, tokens)
        if self.remove_punctuation:
            tokens = [w for w in tokens if w not in self.punctuations]
#         print(4, tokens)

#         padded = self.text_field.pad([tokens])
#         print(5, padded)
#         cap_ids = self.text_field.numericalize(padded)
#         print(6, cap_ids)
        return {
            "roi_feat": precomp_data.astype(np.float32),
            # "cap": cap_ids[0]
            "cap": tokens
        }

# 构建evaluatioin的dataloader,返回原始的text caption
class APeval_Dataset(TorchDataset):
    def __init__(self, ds_hdf5, my_idx, text_field, max_detections, feature_type, lower=False, remove_punctuation=False,
                 tokenize=(lambda s: s.split()), nopoints=True):
        self.ds_hdf5 = ds_hdf5  # hdf5 object
        self.my_idx = my_idx  # np.array
        self.text_field = text_field
        # roi parameters
        self.max_detections = max_detections
        self.feature_type = feature_type
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
#         print(item, self.my_idx[item])
#         print("getitem0", item)
#         print("getitem0", self.my_idx[item])
        if self.feature_type == "grid":
            roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_grids'][()]
        elif self.feature_type == "detector":
            roi_feat = self.ds_hdf5[f'{self.my_idx[item]}_features'][()]
        cap = self.ds_hdf5[f'{self.my_idx[item]}_cap'][()]
#         print("getitem1", cap)

        
        delta = self.max_detections - len(roi_feat)
        if delta > 0:
            precomp_data = np.concatenate([roi_feat, np.zeros((delta, len(roi_feat[0])))], axis=0)
        else:
            precomp_data = roi_feat[:self.max_detections]
        

#         padded = self.text_field.pad([tokens])
#         print(5, padded)
#         cap_ids = self.text_field.numericalize(padded)
#         print(6, cap_ids)
        return {
            "roi_feat": precomp_data.astype(np.float32),
            "cap": cap
        }


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)
            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))
        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCO(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['val'] = ids['val'][:5000]
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None
        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
#         self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []
        for split in ['train', 'val', 'test']:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]['caption']
                img_id = coco.anns[ann_id]['image_id']
                filename = coco.loadImgs(img_id)[0]['file_name']
                
                # caption: 这里的caption变量就是单个caption
                example = Example.fromdict({'image': os.path.join(img_root, filename), 'text': caption})
                if split == 'train':
                    train_samples.append(example)
                elif split == 'val':
                    val_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        return train_samples, val_samples, test_samples

