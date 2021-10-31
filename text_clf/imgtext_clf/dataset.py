from utils import *
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset,SequentialSampler,RandomSampler,DataLoader
# torchvision
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
import json

file_path = "../Datasets/artpedia/artpedia.json"
with open(file_path, 'r') as f:
    artpedia = json.load(f)

ind2count = {}
count = 0
for k in artpedia:
    ind2count[k] = str(count)
    count += 1

class DescriptionDataset(Dataset):
    def __init__(self, img_ids, sents, labels, tokenizer, max_len):
        self.img_ids = img_ids
        self.sents = sents
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        
    def __len__(self):
#         return sum([e is not None for e in self.img_ids])
        return len(self.sents)
        
    def __getitem__(self, item):
        img_path = "../Datasets/artpedia/resize_images/"+self.img_ids[item]+".jpg"
        img = Image.open(img_path)
        img = self.data_transform["val"](img)
        sent = self.sents[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            sent,
            add_special_tokens = True,
            max_length=self.max_len,
            return_token_type_ids = True,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        
        return {
            'images': img,
            'sents': sent,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }
    
    
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = DescriptionDataset(
        img_ids = df['image_index'].values,
        sents = df["sentence"].values,
        labels = df["label"].values,
        tokenizer = tokenizer,
        max_len = max_len,
    )
    
    return DataLoader(ds, batch_size)