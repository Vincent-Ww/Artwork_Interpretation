import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset,SequentialSampler,RandomSampler,DataLoader

from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging

PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

class Imgtext_Lastfour_Clf(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        config.update({'output_hidden_states':True})
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=config)
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        self.linear = nn.Linear(4*self.bert.config.hidden_size+512, 2)
#         self.linear = nn.Linear(4*self.bert.config.hidden_size, 2)
        
    def forward(self, images, input_ids, attention_mask):
        
        outputs = self.bert(input_ids, attention_mask)
        all_hidden_states = torch.stack(outputs[2])
        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )
        concatenate_pooling = concatenate_pooling[:,0]
        img_feats = self.resnet18(images).squeeze()
        combined_feats = torch.cat((concatenate_pooling, img_feats), axis=1)
        output = self.linear(combined_feats)
#         output = self.linear(concatenate_pooling)
        return output