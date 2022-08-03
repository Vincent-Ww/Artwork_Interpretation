import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.m2 import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
# from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttentionMemory, ScaledDotProductAttention
# from models.grid_m2 import Transformer, TransformerEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
# from models.grid_m2_rst import  Transformer, TransformerEncoder, MeshedDecoder, ScaledDotProductAttention
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import itertools



class DataProcessor(nn.Module):
    def __init__(self):
        super(DataProcessor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x)    # [1, d, h, w] => [d, h, w] 
        x = x.permute(1, 2, 0)  # [d, h, w] => [h, w, d]
        return x.view(-1, x.size(-1))   # [h*w, d]


def predict(img_features):
    
    img_features = img_features.to(device)
    out, _ = model.beam_search(img_features, 20, text_field.vocab.stoi['<eos>'], 1, out_size=1, is_sample=False)
    print(out)
    caps_gen = text_field.decode(out, join_words=False)[0]
    caps_gen = ' '.join([k for k, g in itertools.groupby(caps_gen)])
    print(caps_gen)
    print("\n")
    
    
if __name__ == '__main__':    
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--ptm', type=str, default='none')
    args = parser.parse_args()
    print("ptm: ", args.ptm)


    device = torch.device('cuda')
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    print("Creating pipeline: done !")

    # Model and dataloaders
    # region m2
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    
#     # std
#     encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
#     decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
#     model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    
#     # grid m2
#     encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
#     decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
#     model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

#     # grid m2_rst
#     encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
#     decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
#     model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    print("Initialise model: done !")

    params = torch.load(args.ptm)
    model.load_state_dict(params['state_dict'])
    print("Load model parameters: done !")
    
    processor = DataProcessor()  

    dir_path = "../Dataset/artpedia/artpedia_region_features"
#     dir_path = "../Dataset/artpedia/artpedia_grid_features"
    print("feature type:", dir_path)

#     file_path = "../Dataset/artpedia/artpedia_region_features/1420.npz"
#     img = np.load(file_path)
#     print("--------------")
#     print(file_path[:-4])
# #     img = processor(torch.tensor(img)).unsqueeze(0)
#     img = torch.tensor([img['x']])
#     predict(img)
    
    for it, name in enumerate(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, name)
        img = np.load(file_path)
        print("--------------")
        print(name[:-4])
        img = torch.tensor([img['x']])
#         img = processor(torch.tensor(img)).unsqueeze(0)
        predict(img)
        
        
        