# 导入transformers
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging

# 导入torch
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

# 常用包
import re
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from tqdm import tqdm
import json
import random
import time
import warnings
import os
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from model import *
from train import *

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # initialize seeds
    RANDOM_SEED = 42
    set_random_seed(RANDOM_SEED)


    file_path = "../Datasets/artpedia/artpedia.json"
    with open(file_path, 'r') as f:
        artpedia = json.load(f)


    ind2count = {}
    count = 0
    for k in artpedia:
        ind2count[k] = str(count)
        count += 1


    # construct dataframe {img_index, sents, categories}
    sents = []
    cates = []
    imgs = []

    for ind in artpedia:
        img_path = "../Datasets/artpedia/resize_images/"+ind2count[ind]+".jpg"
        if os.path.exists(img_path):
            img_ind = ind2count[ind]
        else:
    #         print(ind2count[ind], end=" ")
            img_ind = "null"

    #     visuals = artpedia[ind]['visual_sentences']
    #     contexts = artpedia[ind]['contextual_sentences']
    #     visual_sents.extend(visuals)
    #     context_sents.extend(contexts)
        for s in artpedia[ind]['visual_sentences']:
            sents.append(s)
            cates.append("visual")
            imgs.append(img_ind)
        for s in artpedia[ind]['contextual_sentences']:
            sents.append(s)
            cates.append("contextual")
            imgs.append(img_ind)


    des_df = pd.DataFrame({"image_index": imgs, "sentence": sents, "category": cates})

    id2label = dict(enumerate(des_df.category.unique()))
    label2id = {v: k for k, v in id2label.items()}

    des_df['label'] = des_df['category'].map(label2id)
    print("dataset: ")
    print(pd.concat((des_df.head(3), des_df.tail(2))))

    # split train, valid, test set.  
    # use the same test set on every experiment
    train, test = train_test_split(des_df, test_size=0.1, random_state=RANDOM_SEED)
    val, test = train_test_split(test, test_size=0.5, random_state=RANDOM_SEED)

    test = test[test['image_index']!="null"]

    print("size of training set: {}".format(train.shape))
    print("size of valid set: {}".format(val.shape))
    print("size of test set: {}".format(test.shape))
    print("\n\n")

    # hyperparameters of dataloader
    MAX_LEN = 64
    BATCH_SIZE = 8
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create dataset loader



    # tokenizer
    PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    train_data_loader = create_data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)
    valid_data_loader = create_data_loader(val, tokenizer, MAX_LEN, BATCH_SIZE)

    # load the model
    model = Imgtext_Lastfour_Clf()
    model.to(device)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load("img_text_lastfour_2000.bin"))

    loss_fn = nn.CrossEntropyLoss().to(device)

    # eval on test
    test_acc, _ = eval_model(
      model,
      test_data_loader,
      loss_fn,
      device,
      len(test)
    )

    print("accuracy on test set: {}".format(test_acc.item()))
    print("\n")


    class_names = train.category.unique()
    y_texts, y_pred, y_pred_probs, y_test = get_predictions(
      model,
      test_data_loader,
      device
    )


    print(classification_report(y_test, y_pred, target_names=[str(label) for label in class_names]))
    
    
def get_predictions(model, data_loader, device):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            images = d["images"].to(device)
            texts = d["sents"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values


if __name__ == '__main__':
    main()