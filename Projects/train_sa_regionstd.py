import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from data.dataset import SA_Dataset, SAeval_Dataset
import evaluation
from evaluation import PTBTokenizer, Cider
# from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttentionMemory, ScaledDotProductAttention
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
import argparse, os, pickle
from tqdm import tqdm
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import h5py
import time
import pandas as pd
import json

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

loss_file = "output_logs/losscurve_sa_multiple_region_std.txt"
with open(loss_file, 'a+') as f:
    f.write("\n\n\n")
    f.write(time.asctime(time.localtime(time.time())))
    f.write("\n")

def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                detections = batch['roi_feat']
                captions = batch['cap']
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(iter(dataloader)):
            images = batch['roi_feat']
            caps_gt = batch['cap']
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, train_data_loader, val_data_loader, optim, loss_fn, text_field):
    # Training with cross-entropy
#     model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(train_data_loader)) as pbar:
        for it, batch in enumerate(train_data_loader):
            model.train()
            detections = batch['roi_feat']
            captions = batch['cap']
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()
            
            if it % 200 == 0:
                val_loss = evaluate_loss(model, val_data_loader, loss_fn, text_field)
                print(f"the {it}th iteration: Val loss {val_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
                with open(loss_file, 'a+') as f:
                    f.write(f'the {it}th iteration, Val loss {val_loss} \n') 
                    
                    
    loss = running_loss / len(train_data_loader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()

    print('Meshed-Memory Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

# #***evaluate on coco dataset
#     # Pipeline for image regions
#     image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', fix_length=30, lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)



    
    # loading files for semart
    sa_train_csv = pd.read_csv("../Dataset/SemArt/prediction_csvs/semart_train_prediction.csv", delimiter='\t')
    sa_train_csv = sa_train_csv[sa_train_csv['prediction']==0]
    sa_val_csv = pd.read_csv("../Dataset/SemArt/prediction_csvs/semart_val_prediction.csv")
    sa_val_csv = sa_val_csv[sa_val_csv['predictioin']==0]
    sa_test_csv = pd.read_csv("../Dataset/SemArt/prediction_csvs/semart_test_prediction.csv")
    sa_test_csv = sa_test_csv[sa_test_csv['predictioin']==0]
    
    train_roi_feats = h5py.File("../Dataset/SemArt/sa_train_roi.hdf5", "r")
    val_roi_feats = h5py.File("../Dataset/SemArt/sa_val_roi.hdf5", "r")
    test_roi_feats = h5py.File("../Dataset/SemArt/sa_test_roi.hdf5", "r")
    
    train_img_names = np.unique(sa_train_csv['img_name'].to_numpy())
    val_img_names = np.unique(sa_val_csv['img_name'].to_numpy())
    test_img_names = np.unique(sa_test_csv['img_name'].to_numpy())

    train_img_caps_map = json.load(open('../Dataset/SemArt/train_img_caps_map.json'))
    val_img_caps_map = json.load(open('../Dataset/SemArt/val_img_caps_map.json'))
    test_img_caps_map = json.load(open('../Dataset/SemArt/test_img_caps_map.json'))
    
    print("loading files: done!!!")
    
        
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    print("Load vocabulary: done!!")
    
    
    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    
    model_params = torch.load('saved_models/saved_models_region_std/region_std_best.pth')
    model.load_state_dict(model_params['state_dict'])
    print("building model: done !!")
    
    
    

    
    semart_train = SA_Dataset(sa_train_csv, train_roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True,
                   tokenize='spacy')
    semart_val = SA_Dataset(sa_val_csv, val_roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True,
                   tokenize='spacy')
    semart_test = SA_Dataset(sa_test_csv, test_roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True,
                   tokenize='spacy')
    print("establish training dataset: done !!")
    
    
    dict_semart_train = SAeval_Dataset(sa_train_csv, train_img_names, train_img_caps_map, test_roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True, tokenize='spacy')
    dict_semart_val = SAeval_Dataset(sa_val_csv, val_img_names, val_img_caps_map, val_roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True, tokenize='spacy')
    dict_semart_test = SAeval_Dataset(sa_test_csv, test_img_names, test_img_caps_map, test_roi_feats, text_field, max_detections=50, lower=True, remove_punctuation=True, tokenize='spacy')
    print("establish evaluation dataset: done !!")
    
    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    best_val_loss = float('inf')
    patience = 0
    start_epoch = 0
    
    def text_progress(minibatch, text_field):
        batch_tokens = [batch['cap'] for batch in minibatch]
        padded_tokens = text_field.pad(batch_tokens)
        token_ids = text_field.numericalize(padded_tokens)
        return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
                "cap": token_ids}

    def text_progress2(minibatch):
        batch_tokens = [batch['cap'] for batch in minibatch]
        return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
                "cap": batch_tokens}

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    print("-----Training starts-----")
    for e in range(start_epoch, start_epoch + 100):
        print("-"*10 + "Epoch" + str(e) + "-"*10)
        with open(loss_file, 'a+') as f:
            f.write(f'+++++++++++++++++the {e}th Epoch+++++++++++++++++++\n') 
  
        # semart dataloader
        train_data_loader = TorchDataLoader(semart_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda x: text_progress(x, semart_train.text_field))
        val_data_loader = TorchDataLoader(semart_val, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress(x, semart_train.text_field))
        
        dict_train_data_loader = TorchDataLoader(dict_semart_train, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress2(x))
        dict_val_data_loader = TorchDataLoader(dict_semart_val, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress2(x))
        dict_test_data_loader = TorchDataLoader(dict_semart_test, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress2(x))
        
        

        
        # print the val loss before training
        val_loss = evaluate_loss(model, val_data_loader, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)
        print(f"Eproch -1: Val loss {val_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3")
        
        if not use_rl:
            print("training with xe")
            train_loss = train_xe(model, train_data_loader, val_data_loader, optim, loss_fn, text_field)
#             print(f"Eproch {e}: Training loss1 {train_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            print("training with rl")
#             train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
#             writer.add_scalar('data/train_loss', train_loss, e)
#             writer.add_scalar('data/reward', reward, e)
#             writer.add_scalar('data/reward_baseline', reward_baseline, e)
        

        # Validation loss
        val_loss = evaluate_loss(model, val_data_loader, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)
        print(f"Eproch {e}: Val loss {val_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!4")
        with open(loss_file, 'a+') as f:
            f.write(f'Epoch {e}, Val loss {val_loss} \n')
        
      
        # training loss
        train_loss1 = evaluate_loss(model, train_data_loader, loss_fn, text_field)
        writer.add_scalar('data/train_loss', train_loss1, e)
        print(f"Eproch {e}: Training loss2 {train_loss1}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!5")
        with open(loss_file, 'a+') as f:
            f.write(f'Epoch {e}, Training loss {train_loss1} \n')
        
        # Validation scores
        scores = evaluate_metrics(model, dict_val_data_loader, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
#         writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_test_data_loader, text_field)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
#         writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        
        print("validation: done !")

        # Prepare for next epoch
        best = False
        if val_loss < best_val_loss:
            print("-"*5 + "reach better loss" + "-"*5)
            with open(loss_file, 'a+') as f:
                    f.write("reach better loss") 
            best_val_loss = val_loss
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/sa_{}_last_{}epoch.pth'.format(args.exp_name, e))
        print("saved model: done !!!!")

        if best:
            copyfile('saved_models/sa_{}_last_{}epoch.pth'.format(args.exp_name, e), 'saved_models/sa_{}_best_{}epoch.pth'.format(args.exp_name, e))

        if exit_train:
            writer.close()
            print("exit at line 521!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            break
