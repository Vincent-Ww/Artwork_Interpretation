import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from data.dataset import AP_Dataset0, AP_Dataset, APeval_Dataset
import evaluation
from evaluation import PTBTokenizer, Cider
from models.grid_m2 import Transformer, TransformerEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
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

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

loss_file = "output_logs/losscurve_ap_multiple_grid_m2.txt"
with open(loss_file, 'a+') as f:
    f.write("\n\n\n")
    f.write("artpedia\n")
    f.write("training using whole dataset\n")
    f.write("shuffle the training data before training\n")
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
#                 if it == 2:
#                     break

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
#             if it == 2:
#                 break

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
#             if it == 2:
#                 break
            if it % 50 == 0:
                val_loss = evaluate_loss(model, val_data_loader, loss_fn, text_field)
                print(f"the {it}th iteration: Val loss {val_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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


    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', fix_length=30, lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    
    # load files for artpedia
    train_myidx = np.load('../Dataset/artpedia/train_myidx.npy')
    val_myidx = np.load('../Dataset/artpedia/val_myidx.npy')
    test_myidx = np.load('../Dataset/artpedia/test_myidx.npy')
    
#     ap_roi_cap = h5py.File("../Dataset/artpedia/ap_roi_cap.hdf5", "r")   

#     # region features
#     ap_train_dataset = h5py.File("../Dataset/artpedia/artpedia_train2.hdf5", "r")
#     ap_val_dataset = h5py.File("../Dataset/artpedia/artpedia_val2.hdf5", "r")
#     ap_test_dataset = h5py.File("../Dataset/artpedia/artpedia_test2.hdf5", "r")
    
    # grid features
    ap_train_dataset = h5py.File("../Dataset/artpedia/ap_train_grid.hdf5", "r")
    ap_val_dataset = h5py.File("../Dataset/artpedia/ap_val_grid.hdf5", "r")
    ap_test_dataset = h5py.File("../Dataset/artpedia/ap_test_grid.hdf5", "r")
    
    train_item2imgCapIdx = np.load("../Dataset/artpedia/train_item2imgCapIdx.npy", allow_pickle=True).item()
    val_item2imgCapIdx = np.load("../Dataset/artpedia/val_item2imgCapIdx.npy", allow_pickle=True).item()
    test_item2imgCapIdx = np.load("../Dataset/artpedia/test_item2imgCapIdx.npy", allow_pickle=True).item()
    print("loading files: done!!!")

    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    print("Load vocabulary: done!!")
    
    # Model and dataloaders
    
#     # meshed memory transformer
#     encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
#                                      attention_module_kwargs={'m': args.m})
#     decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
#     model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    
#     model_params = torch.load('meshed_memory_transformer.pth')
#     model.load_state_dict(model_params['state_dict'])

    # m2 model
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    
    model_params = torch.load('saved_models_grid_m2/grid_m2_best.pth')
    model.load_state_dict(model_params['state_dict'])
    print("building model: done !!")
    
    
#     artpedia_train = AP_Dataset0(ap_roi_cap, my_idx, text_field, max_detections=50, lower=True, remove_punctuation=True,
#                        tokenize='spacy')

#     # one caption each image during training
#     artpedia_train = AP_Dataset0(ap_train_dataset, train_myidx, text_field, max_detections=49, lower=True, remove_punctuation=True,
#                        tokenize='spacy')
#     artpedia_val = AP_Dataset0(ap_val_dataset, val_myidx, text_field, max_detections=49, lower=True, remove_punctuation=True,
#                        tokenize='spacy')
#     artpedia_test = AP_Dataset0(ap_test_dataset, test_myidx, text_field, max_detections=49, lower=True, remove_punctuation=True,
#                        tokenize='spacy')

    # all captions each image during training 
    artpedia_train = AP_Dataset(ap_train_dataset, train_item2imgCapIdx, text_field, max_detections=49, lower=True, remove_punctuation=True, tokenize='spacy')
    artpedia_val = AP_Dataset(ap_val_dataset, val_item2imgCapIdx, text_field, max_detections=49, lower=True, remove_punctuation=True, tokenize='spacy')
    artpedia_test = AP_Dataset(ap_test_dataset, test_item2imgCapIdx, text_field, max_detections=49, lower=True, remove_punctuation=True, tokenize='spacy')
    print("Establish dataset: done !!")
    
#     #***evaluate on coco dataset
#     dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
#     ref_caps_train = list(train_dataset.text)
#     cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
#     dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
#     dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    
    dict_artpedia_train = APeval_Dataset(ap_train_dataset, train_myidx, text_field, max_detections=49, lower=True, remove_punctuation=True, tokenize='spacy')
    dict_artpedia_val = APeval_Dataset(ap_val_dataset, val_myidx, text_field, max_detections=49, lower=True, remove_punctuation=True, tokenize='spacy')
    dict_artpedia_test = APeval_Dataset(ap_test_dataset, test_myidx, text_field, max_detections=49, lower=True, remove_punctuation=True, tokenize='spacy')
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
#         print(f"text_progress {0}", minibatch)
        batch_tokens = [batch['cap'] for batch in minibatch]
#         print(f"text_progress {1}", batch_tokens)
        padded_tokens = text_field.pad(batch_tokens)
#         print(f"text_progress {2}", padded_tokens)
        token_ids = text_field.numericalize(padded_tokens)
#         print(f"text_progress {3}", token_ids)
        return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
                "cap": token_ids}

    def text_progress2(minibatch):
#         print(f"text_progress {0}", minibatch)
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
        
        # artpedia dataloader
        train_data_loader = TorchDataLoader(artpedia_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda x: text_progress(x, artpedia_train.text_field))
        val_data_loader = TorchDataLoader(artpedia_val, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress(x, artpedia_val.text_field))
        
        dict_train_data_loader = TorchDataLoader(dict_artpedia_train, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress2(x))
        dict_val_data_loader = TorchDataLoader(dict_artpedia_val, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress2(x))
        dict_test_data_loader = TorchDataLoader(dict_artpedia_test, batch_size=args.batch_size,
                                  collate_fn=lambda x: text_progress2(x))
        
        
        
        
        # #***evaluate on coco dataset
#         dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
#         dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
#         dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True, num_workers=args.workers)
#         dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
#         dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        
#         print("COCO.splits: Paired_Dataset")
#         cnt = 0
#         for b in dataloader_train:
#             print(b)
#             cnt += 1
#             if cnt >=2:
#                 break
        
        
#         print("COCO.splits.imagedirectory")
#         cnt = 0
#         for b in dict_dataloader_train:
#             print(b)
#             cnt += 1
#             if cnt >=2:
#                 break
        
        # print the val loss before training
        val_loss = evaluate_loss(model, val_data_loader, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)
        print(f"Eproch -1: Val loss {val_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
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
        print(f"Eproch {e}: Val loss {val_loss}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        with open(loss_file, 'a+') as f:
            f.write(f'Epoch {e}, Val loss {val_loss} \n')
        
      
        # training loss
        train_loss1 = evaluate_loss(model, train_data_loader, loss_fn, text_field)
        writer.add_scalar('data/train_loss', train_loss1, e)
        print(f"Eproch {e}: Training loss2 {train_loss1}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
                    f.write("reach better loss\n") 
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
        }, 'saved_models/{}_last_{}epoch.pth'.format(args.exp_name, e))
        print("saved model: done !!!!")

        if best:
            copyfile('saved_models/{}_last_{}epoch.pth'.format(args.exp_name, e), 'saved_models/{}_best_{}epoch.pth'.format(args.exp_name, e))

        if exit_train:
            writer.close()
            print("exit at line 483!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            break
