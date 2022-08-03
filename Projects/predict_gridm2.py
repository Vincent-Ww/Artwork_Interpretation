import random
from data import TextField
from models.grid_m2 import Transformer, TransformerEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
import torch
import pickle
import numpy as np


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


device = torch.device('cuda')



text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                       remove_punctuation=True, nopoints=False)
text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': 40})
decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

# data = torch.load('meshed_memory_transformer.pth')
# model.load_state_dict(data['state_dict'])

print("Initialise model: done !")


# for i in range(10):
#     print("\n")
#     images = torch.tensor(np.load("D:/Vincent/melbuni/NLP_Research/Datasets/artpedia/artpedia_roi_features/" + str(i) + ".npz")['x']).to(device)
#     images = images[:50].unsqueeze(dim=0)
#
#     with torch.no_grad():
#         beam_size = 1
#         out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], beam_size, out_size=1, is_sample=True, top_k=10, top_p=0.9)
#     # print("out: ", out)
#
#     gen_cap = text_field.decode(out, join_words=True)
#     print("generated caption: ", gen_cap)



images0 = torch.tensor(np.load("../Dataset/artpedia/artpedia_features/0.npz")['x']).to(device)
# images1 = torch.tensor(np.load("../Dataset/artpedia/artpedia_features/1.npz")['x']).to(device)
# images = torch.stack([images0[:49], images1[:49]])
images = images0[:49].unsqueeze(dim=0)

print("===============================")
print("test beam search inference")
with torch.no_grad():
    beam_size = 3
    out, _ = model.beam_search(images, 10, text_field.vocab.stoi['<eos>'], beam_size, out_size=1, is_sample=False)
print("out: ", out)
gen_cap = text_field.decode(out, join_words=True)
print("generated caption: ", gen_cap)
print("\n\n\n")


print("===============================")
print("test forward propagation")
with torch.no_grad():
    seq = torch.tensor([[2]], device=device)
    out = model(images, seq)
print("out: ", out)
print("\n\n\n")