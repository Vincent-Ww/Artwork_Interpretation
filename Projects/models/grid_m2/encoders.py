from torch.nn import functional as F
from models.grid_m2.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.grid_m2.attention import MultiHeadAttention, MultiHeadGeometryAttention
from models.grid_m2.grid_aug import BoxRelationalEmbedding


# class EncoderLayer(nn.Module):
#     def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
#                  attention_module=None, attention_module_kwargs=None):
#         super(EncoderLayer, self).__init__()
#         self.identity_map_reordering = identity_map_reordering
#         self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
#                                         attention_module=attention_module,
#                                         attention_module_kwargs=attention_module_kwargs)
#         self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

#     def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
#         att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
#         ff = self.pwff(att)
#         return ff


# class MultiLevelEncoder(nn.Module):
#     def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
#                  identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
#         super(MultiLevelEncoder, self).__init__()
#         self.d_model = d_model
#         self.dropout = dropout
#         self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
#                                                   identity_map_reordering=identity_map_reordering,
#                                                   attention_module=attention_module,
#                                                   attention_module_kwargs=attention_module_kwargs)
#                                      for _ in range(N)])
#         self.padding_idx = padding_idx

#     def forward(self, input, attention_weights=None):
#         # input (b_s, seq_len, d_in)
#         attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

#         outs = []
#         out = input
#         for l in self.layers:
#             out = l(out, out, out, attention_mask, attention_weights)
#             outs.append(out.unsqueeze(1))

#         outs = torch.cat(outs, 1)
#         return outs, attention_mask


# class MemoryAugmentedEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx, d_in=2048, **kwargs):
#         super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
#         self.fc = nn.Linear(d_in, self.d_model)
#         self.dropout = nn.Dropout(p=self.dropout)
#         self.layer_norm = nn.LayerNorm(self.d_model)

#     def forward(self, input, attention_weights=None):
#         out = F.relu(self.fc(input))
#         out = self.dropout(out)
#         out = self.layer_norm(out)
#         return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)



# RSTNet
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):
        # q, k = (queries + pos, keys + pos) if pos is not None else (queries, keys)
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


# RSTNet
class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])


    def forward(self, input, attention_weights=None, pos=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # grid geometry embedding
        # follow implementation of https://github.com/yahoo/object_relation_transformer/blob/ec4a29904035e4b3030a9447d14c323b4f321191/models/RelationTransformerModel.py
        relative_geometry_embeddings = BoxRelationalEmbedding(input)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])    # [b_s, r, r]
        box_size_per_head.insert(1, 1)                                      # [1, b_s, r, r]
        relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)   # λ_ij

        # m2: meshed architecture
        outs = []
        out = input
        for l in self.layers:
            # out = l(out, out, out, attention_mask, attention_weights)    # m2
            out = l(out, out, out, relative_geometry_weights, attention_mask, attention_weights, pos=pos)   # RSTNet
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask
        
        ## non-meshed architecture
        # out = input
        # for layer in self.layers:
        #     out = layer(out, out, out, relative_geometry_weights, attention_mask, attention_weights, pos=pos)
        #
        # return out, attention_mask

# RSTNet
class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None, pos=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights, pos=pos)