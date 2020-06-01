import logging
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from fairseq.modules import SinusoidalPositionalEmbedding

logger = logging.getLogger(__name__)


class CrossmodalTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        emb_dropout,
        attn_dropout,
        res_dropout,
        relu_dropout,
        n_layer,
        attn_mask,
        scale_embedding=True,
    ):
        super(CrossmodalTransformer, self).__init__()
        self.attn_mask = attn_mask
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0
        # 0: padding
        self.pos = SinusoidalPositionalEmbedding(d_model, 0, init_size=128)
        self.emb_dropout = emb_dropout
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(
                d_model, nhead, d_model * 4, attn_dropout, res_dropout, relu_dropout
            )
            self.layers.append(new_layer)

    def forward(self, x_query, x_key=None, x_key_padding_mask = None):
        # Positional Encoder for Inputs -> (B, S) => (B, S, D)
        x_query_pos = self.pos(x_query[:, :, 0])
        
        # (B, S, D) => (S, B, D)
        x_query = F.dropout(
            (self.emb_scale * x_query + x_query_pos), self.emb_dropout, self.training
        ).transpose(0, 1)
        
        if x_key is not None:
            # in the same way
            x_key_pos = self.pos(x_key[:, :, 0])
            x_key = F.dropout(
                (self.emb_scale * x_key + x_key_pos), self.emb_dropout, self.training
            ).transpose(0, 1)
        for layer in self.layers:
            x_query = layer(x_query, x_key, attn_mask=self.attn_mask)
        return x_query


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout, res_dropout, relu_dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            attn_dropout: the dropout value for multihead attention (required).
            res_dropout: the dropout value for residual connection (required).
            relu_dropout: the dropout value for relu (required).
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, nhead, attn_dropout, res_dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, res_dropout, relu_dropout)
    def forward(self, x_query, x_key=None, x_key_padding_mask=None, attn_mask=None):
        """
        x : input of the encoder layer -> (L, B, d)
        """
        if x_key is not None:
            x = self.transformer(
                x_query, x_key, x_key, key_padding_mask=x_key_padding_mask, attn_mask=attn_mask
            )
        else:
            x = self.transformer(
                x_query, x_query, x_query, key_padding_mask=x_key_padding_mask, attn_mask=attn_mask,
            )
        x = self.feedforward(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout, res_dropout):
        super(TransformerBlock, self).__init__()
        self.res_dropout = res_dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=True):
        mask = self.get_future_mask(query, key) if attn_mask else None
        # Do layernorm befor self-attention
        query, key, value = [self.layernorm(x) for x in (query, key, value)]
        x = self.self_attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=mask)[0]
        x = query + F.dropout(x, self.res_dropout, self.training)
        return x
    
    # target(query) / source(key) 
    def get_future_mask(self, q, k=None):
        dim_query = q.shape[0]
        dim_key = dim_query if k is None else k.shape[0]

        # source mask
        '''
        tensor([[0., -inf, -inf],
               [0., 0., -inf],
               [0., 0., 0.]])

        '''
        future_mask = torch.triu(torch.ones(dim_query, dim_key, device = q.device), diagonal=1).float()
        future_mask = future_mask.masked_fill(future_mask == float(1), float('-inf'))
        return future_mask

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, res_dropout, relu_dropout):
        super(FeedForwardBlock, self).__init__()
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Do layernorm befor feed-forward network
        x = self.layernorm(x)
        x2 = self.linear2(F.dropout(F.relu(self.linear1(x)), self.relu_dropout, self.training))
        x = x + F.dropout(x2, self.res_dropout, self.training)
        return x



    
    
# def get_future_mask(q, k=None):
#     dim_query = q.shape[0]
#     dim_key = dim_query if k is None else k.shape[0]
#     future_mask = torch.triu(torch.ones(dim_query, dim_key), diagonal=1)
#     future_mask = future_mask.masked_fill(future_mask == 1.0, float("-inf")).masked_fill(
#         future_mask == 0.0, 1.0
#     )
#     return future_mask
