#code adapted from https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def attention(queries, keys, values, mask=None):
    """ Scaled dot product attention """
    #qkv: batch, seq_len; dim
    dim_q = queries.size()[-1]
    qk = torch.matmul(queries, keys.transpose(-2, -1))
    logits = qk/math.sqrt(dim_q)

    if mask is not None:  #if we stack multiple sequences with different lengths into a batch. 
        # To still benefit from parallelization in PyTorch, we pad the sentences to the same length
        # and mask out the padding tokens during the calculation of the attention values. 
        # This is usually done by setting the respective attention logits to a very low value.
        print("logits before", logits)
        print("mask", mask)

        print("logits", logits.size())
        print("mask", mask.size())

        logits = logits.masked_fill(mask == 0, -1e9)
        print("logits", logits)

    scores_att= F.softmax(logits, dim=-1)
    return torch.matmul(scores_att, values)


class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v):
        super().__init__()
        self.linear_queries = nn.Linear(dim_in, dim_qk)
        self.linear_keys = nn.Linear(dim_in, dim_qk)
        self.linear_values = nn.Linear(dim_in, dim_v)
  
    def forward(self, emb_for_queries,emb_for_keys,emb_for_values): 
        # x corresponds to the inputs (seq_len, emb_dim)
        # linear layers map embeddings to query,keys, values
        return attention(self.linear_queries(emb_for_queries), self.linear_keys(emb_for_keys), self.linear_values(emb_for_values))


class MuiltiHead(nn.Module):
    def __init__(self, n_heads, dim_in, dim_qk, dim_v):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_in, dim_qk, dim_v) for _ in range(n_heads)])
        self.linear = nn.Linear(dim_v*n_heads, dim_in)

    def forward(self, emb_for_queries, emb_for_keys, emb_for_values):
        concat=torch.cat([head(emb_for_queries,emb_for_keys,emb_for_values) for head in self.heads], dim=-1)
        return self.linear(concat)


def position_encoding(seq_len, dim_model):
    pos = torch.arange(seq_len, dtype=torch.float).reshape( -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float).reshape( 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input, dim_feedforward):
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input)
    )

class AddAndNorm(nn.Module):
    def __init__(self, sublayer, dimension, dropout):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *x):
        return self.norm(x[-1] + self.sublayer(*x))


class EncoderLayer(nn.Module):

    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        # dim_model: int = 512, 
        # num_heads: int = 8, 
        # dim_feedforward: int = 2048, 
        # dropout: float = 0.1, 
        
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention=AddAndNorm(MuiltiHead(num_heads, dim_model, dim_k, dim_v), dim_model, dropout)
        self.feed_forward=AddAndNorm(feed_forward(dim_model, dim_feedforward), dim_model, dropout)

    def forward(self, src):
        src = self.attention(src,src,src)
        return self.feed_forward(src)

class Encoder(nn.Module):
    #contains n repeted EncoderLayers

    def __init__(self, n_layers, dim_model, num_heads, dim_feedforward, dropout):
        super().__init__()        
        self.layers = nn.ModuleList([EncoderLayer(dim_model, num_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        
    def forward(self, src):
        seq_len= src.size()[0]
        emb_dimension = src.size()[1]
        src += position_encoding(seq_len, emb_dimension)
        for layer in self.layers:
            src = layer(src)

        return src


class DecoderLayer(nn.Module):

    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        # dim_model: int = 512, 
        # num_heads: int = 8, 
        # dim_feedforward: int = 2048, 
        # dropout: float = 0.1, 
        
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.self_attention=AddAndNorm(MuiltiHead(num_heads, dim_model, dim_k, dim_v), dim_model, dropout)
        self.cross_attention=AddAndNorm(MuiltiHead(num_heads, dim_model, dim_k, dim_v), dim_model, dropout)
        self.feed_forward=AddAndNorm(feed_forward(dim_model, dim_feedforward), dim_model, dropout)

    def forward(self, target, memory):
        target = self.self_attention(target,memory,memory)
        target = self.cross_attention(target,memory,memory) #TODO: cross-attention should have auto-regressive mask
        return self.feed_forward(target)

class Decoder(nn.Module):
    #contains n repeted DecoderLayers

    def __init__(self, n_layers, dim_model, num_heads, dim_feedforward, dropout, vocab_size):
        super().__init__()        
        self.layers = nn.ModuleList([DecoderLayer(dim_model, num_heads, dim_feedforward, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(dim_model, vocab_size)

    def forward(self, target, memory):
        seq_len= target.size()[0]
        emb_dimension = target.size()[1]
        target += position_encoding(seq_len, emb_dimension)
        for layer in self.layers:
            #TODO: auto-regressive
            target = layer(target, memory)
            
        return self.linear(tgt)


class Transformer(nn.Module):

    def __init__(self, n_enc_layers=8,n_dec_layers=8, dim_model=512, num_heads=8, dim_feedforward=2048, dropout=0.5, vocab_size=10):
        super().__init__()        
        self.encoder = Encoder(n_enc_layers, dim_model, num_heads, dim_feedforward, dropout)
        self.decoder = Decoder(n_dec_layers, dim_model, num_heads, dim_feedforward, dropout, vocab_size)

    def forward(self, src, target):
        src=self.encoder(src) #outputs of the encoder
        print("\nself.encoder", src.size())
        return self.decoder(target, src)


if __name__ == "__main__":
    seq_len = 3
    d_k = 2
    d_v = 2

    #supposdly you have before 3 embeddings than you transform into q,k,v
    queries = torch.randn(seq_len, d_k)
    keys = torch.randn(seq_len, d_k)
    values = torch.randn(seq_len, d_v)

    print("queries", queries)
    print("keys", keys)
    print("values", values)

    queries[-1,-1] =0
    keys[-1,-1] =0
    values[-1,-1] =0

    print("\nqueries", queries)
    print("keys", keys)
    print("values", values)

    print("\natt", attention(queries, keys, values))

    src = torch.rand(16, 512)
    tgt = torch.rand(16, 512)
    out = Transformer()(src, tgt)
    print("output", out.size())