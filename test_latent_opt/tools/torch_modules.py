# test_latent_opt/tools/torch_modules.py
'''
* Author: Evan Komp
* Created: 6/25/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Pytorch modules to act as differentiable functions for optimization.

Embeddings are of shape (batch_size, L, D) where L is the sequence length and D is the embedding dimension.

Sequences are variable length, so we must mean pool at some point. Note that to construct batches, we will
pad embeddings with zeros and pass a mask for which embeddings to actually operate on.

Option one: 
 - mean pool over all sequence elements to get a single embedding for the sequence
 - linear layer to get a single output value
 - relu because activity is always positive

Option two:
 - nonlinear learnable projection of each embedding
 - mean pool over all sequence elements to get a single embedding for the sequence
 - linear layer to get a single output value

'''

import torch
import torch.nn as nn

class PoolLinear(nn.Module):

    def __init__(self, input_dim, pool_type='max', dropout=0.0):
        super(PoolLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.linear = nn.Linear(input_dim, 1)
        self.pool_type = pool_type
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask):

        # dropout the embeddings along the D dimension
        embeddings = self.dropout(embeddings)

        # embeddings of shape (batch_size, L, D)
        # for some (i, j) the D vector may be zero and needs to be masked for pooling
        # mask of shape (batch_size, L)
        if self.pool_type == 'mean':
            nonzeros = torch.sum(mask, dim=1).unsqueeze(1)
            # this is of shape (batch_size, 1)
            pooled = torch.sum(embeddings, dim=1) / nonzeros
            # this is of shape (batch_size, D)
        elif self.pool_type == 'max':
            # this is of shape (batch_size, D)
            pooled, _ = torch.max(embeddings, dim=1)
        else:
             raise ValueError(f'Unknown pool type {self.pool_type}')
        
        # linear layer
        projection = self.linear(pooled)
        # this is of shape (batch_size, 1)
        output = torch.relu(projection)
        return output
                
class NonlinearPoolLinear(nn.Module):

    def __init__(self, input_dim, hidden_dim, pool_type='max', dropout=0.0):
        super(NonlinearPoolLinear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        self.pool_type = pool_type
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask):
            
            # embeddings of shape (batch_size, L, D)
            # for some (i, j) the D vector may be zero and needs to be masked for pooling
            # mask of shape (batch_size, L)
            nonzeros = torch.sum(mask, dim=1).unsqueeze(1)
    
            # nonlinear projection
            projected = torch.nn.functional.gelu(self.projection(embeddings))
            # this is of shape (batch_size, L, hidden_dim)
    
            if self.pool_type == 'mean':
                # this is of shape (batch_size, hidden_dim)
                pooled = torch.sum(projected, dim=1) / nonzeros
            elif self.pool_type == 'max':
                # this is of shape (batch_size, hidden_dim)
                pooled, _ = torch.max(projected, dim=1)
            else:
                raise ValueError(f'Unknown pool type {self.pool_type}')
            
            # linear layer
            projection = self.linear(pooled)
            # this is of shape (batch_size, 1)
            output = torch.relu(projection)
            return output
    

class AttnPoolLinear(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.proj_qkv = nn.Linear(input_dim, 3 * hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings, mask):
        # embeddings of shape (batch_size, L, D)
        # for some (i, j) the D vector may be zero and needs to be masked for pooling
        # mask of shape (batch_size, L)
        
        # project to Q, K, V
        qkv = self.proj_qkv(embeddings)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)
        
        # mask out the padding
        mask = mask.unsqueeze(1).unsqueeze(1)
        
        # attention
        attn_output, _ = self.attn(q, k, v, key_padding_mask=mask)

        # pool over all sequence elements
        nonzeros = torch.sum(mask.squeeze(1), dim=1).unsqueeze(1)
        # this is of shape (batch_size, hidden_dim)
        pooled = torch.sum(attn_output, dim=1) / nonzeros
        
        # linear layer
        projection = self.linear(pooled)
        # this is of shape (batch_size, 1)
        output = torch.relu(projection)
        return output
        