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

class MeanPoolLinear(nn.Module):

    def __init__(self, input_dim):
        super(MeanPoolLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, embeddings, mask):

        # embeddings of shape (batch_size, L, D)
        # for some (i, j) the D vector may be zero and needs to be masked for pooling
        # mask of shape (batch_size, L)
        nonzeros = torch.sum(mask, dim=1).unsqueeze(1)

        # mean pool over the sequence length
        pooled = torch.sum(embeddings, dim=1) / nonzeros
        # this is of shape (batch_size, D)

        # linear layer
        projection = self.linear(pooled)
        # this is of shape (batch_size, 1)
        output = torch.relu(projection)
        return output


class NonlinearMeanPoolLinear(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(NonlinearMeanPoolLinear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings, mask):
            
            # embeddings of shape (batch_size, L, D)
            # for some (i, j) the D vector may be zero and needs to be masked for pooling
            # mask of shape (batch_size, L)
            nonzeros = torch.sum(mask, dim=1).unsqueeze(1)
    
            # nonlinear projection
            projected = torch.nn.functional.gelu(self.projection(embeddings))
            # this is of shape (batch_size, L, hidden_dim)
    
            # mean pool over the sequence length
            pooled = torch.sum(projected, dim=1) / nonzeros
            # this is of shape (batch_size, hidden_dim)
    
            # linear layer
            projection = self.linear(pooled)
            # this is of shape (batch_size, 1)
            output = torch.relu(projection)
            return output
