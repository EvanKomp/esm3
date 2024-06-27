# test_latent_opt/tools/data_loader.py
'''
* Author: Evan Komp
* Created: 6/27/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Custom data loader.

This operates over splits on a HD data dict.

Also, we have to pad batches and return a mask since the embeddings are not all the same length.
'''

import torch
from torch.utils.data import DataLoader, Dataset

class LatentOptData(Dataset):

    def __init__(self, dataset_dict, include_splits):
        self.dataset_dict = dataset_dict
        self.include_splits = include_splits

    def __post_init__(self):
        # first compute length
        self._length = 0
        for split in self.include_splits:
            self._length += len(self.dataset_dict[split])
        
        # now get and index mapping
        self._index_mapping = {}
        idx = 0
        for split in self.include_splits:
            for i in range(len(self.dataset_dict[split])):
                self._index_mapping[idx] = (split, i)
                idx += 1
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.dataset_dict[self._index_mapping[idx][0]][self._index_mapping[idx][1]]
    
class LatentOptDataLoader(DataLoader):
    
    def __init__(self, dataset, batch_size, shuffle):
        super(LatentOptDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def collate_fn(self, batch):
        # X is a list of tensor
        # each tensor of shape 1, L, D
        # L may be different between tensors
        # we need to pad to the max L
        # and create a mask for positions on which to operate
        X = batch['z']
        y = torch.tensor(batch['y']).unsqueeze(1)

        # get max L
        max_L = max([x.shape[1] for x in X])
        # pad X
        X_padded = torch.zeros((len(X), max_L, X[0].shape[2]))
        mask = torch.zeros((len(X), max_L))
        for i, x in enumerate(X):
            L = x.shape[1]
            X_padded[i, :L, :] = x
            mask[i, :L] = 1

        return {'z': X_padded, 'mask': mask, 'y': y}
    
    
