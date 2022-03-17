# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:06:28 2021

@author: user
"""

import torch
from torch.utils.data import  Dataset

from PIL import Image
import numpy as np


class customDataset(Dataset):
    
    def __init__(self, data_array, label_array, transform=None):
        
        self.data_array = data_array
        self.label_array = label_array
        #self.label_array = torch.from_numpy(self.label_array).view(len(self.data_array)).long()
        print("label_array: ", type(self.label_array))
        self.transform = transform
        
    def __len__(self):
        
        return len(self.data_array)
        
    def __getitem__(self, idx):
        
        sample = self.data_array[idx]
        label = self.label_array[idx]
        
        #sample = Image.fromarray(np.squeeze(sample), mode = 'L')
        sample = Image.fromarray(np.squeeze(sample))
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label
        
