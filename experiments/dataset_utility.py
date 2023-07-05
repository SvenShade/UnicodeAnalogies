# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    01/11/22
# Purpose: Defines the dataset class to load Unicode Analogies.


# IMPORTS ----------------------------------------------------------------------------------------------------------- #


import os
import glob
import numpy as np
import sys
import torch
import random
from   torch.utils.data import Dataset


# SCRIPT ------------------------------------------------------------------------------------------------------------ #


class dataset(Dataset):
    def __init__(self, path, mode, percent, f_names=None, pad=False):
        self.root_dir = path
        self.mode     = mode
        self.percent  = percent if mode!="test" else 100
        self.pad      = pad
        file_names = [f for f in os.listdir(self.root_dir) if mode in f] if not f_names else f_names
        random.shuffle(file_names)
        self.file_names = file_names[:int(len(file_names)*self.percent/100)]
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data   = np.load(self.root_dir+'/'+self.file_names[idx])
        images = data["images"]
        target = data["target"]
        
        #Shuffle choices.
        if self.mode=="train":
            context = images[:5]
            choices = images[5:]
            indices = list(range(4))
            np.random.shuffle(indices)
            target  = indices.index(target)
            choices = choices[indices]
            #If pad, insert an empty second row between the two context rows, plus 4 extra answers (7 empty frames in total).
            if self.pad:
                row_3   = np.zeros((3, 80, 80))
                extra_a = np.zeros((4, 80, 80))
                images  = np.concatenate((context[:3], row_3, context[3:], choices, extra_a))
            else:
                images = np.concatenate((context, choices))
                
        else:
            if self.pad:
                row_3   = np.zeros((3, 80, 80))
                extra_a = np.zeros((4, 80, 80))
                images  = np.concatenate((images[:3], row_3, images[3:], extra_a))
            
        #Return tensors.
        return torch.tensor(images, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


# END SCRIPT -------------------------------------------------------------------------------------------------------- #