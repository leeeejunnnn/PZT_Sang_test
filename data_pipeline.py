#%%
import os
from glob import glob

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# %%
data_dir = './dataset/'

#%%
class data_pipeline(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.num_data = []
        self.num_target = []
        self.data = []
        self.target = []
        self.data_list = sorted(glob(self.data_dir + '*.npy'))

    def __len__(self):
        return (self.data_list.shape[0])

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        if data_list[idx].find("Int")==0:
            target = [0]
        else: 
            target = [1]
        return data, target
