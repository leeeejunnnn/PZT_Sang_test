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
        self.target = np.array([0, 1])
        self.data_list = sorted(glob(self.data_dir + '/33d/*.npy'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        if self.data_list[idx].find("Int") > 0:
            target = np.array([1,0])
        else: 
            target = np.array([0,1])
        return data, target

# %%
class data_pipeline_1d(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = sorted(glob(self.data_dir + '/1d/*.npy'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        datared = data[:,0:5000]
        if self.data_list[idx].find("int") > 0:
            target = np.array([1, 0])
        else: 
            target = np.array([0, 1])
        return datared, target

# %%
class data_pipeline_3d(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = sorted(glob(self.data_dir + '/3d/*.npy'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
#        data = data[:,0:5000]
        if self.data_list[idx].find("Int") > 0:
            target = np.array([1, 0])
        else: 
            target = np.array([0, 1])
        return data, target


#%%
# %%
class data_pipeline_11d(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = sorted(glob(self.data_dir + '/11d/*.npy'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
#        data = data[:,0:5000]
        if self.data_list[idx].find("Int") > 0:
            target = np.array([1, 0])
        else: 
            target = np.array([0, 1])
        return data, target

#%%
class data_pipeline_1d_nl(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = sorted(glob(self.data_dir + '/1d/nl/tr/*.npy'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        datared = data#[:,0:5000]
        if self.data_list[idx].find("int") > 0:
            target = np.array([1, 0])
        else: 
            target = np.array([0, 1])
        return datared, target

#%%
class data_pipeline_1d_nlte(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = sorted(glob(self.data_dir + '/1d/nl/te/*.npy'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        datared = data#[:,0:5000]
        if self.data_list[idx].find("int") > 0:
            target = np.array([1, 0])
        else: 
            target = np.array([0, 1])
        return datared, target