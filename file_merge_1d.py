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
import numpy as np
import pandas as pd
from natsort import natsorted

# %%
data_dir = './dataset'

intact_list = sorted(glob(data_dir+'/Intact/*/*')) 
damaged_list = sorted(glob(data_dir + '/Damaged/*/*'))
print(len(intact_list), len(damaged_list))
# %%
for i in intact_list:
    dataset = glob(i)
    print(dataset)
    read_data = pd.read_csv(i,index_col=0, sep='\t',header=None, encoding='utf-8')
    data = np.array(read_data.values.tolist())
    res_data = np.reshape(data, (1, -1))
    print(res_data.shape)
    np.save(data_dir+'/1d/'+i.split('/')[-1]+'.npy', res_data)
print('done')
# %%
for i in damaged_list:
    dataset = glob(i)
    print(dataset)
    read_data = pd.read_csv(i,index_col=0, sep='\t',header=None, encoding='utf-8')
    data = np.array(read_data.values.tolist())
    res_data = np.reshape(data, (1, -1))
    print(res_data.shape)
    np.save(data_dir+'/1d/'+i.split('/')[-1]+'.npy', res_data)
print('done')
# %%
