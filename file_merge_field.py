#%%
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

intact_list = sorted(glob(data_dir+'/Frequency domain/Intact/*')) 
damaged_list = sorted(glob(data_dir + '/Frequency domain/Damaged/*'))
print(len(intact_list), len(damaged_list))
# %%
for i in intact_list:
    dataset = glob(i)
    print(dataset)
    dataset_list = natsorted(glob(dataset[0] + '/*'))
    #print(dataset_list)
    file_name = dataset
    new_dataset = []
    for j in dataset_list:
        read_data = pd.read_csv(j, index_col=0, sep='\t',header=None, encoding='utf-8')
        new_dataset.append(read_data.values.tolist())
    data=np.vstack(new_dataset)
    res_data=np.reshape(data,(33,-1))
    print(res_data.shape)
    np.save(data_dir+'/Frequency domain/33d/'+dataset[0].split('/')[-1]+'.npy', res_data)

print('done')
# %%
for i in damaged_list:
    dataset = glob(i)
    print(dataset)
    dataset_list = natsorted(glob(dataset[0] + '/*'))
    #print(dataset_list)
    file_name = dataset
    new_dataset = []
    for j in dataset_list:
        read_data = pd.read_csv(j, index_col=0, sep='\t',header=None, encoding='utf-8')
        new_dataset.append(read_data.values.tolist())
    data=np.vstack(new_dataset)
    res_data=np.reshape(data,(33,-1))
    print(res_data.shape)
    np.save(data_dir+'/Frequency domain/33d/'+dataset[0].split('/')[-1]+'.npy', res_data)

print('done')