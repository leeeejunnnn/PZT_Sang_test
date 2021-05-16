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

intact_list = sorted(glob(data_dir+'/Intact/*')) 
damaged_list = sorted(glob(data_dir + '/Damaged/*'))
print(len(intact_list), len(damaged_list))
# %%
for i in intact_list:
    file_list = natsorted(glob(i+'/*'))
    for j in range(3):
        new_dataset = []
        for k in range(11):
            read_data = pd.read_csv(file_list[j*11+k], index_col=0, sep='\t',header=None, encoding='utf-8')
            #print(file_list[j*11+k])
            new_dataset.append(read_data.values.tolist())
        data=np.vstack(new_dataset)
        res_data=np.reshape(data,(11,-1))        
        #print(res_data.shape)
        x = 182 +j
        np.save(data_dir+'/11d/'+i.split('/')[-1]+'_'+str(x)+'.npy', res_data)
        
print('Done!')
# %%
for i in damaged_list:
    file_list = natsorted(glob(i+'/*'))
    for j in range(3):
        new_dataset = []
        for k in range(11):
            read_data = pd.read_csv(file_list[j*11+k], index_col=0, sep='\t',header=None, encoding='utf-8')
            #print(file_list[j*11+k])
            new_dataset.append(read_data.values.tolist())
        data=np.vstack(new_dataset)
        res_data=np.reshape(data,(11,-1))        
        #print(res_data.shape)
        x = 182+j
        np.save(data_dir+'/11d/'+i.split('/')[-1]+'_'+str(x)+'.npy', res_data)
        

# %%
