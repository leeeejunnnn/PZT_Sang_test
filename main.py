#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import os
from glob import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from data_pipeline import data_pipeline

#%%
# device GPU / CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print ('Available devices ', torch.cuda.device_count())
#print ('Current cuda device ', torch.cuda.current_device())
#print(torch.cuda.get_device_name(device))
device = torch.device('cpu')
#%%
#  Data parameters 
Data_dir = './dataset/'

# NN training parameters
TENSORBOARD_STATE = True
num_epoch = 2048
BATCH_SIZE = 40
#val_ratio = 0.3
Learning_rate = 0.001
L2_decay = 1e-8
LRSTEP = 5
GAMMA = 0.1
#%%
dataset = data_pipeline(Data_dir)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, (40, 20) )
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, shuffle=False, num_workers=0)
# %%
model = nn.Sequential(
#    nn.Linear(1200,600),
#    nn.LeakyReLU(0.2),
    nn.Linear(600,200),
    nn.LeakyReLU(0.2),
    nn.Linear(200,100),
    nn.LeakyReLU(0.2),    
    nn.Linear(100,50),
    nn.LeakyReLU(0.2),        
    nn.Linear(50,25),
    nn.LeakyReLU(0.2),          
    nn.Linear(25,1),
)

#%%
model = model.to(device)
loss_func = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=L2_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=LRSTEP, gamma=GAMMA)

#%%
ckpt_dir = './Checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = '%s%s.pt' % (ckpt_dir, '/Checkpoint_exp')

loss_array = []
for epoch in range(num_epoch):
    for x, target in train_loader:
        optimizer.zero_grad()
        x = x.to(device, dtype=torch.float)
        target = target.to(device)
        output = model(x)

        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print('epoch:', epoch, ' loss:', loss.item())
        loss_array.append(loss)
    np.save('/home/sss-linux1/project/leejun/Thermo/loss.npy',loss_array)    
    
    scheduler.step()
    
ckpt = {'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
torch.save(ckpt,ckpt_path)
print('Higher validation accuracy, Checkpoint Saved!')

plt.plot(loss_array, label='train loss')
plt.legend()
plt.show()


#%%
output_array = []
loss_array = []
target_array = []
model.eval()
for x, target in val_loader:
    x = x.to(device, dtype=torch.float)
    target = target.to(device)
    target_array.append(target.cpu().data.numpy())
    output = model(x)

    loss = output-target
    output_array.append(output.cpu().data.numpy())
    loss_array.append(loss.cpu().data.numpy())


#%%
output_array = np.vstack(output_array)
loss_array = np.vstack(loss_array)
target_array = np.vstack(target_array)
plt.scatter(output_array, target_array, s=1, c="gray")
#plt.plot(output,output, c="red")
plt.show()
plt.plot(loss_array)
plt.show()
plt.plot(sorted(loss_array))
plt.show()
print(np.mean(abs(loss_array)))
# %%
