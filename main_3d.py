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

from data_pipeline import data_pipeline_1d
from model import CNN_1dv

#%%
# device GPU / CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
#device = torch.device('cpu')
#%%
#  Data parameters 
Data_dir = './dataset/'

# NN training parameters
TENSORBOARD_STATE = True
train_num = 5000
num_epoch = 2048
BATCH_SIZE = 1500
model = CNN_1dv()
print(model)
#val_ratio = 0.3
Learning_rate = 0.0001
L2_decay = 1e-8
LRSTEP = 5
GAMMA = 0.1
#%%
dataset = data_pipeline_1d(Data_dir)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, (1500, 480) )
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, shuffle=False, num_workers=0)

#%%
model = model.to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
#scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=LRSTEP, gamma=GAMMA)

#%%
ckpt_dir = './Checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = '%s%s%d.pt' % (ckpt_dir, '/Checkpoint_exp_3d', train_num)
print(ckpt_path)

#%%
loss_array = []
for epoch in range(num_epoch):
    for x, target in train_loader:
        
        x = x.to(device, dtype=torch.float)
        target = np.argmax(target,axis=1)
        target = target.to(device, dtype=torch.long)
        #target = target.view(BATCH_SIZE,1)
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()


    if epoch % 10 == 0:
        print('epoch:', epoch, ' loss:', loss.item())
        loss_array.append(loss)
    np.save('loss.npy',loss_array)    
    
    #scheduler.step()
    
ckpt = {'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
torch.save(ckpt,ckpt_path)
print('Higher validation accuracy, Checkpoint Saved!')

#%%
plt.plot(loss_array, label='train loss')
plt.legend()
plt.show()


#%%

model.eval()
n = 0.
test_loss = 0.
test_acc = 0.
cfm = np.zeros((2,2))
target_array=[]
test_array=[]

for x_test, target_test in val_loader:
    x_test = x_test.to(device, dtype=torch.float)
    target_test = np.argmax(target_test,axis=1)
    target_test = target_test.to(device, dtype=torch.long)
    target_array.append(target_test.cpu().data.numpy())

    logits_test = model(x_test)
    test_loss += F.cross_entropy(logits_test,target_test).item()
    test_acc += (logits_test.argmax(dim=1) == target_test).float().sum().item()
    test_array.append(logits_test.argmax(dim=1).cpu().data.numpy())
    predicted = logits_test.argmax(dim=1).detach().cpu()
    actual = target_test.detach().cpu()
    for i in range(len(predicted)):
        cfm[predicted[i],actual[i]] += 1
    n += x_test.size(0)

test_loss /= n
test_acc /= n

print('Test accuracy is {:.4f}'.format(test_acc))








#%%
import pandas as pd
import seaborn as sn

def plot_confusion(confusion_matrix,classes,vis_format=None):
    plt.figure(figsize=(6,5))
    if vis_format == 'percent':
        confusion_matrix = confusion_matrix/np.sum(confusion_matrix)*100
        vis_fmt = '.2f'
    else:
        vis_fmt = '.0f'
    df_cm_percent = pd.DataFrame(confusion_matrix, classes, classes)
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm_percent, cmap='Blues',
               annot=True, fmt=vis_fmt, annot_kws={"size":16},
               linewidths=.5, linecolor='k', square=True)
    plt.title(str('3d') + ' Confusion Matrix (%)\n' + 'Test accuracy = {:.2f}%'.format(test_acc*100))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.show()

plot_confusion(cfm,list('12'),'percent')







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
