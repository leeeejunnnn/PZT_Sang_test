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

from data_pipeline import data_pipeline_11d
from model import CNN_11dv

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
train_num = 1
num_epoch = 500
BATCH_SIZE = 500
model = CNN_11dv()
print(model)
val_ratio = 0.3
Learning_rate = 0.0001
L2_decay = 1e-8
LRSTEP = 5
GAMMA = 0.1
#%%
dataset = data_pipeline_11d(Data_dir)
print(len(dataset))

#%%
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (100, 40, 40) )
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0)

#%%
model = model.to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
#scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=LRSTEP, gamma=GAMMA)

#%%
ckpt_dir = './Checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = '%s%s%d.pt' % (ckpt_dir, '/Checkpoint_exp_11d', train_num)
print(ckpt_path)

#%%
summary = SummaryWriter()

loss_array = []
train_losses = []
validation_losses = []
best_validation_acc = 0
for epoch in range(num_epoch):
    one_ep_start = time.time()
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
    
    #np.save('loss.npy',loss_array)    
    summary.add_scalar('loss/train_loss', loss.item(), epoch)
    train_losses.append(loss.item())
    #scheduler.step()

    #valdiation
    model.eval()
    n = 0
    validation_loss = 0.
    validation_acc = 0.

    for x_val, target_val in val_loader:
        x_val = x_val.to(device, dtype=torch.float)
        target_val = np.argmax(target_val,axis=1)
        target_val = target_val.to(device, dtype=torch.long)

        pred_val = model(x_val)
        validation_loss += F.cross_entropy(pred_val, target_val).item()
        validation_acc += (pred_val.argmax(dim=1) == target_val).float().sum().item()
        n += x_val.size(0)

    validation_loss /= n
    validation_acc /= n
    if epoch % 10 ==0 :
        print('Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(validation_loss, validation_acc))
    summary.add_scalar('loss/validation_loss',validation_loss, epoch)
    validation_losses.append(validation_loss)

    if validation_acc > best_validation_acc:
        best_validation_acc = validation_acc
        ckpt = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_validation_acc': best_validation_acc}
        torch.save(ckpt,ckpt_path)
        print('Higher validation accuracy, Checkpoint Saved!')

    if epoch % 50 == 0:    
        curr_time = time.time()
        print("one epoch time = %.2f" %(curr_time-one_ep_start))
        print('########################################################')

#%%
plt.plot(loss_array, label='train loss')
plt.legend()
plt.savefig('test_result_loss_11d'+str(train_num)+'.png')
plt.show()


#%% test check point
test_ckpt_path = '%s%s%d.pt' % (ckpt_dir, '/Checkpoint_exp_11d', train_num)
try:
    test_ckpt = torch.load(test_ckpt_path)
    model.load_state_dict(test_ckpt['model'])
    optimizer.load_state_dict(test_ckpt['optimizer'])
    best_validation_acc = test_ckpt['best_validation_acc']
    print('Checkpoint load! Current best validation accuracy is {:.4f}'.format(best_validation_acc))
except:
    print('There is no checkpoint or network has different architecture.')



#%% test
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
    plt.title(str('11d') + ' Confusion Matrix (%)\n' + 'Test accuracy = {:.2f}%'.format(test_acc*100))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    #plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.savefig('test_resul_11d'+str(train_num)+'.png')
    plt.show()

plot_confusion(cfm,['intact', 'damaged'],'percent')




# %%
