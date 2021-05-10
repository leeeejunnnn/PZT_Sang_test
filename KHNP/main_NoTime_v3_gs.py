#%%
import os
import time

import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import pandas as pd
import seaborn as sn

from data_pipeline import KHNPDataset_NoTime
from neuralnet_structure_cnnv2 import NNv1, CNNv1, CNNv2


#%%
# Define parameters
# Global parameters
train_experiment_num = 6
test_experiment_num = 6
NN = CNNv2
model_name = 'CNN'
train_state = True
device = ('cuda' if torch.cuda.is_available() else 'cpu')
if device=='cuda': print('Using GPU, %s' % torch.cuda.get_device_name(0))

# Data parameters
DATA_DIR_train = 'F:/Dropbox/1. Lab_KAIST/6. Project/한수원/P_T_data/data_py/notime/train'
DATA_DIR_valid = 'F:/Dropbox/1. Lab_KAIST/6. Project/한수원/P_T_data/data_py/notime/valid'
DATA_DIR_test = 'F:/Dropbox/1. Lab_KAIST/6. Project/한수원/P_T_data/data_py/notime/test'

# NN training parameters
TENSORBOARD_STATE = True
CHECKPT_DIR = './Checkpoints'
BATCH_SIZE = 16
MAX_EPOCH = 20
LEARNING_RATE = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LRSTEP = 5
GAMMA = 0.1
L2_DECAY = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
result_val_2=[]

# %%
# Construct data pipeline
KHNPdata_train = KHNPDataset_NoTime(DATA_DIR_train)
dataloader_train = DataLoader(KHNPdata_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

KHNPdata_valid = KHNPDataset_NoTime(DATA_DIR_valid)
dataloader_valid = DataLoader(KHNPdata_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

KHNPdata_test = KHNPDataset_NoTime(DATA_DIR_test)
dataloader_test = DataLoader(KHNPdata_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

classes = KHNPdata_train.labels


#%%
#   net = NN().to(device)
# Initialize optimizer and network
for i in range(5):
    for j in range(5):
        neuralnet = NN().to(device)
        optimizer = optim.Adam(neuralnet.parameters(), lr=LEARNING_RATE[i], weight_decay=L2_DECAY[j])
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=LRSTEP, gamma=GAMMA)
        result_val_2.append([LEARNING_RATE[i],L2_DECAY[j]])
        print('Learningrate:{:.0e} / L2_DECAY: {:.0e}'.format(LEARNING_RATE[i], L2_DECAY[j]))


        #%%
        # Checkpoints save path
        ckpt_dir = CHECKPT_DIR
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = '%s%s%d.pt' % (ckpt_dir, '/Checkpoint_exp_', train_experiment_num)


        #%%
        # Train
        if train_state:
            # To use tensorboard
            if TENSORBOARD_STATE:
                summary = SummaryWriter()
            
            it = 0
            train_losses = []
            validation_losses = []
            best_validation_acc = 0
            
            for epoch in range(MAX_EPOCH):      
                # Train
                neuralnet.train()
                one_epoch_start = time.time()
                print('# Epoch {}, Learning Rate: {:.0e}'.format(epoch,scheduler.get_lr()[0]))        
                for x, target in dataloader_train:
                    it += 1

                    # Inputs to device
                    x = x.to(device, dtype=torch.float)
                    target = target.to(device)

                    # Feed data into the network and get outputs
                    logits = neuralnet(x)

                    # Compute loss
                    loss = F.cross_entropy(logits,target)
                    
                    # Flush gradients
                    optimizer.zero_grad()

                    # Back propagtion
                    loss.backward()

                    # Update optimizer
                    optimizer.step()

                    if it == 1:
                        one_iter_elapsed = time.time()-one_epoch_start
                        print('Iter:{} / Train loss: {:.4f}'.format(it, loss.item()))
                    if it % 500 == 0:
                        print('Iter:{} / Train loss: {:.4f}'.format(it, loss.item()))
                        if TENSORBOARD_STATE:
                            summary.add_scalar('loss/train_loss', loss.item(), it)
                train_losses.append(loss.item())
                
                # Update learning rate
                scheduler.step()

                # Validation
                neuralnet.eval()
                n = 0.
                validation_loss = 0.
                validation_acc = 0.

                for x_val, target_val in dataloader_valid:
                    x_val = x_val.to(device, dtype=torch.float)
                    target_val = target_val.to(device)

                    logits_val = neuralnet(x_val)
                    validation_loss += F.cross_entropy(logits_val,target_val).item()
                    validation_acc += (logits_val.argmax(dim=1) == target_val).float().sum().item()
                    n += x_val.size(0)

                validation_loss /= n
                validation_acc /= n
                print('Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(validation_loss, validation_acc))
                result_val_2.append([validation_loss,validation_acc])
                if TENSORBOARD_STATE:
                    summary.add_scalar('loss/validation_loss',validation_loss, it)
                validation_losses.append(validation_loss)

                if validation_acc > best_validation_acc:
                    best_validation_acc = validation_acc
                    ckpt = {'neuralnet': neuralnet.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_validation_acc':best_validation_acc}
                    torch.save(ckpt,ckpt_path)
                    print('Higher validation accuracy, Checkpoint Saved!')
                
                curr_time = time.time()
                print("one epoch time = %.2f" %(curr_time-one_epoch_start))
                print('########################################################')
            
            if TENSORBOARD_STATE:
                summary.close()
            
#            plt.plot(train_losses, label='train loss')
#            plt.plot(validation_losses, label='validation loss')
#            plt.legend()
            
        else:
            print('It is not training state. Go to test section!')

'''#save file 
save = np.array(result_val_2)
np.savetxt("result_val_2.csv", save, delimiter=",")
'''
#%%
with open("file.txt", 'w') as output:
    for row in values:
        output.write(str(row) + '\n')

#%%
# Test
test_ckpt_path = '%s%s%d.pt' % (ckpt_dir, '/Checkpoint_exp_', test_experiment_num)
try:
    test_ckpt = torch.load(test_ckpt_path)
    neuralnet.load_state_dict(test_ckpt['neuralnet'])
    optimizer.load_state_dict(test_ckpt['optimizer'])
    best_validation_acc = test_ckpt['best_validation_acc']
    print('Checkpoint load! Current best validation accuracy is {:.4f}'.format(best_validation_acc))
except:
    print('There is no checkpoint or network has different architecture.')

neuralnet.eval()
n = 0.
test_loss = 0.
test_acc = 0.
cfm = np.zeros((4,4))

for x_test, target_test in dataloader_test:
    x_test = x_test.to(device, dtype=torch.float)
    target_test = target_test.to(device)

    logits_test = neuralnet(x_test)
    test_loss += F.cross_entropy(logits_test,target_test).item()
    test_acc += (logits_test.argmax(dim=1) == target_test).float().sum().item()
    predicted = logits_test.argmax(dim=1).detach().cpu()
    actual = target_test.detach().cpu()
    for i in range(len(predicted)):
        cfm[predicted[i],actual[i]] += 1
    n += x_test.size(0)

test_loss /= n
test_acc /= n

print('Test accuracy is {:.4f}'.format(test_acc))

#%%
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
    plt.title(str(model_name) + ' Confusion Matrix (%)\n' + 'Test accuracy = {:.2f}%'.format(test_acc*100))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_confusion(cfm,list(classes),'percent')

#%%
# neuralnet.layer[0].weight
# neuralnet.output_layer.weight
#W = neuralnet.layer[3].weight.detach().cpu()
#
#fig, axs = plt.subplots(10)
#for i in range(10):
#    axs[i].imshow(np.expand_dims(W[i,:],axis=0))



# %%
