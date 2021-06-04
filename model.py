#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

#%%
class CNNv1(nn.Module):
    def __init__(self):
        super(CNNv1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 10 , (3,1000), stride=(1,50)), #output = (10,30,1980)
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 25, (2, 100), stride=(1,2)), #output = (25, 28, 940)
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 50, (2, 40), stride=(1,2)), #output = (50, 26, 450)
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 100, (2, 30), stride=(1,2)), #output = (100, 26, 210)
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, (2, 30), stride=(1,2)), #output = (100, 24, 90)
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, (2, 30), stride=(1,1)), #output = (100, 22, 60)
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, (2, 30), stride=(1,1)), #output = (100, 25, 33)
            nn.BatchNorm2d(100),
            nn.ReLU()
        )
        self.fc = nn.Sequential( 
            nn.Linear(82500, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(5000,500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500,50),
            nn.BatchNorm1d(50),
            nn.Linear(50, 2)
        )

    def forward(self,x):
        x = x.view(-1, 1, 33, 100000)
        x = self.layer(x)
        x = x.view(-1,82500)
        out = self.fc(x)
        return out 



#

# %%

class CNN_1dv(nn.Module):
    def __init__(self):
        super(CNN_1dv, self).__init__()
        self.layer_1d = nn.Sequential(
            nn.Linear(100000,1000),
#            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(1000, 100),
#            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(100, 10),
#            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    
    def forward(self,x):
        x = x.view(-1, 100000)
        out_1d = self.layer_1d(x)
        return out_1d
# %%
class CNN_3dv(nn.Module):
    def __init__(self):
        super(CNN_3dv, self).__init__()
        self.layer_3d = nn.Sequential(
            nn.Conv2d(1, 10, (3, 1000), stride=(1,50), padding=(1,0)), #output = (10, 3, 1980)
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 25, (3, 100), stride=(1,2), padding=(1,0)), #output = (25, 3, 940)
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 50, (3, 40), stride=(1,2), padding=(1,0)), #output = (50, 3, 450)
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 100, (3, 30), stride=(1,2), padding=(1,0)), #output = (100, 3, 210)
            nn.BatchNorm2d(100),
            nn.ReLU()         
        )

        self.layer_fc = nn.Sequential(
            nn.Linear(63300,1000),
#            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(1000, 100),
#            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(100, 10),
#            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    
    def forward(self,x):
        x = x.view(-1, 1, 3, 100000)
        x = self.layer_3d(x)
        x = x.view(-1, 63300)
        out_3d = self.layer_fc(x)
        return out_3d


#%%
# %%
class CNN_11dv(nn.Module):
    def __init__(self):
        super(CNN_11dv, self).__init__()
        self.layer_11d = nn.Sequential(
            nn.Conv2d(1, 10, (3, 1000), stride=(1,50), padding=(1,0)), #output = (10, 11, 1981)
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 25, (3, 100), stride=(1,2), padding=(1,0)), #output = (25, 11, 942)
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 50, (5, 50), stride=(1,2), padding=(1,0)), #output = (50, 9, 496)
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 100, (5, 50), stride=(1,2), padding=(1,0)), #output = (100, 7, 224)
            nn.BatchNorm2d(100),
            nn.ReLU(),         
            nn.Conv2d(100, 100, (5, 50), stride=(1,2), padding=(1,0)), #output = (100, 5, 88)
            nn.BatchNorm2d(100),
            nn.ReLU()    
        )

        self.layer_fc = nn.Sequential(
            nn.Linear(37500,1000),
#            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(1000, 100),
#            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(100, 10),
#            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    
    def forward(self,x):
        x = x.view(-1, 1, 11, 100000)
        x = self.layer_11d(x)
        x = x.view(-1, 37500)
        out_11d = self.layer_fc(x)
        return out_11d


