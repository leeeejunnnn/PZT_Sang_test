#%%
import shutil
from glob import glob
from os.path import basename

# %%
src = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/'
tr = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/tr/'
te = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/'

l2k = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/2k/'
l4k = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/4k/'
l6k = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/6k/'
l8k = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/8k/'
l10k = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/10k/'



#%%\
for na in glob(tr + '*.npy'):
    n = basename(na)
    shutil.move(tr+n, te+n)
#%%
file_name = glob(te + '*.npy')
print(len(file_name))
# %%
for f_name in file_name:
    filen = basename(f_name)
    if int(f_name.split('_')[-2]) in [1, 7, 13, 19, 25]:
        shutil.copyfile(src+filen,tr+filen)        

    else: shutil.copyfile(src+filen, te+filen)

# %%
for f_name in file_name:
    filen = basename(f_name)
    if int(f_name.split('_')[-2]) in [1, 7, 13, 19, 25]:
        shutil.move(te+filen,tr+filen)        

print(len(glob(tr+'*.npy')))
# %%
for f_name in file_name:
    filen = basename(f_name)
    if int(f_name.split('_')[-2]) in [6, 12, 18, 24, 30]:
        shutil.move(te+filen,tr+filen)        

print(len(glob(tr+'*.npy')))
# %%
# %% loadcondition
for f_name in file_name:
    filen = basename(f_name)
    if int(f_name.split('_')[-2]) in [2, 8, 14, 20, 26]:
        shutil.move(te+filen,l2k+filen)        
    elif int(f_name.split('_')[-2]) in [3, 9, 15, 21, 27]:
        shutil.move(te+filen,l4k+filen)    
    elif int(f_name.split('_')[-2]) in [4, 10, 16, 22, 28]:
        shutil.move(te+filen,l6k+filen)    
    elif int(f_name.split('_')[-2]) in [5, 11, 17, 23, 29]:
        shutil.move(te+filen,l8k+filen)    

print(len(glob(l8k+'*.npy')))
# %%
