#%%
import shutil
from glob import glob
from os.path import basename

# %%
src = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/'
tr = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/tr/'
te = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/1d/nl/te/'

#%%\
for na in glob(tr + '*.npy'):
    n = basename(na)
    shutil.move(tr+n, te+n)

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
