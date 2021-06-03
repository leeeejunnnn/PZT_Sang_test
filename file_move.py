#%%
import shutil
from glob import glob
from os.path import basename

# %%
src = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/33d/'
tr = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/33d/nl/tr/'
te = '/home/sss-linux1/project/leejun/PZT_Sang_test/dataset/33d/nl/te/'

#%%
file_name = glob(src + '*.npy')
print(len(file_name))
# %%
for f_name in file_name:
    filen = basename(f_name)
    if int(f_name.split('_')[-2]) in [1, 7, 13, 19, 25]:
        shutil.copyfile(src+filen,tr+filen)        

    else: shutil.copyfile(src+filen, te+filen)

# %%
