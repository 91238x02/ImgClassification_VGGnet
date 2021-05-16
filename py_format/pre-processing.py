#!/usr/bin/env python
# coding: utf-8

# # Module import

# In[21]:


## Import module 
import Func_1  #Custom Functions 
import os
import re
import cv2
import csv
import numpy as np
import random
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[22]:


get_ipython().run_line_magic('pwd', '')


# # Resize & Align by 1..2..3..

# In[23]:


## Create directory
get_ipython().system('mkdir d:\\\\sample')
get_ipython().system('mkdir d:\\\\sample\\\\raw')
get_ipython().system('mkdir d:\\\\sample\\\\Train_Dataset')


# In[ ]:


"""Input Imgs into 'd:/sample/raw' """


# In[4]:


## Resize & create file by order:asc
"""Warning!!
Your row dataset must be example(1).jpg, example(2).jpg, example(3).jpg ... example(3000).jpg
If your number contained in the filename is overlapped, it will be overwritten.
Meanwhile, raw dataset not must be 1.jpg, 2.jpg, 3.jpg... 3000.jpg."""

path_etc = 'd:\\sample\\raw'
path_re = 'd:\\sample\\Train_Dataset'

Func_1.resize(path_etc, path_re, height=128, width=128)  #filename must me English  #default = 128*128


# # Create Label (csv file)

# In[25]:


## Make csv file in path_re
"""k1, k2, k3, k4, k5 is class.
And each number is number of the class.
So, you must match each class and the number."""

path_re = 'd:\\sample\\label.csv'
Func_1.csv_maker_5(path_re, k1=1800, k2=1800, k3=1800, k4=1800, k5=1800)


#  
#  

#  
#  

#  
#  

#  
#  

# # Generating

# ## Create Generated imgs

# In[ ]:


## Create directory
get_ipython().system('mkdir d:\\\\sample\\\\datagen_raw')
get_ipython().system('mkdir d:\\\\sample\\\\fixed_resize')
get_ipython().system('mkdir d:\\\\sample\\\\datagen')


# In[ ]:


## Resize & create file by order:asc
path_etc = 'd:\\sample\\datagen_raw'
path_re = 'd:\\sample\\fixed_resize'

Func_1.fix_resize(path_etc, path_re, height=224, width=224)  #filename must me English  #default = 128*128


# In[ ]:


## Load IMGs
path = 'd:\\sample\\fixed_resize'
resize = Func_1.image_load(path)


# In[ ]:


## Check out first img(resized)
import matplotlib.pyplot as plt

img = resize[0, : ]
plt.figure()
plt.imshow(img)

print(resize.shape)


# In[ ]:


## Make X (feature)
X = resize
X.shape


# In[ ]:


## Sampling
X = X.astype('float')
X = X/255
X.shape 


# In[ ]:


## Func : datagen
def datagen(p1, p2, data, path):
    #p1 : picture range generating
    #p2 : generating figures per each picture
    X = data
    for k in range(p1):
        X2 = X[k:k+1, : ]
        X2.reshape(1, -1)

        i = 1
        for batch in X_datagen.flow(X2, batch_size=1,
                                    save_to_dir=path, 
                                    save_prefix='Picture'+str(k), 
                                    save_format='jpg'):
            i += 1
            if i > p2:
                break  


# In[ ]:


## Setting options
# Index : url below                
# https://m.blog.naver.com/PostView.nhn?blogId=isu112600&logNo=221582003889&proxyReferer=https:%2F%2Fwww.google.com%2F

from tensorflow.keras.preprocessing.image import ImageDataGenerator

X_datagen = ImageDataGenerator(horizontal_flip = True,
                               vertical_flip = False,
                               #brightness_range = [0.2, 1.0],
                               #width_shift_range = 0.2,
                               #height_shift_range = 0.2,
                               zoom_range = 0.1,
                               rotation_range = 60,
                               fill_mode = 'nearest'
                              )


# In[ ]:


## Execute function : datagen
path = 'd:\\sample\\datagen'
datagen(600,2,X,path)  #Create datagen imgs (50 figs:5*10)


# ## Visualize generated img on jupyter

# In[ ]:


## Move Current directory
get_ipython().run_line_magic('cd', 'd:/sample/datagen')


# In[ ]:


## file name change 
"""Warning
If above code(move dir) not work, Do not execute code below.
Unless your current folder's name will change like 1,2,3,4...etc."""
import os
count = 1
for i in os.listdir():
    os.rename(i, str(count) + '.' + i.split('.')[-1] )
    count += 1


# In[ ]:


## Load IMGs
path_re = 'd:\\sample\\datagen'
sample = Func_1.image_load(path_re)


# In[ ]:


## sampling
sample = sample.astype('float')
sample = sample/255
sample.shape


# In[ ]:


## Visualizing
fig = plt.figure(figsize=(20,20))
for i in range(20):
    plt.subplot(5,4,1+i)
    plt.imshow(sample[i,:])
    plt.show


# # Setting Test dataset

# In[13]:


get_ipython().system('mkdir d:\\\\sample\\\\Test_Dataset')
get_ipython().system('mkdir d:\\\\sample\\\\test_raw')


# In[14]:


## Resize & create file by order:asc
"""Warning!!
Your row dataset must be example(1).jpg, example(2).jpg, example(3).jpg ... example(3000).jpg
If your number contained in the filename is overlapped, it will be overwritten.
Meanwhile, raw dataset not must be 1.jpg, 2.jpg, 3.jpg... 3000.jpg."""

path_etc = 'd:\\sample\\test_raw'
path_re = 'd:\\sample\\Test_Dataset'

Func_1.resize(path_etc, path_re, height=128, width=128)  #filename must me English  #default = 128*128


# In[26]:


## Make csv file for test
path_re = 'd:\\sample\\label_2.csv'
Func_1.csv_maker_5(path_re, k1=10, k2=10, k3=10, k4=10, k5=10)


# In[ ]:


"""Test_Dataset must be contained independant pictures"""


# # create dir
# (for google drive upload)

# In[27]:


get_ipython().system('mkdir d:\\\\sample\\\\DATASETS')


# In[28]:


## If you have dir error, restart kernel and retry

# move datasets you need
get_ipython().system('move d:\\\\sample\\\\label.csv d:\\\\sample\\\\DATASETS')
get_ipython().system('move d:\\\\sample\\\\label_2.csv d:\\\\sample\\\\DATASETS')
get_ipython().system('move d:\\\\sample\\\\Test_Dataset d:\\\\sample\\\\DATASETS')
get_ipython().system('move d:\\\\sample\\\\Train_Dataset d:\\\\sample\\\\DATASETS')


# In[19]:


# copy .py & .ipynb file for traing in colab env.
get_ipython().system('mkdir d:\\\\sample\\\\VGGnet')
get_ipython().system('copy c:\\\\practice\\\\ImgClassification_VGGnet\\\\Func_1.py d:\\\\sample\\\\VGGnet')
get_ipython().system('copy c:\\\\practice\\\\ImgClassification_VGGnet\\\\ImgClassification_VGGnet.ipynb d:\\\\sample\\\\VGGnet')
get_ipython().system('copy c:\\\\practice\\\\ImgClassification_VGGnet\\\\For_Colab_Free.ipynb d:\\\\sample\\\\VGGnet')


# ## Create zipfile 

# In[20]:


## Create Zipfile in jupyter
temp_zip = zipfile.ZipFile('d:\\sample\\DATASETS.zip', 'w')
 
for folder, subfolders, files in os.walk('d:\\sample\\DATASETS'): 
    for file in files:
        temp_zip.write(os.path.join(folder, file), 
                       os.path.relpath(os.path.join(folder,file), 'd:\\sample\\DATASETS'), 
                       compress_type = zipfile.ZIP_DEFLATED)        
temp_zip.close()


# ## Move zipfile

# In[ ]:


get_ipython().system('move d:\\\\sample\\\\DATASETS.zip d:\\\\sample\\\\VGGnet')


# In[ ]:


get_ipython().run_line_magic('cd', 'd:\\\\sample\\\\VGGnet')
get_ipython().run_line_magic('ls', '')

"""Check out files below
DATASETS.zip
Func_1.zip
ImgClassification_VGGnet.ipynb

If you have all, upload VGGnet dir to Google drive main home"""


# In[ ]:


"""Do not change file and directory name!
Unless you will have dir error in colab code"""


# # Align filename with none-resize

# In[ ]:


# ## Move Current directory
# %cd d://sample/Train_Dataset


# In[ ]:


# ## Rename files like 1.jpg, 2.jpg, 3.jpg...
# """Warning
# If above code(move dir) not work, Do not execute code below.
# Unless your current folder's name will change like 1,2,3,4...etc."""
# import os
# count = 1
# for i in os.listdir():
#     os.rename(i, str(count) + '.' + i.split('.')[-1] )
#     count += 1

