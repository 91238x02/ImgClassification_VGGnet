#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""This script is for users having colab free ver.
Load img and Check out the accracy of model with training, validation, testsets"""


# # 1) **Preparing Datas**
# 
# 
# 

# ## Check out : Gpu/Ram

# In[ ]:


## 할당된 GPU 확인
gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)


# In[ ]:


## 할당된 Ram 확인
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
  print('re-execute this cell.')
else:
  print('You are using a high-RAM runtime!')


# ## Mount Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## Unzip picture_data
# (for train&test)

# In[ ]:


## Check out details of current dir
get_ipython().system("ls '/content/drive/MyDrive/VGGnet'")


# In[ ]:


## Copy 'VGGnet' directory to content directory in Colab 
get_ipython().system("cp -r '/content/drive/MyDrive/VGGnet/' '/content/'")


# In[ ]:


## Current directory 
get_ipython().run_line_magic('pwd', '')


# In[ ]:


## Check details
get_ipython().system('ls ./VGGnet')


# In[ ]:


## Unzip 'DATASETS.zip' file
get_ipython().system("unzip '/content/VGGnet/DATASETS.zip' -d '/content/VGGnet'")


# In[ ]:


## Delete zipfile 
get_ipython().system("rm '/content/VGGnet/DATASETS.zip'")


# In[ ]:


## Check Train_Dataset
get_ipython().run_line_magic('cd', "'/content/VGGnet/Train_Dataset'")
get_ipython().system("ls '/content/VGGnet/Train_Dataset'")
print('')
print('cnt of pictures')
get_ipython().system('ls -l | grep ^-.*\\.jpg$ | wc -l')


# In[ ]:


## Check Test_Dataset
get_ipython().run_line_magic('cd', "'/content/VGGnet/Test_Dataset'")
get_ipython().system("ls '/content/VGGnet/Test_Dataset'")
print('')
print('cnt of pictures')
get_ipython().system('ls -l | grep ^-.*\\.jpg$ | wc -l')


# In[ ]:


## Check out details of 'VGGnet' dir
get_ipython().run_line_magic('cd', '/content/VGGnet/')
get_ipython().system('ls ')


# # **2) Load datasets**

# ## Import Modules

# In[ ]:


## Modules Required

import Func_1  #Custom Functions 
import os 
import re
import cv2
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow


# ## Load Train_Dataset
# (to Colab environment from google-drive)

# In[ ]:


## Check out current working directory
get_ipython().run_line_magic('pwd', '')


# In[ ]:


## Check out cv2.__version
cv2.__version__  #4.1.2


# In[ ]:


## Check out state of imgs

#Train_Dataset
img = cv2.imread('/content/VGGnet/Train_Dataset/1.jpg')
cv2_imshow(img)

#Test_Dataset
img2 = cv2.imread('/content/VGGnet/Test_Dataset/1.jpg')
cv2_imshow(img2)


# In[ ]:


## Load X : Features
 
path_train = "/content/VGGnet/Train_Dataset"
resize = Func_1.img_load_c(path_train)


# In[ ]:


## Check out first img(resized) & shape
img = resize[0, : ]
plt.figure()
plt.imshow(img)
print(resize.shape)  


# ## Scailing : Trainset(X)

# In[ ]:


## Make X (feature)

X = resize
X.shape  


# In[ ]:


## Sampling
X = X.astype('float')
X = X/255
X.shape  


# ## Load Label

# In[ ]:


path_label = "/content/VGGnet/label.csv"
y = Func_1.label_load(path_label,label_cnt=5)  #label_cnt = len(신발종류)
y.shape


# ## Check out

# In[ ]:


## Confirm X, y
print(X.shape)  
print(y.shape, end='\n\n\n')  

# print("#####Check out : X#####")
# print(X, end='\n\n\n')
# print("#####Check out : y#####")
# print(y)


# In[ ]:


## Check out Train_Dataset imgs
index = [1,2,1801,1802,3601,3602,5401,5402,7201,7202]

plt.figure(figsize=(10, 10))
for j, i in enumerate(index):
    ax = plt.subplot(3, 4, j+1)
    img = X[i, ]
    plt.imshow(img)
    plt.title(np.argmax(y[i]))
    plt.axis("off")

## Index
# 0 - addidas 
# 1 - converse
# 2 - new balance
# 3 - nike
# 4 - vans


# # **3) Training Models : VGGnet**
# (Training is excluded in this part)

# ## Import Modules

# In[ ]:


## module import
import tensorflow as tf # tensorflow 2.0
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


## Transfer learning (전이학습)
from tensorflow.keras.applications.vgg16 import VGG16                        
#from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.resnet import ResNet50


# In[ ]:


## Check out Tensor version
import tensorflow
print(tensorflow.__version__)  #2.4.1


# ## Set Validation

# In[ ]:


# 훈련/테스트 데이터를 0.7/0.3의 비율로 분리합니다.
x_train, x_val, y_train, y_val = train_test_split(X, y, 
                                                test_size = 0.3, 
                                                random_state = 777)

# Checkout
print("x_train.shape :", x_train.shape)
print("y_train.shape :", y_train.shape)
print("x_val.shape :", x_val.shape)  
print("y_val.shape :", y_val.shape)  


# ## *Get accuracy & loss >>> Colab free ver.*
# (Train_Dataset)

# In[ ]:


## Change current dir.
get_ipython().run_line_magic('cd', '"/content/VGGnet"')


# In[ ]:


## Check out directory
get_ipython().system("ls # confirm the file 'VGG16_Model_v1.h5'")


# In[ ]:


## Load trained model

from keras.models import load_model
model = load_model('VGG16_Model_v1.h5')


# In[ ]:


# Check results of train set
scores = model.evaluate(x_train, y_train, batch_size=16, verbose=1)
print("Vgg16 train Error : %.2f%%" % (100-scores[1]*100))
print("Vgg16 train Loss : %.2f" % (scores[0]))


# In[ ]:


# Check results of validation set
scores2 = model.evaluate(x_val, y_val, batch_size=16, verbose=1)
print("Vgg16 val Error : %.2f%%" % (100-scores2[1]*100))
print("Vgg16 val Loss : %.2f" % (scores2[0]))


# In[ ]:


#https://www.tensorflow.org/api_docs/python/tf/keras/Model


# # **4) Test with Test_Dataset**

# ## Load Test_Dataset 

# In[ ]:


## Check out current working directory
get_ipython().run_line_magic('pwd', '')


# In[ ]:


## Load X : Features
path_test = "/content/VGGnet/Test_Dataset"
resize_t = Func_1.img_load_c(path_test)


# In[ ]:


## Check out first img(resized) & shape
img = resize_t[0, : ]
plt.figure()
plt.imshow(img)
print(resize_t.shape)  


# ## Scailing : Testset(X_t)

# In[ ]:


## Make X_t (feature)

X_t = resize_t
X_t.shape  


# In[ ]:


X_t = X_t.astype('float')
X_t = X_t/255
X_t.shape  


# ## Load Label

# In[ ]:


path_label2 = "/content//VGGnet/label_2.csv"
y_t = Func_1.label_load(path_label2,label_cnt=5)  #label_cnt = len(신발종류)
y_t.shape


# ## Check out

# In[ ]:


## Confirm X, y
print(X_t.shape)  
print(y_t.shape, end='\n\n\n')  

# print("#####Check out : X#####")
# print(X, end='\n\n\n')
# print("#####Check out : y#####")
# print(y)


# In[ ]:


## Check out Test_Dataset imgs
index = [1,2,11,12,21,22,31,32,41,42]

plt.figure(figsize=(10, 10))
for j, i in enumerate(index):
    ax = plt.subplot(3, 4, j+1)
    img = X_t[i, ]
    plt.imshow(img)
    plt.title(np.argmax(y_t[i]))
    plt.axis("off")

## Index
# 0 - addidas 
# 1 - converse
# 2 - new balance
# 3 - nike
# 4 - vans


# ## Get accuracy & loss >>> Colab pro ver.
# (Test_Dataset)

# In[ ]:


## Check out accuracy (with test data)  #모델을 실측데이터로 정확도 측정한 결과 확인
scores3 = model.evaluate(X_t, y_t, batch_size=16, verbose=1)
print("Vgg16 ind_dataset Error : %.2f%%" % (100-scores3[1]*100))
print("Vgg16 ind_dataset loss : %.2f" % (scores3[0]))


# # **5) Extract trained model**
# (to google drive)

# In[ ]:


"""
You already have model in your google drive.
So go to google drive and get it.
"""


# In[ ]:


# ## Save model : .h5(Hdf5 type file)
# save_path = "/content/VGGnet"
# model.save('VGG16_Model_v1.h5', save_path, save_format="h5")  


# In[ ]:


# ## Save model : .data
# model.save('VGG16_Model_v2.tf', save_path, save_format="tf")  


# In[ ]:


# ## Download extracted model
# """If this code takes too much time, Copy model to google drive and download to local in google drive Gui envirnment.
# In my case it takes 10-15 minutes
# Help code is located below.
# I recommand copy model to google drive and download in g-drive."""

# from google.colab import files
# files.download('/content/VGGnet/VGG16_Model_v1.h5') 


# In[ ]:


# ## Copy model colab to google drive
# !cp '/content/VGGnet/VGG16_Model_v1.h5' /content/drive/MyDrive/VGGnet

