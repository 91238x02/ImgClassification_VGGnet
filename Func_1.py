import os 
import re
import cv2
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""  ###함수설명###
resize : raw 파일을 resize 하고 1.jpg, 2.jpg, 3.jpg ... 100.jpg 형태로 저장하는 함수 (raw 파일의 숫자만큼 숫자 부여)
image_load : .jpg file 을 python 으로 load 하는 함수, 4차원 ndarray 를 반환한다
csv_maker_10 : class 가 10개인 label csv file 생성 함수
csv_maker_5 : class 가 5개인 label csv file 생성 함수
label_load : label csv file 을 python 으로 로드하여 one-hot encoding 하여 label 값을 반환하는 함수 (2차원 ndarray 반환)
"""




def resize(raw_path=None, resize_path=None, height=128, width=128):
    file_list = os.listdir(raw_path)
    file_name = []
    for i in file_list:
        a = int(re.sub('[^0-9]', '', i))  
        file_name.append(a)

    for i, k in enumerate(file_list):
        img = cv2.imread(raw_path + '\\' + k)
        resize_img1 = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        x = str(file_name[i])
        cv2.imwrite(resize_path + "\\" + x + ".jpg", resize_img1)     
    plt.imshow(resize_img1)
    plt.show()  
    print("img shape", resize_img1.shape)  
    

def fix_resize(raw_path=None, resize_path=None, height=128, width=128):
    file_list = os.listdir(raw_path)
    file_name = []
    for i in file_list:
        a = int(re.sub('[^0-9]', '', i))  
        file_name.append(a)

    for i, k in enumerate(file_list):
        img = cv2.imread(raw_path + '\\' + k)
        resize_img1 = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        fix_img = cv2.cvtColor(resize_img1, cv2.COLOR_BGR2RGB) # 코드 추가 (색상 변형 문제 해결)
        x = str(file_name[i])
        cv2.imwrite(resize_path + "\\" + x + ".jpg", fix_img)     

    plt.imshow(fix_img)
    plt.show()  
    print("img shape", fix_img.shape)
    
    

def image_load(path):
    file_list = os.listdir(path)
    
    file_name = []
    for i in file_list:
        a = int(re.sub('[^0-9]', '', i)) 
        
        file_name.append(a)
    file_name.sort()  
    
    file_res = []
    for j in file_name:
        file_res.append('%s\\%d.jpg' %(path,j) )
    
    image = []
    for k in file_res:
        img = cv2.imread(k)
        image.append(img)
    
    return np.array(image)


def csv_maker_10(path, k1=None, k2=None, k3=None, k4=None, k5=None, k6=None, k7=None, k8=None, k9=None, k10=None):
    file = open(path, 'w')
    
    for i in range(k1):
        file.write(str(0) + '\n')
    for i in range(k2):
        file.write(str(1) + '\n')
    for i in range(k3):
        file.write(str(2) + '\n')
    for i in range(k4):
        file.write(str(3) + '\n')
    for i in range(k5):
        file.write(str(4) + '\n')      
    for i in range(k6):
        file.write(str(5) + '\n')
    for i in range(k7):
        file.write(str(6) + '\n')
    for i in range(k8):
        file.write(str(7) + '\n')
    for i in range(k9):
        file.write(str(8) + '\n')
    for i in range(k10):
        file.write(str(9) + '\n')  
     
    file.close

    
def csv_maker_5(path, k1=None, k2=None, k3=None, k4=None, k5=None):
    file = open(path, 'w')
    
    for i in range(k1):
        file.write(str(0) + '\n')
    for i in range(k2):
        file.write(str(1) + '\n')
    for i in range(k3):
        file.write(str(2) + '\n')
    for i in range(k4):
        file.write(str(3) + '\n')
    for i in range(k5):
        file.write(str(4) + '\n')      
     
    file.close
    
    

def label_load(path, label_cnt=None):

    file = open(path)
    labeldata = csv.reader(file)
    labellist = list(labeldata)
    label = np.array(labellist)
    label = label.astype(int)
    label = np.eye(label_cnt)[label]
    label = label.reshape(-1,label_cnt)
    return label




## 미사용

def next_batch( data1, data2, init, fina ):
    return data1[ init : fina ], data2[ init : fina ]

def shuffle_batch( data_list, label ) :
    x = np.arange( len( data_list) )
    random.shuffle(x)
    data_list2 = data_list[x]
    label2 = label[x]
    return data_list2,label2


























