# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:01:49 2016

@author: Prathyusha
"""

import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

train = os.listdir("/Users/Prathyusha/machinelearning/Data/train")
train_names = ([x for x in train
           if not (x.startswith('.'))])


train_path = "/Users/Prathyusha/machinelearning/Data/train"
i_imagePath = []

i_classes = np.array(['labels'],dtype = 'object')
surf = cv2.xfeatures2d.SURF_create()
class_id = 0
des_list = []
BOW = cv2.BOWKMeansTrainer(30)

from matplotlib import pyplot as plt
print(datetime.now())
for train_name in train_names:
    dir = os.path.join(train_path,train_name)
    im = os.listdir(dir)
    
    for p in im:
        class_path = train_path +"/"+train_name+"/"+p
        image = cv2.imread(class_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp = surf.detect(gray,None)
        kp,des = surf.compute(gray,kp)
        BOW.add(des)
        #print(len(kp))
        img = cv2.drawKeypoints(gray,kp,image)
       # plt.imshow(img),plt.show()
        des_list.append((class_path,des))
        i_imagePath.append(class_path)
        i_classes = np.vstack([i_classes,train_name])
    class_id += 1
print(datetime.now())

#dictionary created
dictionary = BOW.cluster()
print(datetime.now())

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 30)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
print("line 57")
#surf2 = cv2.BOWImgDescriptorExtractor( surf, flann )
print("line 58")
bowDiction = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
print("line 60")
bowDiction.setVocabulary(dictionary)
print("line 62")
print ("bow dictionary", np.shape(dictionary))

def feature_extract(pth):
    im = cv2.imread(pth, 1)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return bowDiction.compute(gray, surf.detect(gray))

train_desc = []
train_labels = []
i = 0
for p in i_imagePath:
    train_desc.extend(feature_extract(p))
    if train_names[i]=='Balloon':
        train_labels.append(1)
    if train_names[i]=='Beach':
        train_labels.append(2)
    if train_names[i]=='Bird':
        train_labels.append(3)
    if train_names[i]=='Bobsled':
        train_labels.append(4)
    if train_names[i]=='Bonsai': 
        train_labels.append(5)
    if train_names[i]=='Building':
        train_labels.append(6)
    if train_names[i]=='Bus':
        train_labels.append(7)
    if train_names[i]=='Butterfly':
        train_labels.append(8)
    if train_names[i]=='Car':
        train_labels.append(9)
    if train_names[i]=='Cat': 
        train_labels.append(10)
    if train_names[i]=='Cougar':
        train_labels.append(11)
    if train_names[i]=='Dessert':
        train_labels.append(12)
    if train_names[i]=='Dog':
        train_labels.append(13)
    if train_names[i]=='Eagle':
        train_labels.append(14)
    if train_names[i]=='Elephant': 
        train_labels.append(15)
    if train_names[i]=='Firework':
        train_labels.append(16)
    if train_names[i]=='Fitness':
        train_labels.append(17)
    if train_names[i]=='Flag':
        train_labels.append(18)
    if train_names[i]=='Foliage':
        train_labels.append(19)
    if train_names[i]=='Fox': 
        train_labels.append(20)
    if train_names[i]=='Goat':
        train_labels.append(21)
    if train_names[i]=='Horse':
        train_labels.append(22)
    if train_names[i]=='Indoordecorate':
        train_labels.append(23)
    if train_names[i]=='Jewelry':
        train_labels.append(24)
    if train_names[i]=='Lion': 
        train_labels.append(25)
    if train_names[i]=='Model':
        train_labels.append(26)
    if train_names[i]=='Mountain':
        train_labels.append(27)
    if train_names[i]=='Mushroom':
        train_labels.append(28)
    if train_names[i]=='Owl':
        train_labels.append(29)
    if train_names[i]=='Penguin': 
        train_labels.append(30)
    i = i+1
    
print(train_desc)