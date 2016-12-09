# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import pandas as pd

train = os.listdir("/Users/Prathyusha/machinelearning/Data/train")
train_names = ([x for x in train
           if not (x.startswith('.'))])


train_path = "/Users/Prathyusha/machinelearning/Data/train"
i_imagePath = []

i_classes = np.array(['labels'],dtype = 'object')
surf = cv2.xfeatures2d.SURF_create(400)
class_id = 0
des_list = []


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
        #print(len(kp))
        img = cv2.drawKeypoints(gray,kp,image)
        des_list.append((class_path,des))
        i_imagePath.append(class_path)
        i_classes = np.vstack([i_classes,train_name])
    class_id += 1
        #plt.imshow(img)


print(class_id)
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
k = 100
from scipy.cluster.vq import kmeans
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(i_imagePath), k), "float32")


from scipy.cluster.vq import vq

for i in range(len(i_imagePath)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(i_imagePath)+1) / (1.0*nbr_occurences + 1)), 'float32')


# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


print(datetime.now())
        
    