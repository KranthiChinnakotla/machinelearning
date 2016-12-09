
# coding: utf-8

# In[1]:

import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import cv2
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report



# In[2]:

#Writing classes and labels to a text file

directory_train = "/Users/Prathyusha/machinelearning/Data/train"
directory_test = "/Users/Prathyusha/machinelearning/Data/val"
def makeFile(path):
    dictionary_train_images = {}
    classes_list = os.listdir(path)
    for i in classes_list:
        if('.' not in i):
            images_list = os.listdir(path + '/' + i)
            dictionary_train_images[i] = images_list
    return dictionary_train_images

def return_features(dictionary, directory):
    # Extracting 400 features from the images
    surf = cv2.xfeatures2d.SIFT_create(nfeatures=400)
    #surf.extended = True
    training_x = []
    training_y = []
    for (k, v) in dictionary.items():
        for i in v:
            image = cv2.imread(directory + '/' + k +'/'+i,)
            image = cv2.resize(image, (255, 255)) 
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kp, des = surf.detectAndCompute(gray,None)
            training_x.append(des)
            training_y.append(k)
    return training_x, training_y

#Encode labels
def encode_labels(training_y, testing_y):
    le = preprocessing.LabelEncoder()
    le.fit(training_y)
    return le.transform(training_y), le.transform(testing_y)


# In[3]:

# Generating features and labels for a train dataset and the test dataset
training_set_images = makeFile(directory_train)
testing_set_images = makeFile(directory_test)

trainX, trainY = return_features(training_set_images, directory_train)
testX, testY = return_features(testing_set_images, directory_test)

trainY, testY = encode_labels(trainY, testY)


# In[4]:

# This function is to reshape 2D training features in to a 1D
def reshapeTrainX():
    trainX_reshape = []
    for x in trainX:
        m, n = x.shape
        x = np.ravel(np.reshape(x, m * n))
        trainX_reshape.append(x.tolist())
    return trainX_reshape


# In[5]:

#Save the train file in the local machine to be read as an input
a = reshapeTrainX()
import numpy
numpy.savetxt("results.csv", a, delimiter=",",fmt='%s')


# In[6]:

# Read the training data file using pandas
import pandas as pd
from pandas import *
a = pd.read_csv('/Users/Prathyusha/machinelearning/Data/train_data.csv',header=None)
a.head()


# In[8]:

# Make 1D array of training features
b = a.ix['0':'0']
train_X = a.as_matrix(columns=b.columns[0:])


# In[17]:

# Using KFold from sklearn and splitting the training dataset in to training and validation dataset
from sklearn.model_selection import KFold
kf = KFold(n_splits = 4)
print(kf)
for train_index, test_index in kf.split(train_X):
    train_val_X,test_val_X = train_X[train_index],train_X[test_index]
    train_val_Y,test_val_Y = trainY[train_index],trainY[test_index]


# In[9]:


# This function is to reshape 2D testing features in to a 1D
def reshapeTestX():
    testX_reshape = []
    for x in testX:
        m, n = x.shape
        x = np.ravel(np.reshape(x, m * n))
        testX_reshape.append(x.tolist())
    return testX_reshape


# In[10]:

##Save the test file in the local machine to be read as an input
a_test = reshapeTestX()
import numpy
numpy.savetxt("test_data.csv", a_test, delimiter=",",fmt='%s')


# In[11]:

# Read the testing data file using pandas
test_csv = pd.read_csv('/Users/Prathyusha/machinelearning/Data/test_data.csv',header=None)
test_csv.head()


# In[12]:

# Make 1D array of testing features
test_X = test_csv.as_matrix(columns=b.columns[0:])


# In[13]:

# Fit NaiveBayes model
naiveclf = GaussianNB()
naiveclf.fit(train_X, trainY)


# In[18]:

#Print the accuracy score:
from sklearn.metrics import accuracy_score
print(accuracy_score(naiveclf.predict(test_X),testY))


# In[19]:

# Importing and fitting SVM linear SVC
from sklearn.svm import LinearSVC
svmLinearClf = LinearSVC(C=10.0)
svmLinearClf.fit(train_X,trainY)


# In[20]:

#Print the accuracy score:
from sklearn.metrics import accuracy_score
print(accuracy_score(svmLinearClf.predict(test_X),testY))


# In[ ]:



