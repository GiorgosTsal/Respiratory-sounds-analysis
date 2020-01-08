#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:03:37 2020

@author: gtsal
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import wave
import sys
import os
import librosa
import librosa.display

import soundSP as ssp


def create_dataframe(folder):
    soundwav = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                soundwav.append(file)
    data = [];
    
    for file in soundwav:
        if file[:3] not in data:
            data.append(file[:3])
    data = sorted(data)
    
    arr = np.array(data)
    
    row =[]
    
    for i in data :
        r1=[]
        for file in soundwav:
            if file[:3]==i:
                r1.append(file)
        row.append(r1)
                
    
    df=pd.DataFrame(row,index=data)
    col = []
    for i in range(1,df.shape[1]+1):
        col.append('soundtrack-'+str(i))
    
    df.columns = col

    return df

def w_c_dataset(df,subfolder):
    files=[]
    lengths=[]
    for i,j in df.iterrows():
        for l in j:
            if(l!=None):
                files.append(subfolder+str(l))
                a=str(l).replace('.wav','.txt')
                tmp=pd.read_csv(subfolder+a,sep='\t',header = None)
                lengths.append(tmp)
                times,labels=frame(lengths)
    sounds = []
    
    [sound,sr] = ssp.load_sound_files(files)
    return sound,sr,lengths,times,labels

def encode(line):
    if line[0]==0:
        if line[1]==0:
            return 0
        else: 
            return 2
    if line[0]==1:
        if line[1]==0:
            return 1
        else:
            return 3

def frame(lengths):
    times=[]
    labels=[]
    for df in lengths:
        l=df.iloc[:,0:2]
        l=l.to_numpy();      
        l=l.reshape(1,-1)
        l=np.ceil(l*22050)
        l=l.reshape(int(l.size/2),-1)
        times.append(l)
        # 0 present of crackle 1 present of wheeles 2 both
        #0 0 0 or 0 0 1 or 0 1 0 or 1 0 0
        k=df.iloc[:,2:4]
        k=k.to_numpy()
        z=[]
        for i in k:
            z.append(encode(i))
        z=np.asarray(z)
        z=z.reshape(-1,1)
        labels.append(z)
    return times,labels

def split_sounds(sounds,times,labels):
    s=[]
    l=[]
    for i,sound in enumerate(sounds):
        for t,label in zip(times[i],labels[i]):
            s.append(sound[int(t[0]):int(t[1])])
            if label==0:
                a=np.array([1,0,0,0])
            if label==1:
                a=np.array([0,1,0,0])
            if label==2:
                a=np.array([0,0,1,0])    
            if label==3:
                a=np.array([0,0,0,1])
            l.append(a)
    return s,l

def experiment():
    folder = "Respiratory_Sound_Database/Respiratory_Sound_Database/"
    
    subfolder =folder  +'audio_and_txt_files/'
    df = create_dataframe(subfolder)
    df=df.rename_axis('ID')
    soundpath=[]
    for i in df.iloc[:,0]:
        soundpath.append(subfolder+i)
    diagnosis = pd.read_csv(folder+'patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])
    soundtracks=df.count(axis=1)
    #features=ssp.extract_feature(soundpath)
    parent_dir = folder
    tr_sub_dirs = subfolder
    ts_sub_dirs = subfolder+"test/"
    #df2=pd.concat(df[1,:]*5,ignore_index=True)
    #df2=pd.concat(df[7,:]*3,ignore_index=True)
    #df2=pd.concat(df[14,:]*2,ignore_index=True)
    df=df.rename_axis('ID')
    print(df.head(10))
    return df,diagnosis

def split_set(dataset,label):
    a=label[label==0]
    d=dataset[label[:,0]==0,:]
    s=np.size(a,0)
    t=int(np.size(a,0)*0.7)
    x_test = d[t:s,:]
    y_test = a[t:s]
    x_train = d[0:t,:]
    y_train = a[0:t]
    for i in range (1,4):
        a=label[label==i]
        d=dataset[label[:,0]==i,:]
        s=np.size(a,0)
        t=int(np.size(a,0)*0.7)
        x_test = np.vstack((x_test,d[t:s,:]))
        y_test = np.hstack((y_test,a[t:s]))
        x_train = np.vstack((x_train,d[0:t,:]))
        y_train = np.hstack((y_train,a[0:t]))
    return x_train,y_train,x_test,y_test

folder = "Respiratory_Sound_Database/Respiratory_Sound_Database/"   
subfolder =folder  +'audio_and_txt_files/'
df,diagnosis = experiment()
[sound,sr,lengths,times,labels] = w_c_dataset(df,subfolder)
[data,label]=split_sounds(sound,times,labels)





labels

ssp.plot_waves(sound[16],sr[16])
ssp.plot_waves(sound[15],sr[15])
ssp.plot_waves(sound[10],sr[10])

dataset = []
for d in data:
    a = ssp.extract_feature(d,22050)
    dataset.append(a)

data=np.asarray(dataset)
a=np.vstack(labels[:])
l1=np.zeros(np.size(a,0))
l2=np.zeros(np.size(a,0))
for i in range(np.size(a,0)):
    if a[i]==1:
        l1[i]=1
    if a[i]==2:
        l2[i]=1
    if a[i]==3:
        l1[i]=1
        l2[i]=1

x_train1,y_train1,x_test1,y_test1=split_set(data,l1.reshape(-1,1))
x_train2,y_train2,x_test2,y_test2=split_set(data,l2.reshape(-1,1))

from  sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_train1=scaler.fit_transform(x_train1)
x_train2=scaler.fit_transform(x_train2)
x_test1=scaler.fit_transform(x_test1)
x_test2=scaler.fit_transform(x_test2)


from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC


# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}  
  
grid1 = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid1.fit(x_train1, y_train1)

# print best parameter after tuning 
print(grid1.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid1.best_estimator_)

grid_predictions = grid1.predict(x_test1) 
  
# print classification report 
print(classification_report(y_test1, grid_predictions))

#After parameter tuning best parameters for weezles(model1)
#
#{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}

#SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
#              precision    recall  f1-score   support
#
#         0.0       0.66      0.98      0.79      1359
#         1.0       0.42      0.03      0.06       711
#
#    accuracy                           0.65      2070
#   macro avg       0.54      0.50      0.42      2070
#weighted avg       0.58      0.65      0.54      2070


#model1 = sklearn.svm.SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

  
grid2 = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid1.fit(x_train2, y_train2)

# print best parameter after tuning 
print(grid2.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid2.best_estimator_)

grid_predictions = grid2.predict(x_test2) 
  
# print classification report 
print(classification_report(y_test2, grid_predictions))



#after cross validation best parameters for crackles

#{'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}

#SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
#    probability=False, random_state=None, shrinking=True, tol=0.001,
#    verbose=False)
#              precision    recall  f1-score   support
#
#         0.0       0.80      1.00      0.89      1652
#         1.0       0.00      0.00      0.00       418
#
#    accuracy                           0.80      2070
#   macro avg       0.40      0.50      0.44      2070
#weighted avg       0.64      0.80      0.71      2070


#model2 = sklearn.svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
#    probability=False, random_state=None, shrinking=True, tol=0.001,
#    verbose=False)

# x_train1,y_train1= sklearn.utils.shuffle(x_train1,y_train1,random_state=0)
# x_train2,y_train2= sklearn.utils.shuffle(x_train2,y_train2,random_state=0)
# x_test1,y_test1= sklearn.utils.shuffle(x_test1,y_test1,random_state=0)
# x_test2,y_test2= sklearn.utils.shuffle(x_test2,y_test2,random_state=0)

#model1.fit(x_train1,y_train1)
#model2.fit(x_train2,y_train2)
#y_pred1=model1.predict(x_test1)
#y_pred2=model2.predict(x_test2)
#print('Model Wheezes')
#print(sklearn.metrics.accuracy_score(y_test1,y_pred1))
#print(sklearn.metrics.confusion_matrix(y_test1,y_pred1))
#print(sklearn.metrics.classification_report(y_test1, y_pred1))
#print('Model Crackles')
#print(sklearn.metrics.accuracy_score(y_test2,y_pred2))
#print(sklearn.metrics.confusion_matrix(y_test2,y_pred2))
#print(sklearn.metrics.classification_report(y_test2, y_pred2))

#[sr,sound] = cns.read_wav_file(files[1],22050)
#sound=ld[:,0]
#sr=ld[:,1]

