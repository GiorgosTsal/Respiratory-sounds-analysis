#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:03:37 2020

@authors: Panagiotis Kasparidis
	  Georgios   Tsalidis
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
import xgboost as xgb
import soundSP as ssp
from  sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix
from sklearn.svm import SVC
import hdbscan
import seaborn as sn

def cm(y_test,y_pred):
    classes = ["none", "crackles", "wheezes", "both"] # put your class labels
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred,labels=range(4)), index = [i for i in classes],
                      columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    plt.title('Confusion matrix')
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 12},fmt='d')
    
    
    plt.show()

def plot_roc(y_test,y_pred):
    
    test=onehenc(y_test,4)
    pred=onehenc(y_pred,4)
    
    
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_auc_score,roc_curve,auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    
    # Plot of a ROC curve for a specific class
    for i in range(4):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="best")
        plt.show()


def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
        plt.savefig('confusion-matrix.png')
        
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
    [sound,sr] = ssp.load_sound_files(files)
    return sound,sr,lengths,times,labels

'''
    encode  0 ->None
            1 ->Crackle
            2 ->Wheeze
            3 ->Both
'''

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
        
def onehenc(labels,classes=4):
    r=np.zeros((labels.shape[0],classes))
    for i in range(labels.shape[0]):
        for j in range(classes):
            if labels[i]==j:
                r[i,j]=1
    return r
def decode(a):
    for i in range(a.shape[0]):
        if a[i]==1:
            return i

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
    parent_dir = folder
    tr_sub_dirs = subfolder
    ts_sub_dirs = subfolder+"test/"

    df=df.rename_axis('ID')
    print(df.head(10))
    return df,diagnosis

folder = "Respiratory_Sound_Database/Respiratory_Sound_Database/"   
subfolder =folder  +'audio_and_txt_files/'
df,diagnosis = experiment()
[sound,sr,lengths,times,labels] = w_c_dataset(df,subfolder)
[data,label]=split_sounds(sound,times,labels)
a=np.vstack(labels[:])

#np.save('labels',labels)
#np.save('labels',label)

# h label exei ta 4 classes :
# 0 ->None
# 1 ->Crackle 
# 2 ->Wheezes
# 3 ->Both 

from sklearn.cluster import MiniBatchKMeans,DBSCAN
data,samples = ssp.FE(data,sr[0])
data = MinMaxScaler().fit_transform(data)

clustering = MiniBatchKMeans(batch_size=500000, n_clusters=3000, init='k-means++', n_init=50 )
#clustering =hdbscan.HDBSCAN(min_samples=10,prediction_data=True)

clustering.fit(data)


Data = ssp.cluster(clustering.predict(data),samples,3000)
Data = ssp.cluster(clustering,samples,3000)
Data = Data.T
scaler = MinMaxScaler()
scaler.fit(Data)
Data = scaler.transform(Data)
Data = Data.T
Data =ssp.clearEmptyDimensions(Data)


scaler=StandardScaler()

label=np.asarray(label)
#
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(Data,a,test_size=0.3, random_state=42,stratify=a)
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



  
Cs = [2**(-2),2**(-1), 1,2**(1),2**(2),2**(3),2**(4),2**(12),2**(15)]
eps = [2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),2**(0),2**(1),2**(2),2**(3)]


param_grid = {'C': Cs,  
              'gamma': eps, 
              'kernel': ['rbf'],
              'decision_function_shape':['ovr'],
              'class_weight': ['balanced']}  
grid1 = GridSearchCV(SVC(), param_grid,cv=2,n_jobs=6, verbose = 3) 
  
# fitting the model for grid search 
grid1.fit(x_train, y_train)

# print best parameter after tuning 
print(grid1.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid1.best_estimator_)

grid_predictions = grid1.predict(x_test) 
  
# print classification report 
print(classification_report(y_test, grid_predictions))


print(sklearn.metrics.confusion_matrix(y_test,grid_predictions))
plot_roc(y_test,grid_predictions)
cm(y_test,grid_predictions)

#{'C': 4, 'class_weight': 'balanced', 'decision_function_shape': 'ovr', 'gamma': 0.03125, 'kernel': 'rbf'}
svm =SVC(C=400000000,kernel='rbf',gamma=1000000,class_weight='balanced')
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)

print(classification_report(y_test, y_pred))
print(sklearn.metrics.confusion_matrix(y_test,y_pred))

plot_roc(y_test,y_pred)
cm(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier
tree =RandomForestClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=8,max_features=120,class_weight='balanced')
tree.fit(x_train,y_train)
y_pred = tree.predict(x_test)
print(sklearn.metrics.confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
plot_roc(y_test,y_pred)
cm(y_test,y_pred)

