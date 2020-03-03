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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.naive_bayes as nb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
from sklearn.neighbors import kd_tree
import seaborn as sn
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE



def cm(y_test,y_pred):
    classes = ["none", "crackles", "wheezes", "both"] # put your class labels
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred,labels=range(4)), index = [i for i in classes],
                      columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    plt.title('Confusion matrix')
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 12},fmt='d')
    
    
    plt.show()

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
        
        
def logtf(data):
    result = np.zeros((data.shape[0],data.shape[1]))
    for i,d in enumerate(data):
        l = len(d)
        size = np.sum(d)
        for j in range(l):
            result[i,j] = np.log(d[j]/size+1)
    return result
#Data = logtf(Data)

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
        plt.plot([1,0],[1,1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="best")
        plt.show()

def experiment():
    folder = "../Respiratory_Sound_Database/Respiratory_Sound_Database/"   
    
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

folder = "../Respiratory_Sound_Database/Respiratory_Sound_Database/"   
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

dataset = []
for d in data:
    a = ssp.extract_feature(d,sr[0])
    dataset.append(a)

data=np.asarray(dataset)
data = data.reshape([6898,data.shape[1]*data.shape[2],])

label=np.asarray(label)
a=np.zeros(label.shape[0])
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        if label[i][j]==1:
            a[i]=j


np.save('labels',a)
np.save('data',data)


#a=np.load('labels.npy')
#data=np.load('data.npy')

scaler=StandardScaler()

#
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(data,a,test_size=0.3, random_state=42,stratify=a)
scaler.fit(x_train)
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#Logistic Regression

# Decision Tree
min_samples_split=[1*x for x in range(1,21,5)]
max_depth=[x for x in range(10,101,10)]
max_features=[x for x in range(40,251,10)]
min_samples_leaf=[1*x for x in range(1,17,3)]
parameters = {'max_depth':max_depth,'min_samples_split':min_samples_split,
              'max_features':max_features,'min_samples_leaf':min_samples_leaf}
    

rtree = DecisionTreeClassifier(class_weight='balanced')
clf = GridSearchCV(rtree, parameters, cv=5,scoring='accuracy',verbose=2,n_jobs=-1)
clf.fit(x_train, y_train)
print("The best parameters are",clf.best_params_," with a score of ", -clf.best_score_)
y_pred =clf.predict(x_test)
print(classification_report(y_test, y_pred))

cm(y_test,y_pred)
plot_roc(y_test,y_pred)

    # Random Forests tune parameters
min_samples_split=[x for x in range(1,21,8)]
max_depth=[x for x in range(10,101,40)]
max_features=[x for x in range(40,251,40)]
min_samples_leaf=[x for x in range(1,21,10)]
n_estimators=[x for x in range(1,110,40)]
parameters = {'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_split':min_samples_split,
              'max_features':max_features,'min_samples_leaf':min_samples_leaf}

rrforest = RandomForestClassifier(min_weight_fraction_leaf=0.0,class_weight='balanced',
                                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                                          min_impurity_split=None, bootstrap=True, oob_score=True,
                                          n_jobs=1, random_state=0, verbose=0,
                                          warm_start=False)

clf2 = GridSearchCV(rrforest, parameters, cv=3,scoring='accuracy',verbose=2,n_jobs=6)
clf2.fit(x_train, y_train)
print("The best parameters are",clf2.best_params_," with a score of ", clf2.best_score_)
y_pred =clf2.predict(x_test)
print(classification_report(y_test, y_pred))

cm(y_test,y_pred)
plot_roc(y_test,y_pred)

# Logistic Regression
lr =LogisticRegression(C=100,penalty='l2',max_iter=1000,n_jobs=-1,solver='lbfgs',multi_class='multinomial',tol=1e-3,class_weight='balanced')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(classification_report(y_test, y_pred))

cm(y_test,y_pred)
plot_roc(y_test,y_pred)

#gradient boosting
gdc=GradientBoostingClassifier(learning_rate=0.001,n_estimators=120,min_samples_split=5,
                               min_samples_leaf=4,max_features=250)
gdc.fit(x_train,y_train)
y_pred =gdc.predict(x_test)
print(classification_report(y_test, y_pred))

cm(y_test,y_pred)
plot_roc(y_test,y_pred)

#bagging
bc=BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced'),n_estimators=500,max_samples=150,max_features=200,n_jobs=-1)
bc.fit(x_train,y_train)
y_pred = bc.predict(x_test)
print(classification_report(y_test, y_pred))

cm(y_test,y_pred)
plot_roc(y_test,y_pred)

# defining parameter range 

  
Cs = [2**(-5),2**(-4),2**(-3),2**(-2),2**(-1), 1,2**(1),2**(2),2**(3),2**(4),2**(5),2**(6),2**(7),2**(8),2**(9),2**(10),2**(11),2**(12),2**(13),2**(14),2**(15)]
gamma = [2**(-15),2**(-14),2**(-13),2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),2**(-1),2**(0),2**(1),2**(2),2**(3)]


param_grid = {'C': Cs,  
              'gamma': gamma, 
              'kernel': ['rbf'],
              'decision_function_shape':['ovr'],
              'class_weight': ['balanced',{0:1,1:2,2:6,3:12},{0:1,1:4,2:10,3:16}]}  
grid1 = GridSearchCV(SVC(), param_grid,cv=3,n_jobs=6, verbose = 3) 
  
# fitting the model for grid search 
grid1.fit(x_train, y_train)

# print best parameter after tuning 
print(grid1.best_params_) 
#{'C': 4, 'class_weight': 'balanced', 'decision_function_shape': 'ovr', 'gamma': 0.0078125, 'kernel': 'rbf'}
# print how our model looks after hyper-parameter tuning 
print(grid1.best_estimator_)

grid_predictions = grid1.predict(x_test) 
  
# print classification report 
print(classification_report(y_test, grid_predictions))
print(accuracy_score(y_test,grid_predictions))

print(sklearn.metrics.confusion_matrix(y_test,grid_predictions))
#{'C': 4, 'class_weight': 'balanced', 'decision_function_shape': 'ovr', 'gamma': 0.03125, 'kernel': 'rbf'}
cm(y_test,grid_predictions)
plot_roc(y_test,grid_predictions)

