#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:45:34 2019

CNN: Detection of wheezes and crackles

@author: gtsal
"""
#import utilities as utils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.expand_frame_repr', False) #to print all columns

data_path = '/home/gtsal/Desktop/Machine learning/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/'
audio_path = data_path + 'audio_and_txt_files/'

dirs = os.listdir(data_path)
#for file in dirs:
#   print (file)
   
   
   
demo_head =  ['Patient number', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)']
df_no_diagnosis = pd.read_csv(data_path + 'demographic_info.txt', names = demo_head, delimiter = ' ')
  
diagnosis_head = ['Patient number', 'Diagnosis']
diagnosis = pd.read_csv(data_path + 'patient_diagnosis.csv', names = diagnosis_head)

df_diag =  df_no_diagnosis.join(diagnosis.set_index('Patient number'), on = 'Patient number', how = 'left')
df_diag['Diagnosis'].value_counts()

df_diag.groupby('Diagnosis')['Patient number'].nunique().plot(kind='bar')
#plt.show()

print (df_diag['Diagnosis'].value_counts())


filenames = [s.split('.')[0] for s in os.listdir(path = audio_path) if '.txt' in s]


def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)

i_list = []
rec_annotations = []
rec_annotations_dict = {}
for s in filenames:
    (i,a) = Extract_Annotation_Data(s, audio_path)
    i_list.append(i)
    rec_annotations.append(a)
    rec_annotations_dict[s] = a
recording_info = pd.concat(i_list, axis = 0)
recording_info.head()
#pd.set_option('display.max_rows', recording_info.shape[0]+1)   #Code to set the property display.max_rows to just more than total rows

no_label_list = []
crack_list = []
wheeze_list = []
both_sym_list = []
filename_list = []
for f in filenames:
    d = rec_annotations_dict[f]
    no_labels = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
    n_crackles = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
    n_wheezes = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 1)].index)
    both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
    no_label_list.append(no_labels)
    crack_list.append(n_crackles)
    wheeze_list.append(n_wheezes)
    both_sym_list.append(both_sym)
    filename_list.append(f)
    
file_label_df = pd.DataFrame(data = {'filename':filename_list, 'no label':no_label_list, 'crackles only':crack_list, 'wheezes only':wheeze_list, 'crackles and wheezees':both_sym_list})

#Distribution of data classes
w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]
file_label_df.sum()
print (file_label_df.sum())



duration_list = []
for i in range(len(rec_annotations)):
    current = rec_annotations[i]
    duration = current['End'] - current['Start']
    duration_list.extend(duration)

duration_list = np.array(duration_list)
plt.hist(duration_list, bins = 50)
#print('longest cycle:{}'.format(max(duration_list)))
#print('shortest cycle:{}'.format(min(duration_list)))
threshold = 5
#print('Fraction of samples less than {} seconds:{}'.format(threshold, np.sum(duration_list < threshold)/len(duration_list)))

