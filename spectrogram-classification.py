#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:13:04 2019

@author: gtsal
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import glob as gb
import matplotlib.pyplot as plt
import gc
import os, glob, wave
import ntpath

sound_dir_loc=np.array(gb.glob("/home/gtsal/Desktop/Machine learning/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/*.wav"))

data_path = '/home/gtsal/Desktop/Machine learning/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/'
audio_path = data_path + 'audio_and_txt_files/'


dirs = os.listdir(audio_path)

    

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
    


def build_spectogram(file_path,sound_file_name, path):
    plt.interactive(False)
    file_audio_series,sr = librosa.load(file_path,sr=None)    
    spec_image = plt.figure(figsize=[2,2])
    ax = spec_image.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectogram = librosa.feature.melspectrogram(y=file_audio_series, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
    image_name  =  path + sound_file_name + '.jpg'
    plt.savefig(image_name, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    spec_image.clf()
    plt.close(spec_image)
    plt.close('all')
    del file_path,sound_file_name,file_audio_series,sr,spec_image,ax,spectogram
    
    

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)

filenames = [s.split('.')[0] for s in os.listdir(path = audio_path)]



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

    if no_labels!=0:
        print('healthy man...')
        directory = data_path + 'healthy/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory + " created succesfully...")
        else:
            print(directory)
            build_spectogram(data_path + 'audio_and_txt_files/' + f + '.wav', f, directory)
    elif n_crackles!=0:
        print('crackles man...')
        directory = data_path + 'crackles'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory + " created succesfully...")
            
    elif n_wheezes!=0:
        print('wheezes man...')
        directory = data_path + 'wheezes'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory + " created succesfully...")
       
    elif both_sym!=0:
        print('both man...')
        directory = data_path + 'both'
        if not os.path.exists(directory):

            os.makedirs(directory)
            print(directory + " created succesfully...")
       
    
file_label_df = pd.DataFrame(data = {'filename':filename_list, 'no label':no_label_list, 'crackles only':crack_list, 'wheezes only':wheeze_list, 'crackles and wheezees':both_sym_list})
    
w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]
file_label_df.sum()   
print(file_label_df.sum())
