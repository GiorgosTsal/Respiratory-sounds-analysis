#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:34:48 2019

@author: panagiotis
"""
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib.pyplot import specgram
#%matplotlib inline
import cnnsound as cns

from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_data(file):
    df_no_diagnosis = pd.read_csv('demographic_info.txt', names = 
                 ['Patient number', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)'],
                 delimiter = ' ')
    diagnosis = pd.read_csv(file+'patient_diagnosis.csv',
                        names = ['Patient number', 'Diagnosis'])
    

    df =  df_no_diagnosis.join(diagnosis.set_index('Patient number'),
                           on = 'Patient number', how = 'left')
    root =file+'audio_and_txt_files/'
    filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
    i_list = []
    rec_annotations = []
    rec_annotations_dict = {}
    for s in filenames:
        (i,a) = Extract_Annotation_Data(s, root)
        i_list.append(i)
        rec_annotations.append(a)
        rec_annotations_dict[s] = a
    recording_info = pd.concat(i_list, axis = 0)
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
    file_label_df = pd.DataFrame(data = {'filename':filename_list, 'no label':no_label_list,
                                         'crackles only':crack_list,
                                         'wheezes only':wheeze_list,
                                         'crackles and wheezees':both_sym_list})
    w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (file_label_df['wheezes only'] != 0) | (file_label_df['crackles and wheezees'] != 0)]

    return df,recording_info,w_labels,
    
def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)


def load_sound_files(file_paths):
    raw_sounds = []
    raw_sr = []
    for fp in file_paths:
        sr,X = cns.read_wav_file(fp,22050)
        raw_sr.append(sr)
        raw_sounds.append(X)
    return raw_sounds,raw_sr

def plot_waves(raw_sounds,raw_sr):
    i = 1
    #fig = plt.figure(figsize=(8,20))
            #plt.subplot(1,1,i)
    fig = plt.figure(figsize=(20,20))
    librosa.display.waveplot(raw_sounds,sr=raw_sr)
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=7)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds,raw_sr):
    i = 1
    #fig = plt.figure(figsize=(8,20))
    for n,f,s in zip(sound_names,raw_sounds,raw_sr):
        #plt.subplot(len(sound_names),1,i)
        fig = plt.figure(figsize=(20,20))
        specgram(np.array(f), Fs=s)
        plt.title(n.title(),fontsize=12)
        plt.savefig('plots/spectrograms/'+str(100+i)+'_'+n+'_SpecGram.png')  
        plt.close()     
        i += 1

    #plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=7)
   # plt.show()

def plot_log_power_specgram(raw_sounds,raw_sr):
    #fig = plt.figure(figsize=(8,20))
    fig = plt.figure(figsize=(20,20))
    D = librosa.core.amplitude_to_db(np.abs(librosa.stft(raw_sounds))**2, ref=np.max)
    librosa.display.specshow(D,x_axis='time' ,y_axis='log')
    del D    
#    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=7)
    plt.show()

def extract_feature(X,sample_rate):
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    result=mfccs.copy()
    result=np.hstack((result,chroma,mel,contrast,tonnetz))
    #data.append([mfccs,chroma,mel,contrast,tonnetz])
    return result
    #return data

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode