#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 08:59:17 2019

@author: gtsal
"""

import pandas as pd
import numpy as np
import os

from IPython import get_ipython
import matplotlib.pyplot as plt 

# Play an audio file
from pydub import AudioSegment
from pydub.playback import play

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

import soundfile as sf


# Define helper functions

# Load a .wav file. 
# These are 24 bit files. The PySoundFile library is able to read 24 bit files.
# https://pysoundfile.readthedocs.io/en/0.9.0/

def get_wav_info(wav_file):
    data, rate = sf.read(wav_file)
    return data, rate

def graph_spectrogram(wav_file):
    data, rate = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx




get_ipython().run_line_magic('matplotlib', 'inline')
base_path = '/home/gtsal/Desktop/Machine learning/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/'
data_path = base_path + '/audio_and_txt_files/'
dirs = os.listdir( data_path )

# This would print all the files and directories
#for file in dirs:
  # print (file)
    
# Install the pydub library 
#$ conda install -c conda-forge pydub
#$ sudo apt-get install ffmpeg
 
# Will listen to this file:
# 101_1b1_Al_sc_Meditron.wav

audio_file = '101_1b1_Al_sc_Meditron.wav'   
audio_file_1 = '110_1p1_Al_sc_Meditron.wav'   
audio_path = base_path + "/audio_and_txt_files/" + audio_file
audio_path_1 = base_path + "/audio_and_txt_files/" + audio_file_1

song = AudioSegment.from_wav(audio_path)
#play(song)

# Read the Files
 
demographic_info = base_path + 'demographic_info.txt'
patient_diagnosis = base_path + 'patient_diagnosis.csv'
filename_format = base_path + 'filename_format.txt'

# Adult BMI (kg/m2)
# Child Weight (kg)
# Child Height (cm)
demo_col_names = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height'] 

diagn_col_names = ['patient_id','diagnosis']



df_demo = pd.read_csv(demographic_info, sep=" ", header=None, names=demo_col_names)
df_demo.head(10)
print(df_demo)

df_diag = pd.read_csv(patient_diagnosis, header=None, names=diagn_col_names)
df_diag.head(10)
print(df_diag)

file_format_data = open(filename_format, 'r').read()

#Display the contents of one annotation .txt file
annotation = data_path + '101_1b1_Al_sc_Meditron.txt'
print(annotation)
anot_col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']

# Respiratory cycle column values are in 'seconds'.
# Presence = 1
# Absence = 0
df_annot = pd.read_csv(annotation, sep="\t", header=None, names=anot_col_names)
df_annot.head(20)
print(df_annot)

# Install the pysoundfile library
#$ conda install -c conda-forge pysoundfile

# plot the spectrogram

# Time is on the x axis and Frequencies are on the y axis.
# The intensity of the different colours shows the amount of energy i.e. how loud the sound is,
# at different frequencies, at different times


both = graph_spectrogram(audio_path_1)
healthy = graph_spectrogram(audio_path)

