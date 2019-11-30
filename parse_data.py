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

get_ipython().run_line_magic('matplotlib', 'inline')
base_path = '/home/gtsal/Desktop/Machine learning/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/'
data_path = base_path + '/audio_and_txt_files/'
dirs = os.listdir( data_path )

# This would print all the files and directories
for file in dirs:
   print (file)
    
# Install the pydub library 
#$ conda install -c conda-forge pydub
#$ sudo apt-get install ffmpeg
 
# Will listen to this file:
# 101_1b1_Al_sc_Meditron.wav

audio_file = '101_1b1_Al_sc_Meditron.wav'   
audio_path = base_path + "/audio_and_txt_files/" + audio_file

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

df_diag = pd.read_csv(patient_diagnosis, header=None, names=diagn_col_names)
df_diag.head(10)

file_format_data = open(filename_format, 'r').read()

#Display the contents of one annotation .txt file
annotation = data_path + '101_1b1_Al_sc_Meditron.txt'
print(annotation)
anot_col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']

# Respiratory cycle column values are in 'seconds'.
# Presence = 1
# Absence = 0
df_annot = pd.read_csv(annotation, sep="\t", header=None, names=anot_col_names)
df_annot.head(10)
print(df_annot)



