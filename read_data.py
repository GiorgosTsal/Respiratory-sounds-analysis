import pandas as pd
import numpy as np
import os

from IPython import get_ipython
import matplotlib.pyplot as plt 

# Play an audio file
import wave

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
base_path = "/home/gtsal/Desktop/Machine learning/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/"
dirs = os.listdir( base_path )

# This would print all the files and directories
for file in dirs:
   print (file)
   
   
# Install the pydub library => conda install -c conda-forge pydub
   
# We will listen to this file:
# 101_1b1_Al_sc_Meditron.wav


audio_file = '101_1b1_Al_sc_Meditron.wav'   
audio_path = base_path + "/audio_and_txt_files/" + audio_file

waveFile = wave.open(audio_path, 'r')

