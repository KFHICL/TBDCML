#####################################################################
# Description
#####################################################################
'''
This script can be used to generate a summary of all the results in 
the results folder

Note there are manual exceptions due to formatting inconsistencies - 
modify as appropriate


Inputs:
Nonte

Ouputs:
csv file with RMSE scores

'''
#####################################################################
# Imports
#####################################################################
# %%
import sys

import os
import glob
import random
import time
import math
import datetime
import shutil
import json
import scipy
import tensorflow as tf # TODO If we want to load models saved on the HPC we need to use legacy keras 2. See test.py for import of this 
import sklearn
from sklearn import preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import argparse
import csv
#####################################################################
# Settings
#####################################################################
# %%
plt.style.use("seaborn-v0_8-colorblind")

# %%
resultFolder = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults'
summaryFolder = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\RMSESummaries'


ts = datetime.date.today().strftime('%d%m%Y')+str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
summaryPath = 'summary_{ts}.csv'.format(ts=ts) # Timestamped output file
summaryPath = os.path.join(summaryFolder,summaryPath)

resultFolderList = os.listdir(resultFolder) # The first repeat of the sweep contains the files to plot

jobNames = []
repeatNums = []
modelIDs = []
trainingRMSEs = []
valRMSEs = []

for jobname in resultFolderList:
    if any(x in jobname for x in ['TEST','repeat','crossVal2105']): # Skip test jobs and old formatted jobs
        continue
    curName,curRepeat = jobname.split('_') # Name of job currently open and repeat
    curPath = os.path.join(os.path.join(resultFolder,jobname),'dataout') # Path to result folder
    

    for fname in os.listdir(curPath):
        if 'RMSE' in fname:
            curFile = os.path.join(curPath,fname) # Path to result folder
            with open(curFile) as json_file:
                RMSEvalue = json.load(json_file)

            modelID = fname.split('_')[-1].split('.')[0] # Get model ID
            jobNames += [curName]
            repeatNums += [curRepeat]
            modelIDs += [modelID]

            if '_val_' in fname:
                valRMSEs += [RMSEvalue]
                trainingRMSEs += [np.nan]
            else:
                trainingRMSEs += [RMSEvalue]
                valRMSEs += [np.nan]


df = pd.DataFrame({'jobName': jobNames, 'repeat': repeatNums, 'modelID': modelIDs, 'trainingRMSE': trainingRMSEs, 'valRMSE': valRMSEs})
df.to_csv(summaryPath, index=True)  

