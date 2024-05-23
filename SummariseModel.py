#####################################################################
# Description
#####################################################################
# This script provides and overview of the performance and training 
# behaviour of a single model in a sweep. The RMSE is calculated for
# all samples, and the results of the model are visualised in various 
# plots.


#####################################################################
# Imports
#####################################################################
import sys

import os
import random
import time
import math
import datetime
import shutil
import json
import scipy
import tensorflow as tf
import sklearn
from sklearn import preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly

import argparse

#####################################################################
# Settings
#####################################################################

# Set matplotlib style
plt.style.use("seaborn-v0_8-colorblind")

#####################################################################
# Input parsing and paths to training files
#####################################################################

# Add arguments for parallel running and training of several different models 
argParser = argparse.ArgumentParser()

argParser.add_argument("-j", "--jobname", help="Name of job to summarise") # parameter to allow parallel running on the HPC
argParser.add_argument("-rp", "--repeat", help="Number of repeats done of job") # parameter to allow parallel running on the HPC
argParser.add_argument("-i", "--jobindex", help="Index of job to summarise") # parameter to allow parallel running on the HPC

args = argParser.parse_args()
jobName = args.jobname # Jobname e.g. "sweep1403repeat"


if args.repeat is not None: # if there are several repeats (should usually be the case for training sweeps)
    repeat = args.repeat # repeat to summarise
else:
    repeat = ''


jobIndex = int(args.jobindex) # Index of the model to summarise (see sweep definition csv file for indices)

resultPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults'
resultPath = os.path.join(resultPath, '{jn}{rp}'.format(rp = repeat, jn=jobName),'dataout')

histOutName = 'trainHist_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Training history
histOutPath = os.path.join(resultPath,histOutName)
predOutName = 'predictions_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Predictions
predOutPath = os.path.join(resultPath,predOutName)
predOutName_val = 'predictions_val_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Predictions
predOutPath_val = os.path.join(resultPath,predOutName_val)
gtOutName = 'groundTruth_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Ground truths
gtOutPath = os.path.join(resultPath,gtOutName)
gtOutName_val = 'groundTruth_val_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Ground truths
gtOutPath_val = os.path.join(resultPath,gtOutName_val)
paramOutName = 'parameters_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Hyperparameters of model
paramOutPath = os.path.join(resultPath,paramOutName)
inputOutName = 'input_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Inputs for model
inputOutPath = os.path.join(resultPath,inputOutName)


with open(histOutPath) as json_file: # load into dict
    history = json.load(json_file)

with open(paramOutPath) as json_file: # load into dict
    parameters = json.load(json_file)
    parameters = json.loads(parameters)

with open(predOutPath) as json_file: # load into dict
    prediction = np.array(json.load(json_file))
    
if os.path.isfile(predOutPath_val):
    with open(predOutPath_val) as json_file: # load into dict
        prediction_val = np.array(json.load(json_file))
else:
    prediction_val = None
    print('No file for validation data predictions found')

with open(gtOutPath) as json_file: # load into dict
    groundTruth = np.array(json.load(json_file))

if os.path.isfile(gtOutPath_val):
    with open(gtOutPath_val) as json_file: # load into dict
        groundTruth_val = np.array(json.load(json_file))
else:
    groundTruth_val = None
    print('No file for validation data ground truth found')

with open(inputOutPath) as json_file: # load into dict
    inputDat = np.array(json.load(json_file))

#####################################################################
# RMSE (root mean squared error) calculation
#####################################################################

RMSE = tf.keras.metrics.RootMeanSquaredError()
RMSE.update_state(groundTruth,prediction)
print('RMSE  = ' + str(RMSE.result().numpy()))
if groundTruth_val is not None:
    RMSE_val = tf.keras.metrics.RootMeanSquaredError()
    RMSE_val.update_state(groundTruth_val,prediction_val)
    print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))

#####################################################################
# Model summary figure
#####################################################################

# Intantiate figure
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
totalCols = 6 # figure columns
plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme

# Table of hyperparameters for the given model
ax = plt.subplot(1,totalCols,(1,2)) # Axis definition and limits
ncols = 2
nrows = len(parameters)
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_axis_off()

for y in range(0, nrows): # Loop over all hyperparameters
    ax.annotate( # Name of hyperparameter
        xy=(0,y),
        text=list(parameters.keys())[y],
        ha='left'
    )
    ax.annotate( # Value of hyperparameter
        xy=(1,y),
        text=list(parameters.values())[y],
        ha='left'
    )

# Annotate with headers
ax.annotate( 
    xy=(0, nrows),
    text='Parameter',
    weight='bold',
    ha='left'
)
ax.annotate(
    xy=(1, nrows),
    text='Value',
    weight='bold',
    ha='left'
)

# Prediction vs ground truth scatter plot
ax = plt.subplot(2,totalCols,(3,4))
ax.scatter(x = groundTruth, y = prediction, marker = ".")
ax.plot([0,4],[0,4],color = '#04D8B2')
plt.xlim([np.min(groundTruth),np.max(groundTruth)])
plt.ylim([np.min(prediction),np.max(prediction)])
plt.title('Prediction vs ground truth')
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.grid()

# Training history plot
ax = plt.subplot(2,totalCols,(5,6))
ax.plot(history['loss'])
ax.plot(history['val_loss'])
plt.title('model loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')


# Prediction vs ground truth error distribution
absErr = np.array(groundTruth) - np.array(prediction)
absErrDfFlat = pd.DataFrame(absErr.reshape(-1))

ax = plt.subplot(2,totalCols,(9,10))
sns.histplot(absErrDfFlat)
plt.grid()
plt.xlabel('Absolute Error')
plt.title('Distribution of absolute errors')
plt.legend([],[], frameon=False)

# Plot prediction vs actual field
sampleNum = 1 # Choose the sample (out of 100) to be plotted
grid = [inputDat[0,:,:,1],inputDat[0,:,:,2]] # Grid for plotting

ax = plt.subplot(2,totalCols,11) # Ground truth
CS = ax.contourf(grid[0],grid[1],groundTruth[sampleNum,:,:])
plt.title('ground truth')

ax = plt.subplot(2,totalCols,12) # Prediction
CS2 = ax.contourf(grid[0],grid[1],prediction[sampleNum,:,:],  levels = CS.levels)
fig.colorbar(CS2)
plt.title('prediction')

fig.get_constrained_layout

plt.show()


# # Save grid for use in other scripts
# gridout = [grid[0].tolist(),grid[1].tolist()]
# gridPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json'
# with open(gridPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(gridout, f, indent=2)




