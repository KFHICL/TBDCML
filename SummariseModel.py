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
import matplotlib
import pandas as pd
import plotly

import argparse

#####################################################################
# Settings
#####################################################################

# Set matplotlib style
plt.style.use("seaborn-v0_8-colorblind")

sampleNum = 1 # Choose the sample (out of 100) to be plotted

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
# Performance measures calculation
#####################################################################

# Grid required for plotting
grid = np.array([inputDat[0,:,:,1],inputDat[0,:,:,2]])

# RMSE
RMSE = tf.keras.metrics.RootMeanSquaredError()
RMSE.update_state(groundTruth,prediction)
print('RMSE  = ' + str(RMSE.result().numpy()))
if groundTruth_val is not None:
    RMSE_val = tf.keras.metrics.RootMeanSquaredError()
    RMSE_val.update_state(groundTruth_val,prediction_val)
    print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))

# Maximum failure index location, Shape: [samples, 55, 20]
maxGT = np.zeros(groundTruth.shape[0])
maxGTIdx = np.zeros((groundTruth.shape[0],2))
maxGTCoords = np.zeros((groundTruth.shape[0],2))

maxGT_val = np.zeros(groundTruth_val.shape[0])
maxGTIdx_val = np.zeros((groundTruth_val.shape[0],2))
maxGTCoords_val = np.zeros((groundTruth_val.shape[0],2))

maxPred = np.zeros(prediction.shape[0])
maxPredIdx = np.zeros((prediction.shape[0],2))
maxPredCoords = np.zeros((prediction.shape[0],2))

maxPred_val = np.zeros(prediction_val.shape[0])
maxPredIdx_val = np.zeros((prediction_val.shape[0],2))
maxPredCoords_val = np.zeros((prediction_val.shape[0],2))

# Ground truth all samples
for i in range(groundTruth.shape[0]):
    maxGT[i] = np.max(groundTruth[i,:,:]) # Maximum true value
    maxGTIdx[i] = np.unravel_index(groundTruth[i].argmax(), groundTruth[i].shape) # 2D index of maximum value in ground truth
    maxGTCoords[i] = grid[0,int(maxGTIdx[i,0]),int(maxGTIdx[i,1])],grid[1,int(maxGTIdx[i,0]),int(maxGTIdx[i,1])] # Coordinate of maximum FI value

# Ground truth validation samples
for i in range(groundTruth_val.shape[0]):
    maxGT_val[i] = np.max(groundTruth_val[i,:,:]) # Maximum true value
    maxGTIdx_val[i] = np.unravel_index(groundTruth_val[i].argmax(), groundTruth_val[i].shape) # 2D index of maximum value in ground truth
    maxGTCoords_val[i] = grid[0,int(maxGTIdx_val[i,0]),int(maxGTIdx_val[i,1])],grid[1,int(maxGTIdx_val[i,0]),int(maxGTIdx_val[i,1])] # Coordinate of maximum FI value

# Prediction all samples
for i in range(prediction.shape[0]):
    maxPred[i] = np.max(prediction[i,:,:]) # Maximum true value
    maxPredIdx[i] = np.unravel_index(prediction[i].argmax(), prediction[i].shape) # 2D index of maximum value in ground truth
    maxPredCoords[i] = grid[0,int(maxPredIdx[i,0]),int(maxPredIdx[i,1])],grid[1,int(maxPredIdx[i,0]),int(maxPredIdx[i,1])] # Coordinate of maximum FI value

# Prediction validation samples
for i in range(prediction_val.shape[0]):
    maxPred_val[i] = np.max(prediction_val[i,:,:]) # Maximum true value
    maxPredIdx_val[i] = np.unravel_index(prediction_val[i].argmax(), prediction_val[i].shape) # 2D index of maximum value in ground truth
    maxPredCoords_val[i] = grid[0,int(maxPredIdx_val[i,0]),int(maxPredIdx_val[i,1])],grid[1,int(maxPredIdx_val[i,0]),int(maxPredIdx_val[i,1])] # Coordinate of maximum FI value


# Calculate maximum error in distance and print these
x_error = maxGTCoords[:,0] - maxPredCoords[:,0]
y_error = maxGTCoords[:,1] - maxPredCoords[:,1]
errorDist = np.sqrt(np.square(x_error) + np.square(y_error))
x_error_val = maxPredCoords_val[:,0] - maxGTCoords_val[:,0]
y_error_val = maxPredCoords_val[:,1] - maxGTCoords_val[:,1]
errorDist_val = np.sqrt(np.square(x_error_val) + np.square(y_error_val))

# Format into dataframe
ErrDistDf = pd.Series(errorDist.reshape(-1), name='All data')
ErrDistDf_val = pd.Series(errorDist_val.reshape(-1), name='Validation data')
ErrorDistDf = pd.concat([ErrDistDf, ErrDistDf_val], axis=1)

# print([str(max(x_error))])
# print([str(max(y_error))])
# print([str(max(x_error_val))])
# print([str(max(y_error_val))])

# Calculate errors in value of maximum point
maxPointError = maxPred[:] - maxGT[:]
maxPointError_val = maxPred_val[:] - maxGT_val[:]
# Format into dataframe
maxErrDf = pd.Series(maxPointError.reshape(-1), name='All data')
maxErrDf_val = pd.Series(maxPointError_val.reshape(-1), name='Validation data')
maxPointErrorDf = pd.concat([maxErrDf, maxErrDf_val], axis=1)

# print([str(max(maxPointError))])
# print([str(max(maxPointError_val))])

# Intantiate figure
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
totalCols = 2 # figure columns
plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis

# Plot prediction error distributions for maximum FI point
ax = plt.subplot(2,totalCols,1)
sns.kdeplot(maxPointErrorDf, ax=ax)
plt.grid()
plt.xlabel('FI error')
plt.title('Distribution of difference between predicted max and actual max FI')
# plt.legend([],[], frameon=False)

# Plot 
ax = plt.subplot(2,totalCols,2)
bins = np.arange(-5, 150, 10)
sns.histplot(ErrorDistDf, ax=ax, bins = bins)
plt.grid()
plt.xlabel('Distance error')
plt.title('Distribution of error distance to max FI point')
# plt.legend([],[], frameon=False)

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




