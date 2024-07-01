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
import warnings

import argparse

#####################################################################
# Settings
#####################################################################

# Set matplotlib style
plt.style.use("seaborn-v0_8-colorblind")

sampleNum = 1 # Choose the sample (out of 100) to be plotted

trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain' # Path for training data samples
numSamples = len(os.listdir(trainDat_path)) # number of samples is number of files in datain
sampleShape = [55,20]
warnings.warn("Warning: if displaying data generated prior to 14.06.2024 the comparison will be between ALL DATA and validation data even if TRAINING DATA is displayed")
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



if jobName == 'TESTJOB':
    resultPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\dataoutTESTJOB'
    jobIndex = 1
else:
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
print('RMSE for training set = ' + str(RMSE.result().numpy()))
if groundTruth_val is not None:
    RMSE_val = tf.keras.metrics.RootMeanSquaredError()
    RMSE_val.update_state(groundTruth_val,prediction_val)
    print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))

# Maximum failure index location, Shape: [samples, 55, 20]
maxGT = np.zeros(groundTruth.shape[0])
maxGTIdx = np.zeros((groundTruth.shape[0],2))
maxGTCoords = np.zeros((groundTruth.shape[0],2))

if groundTruth_val is not None:
    maxGT_val = np.zeros(groundTruth_val.shape[0])
    maxGTIdx_val = np.zeros((groundTruth_val.shape[0],2))
    maxGTCoords_val = np.zeros((groundTruth_val.shape[0],2))

maxPred = np.zeros(prediction.shape[0])
maxPredIdx = np.zeros((prediction.shape[0],2))
maxPredCoords = np.zeros((prediction.shape[0],2))

if groundTruth_val is not None:
    maxPred_val = np.zeros(prediction_val.shape[0])
    maxPredIdx_val = np.zeros((prediction_val.shape[0],2))
    maxPredCoords_val = np.zeros((prediction_val.shape[0],2))

# Ground truth training samples
for i in range(groundTruth.shape[0]):
    maxGT[i] = np.max(groundTruth[i,:,:]) # Maximum true value
    maxGTIdx[i] = np.unravel_index(groundTruth[i].argmax(), groundTruth[i].shape)[:2] # 2D index of maximum value in ground truth
    maxGTCoords[i] = grid[0,int(maxGTIdx[i,0]),int(maxGTIdx[i,1])],grid[1,int(maxGTIdx[i,0]),int(maxGTIdx[i,1])] # Coordinate of maximum FI value

# Ground truth validation samples
if groundTruth_val is not None:
    for i in range(groundTruth_val.shape[0]):
        maxGT_val[i] = np.max(groundTruth_val[i,:,:]) # Maximum true value
        maxGTIdx_val[i] = np.unravel_index(groundTruth_val[i].argmax(), groundTruth_val[i].shape)[:2] # 2D index of maximum value in ground truth
        maxGTCoords_val[i] = grid[0,int(maxGTIdx_val[i,0]),int(maxGTIdx_val[i,1])],grid[1,int(maxGTIdx_val[i,0]),int(maxGTIdx_val[i,1])] # Coordinate of maximum FI value

# Prediction training samples
for i in range(prediction.shape[0]):
    maxPred[i] = np.max(prediction[i,:,:]) # Maximum true value
    maxPredIdx[i] = np.unravel_index(prediction[i].argmax(), prediction[i].shape)[:2] # 2D index of maximum value in ground truth
    maxPredCoords[i] = grid[0,int(maxPredIdx[i,0]),int(maxPredIdx[i,1])],grid[1,int(maxPredIdx[i,0]),int(maxPredIdx[i,1])] # Coordinate of maximum FI value

# Prediction validation samples
if groundTruth_val is not None:
    for i in range(prediction_val.shape[0]):
        maxPred_val[i] = np.max(prediction_val[i,:,:]) # Maximum true value
        maxPredIdx_val[i] = np.unravel_index(prediction_val[i].argmax(), prediction_val[i].shape)[:2] # 2D index of maximum value in ground truth
        maxPredCoords_val[i] = grid[0,int(maxPredIdx_val[i,0]),int(maxPredIdx_val[i,1])],grid[1,int(maxPredIdx_val[i,0]),int(maxPredIdx_val[i,1])] # Coordinate of maximum FI value

# Format Mx FI into dataframes
maxPredDf = pd.DataFrame(maxPred.reshape(-1))
maxPredDf.columns = ['Failure Index']
maxPredDf['Specimens']='Training data'
maxPredDf['Prediction or GT']='Prediction'

if groundTruth_val is not None:
    maxPredDf_val = pd.DataFrame(maxPred_val.reshape(-1))
    maxPredDf_val.columns = ['Failure Index']
    maxPredDf_val['Specimens']='Validation data'
    maxPredDf_val['Prediction or GT']='Prediction'

maxGTDf = pd.DataFrame(maxGT.reshape(-1))
maxGTDf.columns = ['Failure Index']
maxGTDf['Specimens']='Training data'
maxGTDf['Prediction or GT']='Ground Truth'

if groundTruth_val is not None:
    maxGTDf_val = pd.DataFrame(maxGT_val.reshape(-1))
    maxGTDf_val.columns = ['Failure Index']
    maxGTDf_val['Specimens']='Validation data'
    maxGTDf_val['Prediction or GT']='Ground Truth'

if groundTruth_val is not None:
    MaxFIDf = pd.concat([maxPredDf, maxPredDf_val, maxGTDf, maxGTDf_val])
# print(maxPredDf_val.head)


# Calculate maximum error in distance and print these
x_error = maxGTCoords[:,0] - maxPredCoords[:,0]
y_error = maxGTCoords[:,1] - maxPredCoords[:,1]
errorDist = np.sqrt(np.square(x_error) + np.square(y_error))
if groundTruth_val is not None:
    x_error_val = maxPredCoords_val[:,0] - maxGTCoords_val[:,0]
    y_error_val = maxPredCoords_val[:,1] - maxGTCoords_val[:,1]
    errorDist_val = np.sqrt(np.square(x_error_val) + np.square(y_error_val))

# Format into dataframe
ErrDistDf = pd.Series(errorDist.reshape(-1), name='Training data')
if groundTruth_val is not None:
    ErrDistDf_val = pd.Series(errorDist_val.reshape(-1), name='Validation data')
    ErrorDistDf = pd.concat([ErrDistDf, ErrDistDf_val], axis=1)


# Calculate errors in value of maximum point
maxPointError = maxPred[:] - maxGT[:]
if groundTruth_val is not None:
    maxPointError_val = maxPred_val[:] - maxGT_val[:]
# Format into dataframe
maxErrDf = pd.Series(maxPointError.reshape(-1), name='Training data')
if groundTruth_val is not None:
    maxErrDf_val = pd.Series(maxPointError_val.reshape(-1), name='Validation data')
    maxPointErrorDf = pd.concat([maxErrDf, maxErrDf_val], axis=1)


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
        ha='right'
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
plt.title('Model training history')
plt.grid()
plt.ylabel('Loss')
plt.xlabel('Training Epoch')
plt.legend(['Training Data', 'Validation Data'], loc='upper right')


# Prediction vs ground truth error distribution
absErr = np.array(prediction) - np.array(groundTruth)
absErrDfFlat = pd.DataFrame(absErr.reshape(-1))
absErrDfFlat.columns = ['Error']
absErrDfFlat['Specimens']='Training data'

if groundTruth_val is not None:
    absErr_val = np.array(prediction_val) - np.array(groundTruth_val)
    absErrDfFlat_val = pd.DataFrame(absErr_val.reshape(-1))
    absErrDfFlat_val.columns = ['Error']
    absErrDfFlat_val['Specimens']='Validation data'
    absErrorDf = pd.concat([absErrDfFlat, absErrDfFlat_val])

ax = plt.subplot(2,totalCols,(9,10))
sns.histplot(absErrDfFlat)
plt.grid()
plt.xlabel('Absolute Error')
plt.title('Distribution of absolute errors')
plt.legend([],[], frameon=False)

# Plot prediction vs actual field

ax = plt.subplot(2,totalCols,11) # Ground truth
CS = ax.contourf(grid[0],grid[1],groundTruth[sampleNum,:,:].reshape(groundTruth.shape[1],-1))
plt.title('ground truth')

ax = plt.subplot(2,totalCols,12) # Prediction
CS2 = ax.contourf(grid[0],grid[1],prediction[sampleNum,:,:].reshape(prediction.shape[1],-1),  levels = CS.levels)
fig.colorbar(CS2)
plt.title('prediction')

fig.get_constrained_layout



# # Save grid for use in other scripts
# gridout = [grid[0].tolist(),grid[1].tolist()]
# gridPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json'
# with open(gridPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(gridout, f, indent=2)




#####################################################################
# Misc Plots
#####################################################################

# Ground truth vs prediction figure
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(300*px, 300*px), layout="constrained")
plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis

ax = plt.subplot(1,2,1) # Ground truth
CS = ax.contourf(grid[0],grid[1],groundTruth[sampleNum,:,:].reshape(groundTruth.shape[1],-1))
plt.title('Ground truth')
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(1,2,2) # Prediction
CS2 = ax.contourf(grid[0],grid[1],prediction[sampleNum,:,:].reshape(prediction.shape[1],-1),  levels = CS.levels)

cbar = fig.colorbar(CS2)
cbar.ax.set_ylabel('Failure Index')

ax.set_xticks([])
ax.set_yticks([])
plt.title('Prediction')

fig.get_constrained_layout

# Max FI point error distribution figure
if groundTruth_val is not None:
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
    totalCols = 2 # figure columns
    plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis

    # Plot prediction error distributions for maximum FI point
    ax = plt.subplot(2,totalCols,1)
    sns.kdeplot(maxPointErrorDf, ax=ax)
    plt.grid()
    plt.xlabel('FI error')
    plt.title('Error distribution of peak failure index prediction')


    # Plot distribution of error distance to mx FI
    ax = plt.subplot(2,totalCols,2)
    bins = np.arange(-5, 150, 10)
    sns.histplot(ErrorDistDf, ax=ax, bins = bins)
    plt.grid()
    plt.xlabel('Distance error')
    plt.title('Distribution of distance between predicted FI peak and actual FI peak')

    # Plot prediction error distributions for all points
    ax = plt.subplot(2,totalCols,3)
    # print(absErrorDf)
    g = sns.kdeplot(absErrorDf, ax=ax,x='Error', hue = 'Specimens')
    g.legend_.set_title(None)
    plt.grid()
    plt.xlabel('FI error')
    plt.title('Error distribution of all failure index predictions')
    print('median of training error ' + str(np.percentile(absErr,50)))
    print('median of val error ' + str(np.percentile(absErr_val,50)))
    print('95 percent of all training error fall between ' + str(np.percentile(absErr,2.5)) + ' and ' + str(np.percentile(absErr,97.5)))
    print('95 percent of all validation error fall between ' + str(np.percentile(absErr_val,2.5)) + ' and ' + str(np.percentile(absErr_val,97.5)))
    print('95 percent of all training error is below an absolute value of ' + str(np.percentile(np.absolute(absErr),95)))
    print('95 percent of all validation error is below an absolute value of ' + str(np.percentile(np.absolute(absErr_val),95)))
    print('The maximum training error is '+ str(np.max(np.absolute(absErr))))
    print('The maximum validation error is '+ str(np.max(np.absolute(absErr_val))))
    
    print('The mean true training failure index is '+str(np.mean(np.array(groundTruth))))
    print('The mean true validation failure index is '+str(np.mean(np.array(groundTruth_val))))
    print('The maximum true training failure index is '+str(np.max(np.array(groundTruth))))
    print('The maximum true validation failure index is '+str(np.max(np.array(groundTruth_val))))

    # Plot of Max FI prediction vs ground truth
    ax = plt.subplot(2,totalCols,4)
    bins = np.arange(-5, 150, 10)
    sns.barplot(MaxFIDf, x="Prediction or GT", y="Failure Index", hue="Specimens", capsize=.3, gap=.1, linewidth=1, edgecolor="0", err_kws={"color": "0", "linewidth": 1}, width=.5)
    plt.grid()
    plt.xlabel('Failure Index')
    plt.title('Maximum failure index of all specimens')


# Input data figure
# Reminder: the header index is the following
# [   0        1         2       3     4     5     6     7     8      9      10   11    12    13  ]
# ['label' 'x_coord' 'y_coord' 'e11' 'e22' 'e12' 'S11' 'S22' 'S12' 'SMises' 'FI' 'E11' 'E22' 'E12']
Eyy = inputDat[sampleNum,:,:,12]
eyy = inputDat[sampleNum,:,:,4]
FI = inputDat[sampleNum,:,:,10]
Exx = inputDat[sampleNum,:,:,11]
Gxy = inputDat[sampleNum,:,:,13]
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(300*px, 300*px), layout="constrained")
plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis

ax = plt.subplot(1,3,1) # E_xx
CS = ax.contourf(grid[0],grid[1],Exx/(1000))
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Stiffness [GPa]')
plt.title('E_xx')
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(1,3,2) # E_yy
CS2 = ax.contourf(grid[0],grid[1],Eyy/1000)
cbar = fig.colorbar(CS2)
cbar.ax.set_ylabel('Stiffness [GPa]')
ax.set_xticks([])
ax.set_yticks([])
plt.title('E_yy')

ax = plt.subplot(1,3,3) # Gxy
CS3 = ax.contourf(grid[0],grid[1],Gxy/1000)
cbar = fig.colorbar(CS3)
cbar.ax.set_ylabel('Stiffness [GPa]')
ax.set_xticks([])
ax.set_yticks([])
plt.title('G_xy')

fig.get_constrained_layout






plt.show()
