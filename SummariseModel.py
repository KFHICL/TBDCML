
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



# Add arguments for parallel running and training of several different models 
argParser = argparse.ArgumentParser()

argParser.add_argument("-j", "--jobname", help="Name of job to summarise") # parameter to allow parallel running on the HPC
argParser.add_argument("-rp", "--repeat", help="Number of repeats done of job") # parameter to allow parallel running on the HPC
argParser.add_argument("-i", "--jobindex", help="Index of job to summarise") # parameter to allow parallel running on the HPC

args = argParser.parse_args()
jobName = args.jobname
repeat = args.repeat
jobIndex = int(args.jobindex)

resultPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults'
resultPath = os.path.join(resultPath, '{jn}{rp}'.format(rp = repeat, jn=jobName),'dataout')

histOutName = 'trainHist_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Naming of file out
histOutPath = os.path.join(resultPath,histOutName)
predOutName = 'predictions_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Naming of file out
predOutPath = os.path.join(resultPath,predOutName)
gtOutName = 'groundTruth_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Naming of file out
gtOutPath = os.path.join(resultPath,gtOutName)
paramOutName = 'parameters_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Naming of file out
paramOutPath = os.path.join(resultPath,paramOutName)
inputOutName = 'input_{jn}{rp}_{num}.json'.format(rp = repeat,jn=jobName, num = jobIndex) # Naming of file out
inputOutPath = os.path.join(resultPath,inputOutName)


with open(histOutPath) as json_file: # load into dict
    history = json.load(json_file)

with open(paramOutPath) as json_file: # load into dict
    parameters = json.load(json_file)
    parameters = json.loads(parameters)

with open(predOutPath) as json_file: # load into dict
    prediction = np.array(json.load(json_file))

with open(gtOutPath) as json_file: # load into dict
    groundTruth = np.array(json.load(json_file))

with open(inputOutPath) as json_file: # load into dict
    inputDat = np.array(json.load(json_file))
# print(groundTruth.shape)
# print(prediction.shape)
RMSE = tf.keras.metrics.RootMeanSquaredError() # CNN
RMSE.update_state(groundTruth,prediction)

# print('RMSE  = ' + str(RMSE.result().numpy()))


# Intantiate figure
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
# fig = plt.figure(figsize=(1200*px, 800*px), constrained_layout=True)
fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
totalCols = 6 # figure columns

# Table of parameters for the given model
ax = plt.subplot(1,totalCols,(1,2))
ncols = 2
nrows = len(parameters)
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_axis_off()

for y in range(0, nrows):
    ax.annotate(
        xy=(0,y),
        text=list(parameters.keys())[y],
        ha='left'
    )
    ax.annotate(
        xy=(1,y),
        text=list(parameters.values())[y],
        ha='left'
    )

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

# Prediction vs ground truth value plot
ax = plt.subplot(2,totalCols,(3,4))
ax.scatter(x = groundTruth, y = prediction, marker = ".")
ax.plot([0,4],[0,4],'r')
plt.xlim([np.min(groundTruth),np.max(groundTruth)])
plt.ylim([np.min(prediction),np.max(prediction)])
# ax.set_aspect('equal')
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


# Plot prediction vs actual error
# absErr = pd.DataFrame((yFI_actual - predCNNFI_nonStnd).reshape(100,-1))
absErr = np.array(groundTruth) - np.array(prediction)
absErrDfFlat = pd.DataFrame(absErr.reshape(-1))

ax = plt.subplot(2,totalCols,(9,10))
# sns.displot(absErrDfFlat, kind="kde")
sns.histplot(absErrDfFlat)
plt.grid()
plt.xlabel('Absolute Error')
plt.title('Distribution of aboslute errors')


# Plot prediction vs actual field
sampleNum = 1

grid = [inputDat[0,:,:,1],inputDat[0,:,:,2]] # Grid for plotting

ax = plt.subplot(2,totalCols,11)
CS = ax.contourf(grid[0],grid[1],groundTruth[sampleNum,:,:], cmap = 'jet')
# plt.xlabel('x')
# plt.ylabel('y')
# fig.colorbar(CS)
plt.title('ground truth')

ax = plt.subplot(2,totalCols,12)
CS2 = ax.contourf(grid[0],grid[1],prediction[sampleNum,:,:], cmap = 'jet', levels = CS.levels)
# plt.xlabel('x')
# plt.ylabel('y')
fig.colorbar(CS2)
plt.title('prediction')

fig.get_constrained_layout


plt.show()


# # Save grid for use in other scripts
# gridout = [grid[0].tolist(),grid[1].tolist()]
# gridPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json'
# with open(gridPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(gridout, f, indent=2)



# # Plot prediction vs actual field
# sampleNum = 1

# grid = [inputDat[0,:,:,1],inputDat[0,:,:,2]] # Grid for plotting

# fig, axs = plt.subplots(1,2 ,figsize=[10, 5]) # Create subplots to fit output fields
# ax4 = plt.subplot(2,3,4)
# CS = ax1.contourf(grid[0],grid[1],groundTruth[sampleNum,:,:], cmap = 'jet')
# plt.xlabel('x')
# plt.ylabel('y')
# fig.colorbar(CS)
# plt.title('ground truth')

# ax5 = plt.subplot(2,3,5)
# CS2 = ax2.contourf(grid[0],grid[1],prediction[sampleNum,:,:], cmap = 'jet', levels = CS.levels)
# plt.xlabel('x')
# plt.ylabel('y')
# fig.colorbar(CS2)
# plt.title('prediction')



# plt.show()



