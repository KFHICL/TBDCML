#####################################################################
# Description
#####################################################################
# This script provides and overview of the performance and training 
# behaviour of a several models in a sweep. The RMSE is calculated 
# for all samples, and the results of the models are visualised in 
# various plots.

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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import argparse

#####################################################################
# Settings
#####################################################################
plt.style.use("seaborn-v0_8-colorblind")

trainEpochs = 500 # Maximum number of epochs for trained models
# Results:
resultPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults'
# File indicating which models to be plotted together and which hyperparameters they are sweeps of
# sweepIdxPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\compareIndex_fineTune_2.csv'
sweepIdxPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\crossVal2105\compareIndex_crossVal2105.csv'
baselineIdx = 1 # Index of reference model

# Legacy method of grouping models and displaying them next to each other
# initialLearnRateSweeps = np.linspace(2,4,4-2+1, endpoint=True, dtype = int)
# learnRateDecaySweeps = np.linspace(5,8,8-5+1, endpoint=True, dtype = int)
# batchSizeSweeps = np.linspace(9,10,10-9+1, endpoint=True, dtype = int)
# trainValSweeps = np.linspace(11,12,12-11+1, endpoint=True, dtype = int)
# poolingSweeps = 13
# dropoutSweeps = np.linspace(14,16,16-14+1, endpoint=True, dtype = int)
# kernel1Sweeps = np.linspace(17,18,18-17+1, endpoint=True, dtype = int)
# kernel2Sweeps = np.linspace(19,20,20-19+1, endpoint=True, dtype = int)
# kernel3Sweeps = np.linspace(21,22,22-21+1, endpoint=True, dtype = int)
# smallModelSweeps = np.linspace(23,24,24-23+1, endpoint=True, dtype = int)
# largeModelSweeps = np.linspace(25,27,27-25+1, endpoint=True, dtype = int)
# reluLayerNumSweeps = np.linspace(28,31,31-28+1, endpoint=True, dtype = int)
# reluTanhLayerNumSweeps = np.linspace(32,35,35-32+1, endpoint=True, dtype = int)
# tanhReluLayerNumSweeps = np.linspace(36,39,39-36+1, endpoint=True, dtype = int)
# optimizerSweeps = np.linspace(40,41,41-40+1, endpoint=True, dtype = int)

#####################################################################
# Input parsing and paths to training files
#####################################################################

# Argument to pass - which job should be summarised
argParser = argparse.ArgumentParser()
argParser.add_argument("-j", "--jobname", help="Name of job to summarise") # parameter to allow parallel running on the HPC
argParser.add_argument("-rp", "--repeats", help="Number of repeats of job")

args = argParser.parse_args()
jobName = args.jobname
repeats = args.repeats

# Format paths for data loading
resultPath = os.path.join(resultPath, '{jn}'.format(jn=jobName))
if args.repeats is not None: # if there are several repeats (should usually be the case)
    repeats = int(repeats)
    temp = []
    for i in range(repeats):
        temp += [os.path.join(resultPath + str(i+1), 'dataout')] # Repeats are 1-indexed
    resultPath = temp
else:
    repeats = 1
    resultPath = [os.path.join(resultPath, 'dataout')]

#####################################################################
# Data import and formatting
#####################################################################

# Number of models per repeat
numModels = len([entry for entry in os.listdir(resultPath[0]) if 'predictions_{jn}'.format(jn=jobName) in entry])

for i in range(repeats): # For each repeat (1=indexed)
    for j in range(numModels): # For each model (1-indexed)
        print('Now loading repeat {rp} model number {num}'.format(rp = i+1, num = j+1))

        if repeats == 1:
            rpName = ''
        else:
            rpName = i+1

        histOutName = 'trainHist_{jn}{rp}_{num}.json'.format(jn=jobName, rp = rpName, num = j+1) # Naming of file out
        histOutPath = os.path.join(resultPath[i],histOutName)
        predOutName = 'predictions_{jn}{rp}_{num}.json'.format(jn=jobName, rp = rpName, num = j+1) # Naming of file out
        predOutPath = os.path.join(resultPath[i],predOutName)
        predOutName_val = 'predictions_val_{jn}{rp}_{num}.json'.format(jn=jobName, rp = rpName, num = j+1) # Naming of file out
        predOutPath_val = os.path.join(resultPath[i],predOutName_val)
        gtOutName = 'groundTruth_{jn}{rp}_{num}.json'.format(jn=jobName, rp = rpName, num = j+1) # Naming of file out
        gtOutPath = os.path.join(resultPath[i],gtOutName)
        gtOutName_val = 'groundTruth_val_{jn}{rp}_{num}.json'.format(jn=jobName, rp = rpName, num = j+1) # Ground truths
        gtOutPath_val = os.path.join(resultPath[i],gtOutName_val)
        paramOutName = 'parameters_{jn}{rp}_{num}.json'.format(jn=jobName, rp = rpName, num = j+1) # Naming of file out
        paramOutPath = os.path.join(resultPath[i],paramOutName)

        with open(histOutPath) as json_file: # load into dict
            history = json.load(json_file)
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
        with open(paramOutPath) as json_file: # load into dict
            parameter = json.load(json_file)
            parameter = json.loads(parameter) # reformat
        
        if i == 0 and j == 0: # Initialise arrays on first loop
            trainHist = np.empty(shape = (repeats, numModels, trainEpochs)) # Array of training histories
            valHist = np.empty(shape = (repeats, numModels, trainEpochs)) # Array of validation histories
            parameters = [] # Array of parameters
            RMSEs = np.empty(shape = (repeats, numModels)) # Array of RMSEs
            RMSEs_val = np.empty(shape = (repeats, numModels)) # Array of RMSEs
            groundTruths = np.empty(shape = (repeats, numModels, groundTruth.shape[0],groundTruth.shape[1], groundTruth.shape[2])) # Array of ground truths
            if groundTruth_val is not None:
                groundTruths_val = np.empty(shape = (repeats, numModels, groundTruth_val.shape[0],groundTruth_val.shape[1], groundTruth_val.shape[2])) # Array of ground truths for validation data
            predictions = np.empty(shape = (repeats, numModels, prediction.shape[0],prediction.shape[1], prediction.shape[2])) # Array of predictions
            if prediction_val is not None:
                predictions_val = np.empty(shape = (repeats, numModels, prediction_val.shape[0],prediction_val.shape[1], prediction_val.shape[2])) # Array of predictions for validation data

        loss = history['loss'] # Loss history
        val_loss = history['val_loss'] # Validation loss history
        # Some models are stopped early during training, fill out with NaN to ensure the same dimensionality of all histories
        if len(loss)<trainEpochs:
            loss = np.pad(loss,(0,trainEpochs-len(loss)),'constant', constant_values=np.nan)
        if len(val_loss)<trainEpochs:
            val_loss = np.pad(val_loss,(0,trainEpochs-len(val_loss)),'constant', constant_values=np.nan)

        trainHist[i,j] = loss
        valHist[i,j] = val_loss
        parameters.append(parameter)
        groundTruths[i,j] = groundTruth
        if groundTruth_val is not None:
            groundTruths_val[i,j] = groundTruth_val
        predictions[i,j] = prediction
        if prediction_val is not None:
            predictions_val[i,j] = prediction_val

        RMSE = tf.keras.metrics.RootMeanSquaredError()
        RMSE.update_state(groundTruth,prediction)
        RMSEs[i,j] = RMSE.result().numpy()

        if groundTruth_val is not None:
            RMSE_val = tf.keras.metrics.RootMeanSquaredError()
            RMSE_val.update_state(groundTruth_val,prediction_val)
            RMSEs_val[i,j] = RMSE_val.result().numpy()

# Convert parameter dictionaries to dataframe 
parameters = pd.DataFrame.from_dict(parameters)
parameters.index += 1 # Change to be 1-indexed as the models are this...

#####################################################################
# Plots
#####################################################################

# RMSE plot of all models for spotting errors and trends
axis = plt.subplot(1,1,1)
plt.grid()
g = sns.boxplot(ax=axis, data=RMSEs, medianprops=dict(color="red", alpha=0.7))
if groundTruth_val is not None:
    h = sns.boxplot(ax=axis, data=RMSEs_val, medianprops=dict(color="blue", alpha=0.7))
    legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', lw=4, label='All data'),
    matplotlib.lines.Line2D([0], [0], color='blue', lw=4, label='Validation data')]
    axis.legend(handles=legend_elements)
plt.title('RMSE for each model, all training repeats')
plt.ylabel('RMSE')
plt.xlabel('Model number')
g.set_xticks(range(numModels))
g.set_xticklabels(np.linspace(1, numModels, numModels, dtype=int))

# Legacy indexing of sweeps by name
# baselineIdx = 1
# learnRateSweeps = np.linspace(2,8,8-2, dtype = int)
# batchSizeSweeps = np.linspace(9,10,10-9, dtype = int)
# trainValSweeps = np.linspace(11,12,12-11, dtype = int)
# poolingSweeps = 13
# dropoutSweeps = np.linspace(14,16,16-14, dtype = int)
# kernelSweeps = np.linspace(17,22,22-17, dtype = int)
# architectureSweeps = np.linspace(23,39,39-23, dtype = int)
# optimizerSweeps = np.linspace(40,41,41-40, dtype = int)
# 


def sweepPlot(sweep, paramVariables, figname, sampleNum = 1): 
    '''
    Figure for comparison of several models and their training

    Args
    ----------
    sweep: Index of models to be plotted
    paramVariables: Hyperparamters with importance for the given models
    figname: Name of the figure (i.e. which hyperparameters are sweeped)
    sampleNum: The sample number (out of 100) to be plotted qualitatively

    Returns
    ----------
    Nothing, just plots

    '''
    columns = 8 # Figure columns
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
    fig.suptitle(figname, fontsize=16)

    if groundTruth_val is not None:
        RMSE_limits = [np.min([RMSEs[:,sweep[:]-1],RMSEs_val[:,sweep[:]-1]]),np.max([RMSEs[:,sweep[:]-1],RMSEs_val[:,sweep[:]-1]])] # x axis limits for RMSE plotS
    else:
        RMSE_limits = [np.min(RMSEs[:,sweep[:]-1]),np.max(RMSEs[:,sweep[:]-1])] # x axis limits for RMSE plotS
    # contourLim = np.max(np.square(predictions[0, sweep][sampleNum,:,:]-groundTruths[0, sweep][sampleNum,:,:])) # take 1st repeat only, samplenum is specimen out of 100

    # For each model in the given sweep
    for i in range(len(sweep)):

        ########## Parameter listing 
        nrows = 2
        ax = plt.subplot(len(sweep), columns, i*columns+1)
        ncols = 2
        nrows = len(paramVariables)
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_axis_off()

        for y in range(0, nrows): # For each parameter give name and value
            ax.annotate(
                xy=(0,y+0.5),
                text=list(parameters[paramVariables].keys())[y],
                ha='left'
            )
            ax.annotate(
                xy=(1.5,y+0.5),
                text=list(parameters[paramVariables].loc[sweep[i]].values)[y],
                ha='left'
            )

        if i==0:
            ax.annotate( # headers
                xy=(0, nrows),
                text='Parameter',
                weight='bold',
                ha='left'
            )
            ax.annotate(
                xy=(1.5, nrows),
                text='Value',
                weight='bold',
                ha='left'
            )
        # Annotate with model number
        ax.annotate(
                xy=(0,0),
                text='Model {n}'.format(n = sweep[i]),
                ha='left'
            )

        ######### RMSE plot
        # plt.style.use("seaborn-v0_8-colorblind")
        plt.style.use("viridis")
        ax = plt.subplot(len(sweep), columns,(i*columns+2))
        g = sns.boxplot(ax=ax, data=RMSEs[:,sweep[i]-1], orient="h", width=0.5, medianprops=dict(color="red", alpha=0.7)) # Remember the models are 1-indexed
        if groundTruth_val is not None:
            h = sns.boxplot(ax=ax, data=RMSEs_val[:,sweep[i]-1], orient="h", width=0.5, medianprops=dict(color="blue", alpha=0.7)) # Remember the models are 1-indexed
        ax.grid()
        if i == 0:
            plt.title('RMSE')
        ax.set_xlim(RMSE_limits)
        ax.grid(axis = "x", which = "minor")
        ax.minorticks_on()
        # Annotate RMSE plot with mean RMSE of all model repeats
        ax.annotate('Mean={r}'.format(r = round(np.mean(RMSEs[:,sweep[i]-1]), 4)), xy = (0.5,0.8),xycoords = 'axes fraction', weight='bold', ha = 'center',color="red")
        if groundTruth_val is not None:
            ax.annotate('Val Mean={r}'.format(r = round(np.mean(RMSEs_val[:,sweep[i]-1]), 4)), xy = (0.5,0.1),xycoords = 'axes fraction', weight='bold', ha = 'center',color="blue")
        # plt.ylabel('RMSE')
        # plt.xlabel('Model number')
        # g.set_xticks(range(numModels))
        # g.set_xticklabels(np.linspace(1, numModels, numModels, dtype=int))

        ######### Prediction vs ground truth scatter
        for k in range(repeats):
            # Take a random sample of 1000 points to avoid overplotting
            sampIdx = np.random.choice(groundTruths[k, sweep[i]-1].flatten().shape[0], 1000, replace=False)
            gtsample = groundTruths[k, sweep[i]-1].flatten()[sampIdx]
            predsample = predictions[k, sweep[i]-1].flatten()[sampIdx]
            ax = plt.subplot(len(sweep), columns,i*columns+3)
            # ax = plt.subplot(len(sweep), columns,(i*columns+3,i*columns+4))
            ax.scatter(x = gtsample, y = predsample, marker = ".", s=0.2)
            if i == 0: # Headers
                plt.title('Prediction vs ground truth')
                plt.ylabel('prediction')
                plt.xlabel('ground truth')
            ax.plot([0,4],[0,4],color = '#04D8B2') # Plot straight line for comparison
            # ax.set_xlim([np.min([np.min(gtsample), 0.5]),np.max([np.max(gtsample),4.5])])
            # ax.set_ylim([np.min([np.min(predsample), 0.5]),np.max([np.max(predsample),4.5])])
            ax.set_xlim([0,4])
            ax.set_ylim([0,4])
            ax.grid()


        ######## Training history plot
        cmap = matplotlib.colormaps['viridis']
        # Take colors at regular intervals spanning the colormap for each repeat
        colors = cmap(np.linspace(0, 1, repeats))
        for k in range(repeats):
            trainPlot = trainHist[k,sweep[i]-1]
            valPlot = valHist[k,sweep[i]-1]

            ax = plt.subplot(len(sweep), columns,i*columns+4)
            ax.plot(trainPlot, color=colors[k])
            ax.plot(valPlot, '--', color=colors[k])
            plt.ylim([0,0.35])
        plt.grid()
        if i == 0: # headers
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            

        ######## Error distribution plot
        for k in range(repeats):
            absErr = np.array(groundTruths[k, sweep[i]-1]) - np.array(predictions[k, sweep[i]-1])
            absErrDfFlat = pd.DataFrame(absErr.reshape(-1))

            ax = plt.subplot(len(sweep), columns,i*columns+5)
            # sns.displot(absErrDfFlat, kind="kde")
            sns.histplot(absErrDfFlat, bins=50)
            ax.set_xlim([-1,1])
            plt.grid()
            ax.legend([],[], frameon=False)
        if i == 0: # headers
            plt.xlabel('Absolute Error')
            plt.title('Error distribution')
            
        

        ######## Plot prediction vs actual field for the chosen sample
        plt.style.use("seaborn-v0_8-colorblind")
        # Grid: used for plotting only and exported in another script - keep with raw Abaqus files
        # TODO: make a bit more elegant
        gridPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json'
        with open(gridPath) as json_file: # load into dict
            grid = np.array(json.load(json_file)) # grid for plotting

        # Ground truth field
        ax = plt.subplot(len(sweep), columns,i*columns+6)
        CS = ax.contourf(grid[0],grid[1],groundTruths[0, sweep[i]-1][sampleNum,:,:])
        if i == 0: # header
            plt.title('ground truth')
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Prediction field
        ax = plt.subplot(len(sweep), columns,i*columns+7)
        CS2 = ax.contourf(grid[0],grid[1],predictions[0, sweep[i]-1][sampleNum,:,:], levels = CS.levels)
        if i == 0: # header
            plt.title('prediction')
        fig.colorbar(CS2)
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Squared error between prediction and ground truth plot
        ax = plt.subplot(len(sweep), columns,i*columns+8)
        CS3 = ax.contourf(grid[0],grid[1],np.square(predictions[0, sweep[i]-1][sampleNum,:,:]-groundTruths[0, sweep[i]-1][sampleNum,:,:]))
        if i == 0:
            plt.title('error^2')
        fig.colorbar(CS3)
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    


# Legacy method for definition of sweeps to be plotted
# initialLearnRateIdx = np.hstack((baselineIdx,initialLearnRateSweeps))
# initialLearnRateparamVariables = ['initial_lr']

# learnRateDecayIdx = np.hstack((baselineIdx,learnRateDecaySweeps))
# learnRateDecayparamVariables = ['initial_lr', 'lr_decay_rate']

# batchSizeIdx = np.hstack((baselineIdx,batchSizeSweeps))
# batchSizeparamVariables = ['batchSize']

# trainValIdx = np.hstack((baselineIdx,trainValSweeps))
# trainValparamVariables = ['trainValRatio']

# poolingIdx = np.hstack((baselineIdx,poolingSweeps))
# poolingparamVariables = ['pooling']

# dropoutIdx = np.hstack((baselineIdx,dropoutSweeps))
# dropoutparamVariables = ['dropout']

# kernel1Idx = np.hstack((baselineIdx,kernel1Sweeps))
# kernel1paramVariables = ['layer1Kernel']

# kernel2Idx = np.hstack((baselineIdx,kernel2Sweeps))
# kernel2paramVariables = ['layer2Kernel']

# kernel3Idx = np.hstack((baselineIdx,kernel3Sweeps))
# kernel3paramVariables = ['layer3Kernel']

# smallModelIdx = np.hstack((baselineIdx,smallModelSweeps))
# smallModelparamVariables = ['layer2','layer3', 'conv1Activation', 'conv2Activation', 'conv3Activation']

# largeModelIdx = np.hstack((baselineIdx,largeModelSweeps))
# largeModelparamVariables = ['layer4','layer5', 'layer6', 'conv4Activation', 'conv5Activation', 'conv6Activation']

# largeModelIdx = np.hstack((baselineIdx,largeModelSweeps))
# largeModelparamVariables = ['layer4','layer5', 'layer6', 'conv4Activation', 'conv5Activation', 'conv6Activation']

# reluLayerNumIdx = np.hstack((baselineIdx,reluLayerNumSweeps))
# reluLayerNumparamVariables = ['layer4','layer5', 'layer6', 'conv4Activation', 'conv5Activation', 'conv6Activation']

# reluTanhLayerNumIdx = np.hstack((baselineIdx,reluTanhLayerNumSweeps))
# reluTanhLayerNumparamVariables = ['layer4','layer5', 'layer6','conv3Activation', 'conv4Activation', 'conv5Activation', 'conv6Activation']

# tanhReluLayerNumIdx = np.hstack((baselineIdx,tanhReluLayerNumSweeps))
# tanhReluLayerNumparamVariables = ['layer4','layer5', 'layer6','conv3Activation', 'conv4Activation', 'conv5Activation', 'conv6Activation']

# optimizerIdx = np.hstack((baselineIdx,optimizerSweeps))
# optimizerparamVariables = ['optimizer']

# print(RMSEs)


# Create index of sweeps to be plotted
sweepIdx = pd.read_csv(sweepIdxPath)
# toPlot = [9,10,11]
for i in sweepIdx.index: # Create a figure for each sweep
    plotParams = sweepIdx.apply(lambda row: row[row == 1].index.tolist(), axis=1)[i]
    sweepnums = np.fromstring(sweepIdx['sweepIdx'][i],dtype=int, sep=',')
    sweepnums = np.hstack((baselineIdx,sweepnums))
    sweepPlot(sweep=sweepnums, paramVariables = plotParams, figname = sweepIdx['sweepName'][i])

# sweepPlot(sweep=initialLearnRateIdx, paramVariables = initialLearnRateparamVariables, figname = 'Initial learning rate')
# sweepPlot(sweep=learnRateDecayIdx, paramVariables = learnRateDecayparamVariables, figname = 'Final learning rate')
# sweepPlot(sweep=batchSizeIdx, paramVariables = batchSizeparamVariables, figname = 'Batch size')
# sweepPlot(sweep=trainValIdx, paramVariables = trainValparamVariables, figname = 'Train/validation data split')
# sweepPlot(sweep=poolingIdx, paramVariables = poolingparamVariables, figname = 'Max pooling on or off')
# sweepPlot(sweep=dropoutIdx, paramVariables = dropoutparamVariables, figname = 'Dropout')
# sweepPlot(sweep=kernel1Idx, paramVariables = kernel1paramVariables, figname = 'Convolutional layer 1 kernel size')
# sweepPlot(sweep=kernel2Idx, paramVariables = kernel2paramVariables, figname = 'Convolutional layer 2 kernel size')
# sweepPlot(sweep=kernel3Idx, paramVariables = kernel3paramVariables, figname = 'Convolutional layer 3 kernel size')
# sweepPlot(sweep=smallModelIdx, paramVariables = smallModelparamVariables, figname = 'Smaller models')
# sweepPlot(sweep=largeModelIdx, paramVariables = largeModelparamVariables, figname = 'Larger models')
# sweepPlot(sweep=reluLayerNumIdx, paramVariables = reluLayerNumparamVariables, figname = 'Larger models with only relu activation')
# sweepPlot(sweep=reluTanhLayerNumIdx, paramVariables = reluTanhLayerNumparamVariables, figname = 'Larger models with relu activation in first layer')
# sweepPlot(sweep=tanhReluLayerNumIdx, paramVariables = tanhReluLayerNumparamVariables, figname = 'Larger models with tanh activation in first layer, tanh in rest')
# sweepPlot(sweep=optimizerIdx, paramVariables = optimizerparamVariables, figname = 'Optimizer')

plt.show()
# ax = plt.subplot(2,totalCols,(3,4))
# ax.scatter(x = groundTruth, y = prediction, marker = ".")
# ax.plot([0,4],[0,4],'r')
# plt.xlim([np.min(groundTruth),np.max(groundTruth)])
# plt.ylim([np.min(prediction),np.max(prediction)])
# # ax.set_aspect('equal')
# plt.title('Prediction vs ground truth')
# plt.xlabel('ground truth')
# plt.ylabel('prediction')
# plt.grid()


if groundTruth_val is not None:
    print('Mean of validation RMSE: {r}'.format(r = np.mean(RMSE_val)))
