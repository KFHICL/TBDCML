#####################################################################
# Description
#####################################################################
'''
This script provides and overview of the performance and training 
behaviour of a several models in a sweep. The RMSE is calculated 
for all samples, and the results of the models are visualised in 
various plots.

By default works with datasets LFC18 and MC24
Labels are failure index (FI) fields

Inputs:
-j: jobname, note that jobs run using the routine require the appending of "_" to the jobname when calling
-rp: Total number of repeats done of the sweep

compareIndex and sweep_definition csv files in the directory of the "dataout" folder of the first repetition

Example of call:
CompareModels.py -j JOBNAME_ -rp 3


Ouputs:
Figures visualising model performance and behaviour
'''


#####################################################################
# Imports
#####################################################################
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

#####################################################################
# Settings
#####################################################################
plt.style.use("seaborn-v0_8-colorblind")

trainEpochs = 1000 #500 # Maximum number of epochs for trained models
# Results folder with hyperparameter sweeps and cross-validation results:
resultFolder = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults'

baselineIdx = 1 # Index of reference model
warnings.warn("Warning: if displaying data generated prior to 14.06.2024 the comparison will be between ALL DATA and validation data even if TRAINING DATA is displayed")

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

# Botch, this particular job had one model fail in all repetitions - not expected to be a good model anyway, 
if jobName == 'MC24Standardisation3107_':
    warnings.warn('FOR MC24Standardisation3107 IGNORE MODEL 12 PERFORMANCE, THIS IS A BOTCH TO SEE REMAINING MODELS, MODEL 12 FAILED TRAINING')

# Format paths for data loading
resultFolder = os.path.join(resultFolder, '{jn}'.format(jn=jobName))
if args.repeats is not None: # if there are several repeats (should usually be the case)
    repeats = int(repeats)
    temp = []
    for i in range(repeats):
        temp += [os.path.join(resultFolder + str(i+1), 'dataout')] # Repeats are 1-indexed
    resultPath = temp
else:
    repeats = 1
    resultPath = [os.path.join(resultFolder, 'dataout')]

#####################################################################
# Data import and formatting
#####################################################################

# Create index of sweeps to be plotted
resultFolderList = os.listdir(resultFolder+'1') # The first repeat of the sweep contains csv files

for fname in resultFolderList:
    if 'compareIndex' in fname:
        sweepIdxPath= os.path.join(resultFolder+'1',fname)

for fname in resultFolderList:
    if 'sweep_definition' in fname:
        sweepDefPath= os.path.join(resultFolder+'1',fname)

sweepDef= pd.read_csv(sweepDefPath)
numModels = np.max(sweepDef['Index'])

for i in range(repeats): # For each repeat (1=indexed)
    for j in range(numModels): # For each model (1-indexed)
        print('Now loading repeat {rp} model number {num}'.format(rp = i+1, num = j+1))

        if repeats == 1:
            # rpName = ''
            rpName = 1
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

        # Routine if we are missing a model - i.e. it failed
        # In this case we need the hyperparameters from another repeat of the same model
        if not os.path.isfile(histOutPath):
            print('DATA MISSING FOR repeat {rp} model number {num}'.format(rp = i+1, num = j+1))
            # Attempt to load parameters from other repeat
            success = 0
            for x in range(repeats):
                print('Attempting to get parameters from repeat ' + str(x+1))
                paramOutName = 'parameters_{jn}{rp}_{num}.json'.format(jn=jobName, rp = x+1, num = j+1) # Naming of file out
                paramOutPath = os.path.join(resultPath[x],paramOutName)
                if os.path.isfile(paramOutPath):
                    with open(paramOutPath) as json_file: # load into dict
                        parameter = json.load(json_file)
                        parameter = json.loads(parameter) # reformat
                        parameters.append(parameter)
                        success = 1
                        print('Success!')
                        break
            if success == 0:
                raise Exception('No successful repeats for model number {num}'.format( num = j+1))
            continue
        
        # Load data into variables
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
        

        # On the first iteration we initialise arrays
        if i == 0 and j == 0:
            trainHist = np.empty(shape = (repeats, numModels, trainEpochs)) * np.nan # Array of training histories
            valHist = np.empty(shape = (repeats, numModels, trainEpochs)) * np.nan # Array of validation histories
            parameters = [] # Array of parameters
            RMSEs = np.empty(shape = (repeats, numModels)) * np.nan # Array of RMSEs
            MAEs = np.empty(shape = (repeats, numModels)) * np.nan # Array of MSEs
            R2s = np.empty(shape = (repeats, numModels)) * np.nan # Array of R squared
            RMSEs_val = np.empty(shape = (repeats, numModels)) * np.nan # Array of RMSEs
            MAEs_val = np.empty(shape = (repeats, numModels)) * np.nan # Array of MSEs
            R2s_val = np.empty(shape = (repeats, numModels)) * np.nan # Array of R squared

            if not jobName == 'MC24DatasetSize2908_': # Botch for dataset size sweep where we need larger arrays for more specimens
                groundTruths = np.empty(shape = (repeats, numModels, groundTruth.shape[0],groundTruth.shape[1], groundTruth.shape[2]))* np.nan  # Array of ground truths
                if groundTruth_val is not None:
                    groundTruths_val = np.empty(shape = (repeats, numModels, groundTruth_val.shape[0],groundTruth_val.shape[1], groundTruth_val.shape[2]))* np.nan  # Array of ground truths for validation data
                predictions = np.empty(shape = (repeats, numModels, prediction.shape[0],prediction.shape[1], prediction.shape[2]))* np.nan  # Array of predictions
                if prediction_val is not None:
                    predictions_val = np.empty(shape = (repeats, numModels, prediction_val.shape[0],prediction_val.shape[1], prediction_val.shape[2]))* np.nan  # Array of predictions for validation data
            elif jobName=='MC24DatasetSize2908_':
                numSamples = 1000
                groundTruths = np.empty(shape = (repeats, numModels, numSamples,groundTruth.shape[1], groundTruth.shape[2]))* np.nan  # Array of ground truths
                if groundTruth_val is not None:
                    groundTruths_val = np.empty(shape = (repeats, numModels, numSamples,groundTruth_val.shape[1], groundTruth_val.shape[2]))* np.nan  # Array of ground truths for validation data
                predictions = np.empty(shape = (repeats, numModels, numSamples,prediction.shape[1], prediction.shape[2]))* np.nan  # Array of predictions
                if prediction_val is not None:
                    predictions_val = np.empty(shape = (repeats, numModels, numSamples,prediction_val.shape[1], prediction_val.shape[2]))* np.nan  # Array of predictions for validation data
            

        loss = history['loss'] # training curve
        val_loss = history['val_loss'] # Validation training curve
        # Some models are stopped early during training, fill out with NaN to ensure the same dimensionality of all training curves
        if len(loss)<trainEpochs:
            loss = np.pad(loss,(0,trainEpochs-len(loss)),'constant', constant_values=np.nan)
        if len(val_loss)<trainEpochs:
            val_loss = np.pad(val_loss,(0,trainEpochs-len(val_loss)),'constant', constant_values=np.nan)
        
        # Append training curves and hyperparameters
        trainHist[i,j] = loss
        valHist[i,j] = val_loss
        parameters.append(parameter)



        if not jobName=='MC24DatasetSize2908_': # Botch for dataset size sweep where we need larger arrays for more specimens
            groundTruths[i,j] = groundTruth.reshape(groundTruth.shape[0],groundTruth.shape[1], groundTruth.shape[2])
            if groundTruth_val is not None:
                groundTruths_val[i,j] = groundTruth_val.reshape(groundTruth_val.shape[0],groundTruth_val.shape[1], groundTruth_val.shape[2])
            predictions[i,j] = prediction.reshape(prediction.shape[0],prediction.shape[1], prediction.shape[2])
            if prediction_val is not None:
                predictions_val[i,j] = prediction_val.reshape(prediction_val.shape[0],prediction_val.shape[1], prediction_val.shape[2])
        elif jobName=='MC24DatasetSize2908_':
            temp = groundTruth.reshape(groundTruth.shape[0],groundTruth.shape[1], groundTruth.shape[2])
            temp = np.pad(temp, ((0,numSamples-groundTruth.shape[0]),(0,0),(0,0)), 'constant', constant_values=np.nan)
            groundTruths[i,j] = temp
            if groundTruth_val is not None:
                temp = groundTruth_val.reshape(groundTruth_val.shape[0],groundTruth_val.shape[1], groundTruth_val.shape[2])
                temp = np.pad(temp, ((0,numSamples-groundTruth_val.shape[0]),(0,0),(0,0)), 'constant', constant_values=np.nan)
                groundTruths_val[i,j] = temp
            temp = prediction.reshape(prediction.shape[0],prediction.shape[1], prediction.shape[2])
            temp = np.pad(temp, ((0,numSamples-prediction.shape[0]),(0,0),(0,0)), 'constant', constant_values=np.nan)
            predictions[i,j] = temp
            if prediction_val is not None:
                temp = prediction_val.reshape(prediction_val.shape[0],prediction_val.shape[1], prediction_val.shape[2])
                temp = np.pad(temp, ((0,numSamples-prediction_val.shape[0]),(0,0),(0,0)), 'constant', constant_values=np.nan)
                predictions_val[i,j] = temp


        # RMSE for training specimens
        RMSE = tf.keras.metrics.RootMeanSquaredError()
        RMSE.update_state(groundTruth,prediction)
        RMSEs[i,j] = RMSE.result().numpy()

        # Mean absolute error for training specimens
        MAE = tf.keras.metrics.MeanAbsoluteError()
        MAE.update_state(groundTruth,prediction)
        MAEs[i,j] = MAE.result().numpy()

        # R^2 corr for training specimens
        R2 = tf.keras.metrics.R2Score()
        R2.update_state(groundTruth.reshape(groundTruth.shape[0],-1),prediction.reshape(prediction.shape[0],-1))
        R2s[i,j] = R2.result().numpy()

        # Same metrics for validation specimens
        if groundTruth_val is not None:
            RMSE_val = tf.keras.metrics.RootMeanSquaredError()
            RMSE_val.update_state(groundTruth_val,prediction_val)
            RMSEs_val[i,j] = RMSE_val.result().numpy()

            MAE_val = tf.keras.metrics.MeanAbsoluteError()
            MAE_val.update_state(groundTruth_val,prediction_val)
            MAEs_val[i,j] = MAE_val.result().numpy()

            R2_val = tf.keras.metrics.R2Score()
            R2_val.update_state(groundTruth_val.reshape(groundTruth_val.shape[0],-1),prediction_val.reshape(prediction_val.shape[0],-1))
            R2s_val[i,j] = R2_val.result().numpy()

# Convert parameter dictionaries to dataframe 
parameters = pd.DataFrame.from_dict(parameters)
parameters.index += 1 # Change to be 1-indexed to agree with model numbering

#####################################################################
# Plots
#####################################################################

# RMSE plot of all models for spotting errors and trends
axis = plt.subplot(1,1,1)
if groundTruth_val is not None: # Val RMSE
    modelNums = np.linspace(1,numModels,numModels)
    modelIdx = np.tile(modelNums,repeats)
    RMSEDf = pd.DataFrame(data = RMSEs.reshape(-1))
    RMSEDf.columns = ['RMSE']
    RMSEDf['Model Number'] = modelIdx

    RMSEDf_val = pd.DataFrame(data = RMSEs_val.reshape(-1))
    RMSEDf_val.columns = ["RMSE"]
    RMSEDf_val['Model Number'] = modelIdx

    RMSEDf['Specimens']='Training data'
    RMSEDf_val['Specimens']='Validation data'
    RMSEsDf = pd.concat([RMSEDf, RMSEDf_val])
    RMSEsDf = pd.DataFrame.dropna(RMSEsDf)
    g = sns.boxplot(ax=axis, data=RMSEsDf, x="Model Number", y="RMSE", hue="Specimens", fill=True, medianprops=dict(alpha=0.7))

else: # Train RMSE
    g = sns.boxplot(ax=axis, data=RMSEs, medianprops=dict( alpha=0.7))
    warnings.warn('Only displaying training RMSE')
plt.grid()

plt.title('RMSE for each model, all training repeats')
plt.ylabel('RMSE')
plt.xlabel('Model number')
g.set_xticks(range(numModels))
g.set_xticklabels(np.linspace(1, numModels, numModels, dtype=int))


# MAIN PLOT FOR COMPARING MODELS IN SWEEP
def sweepPlot(sweep, paramVariables, figname, sampleNum = 2): 
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
    # Instantiate figure
    columns = 8 # Figure columns
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
    fig.suptitle(figname, fontsize=16)

    # Calculate limits for RMSE to plot all models with the same axis
    if groundTruth_val is not None:
        RMSE_limits = [np.nanmin([RMSEs[:,sweep[:]-1],RMSEs_val[:,sweep[:]-1]]),np.nanmax([RMSEs[:,sweep[:]-1],RMSEs_val[:,sweep[:]-1]])] # x axis limits for RMSE plotS
    else:
        RMSE_limits = [np.nanmin(RMSEs[:,sweep[:]-1]),np.nanmax(RMSEs[:,sweep[:]-1])] # x axis limits for RMSE plotS

    # For each model in the given sweep
    for i in range(len(sweep)):

        ######### List hyperparameters from compareindex csv file
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
                ha='left',
                size=6
            )
            ax.annotate(
                xy=(1.5,y+0.5),
                text=list(parameters[paramVariables].loc[sweep[i]].values)[y],
                ha='left',
                size=6
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
        plt.style.use("seaborn-v0_8-colorblind")
        if groundTruth_val is not None:
            # Format into dataframe
            RMSEDf = pd.Series(RMSEs[:,sweep[i]-1].reshape(-1), name='All data')
            RMSEDf_val = pd.Series(RMSEs_val[:,sweep[i]-1].reshape(-1), name='Validation data')
            RMSEsDf = pd.concat([RMSEDf, RMSEDf_val], axis=1)
        
        ax = plt.subplot(len(sweep), columns,(i*columns+2))
        if groundTruth_val is not None:
            g = sns.boxplot(ax=ax, data=RMSEsDf, orient="h", width=0.5, medianprops=dict(alpha=0.7)) # Remember the models are 1-indexed
        else:
            g = sns.boxplot(ax=ax, data=RMSEs[:,sweep[i]-1], orient="h", width=0.5, medianprops=dict(alpha=0.7)) # Remember the models are 1-indexed
        ax.grid()
        if i == 0:
            plt.title('RMSE')
        ax.set_xlim(RMSE_limits)
        ax.grid(axis = "x", which = "minor")
        ax.minorticks_on()

        # Annotate RMSE plot with mean RMSE of all model repeats
        ax.annotate('Mean={r}'.format(r = round(np.nanmean(RMSEs[:,sweep[i]-1]), 4)), xy = (0.5,0.8),xycoords = 'axes fraction', weight='bold', ha = 'center',color="red")
        if groundTruth_val is not None:
            ax.annotate('Val Mean={r}'.format(r = round(np.nanmean(RMSEs_val[:,sweep[i]-1]), 4)), xy = (0.5,0.1),xycoords = 'axes fraction', weight='bold', ha = 'center',color="blue")



        ######### Prediction vs ground truth scatter
        for k in range(repeats):
            # Take a random sample of 1000 points to avoid overplotting
            sampIdx = np.random.choice(groundTruths[k, sweep[i]-1].flatten().shape[0], 1000, replace=False)
            gtsample = groundTruths[k, sweep[i]-1].flatten()[sampIdx]
            predsample = predictions[k, sweep[i]-1].flatten()[sampIdx]
            ax = plt.subplot(len(sweep), columns,i*columns+3)
            if i == 0: # Headers
                plt.title('Prediction vs ground truth')
                plt.ylabel('prediction')
                plt.xlabel('ground truth')
            ax.scatter(x = gtsample, y = predsample, marker = ".", s=0.2)
            ax.plot([0,4],[0,4],color = '#04D8B2') # Plot straight line for comparison
            if np.isnan(gtsample).all():
                ax.set_xlim([0,4])
                ax.set_ylim([0,4])
            else:
                ax.set_xlim([0,np.nanmax(gtsample)*1.2])
                ax.set_ylim([0,np.nanmax(gtsample)*1.2])
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
            ymean = np.nanmean(trainHist[:,sweep[i]-1])
            
        plt.ylim([0,2*ymean])
        plt.grid()
        if i == 0: # headers
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            

        ######## Error distribution plot
        for k in range(repeats):
            absErr = np.array(predictions[k, sweep[i]-1]) - np.array(groundTruths[k, sweep[i]-1])
            absErrDfFlat = pd.DataFrame.dropna(pd.DataFrame(absErr.reshape(-1)))

            ax = plt.subplot(len(sweep), columns,i*columns+5)
            sns.histplot(absErrDfFlat, bins=50)
            ax.set_xlim([-1,1])
            plt.grid()
            ax.legend([],[], frameon=False)
        if i == 0: # headers
            plt.xlabel('Absolute Error')
            plt.title('Error distribution')
            
        
        ######## Plot prediction vs actual field for the chosen sample
        plt.style.use("seaborn-v0_8-colorblind")

        # Load grid: used for plotting contours only and exported in another script
        if "Dataset" in parameters:
            print(parameters['Dataset'].loc[sweep[i]])
            if not parameters['Dataset'].loc[sweep[i]] == 'LFC18':
                gridPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\sampleGrid.json"
            else:
                gridPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json'
        else:
            gridPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json'
        with open(gridPath) as json_file: # load into dict
            grid = np.array(json.load(json_file)) # grid for plotting

        # Avoid failed repeat
        for n in range(repeats):
            if np.isnan(groundTruths[n, sweep[i]-1][sampleNum,:,:]).any():
                repeat_field_idx = 0
            else:
                repeat_field_idx = n
                break

        # Ground truth field
        ax = plt.subplot(len(sweep), columns,i*columns+6)
        CS = ax.contourf(grid[0],grid[1],groundTruths[repeat_field_idx, sweep[i]-1][sampleNum,:,:])
        if i == 0: # header
            plt.title('ground truth')
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Prediction field
        ax = plt.subplot(len(sweep), columns,i*columns+7)
        CS2 = ax.contourf(grid[0],grid[1],predictions[repeat_field_idx, sweep[i]-1][sampleNum,:,:], levels = CS.levels)
        if i == 0: # header
            plt.title('prediction')
        fig.colorbar(CS2)
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Squared error between prediction and ground truth plot
        ax = plt.subplot(len(sweep), columns,i*columns+8)
        CS3 = ax.contourf(grid[0],grid[1],np.square(predictions[repeat_field_idx, sweep[i]-1][sampleNum,:,:]-groundTruths[repeat_field_idx, sweep[i]-1][sampleNum,:,:]))
        if i == 0:
            plt.title('error^2')
        fig.colorbar(CS3)
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
   

# Read the compare index file
sweepIdx = pd.read_csv(sweepIdxPath)
# toPlot = [9,10,11] # 

# Create figures for each line in the compare index file
for i in sweepIdx.index: 
    plotParams = sweepIdx.apply(lambda row: row[row == 1].index.tolist(), axis=1)[i] # Parameters of plotted models
    sweepnums = np.fromstring(sweepIdx['sweepIdx'][i],dtype=int, sep=',') # Which models in the sweep to plot in this figure
    sweepnums = np.hstack((baselineIdx,sweepnums))
    sweepPlot(sweep=sweepnums, paramVariables = plotParams, figname = sweepIdx['sweepName'][i]) # Plot


# Heatmap of cross-validation performance metrics (or models) - not super useful
if groundTruth_val is not None:
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(300*px, 300*px), layout="constrained")
    plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis
    heatMap_models = np.linspace(1,numModels,numModels,dtype = int) # Model numbers
    heatMap_RMSE = pd.DataFrame.dropna(pd.DataFrame(data = np.nanmean(RMSEs,axis = 0))) # All RMSE data
    heatMap_RMSE_val = pd.DataFrame.dropna(pd.DataFrame(data = np.nanmean(RMSEs_val,axis = 0))) # Val RMSE data
    heatMap_MAE = pd.DataFrame.dropna(pd.DataFrame(data = np.nanmean(MAEs,axis = 0))) # All MAE data
    heatMap_MAE_val = pd.DataFrame.dropna(pd.DataFrame(data = np.nanmean(MAEs_val,axis = 0))) # Validation MAE data
    heatMap_R2 = pd.DataFrame.dropna(pd.DataFrame(data = 1-np.nanmean(R2s,axis = 0))) # All R squared data
    heatMap_R2_val = pd.DataFrame.dropna(pd.DataFrame(data = 1-np.nanmean(R2s_val,axis = 0))) # Validation R squared data  
    heatmap_data = pd.concat([heatMap_RMSE, heatMap_RMSE_val, heatMap_MAE, heatMap_MAE_val,
                              heatMap_R2, heatMap_R2_val], axis=1)
    heatmap_data.columns = [ 'RMSE', 'RMSE Val', 'MAE', 'MAE Val', '1-R^2', '1-R^2 Val']
    heatmap_data.index = heatMap_models
    ax = plt.subplot(1,1,1) # 
    g = sns.heatmap(heatmap_data)
    plt.xlabel('Metric')
    plt.ylabel('Model')
    plt.title('Model performance comparison')


print(np.mean(RMSEs_val))
print(np.mean(RMSEs))


plt.show()