#####################################################################
# Description
#####################################################################
# This script is used to generate some plots for use in reports and 
# thesis.


#####################################################################
# Imports
####################################################################
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
import skimage as ski
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC


import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import argparse

# %%
#####################################################################
# Settings
#####################################################################
plt.style.use("seaborn-v0_8-colorblind")

trainEpochs = 1000 #500 # Maximum number of epochs for trained models
# Results:
resultFolder = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults'

# Use to point to a specific model to compare on unseen data

# File indicating which models to be plotted together and which hyperparameters they are sweeps of
# sweepIdxPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\compareIndex_fineTune_2.csv'

baselineIdx = 1 # Index of reference model
warnings.warn("Warning: if displaying data generated prior to 14.06.2024 the comparison will be between ALL DATA and validation data even if TRAINING DATA is displayed")

testUnseen = False # We can load a model and test it on an unseen dataset.
# Sweep selection ######################################

# jobName = 'TrainValRatio2308_' # LFC18 and MC24 train-validation split
# compareParam = 'trainValRatio' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'BatchSize1806_' # LFC18 batch size sweep
# compareParam = 'batchSize' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24BatchSize3107_' # MC24 batch size sweep
# compareParam = 'batchSize' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'KernelSize1806_' # LFC18 Kernel size sweep
# compareParam = 'layer1Kernel' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24KernelSize3107_' # MC24 Kernel size sweep
# compareParam = 'layer1Kernel' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'Optimizer1806_' # LFC18 Optimizer sweep
# compareParam = 'optimizer' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24Optimizer3107_' # MC24 Optimizer sweep
# compareParam = 'optimizer' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'ActivationFunction1806_' # LFC18 Activation function sweep
# compareParam = 'conv1Activation' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24ActivationFunction3107_' # MC24 Activation function sweep
# compareParam = 'conv1Activation' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'LossFunc1806_' # LFC18 Loss function sweep
# compareParam = 'loss' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24LossFunc3107_' # MC24 Loss function sweep
# compareParam = 'loss' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'Dropout1806_' # LFC18 dropout sweep
# compareParam = 'dropout' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24Dropout3107_' # MC24 dropout sweep
# compareParam = 'dropout' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'LearnRate1806_' # LFC18 learning rate sweep
# compareParam = 'initial_lr' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24LearnRate3107_' # MC24 learning rate sweep
# compareParam = 'initial_lr' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'LearnRate1806_' # LFC18 learning rate sweep
# compareParam = 'lr_decay_rate' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24LearnRate3107_' # MC24 learning rate sweep
# compareParam = 'lr_decay_rate' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'MaxPooling1806_' # LFC18 max pooling sweep
# compareParam = 'pooling' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24MaxPooling3107_' # MC24 max pooling sweep
# compareParam = 'pooling' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'BatchNorm1806_' # LFC18 batch norm sweep
# compareParam = 'batchNorm' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24BatchNorm3107_' # MC24 batch norm sweep
# compareParam = 'batchNorm' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'ModelDepth1806_' # LFC18 model depth sweep
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24ModelDepth3107_' # MC24 model depth sweep
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'DataAug1806_' # LFC18 data augmentation sweep
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24DataAug3107_' # MC24 data augmentation sweep
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'SkipCons1806_' # LFC18 Skip connection sweep
# compareParam = 'skipConnections' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24SkipCons3107_' # MC24 skip connections sweep
# compareParam = 'skipConnections' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'Standardisation1806_' # LFC18 scaling sweep
# compareParam = 'standardisation' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24Standardisation3107_' # MC24 scaling sweep
# compareParam = 'standardisation' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'ActivationUp2106_' # LFC18 Decoder activation sweep
# compareParam = 'ActivationUp' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24ActivationUp3107_' # MC24 Decoder activation sweep
# compareParam = 'ActivationUp' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'Epsilon2507_' # LFC18 Epsilon sweep
# compareParam = 'epsilon' # parameter to plot comparison of
# dataset = 'LFC18'


# jobName = 'MC24Epsilon3107_' # MC24 Epsilon sweep
# compareParam = 'epsilon' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'Epsilon2908_' # LFC18 Epsilon sweep with default values
# compareParam = 'epsilon' # parameter to plot comparison of
# dataset = 'LFC18'


# jobName = 'CrossValidation2808_' # LFC18 CrossValidation after hyperparameter optimisation
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'MC24CrossValidation2808_' # MC24 CrossValidation after hyperparameter optimisation
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'MC24ConstVf2908_' # MC24 constant Vf sweep
# compareParam = 'Dataset' # parameter to plot comparison of
# dataset = 'MC24'

jobName = 'MC24DatasetSize2908_' # MC24 DatasetSize sweep
compareParam = 'Dataset' # parameter to plot comparison of
dataset = 'MC24'

# jobName = 'Epsilon2507_' # MC24 DatasetSize sweep
# compareParam = 'epsilon' # parameter to plot comparison of
# dataset = 'LFC18'








# jobName = 'MC24CrossValidation1408_' # MC24 CrossValidation after hyperparameter optimisation
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'MC24'

# jobName = 'SkipCons1806_' # LFC18 SKip connection sweeps with 1-6 layer models
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'DataAug1806_' # Data augmentation sweep with 1-6 layers with and without skip connections
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'LFC18BaselineCrossValidation2208_' # LFC18 cross validation before hyperparameter optimisation
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'

# jobName = 'ModelDepth1806_' # LFC18 model depth sweep
# compareParam = 'Index' # parameter to plot comparison of
# dataset = 'LFC18'



repeats = 3
numSamples = 100
train_valSplit = 0.9
# ######################################################




# Format paths for data loading
resultFolder = os.path.join(resultFolder, '{jn}'.format(jn=jobName))
if repeats is not None: # if there are several repeats (should usually be the case)
    repeats = int(repeats)
    temp = []
    for i in range(repeats):
        temp += [os.path.join(resultFolder + str(i+1), 'dataout')] # Repeats are 1-indexed
    resultPath = temp
else:
    repeats = 1
    resultPath = [os.path.join(resultFolder, 'dataout')]



# %% Extras to be loaded in

# LFC18 grid
gridPathOld = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json"
with open(gridPathOld) as json_file: # load into dict
    LFC18_grid = np.array(json.load(json_file)) # grid for plotting

# MC24 grid
gridPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\sampleGrid.json"
with open(gridPath) as json_file: # load into dict
    MC24_grid = np.array(json.load(json_file)) # grid for plotting

if dataset == 'LFC18':
    grid = LFC18_grid
elif dataset == 'MC24':
    grid = MC24_grid

# MC24 dataset used to train model which is evaluated on unseen specimens

MC24_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240703_1417_100Samples"




# MC24 dataset not seen by any model
# 100 unseen samples
# MC24_path_Unseen = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240819_0954_100UnseenSamples"
# 1000 unseen samples
MC24_path_Unseen = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240905_1625_1000UnseenSamples"


MC24_numSamples = len(os.listdir(MC24_path))
MC24_numSamples_unseen = len(os.listdir(MC24_path_Unseen)) # Number of data samples (i.e. TBDC specimens)
MC24_sampleShape = [60,20]
MC24_xNames = ['Ex','Ey','Gxy','c2','Vf'] # Names of input features in input csv
MC24_yNames = ['FI'] # Names of ground truth features in input csv


def formatCoords(values,coordIdx):
    coords = [[x for x in values[:,coordIdx][y].split(' ') if x] for y in range(len(values[:,coordIdx]))] # Split coordinates by delimiter (space)
    coords = [np.char.strip(x, '[') for x in coords] # Coordinate output from abaqus has leading "["
    coords = [[x for x in coords[y] if x] for y in range(len(values[:,coordIdx]))] # remove empty array elements
    coords = np.array([[float(x) for x in coords[y][0:2]] for y in range(len(values[:,coordIdx]))]) # Take 2d coordinates and convert to float
    return coords
def loadSample(path = str):
  '''
  Imports data in csv and formats into a tensor
  Data from Abaqus comes in a slightly bothersome format, this 
  function manually reformats it
  '''
  # Read sample csv data
  sample = pd.read_csv(path)
  headers = sample.columns.values.tolist()
  values = np.array(sample)
  if "coordinates" in headers: 
        coordIdx = headers.index("coordinates")
        if '[' in values[0,1]: # Some coordinates will be formatted with brackets from abaqus export
            coords = formatCoords(values,coordIdx)
            values = np.column_stack((values[:,0],coords,values[:,2:])).astype(float) # Create a new values vector which contains the coordinates

        headers = np.concatenate(([[headers[0],'x_coord','y_coord'],headers[2:]])) # rectify the headers to include x and y coordinates separately
  headers = np.array(headers)
  return headers, values

# Import MC24 Sanmples used to train the Crossval1408 model
for i,file in enumerate(os.listdir(MC24_path)):
    filepath = os.path.join(MC24_path,file)
    if i==0:
        MC24_headers, MC24_samples = loadSample(filepath)
        MC24_samples = MC24_samples.reshape(1, np.shape(MC24_samples)[0],np.shape(MC24_samples)[1])
    else:
        addSamp = loadSample(filepath)[1]
        MC24_samples = np.concatenate((MC24_samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
# Reshape sample variable to have shape (samples, row, column, features)
MC24_samples2D = MC24_samples.reshape(MC24_numSamples,MC24_sampleShape[0],MC24_sampleShape[1],MC24_samples.shape[-1])



# Import MC24 Unseen Sanmples
for i,file in enumerate(os.listdir(MC24_path_Unseen)):
    filepath = os.path.join(MC24_path_Unseen,file)
    if i==0:
        MC24_headers_Unseen, MC24_samples_Unseen = loadSample(filepath)
        MC24_samples_Unseen = MC24_samples_Unseen.reshape(1, np.shape(MC24_samples_Unseen)[0],np.shape(MC24_samples_Unseen)[1])
    else:
        addSamp = loadSample(filepath)[1]
        MC24_samples_Unseen = np.concatenate((MC24_samples_Unseen,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
# Reshape sample variable to have shape (samples, row, column, features)
MC24_samples2D_Unseen = MC24_samples_Unseen.reshape(MC24_numSamples_unseen,MC24_sampleShape[0],MC24_sampleShape[1],MC24_samples_Unseen.shape[-1])

# Find indeces of input features 
MC24_featureIdx_Unseen = []
for name in MC24_xNames:
   MC24_featureIdx_Unseen += [np.where(MC24_headers_Unseen == name)[0][0]]

# Find indeces of ground truth features 
MC24_gtIdx_Unseen = []
for name in MC24_yNames:
   MC24_gtIdx_Unseen += [np.where(MC24_headers_Unseen == name)[0][0]]
   
MC24_X_Unseen = MC24_samples2D_Unseen[:,:,:,MC24_featureIdx_Unseen]  # Input features
MC24_Y_Unseen = MC24_samples2D_Unseen[:,:,:,MC24_gtIdx_Unseen] # Labels

MC24_X = MC24_samples2D[:,:,:,MC24_featureIdx_Unseen]  # Input features
MC24_Y = MC24_samples2D[:,:,:,MC24_gtIdx_Unseen] # Labels

MC24_X_Unseen_shape = MC24_X_Unseen.shape
MC24_Y_Unseen_shape = MC24_Y_Unseen.shape

MC24_X_shape = MC24_X.shape
MC24_Y_shape = MC24_Y.shape


Xscaler = preprocessing.MinMaxScaler() # Do a scaler for the in and outputs separately (to be able to inversely standardise predictions)
Yscaler = preprocessing.MinMaxScaler()
# Xscaler = preprocessing.StandardScaler() # Default is scale by mean and divide by std
# Yscaler = preprocessing.StandardScaler()

Xscaler.fit(MC24_X_Unseen.reshape(MC24_X_Unseen.shape[0]*MC24_X_Unseen.shape[1]*MC24_X_Unseen.shape[2],-1)) # Scaler only takes input of shape (data,features)
MC24_X_Unseen = Xscaler.transform(MC24_X_Unseen.reshape(MC24_X_Unseen.shape[0]*MC24_X_Unseen.shape[1]*MC24_X_Unseen.shape[2],-1))
MC24_X_Unseen = MC24_X_Unseen.reshape(MC24_X_Unseen_shape) # reshape to 2D samples

# No need to rescale ground truths, but is needed to unscale predictions (Note there is a little bit of data leakage since we do unscaling based on the whole dataset used in training (also the val set))
# Yscaler.fit(MC24_Y.reshape(MC24_Y.shape[0]*MC24_Y.shape[1]*MC24_Y.shape[2],-1)) # Scaler only takes input of shape (data,features)
Yscaler.fit(MC24_Y_Unseen.reshape(MC24_Y_Unseen.shape[0]*MC24_Y_Unseen.shape[1]*MC24_Y_Unseen.shape[2],-1)) # Scaler only takes input of shape (data,features)
# MC24_Y_Unseen = Yscaler.transform(MC24_Y_Unseen.reshape(MC24_Y_Unseen.shape[0]*MC24_Y_Unseen.shape[1]*MC24_Y_Unseen.shape[2],-1))
# MC24_Y_Unseen = MC24_Y_Unseen.reshape(MC24_Y_Unseen_shape) # reshape to 2D samples


#####################################################################
# Data import and formatting
#####################################################################
# Create index of sweeps to be plotted
resultFolderList = os.listdir(resultFolder+'1') # The first repeat of the sweep contains the files to plot

for fname in resultFolderList:
    if 'compareIndex' in fname:
        sweepIdxPath= os.path.join(resultFolder+'1',fname)

for fname in resultFolderList:
    if 'sweep_definition' in fname:
        sweepDefPath= os.path.join(resultFolder+'1',fname)

sweepDef= pd.read_csv(sweepDefPath)
numModels = np.max(sweepDef['Index'])


if jobName=='TrainValRatio2308_':
    numModels=3
    sweepDef = sweepDef.loc[:2]


# numModels = len([entry for entry in os.listdir(resultPath[0]) if 'predictions_{jn}'.format(jn=jobName) in entry])

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

        # We can test a loaded model on unseen data
        if testUnseen:
            modelName = 'model_MC24DatasetSize2908_{rp}_{m}.keras'.format(rp = i+1, m = j+1)
            modelPath = os.path.join(resultPath[i],modelName)

            loaded_model = keras.models.load_model(modelPath)


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
        
        if i == 0 and j == 0:  # Initialise arrays on first loop
            trainHist = np.empty(shape = (repeats, numModels, trainEpochs)) * np.nan # Array of training histories
            valHist = np.empty(shape = (repeats, numModels, trainEpochs)) * np.nan # Array of validation histories
            parameters = [] # Array of parameters
            # parameters = np.empty(shape = (numModels))
            RMSEs_Unseen = np.empty(shape = (repeats, numModels, MC24_Y_Unseen.shape[0])) * np.nan # Array of RMSEs
            SSIMs_Unseen = np.empty(shape = (repeats, numModels, MC24_Y_Unseen.shape[0])) * np.nan # Array of SSIMs
            RMSEs_Unseen2 = np.empty(shape = (repeats, numModels)) * np.nan
            # MRMSE = np.empty(shape = (repeats, numModels)) * np.nan # Array of RMSEs
            # MSSIM = np.empty(shape = (repeats, numModels)) * np.nan # Array of SSIMs
            MAEs = np.empty(shape = (repeats, numModels)) * np.nan # Array of MSEs
            R2s = np.empty(shape = (repeats, numModels)) * np.nan # Array of R squared
            RMSEs_val2= np.empty(shape = (repeats, numModels)) * np.nan # Array of RMSEs
            MAEs_val = np.empty(shape = (repeats, numModels)) * np.nan # Array of MSEs
            R2s_val = np.empty(shape = (repeats, numModels)) * np.nan # Array of R squared
            RMSEs_train2 = np.empty(shape = (repeats, numModels)) * np.nan # Array of RMSEs


            if not (jobName=='TrainValRatio2308_' or jobName=='MC24DatasetSize2908_'): # Train val ratio sweeps have different shapes
                RMSEs = np.empty(shape = (repeats, numModels, groundTruth.shape[0])) * np.nan # Array of RMSEs
                SSIMs = np.empty(shape = (repeats, numModels, groundTruth.shape[0])) * np.nan # Array of SSIMs

                RMSEs_val = np.empty(shape = (repeats, numModels, groundTruth_val.shape[0])) * np.nan # Array of RMSEs
                SSIMs_val = np.empty(shape = (repeats, numModels, groundTruth_val.shape[0])) * np.nan # Array of SSIMs
                

                groundTruths = np.empty(shape = (repeats, numModels, groundTruth.shape[0],groundTruth.shape[1], groundTruth.shape[2]))* np.nan  # Array of ground truths
                if groundTruth_val is not None:
                    groundTruths_val = np.empty(shape = (repeats, numModels, groundTruth_val.shape[0],groundTruth_val.shape[1], groundTruth_val.shape[2]))* np.nan  # Array of ground truths for validation data
                predictions = np.empty(shape = (repeats, numModels, prediction.shape[0],prediction.shape[1], prediction.shape[2]))* np.nan  # Array of predictions
                if prediction_val is not None:
                    predictions_val = np.empty(shape = (repeats, numModels, prediction_val.shape[0],prediction_val.shape[1], prediction_val.shape[2]))* np.nan  # Array of predictions for validation data
            elif (jobName=='TrainValRatio2308_' or jobName=='MC24DatasetSize2908_'):
                if jobName=='MC24DatasetSize2908_':
                    numSamples = 1000
                RMSEs = np.empty(shape = (repeats, numModels, numSamples)) * np.nan # Array of RMSEs
                SSIMs = np.empty(shape = (repeats, numModels, numSamples)) * np.nan # Array of SSIMs
                RMSEs_val = np.empty(shape = (repeats, numModels, numSamples)) * np.nan # Array of RMSEs
                SSIMs_val = np.empty(shape = (repeats, numModels, numSamples)) * np.nan # Array of SSIMs
                
                groundTruths = np.empty(shape = (repeats, numModels, numSamples,groundTruth.shape[1], groundTruth.shape[2]))* np.nan  # Array of ground truths
                if groundTruth_val is not None:
                    groundTruths_val = np.empty(shape = (repeats, numModels, numSamples,groundTruth_val.shape[1], groundTruth_val.shape[2]))* np.nan  # Array of ground truths for validation data
                predictions = np.empty(shape = (repeats, numModels, numSamples,prediction.shape[1], prediction.shape[2]))* np.nan  # Array of predictions
                if prediction_val is not None:
                    predictions_val = np.empty(shape = (repeats, numModels, numSamples,prediction_val.shape[1], prediction_val.shape[2]))* np.nan  # Array of predictions for validation data
            
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

        if not (jobName=='TrainValRatio2308_' or jobName=='MC24DatasetSize2908_'):
            groundTruths[i,j] = groundTruth.reshape(groundTruth.shape[0],groundTruth.shape[1], groundTruth.shape[2])
            if groundTruth_val is not None:
                groundTruths_val[i,j] = groundTruth_val.reshape(groundTruth_val.shape[0],groundTruth_val.shape[1], groundTruth_val.shape[2])
            predictions[i,j] = prediction.reshape(prediction.shape[0],prediction.shape[1], prediction.shape[2])
            if prediction_val is not None:
                predictions_val[i,j] = prediction_val.reshape(prediction_val.shape[0],prediction_val.shape[1], prediction_val.shape[2])
        elif (jobName=='TrainValRatio2308_' or jobName=='MC24DatasetSize2908_'):
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

        RMSE = tf.keras.metrics.RootMeanSquaredError()

        MAE = tf.keras.metrics.MeanAbsoluteError()
        MAE.update_state(groundTruth,prediction)
        MAEs[i,j] = MAE.result().numpy()

        R2 = tf.keras.metrics.R2Score()
        R2.update_state(groundTruth.reshape(groundTruth.shape[0],-1),prediction.reshape(prediction.shape[0],-1))
        R2s[i,j] = R2.result().numpy()

        # harMean = 2/(1/8 + 1/50) # Harmonic mean of tow dimensions (based on MC24 for now)
        # pixelLength = 2.5
        # charLengthPixels = harMean/pixelLength
        # winKernel = int(np.floor(charLengthPixels)) # Takwe as material characteristic length - is 7 pixels for both datasets currently (i.e. harmonic mean for MC24 and tow width rounded down to nearest integer for LFC18)
        if dataset == 'LFC18':
            winKernel = 5
        elif dataset == 'MC24':
            winKernel = 7
        # SSIM = np.empty(shape = (groundTruths.shape[2]))
        # SSIM_val = np.empty(shape = (groundTruths_val.shape[2]))

        
        for sp in range(groundTruths.shape[2]): # Loop over specimens
            im1 = groundTruths[i,j,sp]
            im2 = predictions[i,j,sp]
            rangedat = np.max(im1) - np.min(im1)

            SSIMs[i,j,sp], simIm = ski.metrics.structural_similarity(im1, im2, win_size=winKernel, gradient=False, data_range=rangedat, channel_axis=None, gaussian_weights=False, full=True)
            
            RMSE.update_state(im1,im2)
            RMSEs[i,j,sp] = RMSE.result().numpy()
        
        
        # SSIMs[i,j,sp] = np.mean(SSIM)
        for sp in range(groundTruths_val.shape[2]): # Loop over specimens
            im1 = groundTruths_val[i,j,sp]
            im2 = predictions_val[i,j,sp]
            rangedat = np.max(im1) - np.min(im1)

            SSIMs_val[i,j,sp], simIm = ski.metrics.structural_similarity(im1, im2, win_size=winKernel, gradient=False, data_range=rangedat, channel_axis=None, gaussian_weights=False, full=True)
            
            RMSE.update_state(im1,im2)
            RMSEs_val[i,j,sp] = RMSE.result().numpy()
        # SSIMs_val[i,j] = np.mean(SSIM_val)


        # Check models on unseen data samples
        if 'loaded_model' in locals():
                pred_Unseen = loaded_model.predict(MC24_X_Unseen)
                # pred_Unseen = pred_Unseen.reshape(MC24_Y_Unseen[sp].shape[0],MC24_Y_Unseen[sp].shape[1],MC24_Y_Unseen[sp].shape[2]) # Make prediction on a single specimen (we need to resize as a batch of size 1)
                pred_Unseen_shape = pred_Unseen.shape
                gt_Unseen = MC24_Y_Unseen
                gt_Unseen_shape = gt_Unseen.shape

                pred_Unseen_invStandard = Yscaler.inverse_transform(pred_Unseen.reshape(pred_Unseen_shape[0]*pred_Unseen_shape[1]*pred_Unseen_shape[2],-1))
                pred_Unseen_invStandard = pred_Unseen_invStandard.reshape(pred_Unseen_shape)
        
        if 'loaded_model' in locals():
            for sp in range(gt_Unseen_shape[0]): #
            
                
                im1 = gt_Unseen[sp].reshape(gt_Unseen_shape[1:3])
                im2 = pred_Unseen_invStandard[sp].reshape(pred_Unseen_shape[1:3])
                
                rangedat = np.max(im1) - np.min(im1)

                SSIMs_Unseen[i,j,sp], simIm = ski.metrics.structural_similarity(im1, im2, win_size=winKernel, gradient=False, data_range=rangedat, channel_axis=None, gaussian_weights=False, full=True)
                
                RMSE.update_state(im1,im2)
                RMSEs_Unseen[i,j,sp] = RMSE.result().numpy()

        RMSE_train2 = tf.keras.metrics.RootMeanSquaredError()
        RMSE_train2.update_state(groundTruth,prediction)
        RMSEs_train2[i,j] = RMSE_train2.result().numpy()

        if groundTruth_val is not None:
            RMSE_val2 = tf.keras.metrics.RootMeanSquaredError()
            RMSE_val2.update_state(groundTruth_val,prediction_val)
            RMSEs_val2[i,j] = RMSE_val2.result().numpy()

            

            if 'loaded_model' in locals():
                RMSE_Unseen2 = tf.keras.metrics.RootMeanSquaredError()
                RMSE_Unseen2.update_state(gt_Unseen,pred_Unseen_invStandard)
                RMSEs_Unseen2[i,j] = RMSE_Unseen2.result().numpy()

            MAE_val = tf.keras.metrics.MeanAbsoluteError()
            MAE_val.update_state(groundTruth_val,prediction_val)
            MAEs_val[i,j] = MAE_val.result().numpy()

            R2_val = tf.keras.metrics.R2Score()
            R2_val.update_state(groundTruth_val.reshape(groundTruth_val.shape[0],-1),prediction_val.reshape(prediction_val.shape[0],-1))
            R2s_val[i,j] = R2_val.result().numpy()

# Convert parameter dictionaries to dataframe 
parameters = pd.DataFrame.from_dict(parameters)
parameters.index += 1 # Change to be 1-indexed as the models are this...

# Convert to dataframes
MRMSE_val = np.nanmean(RMSEs_val, axis = 2) # take mean along specimen axis
MSSIM_val = np.nanmean(SSIMs_val, axis = 2) # take mean along specimen axis
MRMSE_Unseen = np.nanmean(RMSEs_Unseen, axis = 2)
MSSIM_Unseen = np.nanmean(SSIMs_Unseen, axis = 2)




if 'ModelDepth' in jobName: # Botch for model depth jobs
    sweepDef[compareParam] = [3,1,2,4,5,6] # Model depths are swept like this with the 3 layer as the baseline
    sweepDef = sweepDef.rename(columns={compareParam: 'Model Depth'})
    compareParam = 'Model Depth'

if 'KernelSize' in jobName:
    sweepDef = sweepDef.rename(columns={compareParam: 'Kernel Size'})
    compareParam = 'Kernel Size'

# if 'SkipCons' in jobName: # Botch for model depth jobs
#     sweepDef[compareParam] = [3,3,1,2,4,5,6] # Model depths are swept like this with the 3 layer as the baseline
#     sweepDef = sweepDef.rename(columns={compareParam: 'Model Depth'})
#     compareParam = 'Model Depth'

if 'Cross' in jobName: # Botch for model depth jobs
    compareParam = 'Index'
    sweepDef = sweepDef.rename(columns={compareParam: 'Fold'})
    compareParam = 'Fold'

# if 'DataAug' in jobName: # Botch for specific jobs


#     compareParam = 'Index'
#     sweepDef[compareParam] = [3,1,2,4,5,6,3,1,2,4,5,6]
#     sweepDef = sweepDef.rename(columns={compareParam: 'Model Depth'})
#     compareParam = 'Model Depth'

if 'ActivationFunction' in jobName:
    sweepDef = sweepDef.rename(columns={compareParam: 'Activation Function'})
    compareParam = 'Activation Function'


# RMSEDf_val2 IS DONE ON A POINT-BY-POINT BASIS FOR CONSISTENCY WITH COMPAREMODELS AND MODELSUMMARY
# SSIM CALCULATIONS NEED TO BE DONE ON A SPECIMEN-BY-SPECIMEN BASIS AND THEN AVERAGED
RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
RMSEDf_val = pd.DataFrame(MRMSE_val,columns = sweepDef[compareParam])
SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])
RMSEDf_Unseen = pd.DataFrame(MRMSE_Unseen,columns = sweepDef[compareParam])
RMSEDf_Unseen2 = pd.DataFrame(RMSEs_Unseen2,columns = sweepDef[compareParam])
SSIMDf_Unseen = pd.DataFrame(MSSIM_Unseen,columns = sweepDef[compareParam])
RMSEDf_train2 = pd.DataFrame(RMSEs_train2,columns = sweepDef[compareParam])



if 'DataAug' in jobName:
    sweepDef = sweepDef.loc[[0,1]]
    RMSEs_val2 = RMSEs_val2[:,[0,1]]
    MSSIM_val = MSSIM_val[:,[0,1]]
    sweepDef = sweepDef.rename(columns={compareParam: 'Data augmentation'})
    compareParam = 'Data augmentation'
    sweepDef[compareParam] = [0,1]
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])


if 'Optimizer' in jobName:
    sweepDef = sweepDef.loc[[0,1,8]]
    RMSEs_val2 = RMSEs_val2[:,[0,1,8]]
    MSSIM_val = MSSIM_val[:,[0,1,8]]
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])
    
if 'MaxPooling' in jobName:
    sweepDef = sweepDef.loc[[0,1]]
    RMSEs_val2 = RMSEs_val2[:,[0,1]]
    MSSIM_val = MSSIM_val[:,[0,1]]
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])

if 'BatchNorm' in jobName:
    sweepDef = sweepDef.loc[[0,1]]
    RMSEs_val2 = RMSEs_val2[:,[0,1]]
    MSSIM_val = MSSIM_val[:,[0,1]]
    valHist = valHist[:,[0,1]]
    trainHist = trainHist[:,[0,1]]
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])

if 'SkipCons' in jobName:
    sweepDef = sweepDef.loc[[0,1]]
    RMSEs_val2 = RMSEs_val2[:,[0,1]]
    MSSIM_val = MSSIM_val[:,[0,1]]
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])

if 'Standardisation' in jobName:
    sweepDef = sweepDef.loc[[0,1]]
    RMSEs_val2 = RMSEs_val2[:,[0,1]]
    MSSIM_val = MSSIM_val[:,[0,1]]
    sweepDef = sweepDef.rename(columns={compareParam: 'Scaling'})
    compareParam = 'Scaling'
    sweepDef['Scaling'] = ['Standardisation','Normalisation']
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])

if 'ActivationUp21' in jobName:
    sweepDef = sweepDef.loc[[0,19]]
    RMSEs_val2 = RMSEs_val2[:,[0,19]]
    MSSIM_val = MSSIM_val[:,[0,19]]
    sweepDef = sweepDef.rename(columns={compareParam: 'Decoder activation'})
    compareParam = 'Decoder activation'
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])


if 'ActivationUp3107' in jobName: # Botch (no baseline model included in sweep - taking model with 3 layers (todo))
    sweepDef = sweepDef.loc[[0,1]]
    RMSEs_val2 = RMSEs_val2[:,[0,1]]
    MSSIM_val = MSSIM_val[:,[0,1]]
    sweepDef = sweepDef.rename(columns={compareParam: 'Decoder activation'})
    compareParam = 'Decoder activation'
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])


# if 'DataAug1806' in jobName: # Botch for specific jobs
#     SSIMDf_val = SSIMDf_val.iloc[:,:6] # Take only the models with both skip connections and data augmentation
#     RMSEDf_val2 = RMSEDf_val2.iloc[:,:6]

if compareParam == 'initial_lr':
    sweepDef = sweepDef.loc[[0,1,2,3,4]]
    RMSEs_val2 = RMSEs_val2[:,[0,1,2,3,4]]
    MSSIM_val = MSSIM_val[:,[0,1,2,3,4]]
    sweepDef = sweepDef.rename(columns={compareParam: 'Initial learning rate'})
    compareParam = 'Initial learning rate'
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])
    

if compareParam == 'lr_decay_rate':
    sweepDef = sweepDef.loc[[0,5,10,15]]
    RMSEs_val2 = RMSEs_val2[:,[0,5,10,15]]
    MSSIM_val = MSSIM_val[:,[0,5,10,15]]
    sweepDef = sweepDef.rename(columns={compareParam: 'Learning rate decay'})
    compareParam = 'Learning rate decay'
    RMSEDf_val2 = pd.DataFrame(RMSEs_val2,columns = sweepDef[compareParam])
    SSIMDf_val = pd.DataFrame(MSSIM_val,columns = sweepDef[compareParam])
    

# Format loss into dataframe
# Take mean across repeats
mTrainHist = np.nanmean(trainHist,axis = 0)
mValHist = np.nanmean(valHist,axis = 0)

# mTrainHist = trainHist[0]
# mValHist = valHist[0]

# mTrainHist = trainHist
# mValHist = valHist

# Into dataframes
mTrainHistDf = pd.DataFrame(mTrainHist.T, columns= sweepDef[compareParam])
mTrainHistDf['Data'] = 'Training'
mTrainHistDf['Epoch'] = mTrainHistDf.index
mValHistDf = pd.DataFrame(mValHist.T, columns= sweepDef[compareParam])
mValHistDf['Data'] = 'Validation'
mValHistDf['Epoch'] = mValHistDf.index

mHistDf =  pd.concat([mTrainHistDf, mValHistDf])
mHistDf = pd.melt(mHistDf, id_vars=['Data','Epoch'], value_vars= sweepDef[compareParam])
mHistDf = mHistDf.rename(columns={"value": "Loss"})





# %% Model comparison figure
plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

columns = 2 # Figure columns
rows = 2


# RMSE and SSIM plot
ax = plt.subplot(2,2,(1,2))
ax2 = plt.twinx()
# g = sns.boxplot(ax=ax, data=RMSEDf_val, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#029E73") # Remember the models are 1-indexed

# RMSE is done for all points individually, not a mean of all the specimens
g = sns.boxplot(ax=ax, data=RMSEDf_val2, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#029E73") # Remember the models are 1-indexed
# g = sns.boxplot(ax=ax, data=RMSEDf_val3.loc[RMSEDf_val3['Model Depth'].isin([2,3,4,5])], x = 'Model Depth', y = 'value',orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), hue = 'Data Augmentation',palette   = 'viridis') # Remember the models are 1-indexed
# g = sns.boxplot(ax=ax, data=SSIMDf_val3.loc[SSIMDf_val3['Model Depth'].isin([2,3,4,5])], x = 'Model Depth', y = 'value', orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), hue = 'Data Augmentation',palette   = 'viridis')
# g2 = sns.pointplot(ax = ax,data=RMSEDf_val3.loc[RMSEDf_val3['Model Depth'].isin([2,3,4,5])], x = 'Model Depth', y = 'value', estimator = 'median', ci=None, scale=0.3, marker='D',linewidth = 0.25,hue = 'Data Augmentation',palette   = 'viridis')
# h2 = sns.pointplot(ax = ax2,data=SSIMDf_val3.loc[SSIMDf_val3['Model Depth'].isin([2,3,4,5])], x = 'Model Depth', y = 'value', estimator = 'median', ci=None, scale=0.3, marker='o',linewidth = 0.25,hue = 'Data Augmentation')
# extract the existing handles and labels
# h, l = ax.get_legend_handles_labels()

# slice the appropriate section of l and h to include in the legend
# ax.legend(h[0:2], l[0:2], bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.,  title = 'Data \nAugmentation')


# ax2.get_legend().remove()
h = sns.boxplot(ax=ax2, data=SSIMDf_val, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#0173B2") # Remember the models are 1-indexed
# g2 = sns.pointplot(ax = ax,data=RMSEDf_val, estimator = 'median', ci=None, scale=0.3, color="#029E73", marker='D',linewidth = 0.25)
g2 = sns.pointplot(ax = ax,data=RMSEDf_val2, estimator = 'median', ci=None, scale=0.3, color="#029E73", marker='D',linewidth = 0.25)
h2 = sns.pointplot(ax = ax2,data=SSIMDf_val, estimator = 'median', ci=None, scale=0.3, color="#0173B2", marker='o',linewidth = 0.25)
ax.set_ylim([0.99*np.min(RMSEDf_val2),np.max(RMSEDf_val2)+np.max(RMSEDf_val2)-0.9*np.min(RMSEDf_val2)])
ax2.set_ylim([np.min(SSIMDf_val)-np.max(SSIMDf_val)+np.min(SSIMDf_val),1.01*np.max(SSIMDf_val)])



ax.grid()
# plt.title('RMSE')
ax.grid(axis = "y", which = "minor")
ax.minorticks_on()
# ax.set_ylabel('SSIM',color = 'black')
ax.set_ylabel('RMSE',color = 'black',bbox=dict(facecolor="#029E73", edgecolor="#029E73", pad=0.2, alpha=0.5, boxstyle = 'Round'))
ax2.set_ylabel('SSIM',bbox=dict(facecolor="#0173B2", edgecolor="#0173B2", pad=0.2, alpha=0.5, boxstyle = 'Round'))


# plt.savefig('BatchSize.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
# plt.savefig('ModelDepth_RMSESSIM.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
# plt.savefig('DataAug1806_noSkip.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
# plt.savefig('LFC18BaselineCrossVal.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
# plt.savefig('LFC18ModelDepth_DataAugImprovement_SSIM.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.savefig('HP_'+dataset+'_'+compareParam+'.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.show()

# %% Train val RMSE comparison
RMSEDf_train2['Specimens'] = 'Training'
RMSEDf_val2['Specimens'] = 'Validation'

RSMEs_Df_all = pd.concat([RMSEDf_train2,RMSEDf_val2])
RSMEs_Df_all = pd.melt(RSMEs_Df_all, id_vars=['Specimens'])

# plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "12"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

ax = plt.subplot(1,1,1)
g = sns.boxplot(ax=ax, data=RSMEs_Df_all, x="Dataset", y="value", hue="Specimens", fill=True, medianprops=dict(alpha=0.7),palette = 'viridis')
ax.set_ylabel('RMSE')
plt.grid()
h,l = ax.get_legend_handles_labels()
# ax.get_legend().remove()
# ax.set_yscale('log')
# fig.legend(title='Dataset',handles = h,labels=l, 
#            loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.2))

plt.savefig('Train_val_RMSE_DatasetSize.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)

plt.show()


#%% Export data so the csv file
HP_Res_Path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\Hyperparameter_Optimisation_results'

# Take means for printing before formatting for export
meanRMSEs = RMSEDf_val2.mean(axis=0)
meanSSIMs = SSIMDf_val.mean(axis=0)



SSIMDf_val['Statistic'] = 'SSIM'
RMSEDf_val2['Statistic'] = 'RMSE'

SSIMDf_val['Hyperparameter'] = compareParam
RMSEDf_val2['Hyperparameter'] = compareParam
SSIMDf_val.columns.name = 'HP Value'
RMSEDf_val2.columns.name = 'HP Value'

SSIMDf_val['Repeat'] = SSIMDf_val.index
RMSEDf_val2['Repeat'] = RMSEDf_val2.index

exportDf = pd.concat([RMSEDf_val2,SSIMDf_val])
exportDf = pd.melt(exportDf, id_vars=['Statistic','Hyperparameter','Repeat'])
exportDf = exportDf.rename(columns={"value": "Value"})

# Print formatted text to paste into report
hpvals, indeces = np.unique(exportDf['HP Value'],return_index=True)

# Use to unsort list (uncomment)
# hpOrder = np.unique(indeces,return_index=True)[1]
# hpvals = [hpvals[index] for index in hpOrder]
# meanRMSEs_sortbytHP = meanRMSEs
# meanSSIMs_sortbytHP = meanSSIMs

# Comment out when unsorting list
meanRMSEs_sortbytHP = meanRMSEs.sort_index()
meanSSIMs_sortbytHP = meanSSIMs.sort_index()

hpvals = np.pad(hpvals,(0,7-len(hpvals)),'constant',constant_values=np.nan)


meanRMSEs_sortbytHP = np.pad(meanRMSEs_sortbytHP,(0,7-len(meanRMSEs_sortbytHP)),'constant',constant_values=np.nan)
meanRMSEs_sortbytHP = np.round(meanRMSEs_sortbytHP,4)
meanSSIMs_sortbytHP = np.pad(meanSSIMs_sortbytHP,(0,7-len(meanSSIMs_sortbytHP)),'constant',constant_values=np.nan)
meanSSIMs_sortbytHP = np.round(meanSSIMs_sortbytHP,4)

print('HYPERPARAM & {val1} & {val2} & {val3} & {val4} & {val5} & {val6} & {val7} '.format(val1 = hpvals[0], val2 = hpvals[1], val3 = hpvals[2], val4 = hpvals[3], val5 = hpvals[4], val6 = hpvals[5], val7 = hpvals[6]))
print('RMSE & {val1} & {val2} & {val3} & {val4} & {val5} & {val6} & {val7} '.format(val1 = meanRMSEs_sortbytHP[0], val2 = meanRMSEs_sortbytHP[1], val3 = meanRMSEs_sortbytHP[2], val4 = meanRMSEs_sortbytHP[3], val5 = meanRMSEs_sortbytHP[4], val6 = meanRMSEs_sortbytHP[5], val7 = meanRMSEs_sortbytHP[6]))
print('SSIM & {val1} & {val2} & {val3} & {val4} & {val5} & {val6} & {val7} '.format(val1 = meanSSIMs_sortbytHP[0], val2 = meanSSIMs_sortbytHP[1], val3 = meanSSIMs_sortbytHP[2], val4 = meanSSIMs_sortbytHP[3], val5 = meanSSIMs_sortbytHP[4], val6 = meanSSIMs_sortbytHP[5], val7 = meanSSIMs_sortbytHP[6]))

exportDf.to_csv(os.path.join(HP_Res_Path,'HP_'+dataset+'.csv'), index=True,mode = 'a', header = True)  




# %%
# Training curve comparison figure
plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)


# ax = plt.subplot(1,2,1)

# lp = sns.lineplot(data = mHistDf.loc[mHistDf['batchNorm'].isin([0])],x="Epoch", y="Loss",
#               hue="Data", palette = 'colorblind',linewidth = 0.5)
# ax.set_ylim([0.1,1.2])
# # ax.set_yscale('log')
# ax.grid(axis = 'y',which = 'major')
# # ax.minorticks_on()
# ax.get_legend().remove()
# plt.title('Batchnorm off' )

# ax = plt.subplot(1,2,2)

# lp = sns.lineplot(data = mHistDf.loc[mHistDf['batchNorm'].isin([1])],x="Epoch", y="Loss",
#               hue="Data", palette = 'colorblind',linewidth = 0.5)
# ax.set_ylim([0.1,1.2])
# # ax.set_yscale('log')
# ax.grid(axis = 'y',which = 'major')
# # ax.minorticks_on()
# # ax.get_legend().remove()
# plt.title('Batchnorm on' )
# ax.set_ylabel('')

ax = plt.subplot(1,1,1)

# lp = sns.lineplot(data = mHistDf,x="Epoch", y="Loss",
#               hue="Dataset",style = 'Data', palette = 'colorblind',linewidth = 0.5)

# lp = sns.lineplot(data = mHistDf.loc[mHistDf['Dataset'].isin(['MC24','MC24_1000'])],x="Epoch", y="Loss",
#               hue="Dataset",style = 'Data', palette = 'colorblind',linewidth = 0.5,size="Data",sizes = [0.4,0.2])
lp = sns.lineplot(data = mHistDf.loc[mHistDf['epsilon'].isin([0.0000001,0.001,0.1])],x="Epoch", y="Loss",
              hue="epsilon",style = 'Data', palette = 'colorblind',linewidth = 0.5,size="Data",sizes = [0.4,0.2])
ax.set_ylim([0.1,1])
ax.set_yscale('log')
ax.grid(axis = 'y',which = 'minor')
plt.grid()
ax.minorticks_on()
ax.set_ylabel('Loss')
# plt.title('Dropout = 0.3' )

h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()

# fig.legend(title='Dataset',handles = h,labels=l, 
#            loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.3))
fig.legend(title='',handles = h,labels=l, 
           loc="lower center", ncol=2,bbox_to_anchor=(0.58, -0.35))
# lp = sns.lineplot(data = mHistDf.loc[mHistDf['Model Depth'].isin([1,3,6])],x="Epoch", y="Loss",
#              hue="Model Depth", style="Data", palette = 'colorblind',linewidth = 0.5)
# lp = sns.lineplot(data = mHistDf,x="Epoch", y="Loss",
#              hue="loss", style="Data", palette = 'colorblind',linewidth = 0.5)
# lp = sns.lineplot(data = mHistDf,x="Epoch", y="Loss",
#               hue="Data", palette = 'colorblind',linewidth = 0.5)



# lp = sns.lineplot(data = mHistDf.loc[mHistDf['dropout'].isin([0.0,0.3])],x="Epoch", y="Loss",
#               hue="dropout", style="Data", palette = 'colorblind',linewidth = 0.5,size="Data",sizes = [0.5,0.35])

# sns.move_legend(
#     lp, "center left",
#     bbox_to_anchor=(1.05, 0.5), ncol=1, title=None, frameon=False,
# )
# ax.set_ylim([0.1,0.4])
# ax.set_yscale('log')
# ax.grid(axis = 'y',which = 'minor')
# ax.minorticks_on()

plt.savefig('Epsilon_TrainingCurves.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.show()


# %% Fold comparison figures
plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly


# RMSE
fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

# RMSE 
ax = plt.subplot(2,2,(1,2))

g = sns.boxplot(ax=ax, data=RMSEDf_val2, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#029E73",boxprops=dict(alpha=1)) # Remember the models are 1-indexed
h = sns.boxplot(ax=ax, data=RMSEDf_Unseen2, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#0173B2",boxprops=dict(alpha=1)) # Remember the models are 1-indexed

g2 = sns.pointplot(ax = ax,data=RMSEDf_val2, estimator = 'median', ci=None, scale=0.3, color="#029E73", marker='D',linewidth = 0.25)
h2 = sns.pointplot(ax = ax,data=RMSEDf_Unseen2, estimator = 'median', ci=None, scale=0.3, color="#0173B2", marker='o',linewidth = 0.25)

ax.grid(visible=True,which = 'both',axis = 'y')
# plt.title('RMSE')
ax.minorticks_on()
ax.set_ylabel('RMSE',color = 'black')
ax.set_xlabel('',color = 'black')

# RMSE 
ax = plt.subplot(2,2,(3,4))

g = sns.boxplot(ax=ax, data=SSIMDf_val, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#029E73",boxprops=dict(alpha=1)) # Remember the models are 1-indexed
h = sns.boxplot(ax=ax, data=SSIMDf_Unseen, orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#0173B2",boxprops=dict(alpha=1)) # Remember the models are 1-indexed

g2 = sns.pointplot(ax = ax,data=SSIMDf_val, estimator = 'median', ci=None, scale=0.3, color="#029E73", marker='D',linewidth = 0.25)
h2 = sns.pointplot(ax = ax,data=SSIMDf_Unseen, estimator = 'median', ci=None, scale=0.3, color="#0173B2", marker='o',linewidth = 0.25)

ax.grid(visible=True,which = 'both',axis = 'y')
# plt.title('SSIM')
ax.minorticks_on()
ax.set_ylabel('SSIM',color = 'black')
# ax.set_xlabel('Model fold',color = 'black')
# ax.set_xlabel('Model fold',color = 'black')

han1 = matplotlib.patches.Patch(color="#029E73", label='Validation')
han2 = matplotlib.patches.Patch(color="#0173B2", label='Test')
fig.legend(title='Validation or test set',handles = [han1,han2], 
           loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.2))


plt.savefig('ComparisonWithUnseen_datasetSize.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.show()


# plotvals = np.sort(sweepDef[compareParam])
# bins = np.arange(np.min(mErrorDistDf['Distance error [pixels]'])-2, np.max(mErrorDistDf['Distance error [pixels]']), 4)


# # Errors in FI prediction
# fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight)
# fig.set_figwidth(figWidth)

# for i in range(numModels):
#     if i == 0:
#         ax = plt.subplot(numModels,1,i+1)
#         # g = sns.histplot(data = mErrorDistDf[mErrorDistDf[compareParam]==plotvals[i]], ax=ax,x='Distance error [pixels]', hue = 'Data', bins = bins)
#         g = sns.kdeplot(data = errorDf[errorDf[compareParam]==plotvals[i]], ax=ax,x='FI error', hue = 'Data')
#         ax.grid()
#         ax.get_legend().remove()
#         plt.xlabel('')
#         plt.ylabel('')
#         plt.setp(ax.get_xticklabels(), visible=False)
#         ax.set_yticklabels([])
#         plt.xlim([-0.25,0.25])
        
#     else:
#         ax1 = plt.subplot(numModels,1,i+1, sharex = ax)
#         # g = sns.histplot(data = mErrorDistDf[mErrorDistDf[compareParam]==plotvals[i]], ax=ax1,x='Distance error [pixels]', hue = 'Data', bins = bins)
#         g = sns.kdeplot(data = errorDf[errorDf[compareParam]==plotvals[i]], ax=ax1,x='FI error', hue = 'Data')
#         ax1.grid()
#         ax1.get_legend().remove()
#         plt.xlabel('')
#         plt.ylabel('')
#         ax1.set_yticklabels([])
#         plt.xlim([-0.25,0.25])
#         if not i == numModels-1:
#             plt.setp(ax1.get_xticklabels(), visible=False)
#         if i == numModels-1:
#             plt.xlabel('FI error')



# # Error at maximum FI point
# fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight)
# fig.set_figwidth(figWidth)

# for i in range(numModels):
#     if i == 0:
#         ax = plt.subplot(numModels,1,i+1)
#         # g = sns.histplot(data = mErrorDistDf[mErrorDistDf[compareParam]==plotvals[i]], ax=ax,x='Distance error [pixels]', hue = 'Data', bins = bins)
#         g = sns.kdeplot(data = maxPointErrorDf[maxPointErrorDf[compareParam]==plotvals[i]], ax=ax,x='Error at peak FI', hue = 'Data')
#         ax.grid()
#         ax.get_legend().remove()
#         plt.xlabel('')
#         plt.ylabel('')
#         plt.setp(ax.get_xticklabels(), visible=False)
#         ax.set_yticklabels([])
#         plt.xlim([-0.75,0.25])
        
#     else:
#         ax1 = plt.subplot(numModels,1,i+1, sharex = ax)
#         # g = sns.histplot(data = mErrorDistDf[mErrorDistDf[compareParam]==plotvals[i]], ax=ax1,x='Distance error [pixels]', hue = 'Data', bins = bins)
#         g = sns.kdeplot(data = maxPointErrorDf[maxPointErrorDf[compareParam]==plotvals[i]], ax=ax1,x='Error at peak FI', hue = 'Data')
#         ax1.grid()
#         ax1.get_legend().remove()
#         plt.xlabel('')
#         plt.ylabel('')
#         ax1.set_yticklabels([])
#         plt.xlim([-0.75,0.25])
#         if not i == numModels-1:
#             plt.setp(ax1.get_xticklabels(), visible=False)
#         if i == numModels-1:
#             plt.xlabel('Error at peak FI')



# plt.savefig('BatchSize.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
# # plt.show()
# %%
#%% Calculate maximum FI point errors:

# Maximum failure index location
maxGT = np.empty(shape = (repeats, numModels, groundTruths.shape[2]))* np.nan  # Maximum value
maxGTIdx = np.empty(shape = (repeats, numModels, groundTruths.shape[2],2))* np.nan # Maximum value indices
maxGTCoords = maxGTIdx.copy()

maxGT_val = np.empty(shape = (repeats, numModels, groundTruths_val.shape[2]))* np.nan  # Maximum value
maxGTIdx_val = np.empty(shape = (repeats, numModels, groundTruths_val.shape[2],2))* np.nan # Maximum value indices
maxGTCoords_val = maxGTIdx_val.copy()

maxPred = maxGT.copy()
maxPredIdx = maxGTIdx.copy()
maxPredCoords = maxGTCoords.copy()

maxPred_val = maxGT_val.copy()
maxPredIdx_val = maxGTIdx_val.copy()
maxPredCoords_val = maxGTCoords_val.copy()

# Ground truth training samples
for r in range(repeats):
    for m in range(numModels):
        for sp in range(groundTruths.shape[2]):
            maxGT[r,m,sp] = np.max(groundTruths[r,m,sp,:,:]) # Maximum true value
            maxGTIdx[r,m,sp] = np.unravel_index(groundTruths[r,m,sp,:,:].argmax(), groundTruths[r,m,sp,:,:].shape) # 2D index of maximum value in ground truth
            maxGTCoords[r,m,sp] = grid[0,int(maxGTIdx[r,m,sp,0]),int(maxGTIdx[r,m,sp,1])],grid[1,int(maxGTIdx[r,m,sp,0]),int(maxGTIdx[r,m,sp,1])] # Coordinate of maximum FI value

# Ground truth validation samples
for r in range(repeats):
    for m in range(numModels):
        for sp in range(groundTruths_val.shape[2]):
            maxGT_val[r,m,sp] = np.max(groundTruths_val[r,m,sp,:,:]) # Maximum true value
            maxGTIdx_val[r,m,sp] = np.unravel_index(groundTruths_val[r,m,sp,:,:].argmax(), groundTruths_val[r,m,sp,:,:].shape) # 2D index of maximum value in ground truth
            maxGTCoords_val[r,m,sp] = grid[0,int(maxGTIdx_val[r,m,sp,0]),int(maxGTIdx_val[r,m,sp,1])],grid[1,int(maxGTIdx_val[r,m,sp,0]),int(maxGTIdx_val[r,m,sp,1])] # Coordinate of maximum FI value

# Prediction training samples
for r in range(repeats):
    for m in range(numModels):
        for sp in range(predictions.shape[2]):
            maxPred[r,m,sp] = np.max(predictions[r,m,sp,:,:]) # Maximum true value
            maxPredIdx[r,m,sp] = np.unravel_index(predictions[r,m,sp,:,:].argmax(), predictions[r,m,sp,:,:].shape) # 2D index of maximum value in ground truth
            maxPredCoords[r,m,sp] = grid[0,int(maxPredIdx[r,m,sp,0]),int(maxPredIdx[r,m,sp,1])],grid[1,int(maxPredIdx[r,m,sp,0]),int(maxPredIdx[r,m,sp,1])] # Coordinate of maximum FI value

# Prediction validation samples
for r in range(repeats):
    for m in range(numModels):
        for sp in range(predictions_val.shape[2]):
            maxPred_val[r,m,sp] = np.max(predictions_val[r,m,sp,:,:]) # Maximum true value
            maxPredIdx_val[r,m,sp] = np.unravel_index(predictions_val[r,m,sp,:,:].argmax(), predictions_val[r,m,sp,:,:].shape) # 2D index of maximum value in ground truth
            maxPredCoords_val[r,m,sp] = grid[0,int(maxPredIdx_val[r,m,sp,0]),int(maxPredIdx_val[r,m,sp,1])],grid[1,int(maxPredIdx_val[r,m,sp,0]),int(maxPredIdx_val[r,m,sp,1])] # Coordinate of maximum FI value


# Calculate maximum error in distance and print these
x_error = maxGTCoords[:,:,:,0] - maxPredCoords[:,:,:,0]
y_error = maxGTCoords[:,:,:,1] - maxPredCoords[:,:,:,1]
errorDist = np.sqrt(np.square(x_error) + np.square(y_error))
x_error_val = maxPredCoords_val[:,:,:,0] - maxGTCoords_val[:,:,:,0]
y_error_val = maxPredCoords_val[:,:,:,1] - maxGTCoords_val[:,:,:,1]
errorDist_val = np.sqrt(np.square(x_error_val) + np.square(y_error_val))

# Take mean across repeats
mErrorDist = np.mean(errorDist,axis = 0)
mErrorDist_val = np.mean(errorDist_val,axis = 0)

# Into dataframes
mErrorDistDf = pd.DataFrame(mErrorDist.T, columns= sweepDef[compareParam])
mErrorDistDf['Data'] = 'Training'
mErrorDistDf_val = pd.DataFrame(mErrorDist_val.T, columns= sweepDef[compareParam])
mErrorDistDf_val['Data'] = 'Validation'

mErrorDistDf =  pd.concat([mErrorDistDf, mErrorDistDf_val])
mErrorDistDf = pd.melt(mErrorDistDf, id_vars=['Data'], value_vars= sweepDef[compareParam])
mErrorDistDf = mErrorDistDf.rename(columns={"value": "Distance error [pixels]"})



# Calculate errors in value of maximum point and put on dataframe
maxPointError = np.mean(maxPred[:] - maxGT[:],axis = 0) # mean across repeats
maxPointError_val = np.mean(maxPred_val[:] - maxGT_val[:], axis = 0)

maxPointErrorDf = pd.DataFrame(maxPointError.T, columns= sweepDef[compareParam])
maxPointErrorDf['Data'] = 'Training'
maxPointErrorDf_val = pd.DataFrame(maxPointError_val.T, columns= sweepDef[compareParam])
maxPointErrorDf_val['Data'] = 'Validation'

maxPointErrorDf =  pd.concat([maxPointErrorDf, maxPointErrorDf_val])
maxPointErrorDf = pd.melt(maxPointErrorDf, id_vars=['Data'], value_vars= sweepDef[compareParam])
maxPointErrorDf = maxPointErrorDf.rename(columns={"value": "Error at peak FI"})


# Calculate errors in FI and put in dataframe
error = (predictions - groundTruths).reshape(numModels,-1) # Do NOT take mean across repeats, flatten specimens
error_val = (predictions_val - groundTruths_val).swapaxes(0, 1).reshape(numModels,-1)

errorDf = pd.DataFrame(error.T, columns= sweepDef[compareParam])
errorDf['Data'] = 'Training'
errorDf_val = pd.DataFrame(error_val.T, columns= sweepDef[compareParam])
errorDf_val['Data'] = 'Validation'

errorDf =  pd.concat([errorDf, errorDf_val])
errorDf = pd.melt(errorDf, id_vars=['Data'], value_vars= sweepDef[compareParam])
errorDf = errorDf.rename(columns={"value": "FI error"})


# %% Data distribution figures
plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

# MC24 fold 4

# Plot of Max FI prediction vs ground truth


gt_train = groundTruths[0].reshape(4,-1) # Should be the same for all repetitions (since we do manual folds and don't shuffle beforehand)
gt_val = groundTruths_val[0].reshape(4,-1) # Should be the same for all repetitions (since we do manual folds and don't shuffle beforehand)

gt_trainDf = pd.DataFrame(gt_train.T, columns= sweepDef[compareParam])
gt_trainDf['Data'] = 'Training'
gt_trainDf['Datapoint'] = gt_trainDf.index
gt_valDf = pd.DataFrame(gt_val.T, columns= sweepDef[compareParam])
gt_valDf['Data'] = 'Validation'
gt_valDf['Datapoint'] = gt_valDf.index

gtDf =  pd.concat([gt_trainDf, gt_valDf])
gtDf = pd.melt(gtDf, id_vars=['Data','Datapoint'], value_vars= sweepDef[compareParam])
gtDf = gtDf.rename(columns={"value": "FI"})


ks_stat = np.zeros(gt_train.shape[0])
ks_p = np.zeros(gt_train.shape[0])
for i in range(gt_train.shape[0]):
    val =gt_val[i] 
    tr = gt_train[i]
    ks = scipy.stats.kstest(val[~np.isnan(val)], tr[~np.isnan(tr)])
    # ks = scipy.stats.ks_2samp(gt_val[i], gt_train[i])
    ks_stat[i] = ks.statistic
    ks_p[i] = ks.pvalue


fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(1,1,1)

train_sort = np.sort(gt_train[0])
val_sort = np.sort(gt_val[0])

# calculate the proportional values of samples
prop_train = 1. * np.arange(len(train_sort)) / (len(train_sort) - 1)
prop_val = 1. * np.arange(len(val_sort)) / (len(val_sort) - 1)



lp = ax.plot(train_sort,prop_train )
lp2 = ax.plot(val_sort,prop_val)
plt.show()


# %% Violin plots of distributions





fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(1,1,1)
bins = np.arange(-5, 150, 10)
plt.grid()
g = sns.violinplot(ax=ax, data=gtDf,x = 'Fold',y = 'FI',hue="Data", split=True, gap=.2, inner="quart",linewidth=1,saturation = 1)
# g = sns.stripplot(ax=ax, data=gtDf,x = 'Fold',y = 'FI',hue="Data",dodge=True, alpha=.25)
# plt.title('Pearson correlation with failure index')
plt.ylabel('Failure Index')
plt.xlabel('')
g.set_xticks(range(len(sweepDef[compareParam])))
g.set_xticklabels(sweepDef[compareParam])
g.set_xlabel('Fold')
# ax.get_legend().remove()
for collection in ax.collections:
    if isinstance(collection, matplotlib.collections.PolyCollection):
        collection.set_edgecolor(matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=1))
        collection.set_facecolor(matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=0.5))
        edgecol = matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=1)
        facecol = matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=0.5)
for han in ax.get_legend_handles_labels():
    for hand in han:
        if isinstance(hand, matplotlib.patches.Rectangle):
            hand.set_edgecolor(hand.get_facecolor())
            hand.set_facecolor(matplotlib.colors.to_rgba(hand.get_facecolor()[:-1], alpha=0.5))
            # h.set_linewidth(1.5)

h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()

fig.legend(title='Dataset',handles = h,labels=l, 
           loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.15))


plt.savefig('MC24_TrainValDistributions.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)

plt.show()






# %% KS plot
plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

ks_LFC18 = np.array([0.0159596 , 0.01266667, 0.02237374, 0.01494949, 0.02290909,
       0.01608081, 0.02179798, 0.01757576, 0.01980808, 0.01336364])
ks_MC24 = np.array([0.07292593, 0.096     , 0.03533333, 0.06374074, 0.04464815,
       0.0567037 , 0.11962963, 0.09542593, 0.03935185, 0.11164815])


ksP_LFC18 = np.array([1.27669193e-02, 8.27269170e-02, 9.76406316e-05, 2.37068630e-02,
       6.03909958e-05, 1.18209402e-02, 1.61633791e-04, 4.36028759e-03,
       8.33987545e-04, 5.77365143e-02])

ksP_MC24 = np.array([2.15331604e-050, 4.40472287e-087, 3.77074448e-012, 1.36749997e-038,
       3.80226817e-019, 1.26349983e-030, 3.87596003e-135, 4.77716130e-086,
       5.73031658e-015, 1.02525092e-117])



ks_LargerDataset = np.array([0.06314815, 0.0857963 , 0.02971111, 0.01745185])

ks_LargerDatasetDf = pd.DataFrame(ks_LargerDataset)
ks_LargerDatasetDf['Specimens'] = [100,200,500,1000]
ks_LargerDatasetDf = pd.melt(ks_LargerDatasetDf, id_vars=['Specimens'])


ks_LFC18Df = pd.DataFrame(ks_LFC18)
ks_LFC18Df['Dataset'] = 'LFC18'
ks_LFC18Df['Fold'] = ks_LFC18Df.index+1
# ks_LFC18Df = ks_LFC18Df.rename({0:'D'})
ks_MC24Df = pd.DataFrame(ks_MC24)
ks_MC24Df['Dataset'] = 'MC24'
ks_MC24Df['Fold'] = ks_MC24Df.index+1
# ks_MC24Df = ks_MC24Df.rename({0:'D'})

ksDf =  pd.concat([ks_LFC18Df, ks_MC24Df])
ksDf = pd.melt(ksDf, id_vars=['Dataset','Fold'])
ksDf = ksDf.rename({'variable':'D'})
ksDf = ksDf.rename({'value':'D'})

ksP_LFC18Df = pd.DataFrame(ksP_LFC18)
ksP_LFC18Df['Dataset'] = 'LFC18'
ksP_LFC18Df['Fold'] = ksP_LFC18Df.index+1
# ks_LFC18Df = ks_LFC18Df.rename({0:'D'})
ksP_MC24Df = pd.DataFrame(ksP_MC24)
ksP_MC24Df['Dataset'] = 'MC24'
ksP_MC24Df['Fold'] = ksP_MC24Df.index+1
# ks_MC24Df = ks_MC24Df.rename({0:'D'})

ksPDf =  pd.concat([ksP_LFC18Df, ksP_MC24Df])
ksPDf = pd.melt(ksPDf, id_vars=['Dataset','Fold'])
ksPDf = ksPDf.rename({'variable':'D'})
ksPDf = ksPDf.rename({'value':'P value'})




fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(1,1,1)
# bp = sns.barplot(data=ksPDf,ax = ax, x = 'Fold',y = "value" ,orient = 'v',hue = 'Dataset')
bp = sns.barplot(data=ks_LargerDatasetDf,ax = ax, x = 'Specimens',y = "value" ,orient = 'v')
ks_LargerDatasetDf
# ax.set_ylabel('D statistic')
ax.set_ylabel('D statistic')
plt.grid()
h,l = ax.get_legend_handles_labels()
# ax.get_legend().remove()
# ax.set_yscale('log')
# fig.legend(title='Dataset',handles = h,labels=l, 
#            loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.2))

plt.savefig('Kolmorogov_stat_DatasetSize.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)

plt.show()

# %% Mean HP improvements plot

plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

LFC18_Meanpath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\Hyperparameter_Optimisation_results\LFC18_HP_Res_Means.xlsx"
MC24_Meanpath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\Hyperparameter_Optimisation_results\MC24_HP_Res_Means.xlsx"
LFC18_Means = pd.read_excel(LFC18_Meanpath,header=None)
MC24_Means = pd.read_excel(MC24_Meanpath,header=None)
LFC18_Means = LFC18_Means.rename(columns={0:'HP'})
MC24_Means = MC24_Means.rename(columns={0:'HP'})


LFC18_default = pd.DataFrame({'Train-val ratio': [0.9], 
                            'Batch size': [8],
                            'Kernel size': [3],
                            'Optimizer': ['Adam'],
                            'Activation Func.': ['tanh'],
                            'Loss': ['MSE'],
                            'Dropout rate': [0.1],
                            'Initial LR': [0.001],
                            'LR decay': [1],
                            'Max pooling': ['On'],
                            'Model depth': [3],
                            'Skip connections': ['Off'],
                            'Scaling': ['Std.'],
                            'Decoder activation': ['Off'],
                            'Epsilon': [1e-07]})

MC24_default = pd.DataFrame({'Train-val ratio': [0.9], 
                            'Batch size': [8],
                            'Kernel size': [3],
                            'Optimizer': ['Adam'],
                            'Activation Func.': ['relu'],
                            'Loss': ['MSE'],
                            'Dropout rate':[0.1],
                            'Initial LR': [0.001],
                            'LR decay': [1],
                            'Max pooling': ['On'],
                            'Model depth': [4],
                            'Skip connections': ['Off'],
                            'Scaling': ['Std.'],
                            'Decoder activation': ['On'],
                            'Epsilon': [1e-03],
                            })

LFC18_defRMSE = LFC18_default.copy()
MC24_defRMSE = MC24_default.copy()
LFC18_defSSIM = LFC18_default.copy()
MC24_defSSIM = MC24_default.copy()

LFC18_optRMSE = LFC18_default.copy()
MC24_optRMSE = MC24_default.copy()
LFC18_optSSIM = LFC18_default.copy()
MC24_optSSIM = MC24_default.copy()

for i in range(len(MC24_default.iloc[0])):
    hpIdx = np.where(LFC18_Means['HP'] == LFC18_default.iloc[0].index[i])
    RMSEIdx = hpIdx[0]+1
    SSIMIdx = hpIdx[0]+2
    defcol_LFC18 = np.where(LFC18_Means.loc[hpIdx] == LFC18_defRMSE.loc[0][i])[1]
    defcol_MC24 = np.where(MC24_Means.loc[hpIdx] == MC24_defRMSE.loc[0][i])[1]

    # Default values
    LFC18_defRMSE.iloc[0,i] = LFC18_Means.iloc[RMSEIdx[0],defcol_LFC18[0]]
    MC24_defRMSE.iloc[0,i] = MC24_Means.iloc[RMSEIdx[0],defcol_MC24[0]]
    LFC18_defSSIM.iloc[0,i] = LFC18_Means.iloc[SSIMIdx[0],defcol_LFC18[0]]
    MC24_defSSIM.iloc[0,i] = MC24_Means.iloc[SSIMIdx[0],defcol_MC24[0]]

    # Take minimum values
    LFC18_optRMSE.iloc[0,i] = np.nanmin(np.array([x for x in LFC18_Means.iloc[RMSEIdx[0]].values if isinstance(x, (int, float))]))
    LFC18_optSSIM.iloc[0,i] = np.nanmax(np.array([x for x in LFC18_Means.iloc[SSIMIdx[0]].values if isinstance(x, (int, float))]))
    MC24_optRMSE.iloc[0,i] = np.nanmin(np.array([x for x in MC24_Means.iloc[RMSEIdx[0]].values if isinstance(x, (int, float))]))
    MC24_optSSIM.iloc[0,i] = np.nanmax(np.array([x for x in MC24_Means.iloc[SSIMIdx[0]].values if isinstance(x, (int, float))]))


LFC18_defRMSE['Value'] = 'Default'
LFC18_defRMSE['Dataset'] = 'LFC18'
LFC18_defSSIM['Value'] = 'Default'
LFC18_defSSIM['Dataset'] = 'LFC18'
MC24_defRMSE['Value'] = 'Default'
MC24_defRMSE['Dataset'] = 'MC24'
MC24_defSSIM['Value'] = 'Default'
MC24_defSSIM['Dataset'] = 'MC24'

LFC18_optRMSE['Value'] = 'Best'
LFC18_optRMSE['Dataset'] = 'LFC18'
LFC18_optSSIM['Value'] = 'Best'
LFC18_optSSIM['Dataset'] = 'LFC18'
MC24_optRMSE['Value'] = 'Best'
MC24_optRMSE['Dataset'] = 'MC24'
MC24_optSSIM['Value'] = 'Best'
MC24_optSSIM['Dataset'] = 'MC24'

RMSE_Means = pd.concat([LFC18_defRMSE,MC24_defRMSE,LFC18_optRMSE,MC24_optRMSE])
SSIM_Means = pd.concat([LFC18_defSSIM,MC24_defSSIM,LFC18_optSSIM,MC24_optSSIM])

RMSE_Means = pd.melt(RMSE_Means, id_vars=['Value','Dataset'], value_name='RMSE')
SSIM_Means = pd.melt(SSIM_Means, id_vars=['Value','Dataset'], value_name='SSIM')

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(1,1,1)


plotparam = 'SSIM'
plotDs = 'LFC18'
if plotparam == 'RMSE':
    plotDf = RMSE_Means.loc[RMSE_Means['Dataset'] == plotDs]
else:
    plotDf = SSIM_Means.loc[RMSE_Means['Dataset'] == plotDs]

p = sns.pointplot(ax = ax,data=plotDf,x='variable', y=plotparam, hue='Value',dodge=.2, linestyle="none", errorbar=None,marker="_")

for n in range(len(MC24_default.iloc[0])):
    difVals = plotDf.loc[plotDf['variable'] == LFC18_default.iloc[0].index[n]]
    if not difVals.iloc[0][plotparam] == difVals.iloc[1][plotparam]:
        # if plotparam == 'RMSE': # RMSE is better if it's lower
        ax.annotate("",
        xy=(n, difVals.iloc[0][plotparam]), xycoords='data',
        xytext=(n, difVals.iloc[1][plotparam]), textcoords='data',
        arrowprops=dict(arrowstyle="<-",
                        connectionstyle="arc3", color='k', lw=0.5),
        )
            # ax.arrow(x = n, y = difVals.iloc[0][plotparam], dx = 0, dy = difVals.iloc[1][plotparam]-difVals.iloc[0][plotparam],width=.002)

        # else 
        # ax.annotate("",
        #     xy=(n, difVals.iloc[1][plotparam]), xycoords='data',
        #     xytext=(n, difVals.iloc[0][plotparam]), textcoords='data',
        #     arrowprops=dict(arrowstyle="<-",
        #                     connectionstyle="arc3", color='k', lw=0.5),
        #     )



ax.set_xticklabels( 
    labels=ax.get_xticklabels(), rotation=90) 
ax.set_xlabel('')
plt.grid()
h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()
# ax.set_yscale('log')
fig.legend(title='Score',handles = h,labels=l, 
           loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.15))

# plt.savefig('LFC18_MeanRMSE_Improvement.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.savefig('LFC18_MeanSSIM_Improvement.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)

# plt.savefig('MC24_MeanRMSE_Improvement.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
# plt.savefig('MC24_MeanSSIM_Improvement.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)

plt.show()
# %% Mean HP improvements plot

plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

OptGains = pd.DataFrame({'Dataset':['LFC18']*3+['MC24']*3+['LFC18']*3+['MC24']*3,'Model Version':['Baseline','Best Hyperparameters','Cross-validation']*4,'Measure':['RMSE']*6+['SSIM']*6,'Score':[0.13674,0.13529,0.1333,0.08324,0.07817,0.0815,0.86309,0.86465,0.8692,0.73163,0.74835,0.7477]})

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
# g = sns.catplot(
#     data=OptGains, x="Dataset", y="Score", col="Measure",hue='Model Version',
#     kind="strip", height=4, aspect=.6,
# )
p = sns.pointplot(ax = ax,data=OptGains.loc[OptGains['Measure'] == 'RMSE'],x='Dataset', y='Score', hue='Model Version',dodge=0.4, linestyle="none", errorbar=None,marker="_")
p2 = sns.pointplot(ax = ax2,data=OptGains.loc[OptGains['Measure'] == 'SSIM'],x='Dataset', y='Score', hue='Model Version',dodge=0.4, linestyle="none", errorbar=None,marker="_")
ax.grid()
ax2.grid()
ax.set_ylabel('RMSE')
ax2.set_ylabel('SSIM')
h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()
ax2.get_legend().remove()
# ax.set_yscale('log')
fig.legend(title='Model version',handles = h,labels=l, 
           loc="lower center", ncol=3,bbox_to_anchor=(0.5, -0.2))



plt.savefig('Total_Improvement_HP.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)



# %%

def plot_contour(grid, samples2D,  ax, xlab = None, ylab = None, cbarlab = None, cBarBins = 5):
    if np.min(samples2D) == np.max(samples2D):
         CS = ax.contourf(grid[0],grid[1],samples2D)
         cbar = fig.colorbar(CS,ticks=[], shrink = 0.85)
         cbar.ax.text(0.1, -0.03, round(np.min(samples2D),2), transform=cbar.ax.transAxes, 
            va='top', ha='left')
         cbar.set_label(cbarlab, rotation=270,labelpad=7)
    else:
        CS = ax.contourf(grid[0],grid[1],samples2D,levels=np.linspace(np.min(samples2D), np.max(samples2D), 10))
        cbar = fig.colorbar(CS,ticks=[], shrink = 0.85)
        cbar.ax.text(0.1, -0.03, round(np.min(samples2D),2), transform=cbar.ax.transAxes, 
            va='top', ha='left')
        cbar.ax.text(0.1, 1.03, round(np.max(samples2D),2), transform=cbar.ax.transAxes, 
            va='bottom', ha='left')
        cbar.set_label(cbarlab, rotation=270,labelpad=7)
    
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # cbar = fig.colorbar(CS,ticks=[np.min(samples2D), np.max(samples2D)], shrink = 0.8)
    
    # cbar.ax.locator_params(nbins=cBarBins)


# ModelFCNN # FFNN model
Yscaler = preprocessing.StandardScaler() # Do a scaler for the in and outputs separately (to be able to inversely standardise predictions)

Yscaler.fit(LFC18_Y.reshape(LFC18_Y.shape[0]*LFC18_Y.shape[1]*LFC18_Y.shape[2],-1)) # Scaler only takes input of shape (data,features)
# MC24_Y_Unseen = Yscaler.transform(MC24_Y_Unseen.reshape(MC24_Y_Unseen.shape[0]*MC24_Y_Unseen.shape[1]*MC24_Y_Unseen.shape[2],-1))
# MC24_Y_Unseen = MC24_Y_Unseen.reshape(MC24_Y_Unseen_shape) # reshape to 2D samples




FFNNPred = modelFCNN.predict(LFC18_X)
FFNNPred = Yscaler.inverse_transform(FFNNPred.reshape(FFNNPred.shape[0]*FFNNPred.shape[1]*FFNNPred.shape[2],-1))
FFNNPred = FFNNPred.reshape(100,55,20,1)

CNN_model = keras.models.load_model(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\LFC18BaselineCrossValidation2208_1\dataout\model_LFC18BaselineCrossValidation2208_1_1.keras")

CNNPred = CNN_model.predict(LFC18_X)
CNNPred = Yscaler.inverse_transform(CNNPred.reshape(CNNPred.shape[0]*CNNPred.shape[1]*CNNPred.shape[2],-1))
CNNPred = CNNPred.reshape(100,55,20,1)

ax = plt.subplot(1,3,1)
plot_contour(grid=grid, samples2D=LFC18_Y[0].reshape(55,20),  ax=ax, xlab = None, ylab = None, cbarlab = 'FI', cBarBins = 5)
ax = plt.subplot(1,3,2)
# pred_Unseen_temp = temp_model.predict(MC24_X_Unseen)
plot_contour(grid=grid, samples2D=FFNNPred[0].reshape(55,20),  ax=ax, xlab = None, ylab = None, cbarlab = 'FI', cBarBins = 5)
ax = plt.subplot(1,3,3)
# pred_Unseen_temp = temp_model.predict(MC24_X_Unseen)
plot_contour(grid=grid, samples2D=CNNPred[0].reshape(55,20),  ax=ax, xlab = None, ylab = None, cbarlab = 'FI', cBarBins = 5)

plt.savefig('Total_Improvement_HP.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.show()
# pred_Unseen_temp = Yscaler.inverse_transform(pred_Unseen_temp.reshape(pred_Unseen_temp.shape[0]*pred_Unseen_temp.shape[1]*pred_Unseen_temp.shape[2],-1))
# pred_Unseen_temp = pred_Unseen_temp.reshape(1000,60,20,1)

# ax = plt.subplot(1,2,1)
# plot_contour(grid=grid, samples2D=MC24_Y[0].reshape(60,20),  ax=ax, xlab = None, ylab = None, cbarlab = None, cBarBins = 5)
# ax = plt.subplot(1,2,2)
# # temp_model = keras.models.load_model(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24DatasetSize2908_1\dataout\model_MC24DatasetSize2908_1_4.keras")
# # pred_Unseen_temp = temp_model.predict(MC24_X_Unseen)
# plot_contour(grid=grid, samples2D=pred_Unseen_temp2[0].reshape(60,20),  ax=ax, xlab = None, ylab = None, cbarlab = None, cBarBins = 5)


# temp_model = keras.models.load_model(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24DatasetSize2908_1\dataout\model_MC24DatasetSize2908_1_4.keras")
# pred_Unseen_temp2 = temp_model.predict(MC24_X)
plt.show()



# %% KS plot for the 1000 dataset samples

MC24_ks_trainsets = np.empty(shape = (10,900, 60, 20)) * np.nan
MC24_ks_valsets = np.empty(shape = (10,100, 60, 20)) * np.nan

 # The k fold variable is 1-indexed, valIdx marks the start of the validation set in this fold





for i in range(10):
    valIdx = int((i)*(1000/10))
    valIndeces = [*range(valIdx,valIdx+int(1000/10),1)]
    trainIndeces =[*range(0,valIdx,1),*range(valIdx+int(numSamples/10),numSamples,1)]
    MC24_ks_trainsets[i] = MC24_Y_Unseen[trainIndeces].reshape(-1,60,20)
    MC24_ks_valsets[i] = MC24_Y_Unseen[valIndeces].reshape(-1,60,20)

ks_stat = np.zeros(10)
ks_p = np.zeros(10)
for i in range(10):
    val =MC24_ks_valsets[i] 
    tr = MC24_ks_trainsets[i]
    ks = scipy.stats.kstest(val[~np.isnan(val)], tr[~np.isnan(tr)])
    # ks = scipy.stats.ks_2samp(gt_val[i], gt_train[i])
    ks_stat[i] = ks.statistic
    ks_p[i] = ks.pvalue



    
fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(1,1,1)

train_sort = np.sort(MC24_ks_trainsets[0])
val_sort = np.sort(MC24_ks_valsets[0])

# calculate the proportional values of samples
prop_train = 1. * np.arange(len(train_sort)) / (len(train_sort) - 1)
prop_val = 1. * np.arange(len(val_sort)) / (len(val_sort) - 1)



lp = ax.plot(train_sort,prop_train )
lp2 = ax.plot(val_sort,prop_val)
plt.show()