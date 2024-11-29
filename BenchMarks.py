# %%

#####################################################################
# Description
#####################################################################
'''
This script allows training and evaluation of Benchmark models locally
using the LFC18 and MC24 datasets


Inputs:
sweep_definition_{jn}.csv: sweep definition file with same jobname as -j, placed in same directory as this script

Ouputs:
All saved to outputs folder where 
{jn} = jobname
{num} = index of model in sweep definition

trainHist_{jn}_{num}.json: training history (curves)
predictions_{jn}_{num}.json: Inversely scaled training specimen predictions - i.e. predictions in the real label scale
predictions_val_{jn}_{num}.json: Same for validation
groundTruth_{jn}_{num}.json: Ground truths in the true label scale
groundTruth_val_{jn}_{num}.json: same for validation
parameters_{jn}_{num}.json: model hyperparameters
input_{jn}_{num}.json: Input features used in model
RMSE_{jn}_{num}.json: Training RMSE across whole dataset
RMSE_val_{jn}_{num}.json: Validation RMSE across whole dataset
model_{jn}_{num}.keras: ".keras" file with trained model
'''
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
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

import seaborn as sns
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

#%% Test model loading 
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\fullSweep1106_repeat1\dataout\model_fullSweep1106_repeat1_1.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\dataoutTESTJOB\model_TESTJOB_1.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24CrossValidation2808_1\dataout\model_MC24CrossValidation2808_1_1.keras"

# loaded_model = keras.models.load_model(modelPath)
# loaded_model.summary()

#%% Settings for test script
sweep_defPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\sweep_definition_benchmarks.csv'
jobname = 'UNet_Test'
sweepIdx = 1
yNames = ['FI'] # Names of ground truth features in input csv
normalizerLength = 20 # Number of random samples used for computation of mean and variance used in data normalisation 

# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)

#%% Automatic setting of variables
sweep_params = pd.read_csv(sweep_defPath)
sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[sweepIdx]
parallel = 1

# %% Various settings
timeStamp = datetime.datetime.now().strftime("%Y%m%d%H%M") # Not currently used
histOutName = 'trainHist_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Training history file
histOutPath = os.path.join('dataoutTESTJOB',histOutName)
predOutName = 'predictions_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Predictions
predOutPath = os.path.join('dataoutTESTJOB',predOutName)
predOutName_val = 'predictions_val_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Predictions
predOutPath_val = os.path.join('dataoutTESTJOB',predOutName_val)
gtOutName = 'groundTruth_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Ground truths
gtOutPath = os.path.join('dataoutTESTJOB',gtOutName)
gtOutName_val = 'groundTruth_val_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Ground truths
gtOutPath_val = os.path.join('dataoutTESTJOB',gtOutName_val)
paramOutName = 'parameters_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Hyperparameters
paramOutPath = os.path.join('dataoutTESTJOB',paramOutName)
inputOutName = 'input_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Model inputs
inputOutPath = os.path.join('dataoutTESTJOB',inputOutName)
modelOutPath = 'model_{jn}_{num}.keras'.format(jn=jobname, num = parallel) # Model architecture and weights
modelOutPath = os.path.join('dataoutTESTJOB',modelOutPath)
RMSEOutPath = 'RMSE_{jn}_{num}.json'.format(jn=jobname, num = parallel) # RMSE of model to quickly compare models when many are trained in a sweep
RMSEOutPath = os.path.join('dataoutTESTJOB',RMSEOutPath)
RMSEOutPath_val = 'RMSE_val_{jn}_{num}.json'.format(jn=jobname, num = parallel) # RMSE of model to quickly compare models when many are trained in a sweep
RMSEOutPath_val = os.path.join('dataoutTESTJOB',RMSEOutPath_val)

#%% Import data

if params['Dataset'] == 'MC24_224': # High-resolution MC24 dataset
  trainDat_name = 'MatLabModel2024' 
  sampleShape = [224,224]
  xNames = ['Ex','Ey','Gxy','Vf','c2'] # Names of input features in input csv
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples' # Path for training data samples
  samplesPerFile = 40
elif params['Dataset'] == 'LFC18': # ABAQUS DATA FROM GAUDRON2018
  trainDat_name = 'Gaudron2018' 
  sampleShape = [55,20]
  xNames = ['E11','E22','E12'] # Names of input features in input csv
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain' # Path for training data samples
  samplesPerFile = 1
elif params['Dataset'] == 'MC24': # MECOMPOSITES MODEL FROM 2024 (100 samples)
  trainDat_name = 'MatLabModel2024' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240703_1417_100Samples'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
elif params['Dataset'] == 'MC24_200': # MECOMPOSITES MODEL FROM 2024 (1000 samples)
  trainDat_name = 'MatLabModel2024_200' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
elif params['Dataset'] == 'MC24_500': # MECOMPOSITES MODEL FROM 2024 (1000 samples)
  trainDat_name = 'MatLabModel2024_500' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
elif params['Dataset'] == 'MC24_1000': # MECOMPOSITES MODEL FROM 2024 (1000 samples)
  trainDat_name = 'MatLabModel2024_1000' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1233_1kSamples'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
elif params['Dataset'] == 'MC24_10000': # MECOMPOSITES MODEL FROM 2024 (10,000 samples)
  trainDat_name = 'MatLabModel2024_10000' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1239_10kSamples'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
elif params['Dataset'] == 'MC24_100000': # MECOMPOSITES MODEL FROM 2024 (100,000 samples)
  trainDat_name = 'MatLabModel2024_100000' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1439_100kSamples'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1

# %%

yNames = ['FI'] # Names of ground truth features in input csv
numSamples = len(os.listdir(trainDat_path))*samplesPerFile # number of samples is number of files in datain
batchSize = params['batchSize'] # Batch size for training
valSize = math.floor(params['valSize']*numSamples) # Training and validation data split ratio
testSize = math.floor(params['testSize']*numSamples)
train_length = numSamples-valSize-testSize # Number of training samples 
epochs = params['Epochs'] # Max epochs for training
# epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = math.ceil((numSamples-train_length) / batchSize)

# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)


def loadSample(path = str):
  '''
  Imports data in parquet and formats into a tensorflow dataset
  '''
  # Read sample csv data
  sample = pd.read_parquet(path, engine='auto')
  headers = np.array(sample.columns.values.tolist())
  samples = [y for x, y in sample.groupby('specimen')] # Group by specimen
  samples = np.array(list(map(lambda x: x.to_numpy(), samples))) # Put in 3D array
  samples = samples.reshape(samples.shape[0],sampleShape[0],sampleShape[1],-1) # Reshape to 2D


  # Find indeces of input features 
  featureIdx = []
  for name in xNames:
    featureIdx += [np.where(headers == name)[0][0]]

  # Find indeces of ground truth features 
  gtIdx = []
  for name in yNames:
    gtIdx += [np.where(headers == name)[0][0]]

  X = samples[:,:,:,featureIdx] # Input features

  Y = samples[:,:,:,gtIdx] # Labels

  ds = tf.data.Dataset.from_tensor_slices((X, Y))

  return headers, ds

# Import all data samples and store in one dataset
for i,file in enumerate(os.listdir(trainDat_path)):
    print('Now loading file number {num} out of {total}'.format(num = i+1, total = numSamples/samplesPerFile))
    filepath = os.path.join(trainDat_path,file)
    if i==0:
        headers, samples = loadSample(filepath)
    else:
        addSamp = loadSample(filepath)[1]
        samples = samples.concatenate(addSamp)
samples = samples.shuffle(buffer_size=len(samples)) # Shuffle set
train_ds = samples.take(train_length)
remaining = samples.skip(train_length)
val_ds = remaining.take(valSize)
test_ds = remaining.skip(valSize)

X_trainShape = (train_ds.cardinality().numpy(),sampleShape[0],sampleShape[1],len(xNames))
X_valShape = (val_ds.cardinality().numpy(),sampleShape[0],sampleShape[1],len(xNames))
X_testShape = (test_ds.cardinality().numpy(),sampleShape[0],sampleShape[1],len(xNames))
y_trainShape = (train_ds.cardinality().numpy(),sampleShape[0],sampleShape[1],len(yNames))
y_valShape = (val_ds.cardinality().numpy(),sampleShape[0],sampleShape[1],len(yNames))
y_testShape = (test_ds.cardinality().numpy(),sampleShape[0],sampleShape[1],len(yNames))
    
# %%
# Define mean and variance for normalization based only on training set
feature_ds = train_ds.take(normalizerLength).map(lambda x, y: x) 
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(feature_ds)


# %% Load models


def Xception_Model(inputShape, outputShape, params):
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  x = tf.keras.applications.xception.preprocess_input(input)
  
  base_model = tf.keras.applications.Xception(
      include_top=False,
      weights=None,
      input_tensor=None,
      input_shape=inputShape,
      pooling=None,
      classes=1000,
      classifier_activation="softmax",
      name="xception",
  )
  x = base_model(inputs = x)

  
  x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',strides = 2,activation=params['conv1Activation'])(x)
  x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',strides = 2,activation=params['conv1Activation'])(x)
  x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',strides = 2,activation=params['conv1Activation'])(x)
  x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',strides = 2,activation=params['conv1Activation'])(x)
  x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',strides = 2,activation=params['conv1Activation'])(x)

  outputs = tf.keras.layers.Conv2D(outputShape[-1], 3, activation="linear", padding="same")(x)
  model = tf.keras.Model(input, outputs)
  return model

Xception = Xception_Model(inputShape = X_trainShape[1:], outputShape=y_trainShape[1:], params = params)



# %%





randAug = tf.random.Generator.from_seed(seed) # Random number generator used for random augmentations
def augmentImage(inputMatrices,gtMatrix):
    '''
  Apply augmentations to increase the dataset size

  Args
  ----------
  inputMatrices: the Batchx55x20x3 input
  gtMatrix: the Batchx55x30 ground truth

  Returns
  ----------
  inputMatrices,gtMatrix with consistent augmentations applied

  '''
    height, width = sampleShape[0], sampleShape[1] # image dimensions

    if randAug.normal([]) > 0: # Randomly flip an image horizontally 50% of the time
      inputMatrices = tf.image.flip_left_right(inputMatrices)
      gtMatrix = tf.image.flip_left_right(gtMatrix)
      # gtMatrix = tf.image.flip_left_right(tf.reshape(gtMatrix,[-1,height,width,1]))
      # gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

    if  randAug.normal([]) > 0: # Randomly flip an image vertically 50% of the time
      inputMatrices = tf.image.flip_up_down(inputMatrices)
      gtMatrix = tf.image.flip_up_down(gtMatrix)
      # gtMatrix = tf.image.flip_up_down(tf.reshape(gtMatrix,[-1,height,width,1]))
      # gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

    # We can crop and resize but this messes with the boundary conditions hence not done right now
    # if randAug.normal([]) > 0.67: # Scale to a random size within the bounding box and fit to a random location within this
    #   crop_width = randAug.uniform(shape=(), minval=math.floor(0.7 * width), maxval=math.floor(0.9 * width), dtype = tf.int32)
    #   crop_height = randAug.uniform(shape=(), minval=math.floor(0.7 * height), maxval=math.floor(0.9 * height), dtype = tf.int32)
    #   offset_x = randAug.uniform(shape=(), minval=0, maxval=(width - crop_width), dtype = tf.int32)
    #   offset_y = randAug.uniform(shape=(), minval=0, maxval=(height - crop_height), dtype = tf.int32)

    #   inputMatrices = tf.image.crop_to_bounding_box(inputMatrices, offset_y, offset_x, crop_height, crop_width) # Crop to bounding box
    #   gtMatrix = tf.image.crop_to_bounding_box(tf.reshape(gtMatrix,[-1,height,width,1]), offset_y, offset_x, crop_height, crop_width)
    #   newHeight = crop_height
    #   newWidth = crop_width
    #   gtMatrix = tf.reshape(gtMatrix,[-1,newHeight,newWidth]) # Reshape ground truth back to not have channels dimensions
    
    #   inputMatrices = tf.image.resize(inputMatrices, (height, width)) # Resize to original size (we want all images same size) - this distorts the image
    #   gtMatrix = tf.image.resize(tf.reshape(gtMatrix,[-1,newHeight,newWidth,1]), (height, width), method='nearest')
    #   gtMatrix = tf.reshape(gtMatrix,[-1,height,width])
    #   inputMatrices = tf.cast(inputMatrices, tf.float64)
    #   gtMatrix = tf.cast(gtMatrix, tf.float64)

      
    return (inputMatrices,gtMatrix)


def show_prediction(sample, predictions, names, ground_truth, grid):
  '''
  For a given dataset plots one image, its true mask, and predicted mask

  Args
  ----------
  sample: the input
  prediction: the predicted field
  ground_truth: the ground truth field

  Returns
  ----------
  Nothing, graph will be displayed

  '''
  # Plot  inputs
  fig, axs = plt.subplots(1, sample.shape[-1],figsize=[10,5]) # Create subplots to fit input fields
  for i in range(sample.shape[-1]):
    ax = plt.subplot(1, sample.shape[-1], i+1)
    CS = ax.contourf(grid[0],grid[1],sample[:,:,i], cmap = 'jet')
    plt.xlabel('x')
    plt.ylabel('y')
    fig.colorbar(CS)
    plt.title('input'+str(i+1))

  # Plot outputs
  fig, axs = plt.subplots(1, len(names)+1,figsize=[5*len(names), 5]) # Create subplots to fit output fields

  ax = plt.subplot(1, len(names)+1, 1)
  CS = ax.contourf(grid[0],grid[1],ground_truth, cmap = 'jet')
  plt.xlabel('x')
  plt.ylabel('y')
  fig.colorbar(CS)
  plt.title('ground truth')

  for idx, pred in enumerate(predictions):
    ax = plt.subplot(1, len(names)+1, idx+2)
    CS2 = ax.contourf(grid[0],grid[1],predictions[idx], cmap = 'jet', levels = CS.levels)
    plt.xlabel('x')
    plt.ylabel('y')
    fig.colorbar(CS2)
    plt.title(names[idx])

# Format is csv files with columns 
# Try new method of loading samples


# The below are just used for validation and shape (TODO: replace validation method with tf tensors)
# X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, Y, train_size=trainValRatio, shuffle = True)
# X_trainShape = X_train.shape
# X_valShape = X_val.shape
# y_trainShape = y_train.shape
# y_valShape = y_val.shape




# Training preprocessing
train_ds = train_ds.cache() # cache dataset for it to be used over iterations. Any operation before this will not be reapplied each iteration
train_ds = train_ds.shuffle(buffer_size = len(train_ds)).batch(batchSize) # Shuffle for random order
train_ds = train_ds.repeat() # Repeats dataset indefinitely to avoid errors
# if params['dsAugmentation'] == 1: # We can apply dataset augmentation to effectively increase the dataset size
#    train_ds = train_ds.map(lambda x,y: augmentImage(x,y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Validation preprocessing
val_ds = val_ds.cache() # cache dataset for it to be used over iterations
val_ds = val_ds.shuffle(buffer_size = len(val_ds)).batch(batchSize)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Test preprocessing
test_ds = test_ds.cache() # cache dataset for it to be used over iterations
test_ds = test_ds.shuffle(buffer_size = len(test_ds)).batch(batchSize)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared



# Get shapes for later use
# train_in_shape = X_train.shape
# val_in_shape = X_val.shape
# train_out_shape = y_train.shape
# val_out_shape = y_val.shape

# Currently there's a bug where we need to define the shape manually...
# def set_shapes(image, label):
#     image.set_shape((batchSize,train_in_shape[1],train_in_shape[2],train_in_shape[3]))
#     label.set_shape((batchSize,train_out_shape[1],train_out_shape[2],train_out_shape[3]))
#     return image, label


# train_ds = train_ds.map(set_shapes)
# val_ds = val_ds.map(set_shapes)


#%%
#####################################################################
# CNN Model definition
#####################################################################

def TBDCNet_modelCNN(inputShape, outputShape, params):
  '''
  This function returns a model based on the hyperparameters in the
  sweep definition

  Args
  ----------
  inputShape: the input image shape
  outputShape: the prediction image shape (currently unused)
  params: The hyperparameters for the given sweep index

  Returns
  ----------
  model: tensorflow model

  '''
  # Kernel regularizer (both linear and quadratic)
  if params['L1kernel_regularizer'] > 0 and params['L2kernel_regularizer'] > 0: 
     regularizer = tf.keras.regularizers.L1L2(l1=params['L1kernel_regularizer'], l2=params['L2kernel_regularizer'])
  elif params['L1kernel_regularizer'] > 0:
     regularizer = tf.keras.regularizers.L1(params['L1kernel_regularizer'])
  elif params['L2kernel_regularizer'] > 0:
     regularizer = tf.keras.regularizers.L2(params['L2kernel_regularizer'])
  else:
    regularizer = None

  # Define the model architecture. Each convolutional layer has settings related to the kernel size, 
  # activation function, and regularizer. After each convolutional layer there may be a batch-
  # normalization layer, a max pooling layer, and a dropout layer. The number of convolutional layers
  # is given in the sweep definition



  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  x = normalizer(inputs)

  if params['dsAugmentation'] == 1:
    x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)(x)

  

  x = tf.keras.layers.Conv2D(filters = 32, kernel_size=(int(params['layer1Kernel']), int(params['layer1Kernel'])),activation=params['conv1Activation'], data_format='channels_last', padding='same', kernel_regularizer=regularizer) (x)
  if params['batchNorm'] == 1:
    x = tf.keras.layers.BatchNormalization()(x)
  if params['pooling'] == 1:
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
  if params['dropout'] > 0:
    x = tf.keras.layers.SpatialDropout2D(rate = params['dropout'])(x)
  encoder1 = x # Use this if skip connections need to be used


  if params['layer2'] == 1:
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size=(int(params['layer2Kernel']), int(params['layer2Kernel'])),activation=params['conv2Activation'], data_format='channels_last', padding='same', kernel_regularizer=regularizer) (x)
    if params['batchNorm'] == 1:
      x = tf.keras.layers.BatchNormalization()(x)
    if params['pooling'] == 1:
       x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
    if params['dropout'] > 0:
       x = tf.keras.layers.SpatialDropout2D(rate = params['dropout'])(x)
    encoder2 = x
        

    if params['layer3'] == 1:
        x = tf.keras.layers.Conv2D(filters = 128, kernel_size=(int(params['layer3Kernel']), int(params['layer3Kernel'])),activation=params['conv3Activation'], data_format='channels_last', padding='same', kernel_regularizer=regularizer) (x)
        if params['batchNorm'] == 1:
           x = tf.keras.layers.BatchNormalization()(x)
        if params['pooling'] == 1:
           x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
        if params['dropout'] > 0:
           x = tf.keras.layers.SpatialDropout2D(rate = params['dropout'])(x)
        encoder3 = x


        if params['layer4'] == 1:
            x = tf.keras.layers.Conv2D(filters = 256, kernel_size=(int(params['layer4Kernel']), int(params['layer4Kernel'])),activation=params['conv4Activation'], data_format='channels_last', padding='same', kernel_regularizer=regularizer) (x)
            if params['batchNorm'] == 1:
                x = tf.keras.layers.BatchNormalization()(x)
            if params['pooling'] == 1:
                x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
            if params['dropout'] > 0:
                x = tf.keras.layers.SpatialDropout2D(rate = params['dropout'])(x)
            encoder4 = x


            if params['layer5'] == 1:
                x = tf.keras.layers.Conv2D(filters = 512, kernel_size=(int(params['layer5Kernel']), int(params['layer5Kernel'])),activation=params['conv5Activation'], data_format='channels_last', padding='same', kernel_regularizer=regularizer) (x)
                if params['batchNorm'] == 1:
                    x = tf.keras.layers.BatchNormalization()(x)
                if params['pooling'] == 1:
                    x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
                if params['dropout'] > 0:
                    x = tf.keras.layers.SpatialDropout2D(rate = params['dropout'])(x)
                encoder5 = x


                if params['layer6'] == 1:
                    x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=(int(params['layer6Kernel']), int(params['layer6Kernel'])),activation=params['conv6Activation'], data_format='channels_last', padding='same', kernel_regularizer=regularizer) (x)
                    if params['batchNorm'] == 1:
                        x = tf.keras.layers.BatchNormalization()(x)
                    if params['pooling'] == 1:
                        x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(x)
                    if params['dropout'] > 0:
                        x = tf.keras.layers.SpatialDropout2D(rate = params['dropout'])(x)
                    encoder6 = x

                    if params['ActivationUp'] == 0:
                       temp_activation = 'linear'
                    else:
                       temp_activation = params['conv6Activation']

                    x = tf.keras.layers.Conv2DTranspose(filters = 512, kernel_size = (int(params['layer6Kernel']),int(params['layer6Kernel'])),  padding='same',activation=temp_activation)(x)
                    if params['skipConnections'] == 1:
                      x = tf.keras.layers.Concatenate()([x, encoder5])


                if params['ActivationUp'] == 0:
                    temp_activation = 'linear'
                else:
                    temp_activation = params['conv5Activation']

                x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (int(params['layer5Kernel']),int(params['layer5Kernel'])),  padding='same',activation=temp_activation)(x)
                if params['skipConnections'] == 1:
                  x = tf.keras.layers.Concatenate()([x, encoder4])
            
            
            if params['type'] == 'dense':
              y = tf.keras.layers.Flatten()(x)
              y = tf.keras.layers.Dense(64, activation='relu')(y)
              y = tf.keras.layers.Dense(outputShape[0]*outputShape[1])(y)
              y = tf.keras.layers.Reshape(outputShape)(y)

            if params['ActivationUp'] == 0:
                temp_activation = 'linear'
            else:
                temp_activation = params['conv4Activation']
            
            x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (int(params['layer4Kernel']),int(params['layer4Kernel'])),  padding='same',activation=temp_activation)(x)
            if params['skipConnections'] == 1:
              x = tf.keras.layers.Concatenate()([x, encoder3])

        if params['ActivationUp'] == 0:
            temp_activation = 'linear'
        else:
            temp_activation = params['conv3Activation']
        
        x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (int(params['layer3Kernel']),int(params['layer3Kernel'])),  padding='same',activation=temp_activation)(x)
        if params['skipConnections'] == 1:
          x = tf.keras.layers.Concatenate()([x, encoder2])

    if params['ActivationUp'] == 0:
        temp_activation = 'linear'
    else:
        temp_activation = params['conv2Activation']
    
    x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer2Kernel']),int(params['layer2Kernel'])),  padding='same',activation=temp_activation)(x)
    if params['skipConnections'] == 1:
      x = tf.keras.layers.Concatenate()([x, encoder1])

   # Custom activation function is linear between 0 and 1 and otherwise constant
  def custom_activation(x):
      return tf.math.minimum(K.relu(x), 1)

  x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',activation='linear')(x)

  if params['type'] == 'dense': # For the dense model we jsut pull the output y
     output = y
  else:
    output = x

  model = tf.keras.Model(inputs=input, outputs=output) # Create model

  # Default initial learning rate is 0.001. If the the decay rate is 1 this will be held constant.
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['initial_lr'],
    decay_steps=steps_per_epoch*epochs,
    decay_rate=params['lr_decay_rate'])

  def custom_loss(y_true,y_pred):
    SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
    loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.nn.relu(y_true))))
    loss = tf.reduce_mean(loss)
    return loss
  
  def custom_loss5(y_true,y_pred):
    SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
    loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.math.multiply(tf.nn.relu(y_true),5))))
    loss = tf.reduce_mean(loss)
    return loss

  def peak_loss(y_true,y_pred):
    peakVal = tf.reduce_max(y_true, keepdims=True)
    cond = tf.equal(y_true, peakVal)
    # peakLoc = tf.where(cond)
    # peakLoc_1d = tf.squeeze(peakLoc)
    errorGrid = tf.math.subtract(y_true,y_pred)
    zeroGrid = tf.math.subtract(y_true,y_true) # Grid of zeros so we only get loss in peak location
    # peakPred = y_pred[peakLoc_1d.numpy()[0]]
    # peakPred = tf.slice(y_pred, peakLoc, [1,1])
    loss = tf.where(cond, errorGrid, zeroGrid)
    loss = tf.reduce_mean(loss)

    # loss = peakPred-peakVal
    return loss


#   Loss functions can be swept
  if params['loss'] == 'MSE':
    lossfunc = tf.keras.losses.MeanSquaredError()
  elif params['loss'] == 'MAE':
    lossfunc = tf.keras.losses.MeanAbsoluteError()
  elif params['loss'] == 'Custom':
    lossfunc = custom_loss
  elif params['loss'] == 'Peak':
    lossfunc = peak_loss
  elif params['loss'] == 'Custom5':
    lossfunc = custom_loss5
    



  # Compile model with the optimizer in the sweep definition
  if params['optimizer'] == 'Adadelta':
     model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])
  elif params['optimizer'] == 'Nadam':
     model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])
  else:
     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])

  return model

def TBDCNet_UNet(inputShape, outputShape, params):
   '''
  This function returns a UNet model based on the hyperparameters in the
  sweep definition
  

  Args
  ----------
  inputShape: the input image shape
  outputShape: the prediction image shape (currently unused)
  params: The hyperparameters for the given sweep index

  Returns
  ----------
  model: tensorflow model

  '''
   def double_convBlock(x,filters, params): # Convolutional block
      x = tf.keras.layers.Conv2D(filters, kernel_size = 3, strides = 1, padding = "same", activation = "relu", kernel_initializer = "glorot_uniform")(x)
      x = tf.keras.layers.Conv2D(filters, kernel_size = 3, strides = 1, padding = "same", activation = "relu", kernel_initializer = "glorot_uniform")(x)
      return x
   

   def downSamplingBlock(x,filters, params): # Downsampling block in the encoder
      skip = double_convBlock(x, filters, params)
      x = tf.keras.layers.MaxPool2D(2)(skip)
      x = tf.keras.layers.Dropout(params['dropout'])(x)
      return skip,x
   
   def upSamplingBlock(x,skip,filters, params): # Upsampling block in the decoder
      x = tf.keras.layers.Conv2DTranspose(filters, kernel_size = 3, strides = 2, padding="same")(x)
      x = tf.keras.layers.concatenate([x, skip]) # Skip connection
      x = tf.keras.layers.Dropout(params['dropout'])(x)
      x = double_convBlock(x, filters, params)
      return x


   input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
   x = input

   # Encoder
   skip1, x1 = downSamplingBlock(x, 64, params)
   skip2, x2 = downSamplingBlock(x1, 128, params)
   skip3, x3 = downSamplingBlock(x2, 256, params)
   skip4, x4 = downSamplingBlock(x3, 512, params)

   # Bottlenexk
  
   bottleneck = double_convBlock(x4, 1024, params)

   # Decoder
   print(x1)
   print(x2)
   print(x3)
   print(x4)
   print(skip1)
   print(skip2)
   print(skip3)
   print(bottleneck)
   u6 = upSamplingBlock(bottleneck, skip4, 512, params)
   u7 = upSamplingBlock(u6, skip3, 256, params)
   u8 = upSamplingBlock(u7, skip2, 128, params)
   u9 = upSamplingBlock(u8, skip1, 64, params)

   # Final layer
   outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation = "linear")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(input, outputs, name="U-Net")
   # Default initial learning rate is 0.001. If the the decay rate is 1 this will be held constant.
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['initial_lr'],
    decay_steps=steps_per_epoch*epochs,
    decay_rate=params['lr_decay_rate'])
   def custom_loss(y_true,y_pred):
    SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
    loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.nn.relu(y_true))))
    loss = tf.reduce_mean(loss)
    return loss
   def custom_loss5(y_true,y_pred):
    SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
    loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.math.multiply(tf.nn.relu(y_true),5))))
    loss = tf.reduce_mean(loss)
    return loss
   def peak_loss(y_true,y_pred):
    peakVal = tf.reduce_max(y_true, keepdims=True)
    cond = tf.equal(y_true, peakVal)
    # peakLoc = tf.where(cond)
    # peakLoc_1d = tf.squeeze(peakLoc)
    errorGrid = tf.math.subtract(y_true,y_pred)
    zeroGrid = tf.math.subtract(y_true,y_true) # Grid of zeros so we only get loss in peak location
    # peakPred = y_pred[peakLoc_1d.numpy()[0]]
    # peakPred = tf.slice(y_pred, peakLoc, [1,1])
    loss = tf.where(cond, errorGrid, zeroGrid)
    loss = tf.reduce_mean(loss)

    # loss = peakPred-peakVal
    return loss


  #   Loss functions can be swept
   if params['loss'] == 'MSE':
     lossfunc = tf.keras.losses.MeanSquaredError()
   elif params['loss'] == 'MAE':
     lossfunc = tf.keras.losses.MeanAbsoluteError()
   elif params['loss'] == 'Custom':
     lossfunc = custom_loss
   elif params['loss'] == 'Peak':
     lossfunc = peak_loss
   elif params['loss'] == 'Custom5':
     lossfunc = custom_loss5
    

  # Compile model with the optimizer in the sweep definition
   if params['optimizer'] == 'Adadelta':
      unet_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])
   elif params['optimizer'] == 'Nadam':
      unet_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])
   else:
      unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])

   return unet_model
   


#%%
#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
# checkpoint_path = 'training_checkpoints_{jn}_{num}/cp.ckpt'.format(jn=jobName, num = 1)
checkpoint_path = 'epoch-{epoch:02d}.weights.h5'
checkpoint_dir = 'training_checkpoints_{jn}_{num}'.format(jn=jobname, num = 1)

try:
  os.mkdir(checkpoint_dir) # Make checkpoint directory
except:
  pass

cp_savepath = os.path.join(checkpoint_dir,checkpoint_path)

# Save best weights to checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_savepath,
                                                 save_weights_only=True,
                                                 save_best_only = True,
                                                 monitor = 'val_loss',
                                                 verbose=1)

# Early stopping callback which monitors improvements and stops training if
# it stagnates.
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    min_delta=0, # Minimum improvement to consider an improvement
    patience=60, # Number of epochs with no improvement before stopping training
    verbose=1, # Records message when earlystopping
    mode='auto',
    baseline=None, 
    restore_best_weights=False # Do not restore best weights after early stopping, we do this manually to allow recording of the full training history
)

#%%
#####################################################################
# Model instantiation
#####################################################################

tf.keras.backend.clear_session() # Clear the state and frees up memory

# CNN Model creation
# modelCNN = TBDCNet_modelCNN(inputShape = train_in_shape[1:], outputShape = train_out_shape[1:], params = params)
modelCNN = TBDCNet_UNet(inputShape = train_in_shape[1:], outputShape = train_out_shape[1:], params = params)
modelCNNname = 'CNNModel1'

#%%
#####################################################################
# Model training
#####################################################################

# Known issue: sometimes throws error related to the shape of the labels...
# Fit model to Failure index
epochs = 20
modelCNN_history = modelCNN.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_ds,
                                validation_steps = validation_steps,
                                callbacks=[early_stopping_monitor, cp_callback]
                                )

#%%
#####################################################################
# Data export
#####################################################################

trainingHist = modelCNN_history.history # save training history
modelCNN.load_weights(checkpoint_path) # load best model weights

predCNN = modelCNN.predict(X_train) # Make prediction
predCNN_val = modelCNN.predict(X_val) # Prediction of only validation data
predCNNShape = predCNN.shape
predCNN_valShape = predCNN_val.shape

# Inverse standardisation
if params['standardisation'] == 0:
   predCNN_invStandard = predCNN
   predCNN_val_invStandard = predCNN_val
   ground_truth_invStandard = y_train
   ground_truth_val_invStandard = y_val
else:
   predCNN_invStandard = Yscaler.inverse_transform(predCNN.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
   predCNN_invStandard = predCNN_invStandard.reshape(predCNNShape)
   predCNN_val_invStandard = Yscaler.inverse_transform(predCNN_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
   predCNN_val_invStandard = predCNN_val_invStandard.reshape(predCNN_valShape)
   
   ground_truth_invStandard =  Yscaler.inverse_transform(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
   ground_truth_invStandard = ground_truth_invStandard.reshape(y_trainShape)
   ground_truth_val_invStandard = Yscaler.inverse_transform(y_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
   ground_truth_val_invStandard = ground_truth_val_invStandard.reshape(y_valShape)


# RMSE
RMSE = tf.keras.metrics.RootMeanSquaredError()
RMSE.update_state(ground_truth_invStandard,predCNN_invStandard)
print('RMSE for training set = ' + str(RMSE.result().numpy()))
if ground_truth_val_invStandard is not None:
    RMSE_val = tf.keras.metrics.RootMeanSquaredError()
    RMSE_val.update_state(ground_truth_val_invStandard,predCNN_val_invStandard)
    print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))


#%% Save outputs
inputDat = np.zeros(samples2D.shape)
for i in range(0,samples2D.shape[-1]): # Un-normalise all input features
    inputDat[:,:,:,i] = samples2D[:,:,:,i]

with open(histOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(trainingHist, f, indent=2)

with open(predOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(predCNN_invStandard.tolist(), f, indent=2)

with open(predOutPath_val, 'w') as f: # Dump data to json file at specified path
    json.dump(predCNN_val_invStandard.tolist(), f, indent=2)

with open(gtOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(ground_truth_invStandard.tolist(), f, indent=2)

with open(gtOutPath_val, 'w') as f: # Dump data to json file at specified path
    json.dump(ground_truth_val_invStandard.tolist(), f, indent=2)

with open(paramOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(params.to_json(), f, indent=2)

with open(inputOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(inputDat.tolist(), f, indent=2)

with open(RMSEOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(RMSE.result().numpy().tolist(), f, indent=2)

with open(RMSEOutPath_val, 'w') as f: # Dump data to json file at specified path
    json.dump(RMSE_val.result().numpy().tolist(), f, indent=2)

# Save the model
modelCNN.save(modelOutPath)
# %% Various tests
# numSamples = 100
# kFold = 10
# k = 10
# count = np.linspace(1,100,100)
# trainValRatio = (k-1)/k # Training and validation data split ratio
# train_length = round(numSamples * trainValRatio)
# X = count  # Input data is always stiffness components
# yFI = np.linspace(1001,1100,100) # Labels for failure index

# # Create tensor datasets
# # ds_e22 = tf.data.Dataset.from_tensor_slices((X, ye22))
# # ds_FI = tf.data.Dataset.from_tensor_slices((X, yFI))
# ds_FI = tf.data.Dataset.from_tensor_slices((X,yFI))
# # for x in ds_FI:
# #     print(x)
# # Split into training and validation datasets using k-fold
# valIdx = int((kFold-1)*(numSamples/k)) # The k fold variable is 1-indexed, valIdx marks the start of the validation set in this fold

# # val_ds_e22 = ds_e22.skip(valIdx) # Skip until the point where the validation set starts
# # val_ds_e22 = val_ds_e22.take(int(numSamples/k)) # Take the validation set
# val_ds_FI = ds_FI.skip(valIdx) # Skip until the point where the validation set starts

# val_ds_FI = val_ds_FI.take(int(numSamples/k)) # Take the validation set
# for x in val_ds_FI:
#     print(x)
# # train_ds_e22_first = ds_e22.take(valIdx) # Take samples up to the start of the validation set
# # train_ds_e22_second = ds_e22.skip(int(valIdx+(numSamples/k))) # Take samples after the end of the validation set
# # train_ds_e22 = train_ds_e22_first.concatenate(train_ds_e22_second) # Concatenate the two parts of the training dataset

# train_ds_FI_first = ds_FI.take(valIdx) # Take samples up to the start of the validation set
# train_ds_FI_second = ds_FI.skip(int(valIdx+(numSamples/k))) # Take samples after the end of the validation set
# train_ds_FI = train_ds_FI_first.concatenate(train_ds_FI_second) # Concatenate the two parts of the training dataset
# for x in train_ds_FI:
#     print(x)
# # %%
# tips = sns.load_dataset("tips")
# # sns.kdeplot(data=tips)


# maxPointError =[-0.37377048, -0.37455583,  0.02026224]
# maxPointErrorDf = pd.DataFrame(maxPointError)



# %%
