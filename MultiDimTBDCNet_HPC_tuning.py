#####################################################################
# Description
#####################################################################
# This script is run on the computing cluster (HPC) and contains the
# model definition and training processes. It must be pointed to a 
# sweep definition csv wherein the model hyperparamters are tabulated

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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import argparse

#####################################################################
# Settings
#####################################################################

# Add arguments for parallel running and training of several different models 
argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--parallel", help="Index for parallel running on HPC") # parameter to allow parallel running on the HPC
argParser.add_argument("-j", "--jobname", help="Job name") # Name of job passed when calling script
args = argParser.parse_args()
sweepIdx = int(args.parallel)

# Sweep definition of hyperparameters
sweepPath = 'sweep_definition_Augmentation.csv' # Name of sweep definition file
sweep_params = pd.read_csv(sweepPath)
sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[sweepIdx]

# Various settings
numSamples = 100 # Number of data samples (i.e. TBDC specimens)
batchSize = params['batchSize'] # Batch size for training
trainValRatio = params['trainValRatio'] # Training and validation data split ratio
train_length = round(numSamples * trainValRatio) # Number of training samples 
epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = math.ceil((100-train_length) / batchSize)

# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)

timeStamp = datetime.datetime.now().strftime("%Y%m%d%H%M") # Not currently used
histOutName = 'trainHist_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Training history file
histOutPath = os.path.join('dataout',histOutName)
predOutName = 'predictions_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Predictions
predOutPath = os.path.join('dataout',predOutName)
predOutName_val = 'predictions_val_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Predictions
predOutPath_val = os.path.join('dataout',predOutName_val)
gtOutName = 'groundTruth_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Ground truths
gtOutPath = os.path.join('dataout',gtOutName)
gtOutName_val = 'groundTruth_val_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Ground truths
gtOutPath_val = os.path.join('dataout',gtOutName_val)
paramOutName = 'parameters_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Hyperparameters
paramOutPath = os.path.join('dataout',paramOutName)
inputOutName = 'input_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # Model inputs
inputOutPath = os.path.join('dataout',inputOutName)

#####################################################################
# Data functions
#####################################################################

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

  coords = [[x for x in values[:,1][y].split(' ') if x] for y in range(len(values[:,1]))] # Split coordinates by delimiter (space)
  coords = [np.char.strip(x, '[') for x in coords] # Coordinate output from abaqus has leading "["
  coords = [[x for x in coords[y] if x] for y in range(len(values[:,1]))] # remove empty array elements
  coords = np.array([[float(x) for x in coords[y][0:2]] for y in range(len(values[:,1]))]) # Take 2d coordinates and convert to float

  values = np.column_stack((values[:,0],coords,values[:,2:])).astype(float) # Create a new values vector which contains the coordinates

  headers = np.concatenate(([[headers[0],'x_coord','y_coord'],headers[2:]])) # rectify the headers to include x and y coordinates separately

  return headers, values




def normalise(input_matrix) -> tuple:
    '''
    Normalise values in matrix by removing the mean and scaling 
    to one standard deviation

    Args
    ----------
    input_matrix : np.array
        Numpy array of data in shape [55,20,14]
        OR
        Numpy array of data in shape [1100,14] if not reshaped
        There are 11 features which are each normalised, features 1, 2, and 3 are labels and x/y coordinates

    Returns
    ----------
    tuple
        Normalised array
    '''

    scaler = preprocessing.StandardScaler()
    if len(np.shape(input_matrix)) == 3:
      scaler.fit(input_matrix[:,:,3:])
      input_matrix[:,:,3:] = scaler.transform(input_matrix[:,:,3:])
    elif len(np.shape(input_matrix)) == 2:
      scaler.fit(input_matrix[:,3:])
      input_matrix[:,3:] = scaler.transform(input_matrix[:,3:])
    else:
        raise Exception("Data to be normalised not of correct shape")

    return input_matrix, scaler


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
    height, width = 55, 20 # image dimensions

    if randAug.normal([]) > 0: # Randomly flip an image horizontally 50% of the time
      inputMatrices = tf.image.flip_left_right(inputMatrices)
      gtMatrix = tf.image.flip_left_right(tf.reshape(gtMatrix,[-1,height,width,1]))
      gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

    if  randAug.normal([]) > 0: # Randomly flip an image vertically 50% of the time
      inputMatrices = tf.image.flip_up_down(inputMatrices)
      gtMatrix = tf.image.flip_up_down(tf.reshape(gtMatrix,[-1,height,width,1]))
      gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

    if randAug.normal([]) > -0.3: # Scale to a random size within the bounding box and fit to a random location within this
      crop_width = randAug.uniform(shape=(), minval=math.floor(0.7 * width), maxval=math.floor(0.9 * width), dtype = tf.int32)
      crop_height = randAug.uniform(shape=(), minval=math.floor(0.7 * height), maxval=math.floor(0.9 * height), dtype = tf.int32)
      offset_x = randAug.uniform(shape=(), minval=0, maxval=(width - crop_width), dtype = tf.int32)
      offset_y = randAug.uniform(shape=(), minval=0, maxval=(height - crop_height), dtype = tf.int32)

      inputMatrices = tf.image.crop_to_bounding_box(inputMatrices, offset_y, offset_x, crop_height, crop_width) # Crop to bounding box
      gtMatrix = tf.image.crop_to_bounding_box(tf.reshape(gtMatrix,[-1,height,width,1]), offset_y, offset_x, crop_height, crop_width)
      newHeight = crop_height
      newWidth = crop_width
      gtMatrix = tf.reshape(gtMatrix,[-1,newHeight,newWidth]) # Reshape ground truth back to not have channels dimensions
    
      inputMatrices = tf.image.resize(inputMatrices, (height, width)) # Resize to original size (we want all images same size) - this distorts the image
      gtMatrix = tf.image.resize(tf.reshape(gtMatrix,[-1,newHeight,newWidth,1]), (height, width), method='nearest')
      gtMatrix = tf.reshape(gtMatrix,[-1,height,width])
      inputMatrices = tf.cast(inputMatrices, tf.float64)
      gtMatrix = tf.cast(gtMatrix, tf.float64)

      
    return (inputMatrices,gtMatrix)


def show_prediction(sample, predictions, names, ground_truth, grid):
  '''
  For a given dataset plots one image, its true mask, and predicted mask

  Args
  ----------
  sample: the 55x20x3 input
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

#####################################################################
# Data import, normalisation, and reshaping
#####################################################################

# Import all data samples
for i in range(numSamples):
  path = 'datain/Unnotched_TBDC_2022_'+str(i)+'.csv' # The data format after extracting from Abaqus
  if i==0:
    headers, samples = loadSample(path)
    samples = samples.reshape(1, np.shape(samples)[0],np.shape(samples)[1])
  else:
    addSamp = loadSample(path)[1]
    samples = np.concatenate((samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))

samples, scaler = normalise(samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])) # retain the scaler parameters such that inverse scaling can be done
samples = samples.reshape(numSamples,-1,samples.shape[-1])
# Reshape sample variable to have shape (samples, row, column, features)
samples2D = samples.reshape(numSamples,55,20,14)




#####################################################################
# ML training Settings and dataset preprocessing
#####################################################################

# Reminder: the header index is the following
# [   0        1         2       3     4     5     6     7     8      9      10   11    12    13  ]
# ['label' 'x_coord' 'y_coord' 'e11' 'e22' 'e12' 'S11' 'S22' 'S12' 'SMises' 'FI' 'E11' 'E22' 'E12']

X = samples2D[:,:,:,[11,12,13]]  # Input data is always stiffness components

ye11 = samples2D[:,:,:,3] # Labels for horizontal strain
ye22 = samples2D[:,:,:,4] # Labels for vertical strain
yS11 = samples2D[:,:,:,6] # Labels for horizontal stress
yS22 = samples2D[:,:,:,7] # Labels for vertical stress
ymises = samples2D[:,:,:,9] # Labels for mises stress
yFI = samples2D[:,:,:,10] # Labels for failure index

# Create tensor datasets
ds_e22 = tf.data.Dataset.from_tensor_slices((X, ye22))
ds_FI = tf.data.Dataset.from_tensor_slices((X, yFI))

# Split into training and validation datasets
train_ds_e22 = ds_e22.take(train_length) # Currently not shuffling beforehand, should be done in the future TODO
val_ds_e22 = ds_e22.skip(train_length)
train_ds_FI = ds_FI.take(train_length)
val_ds_FI = ds_FI.skip(train_length)

# Training dataset preprocessing
train_ds_e22 = train_ds_e22.cache() # cache dataset for it to be used over iterations
train_ds_e22 = train_ds_e22.shuffle(buffer_size=1000).batch(batchSize)
train_ds_e22 = train_ds_e22.repeat() # Repeats dataset indefinitely to avoid errors
if params['dsAugmentation'] == 1: # We can apply dataset augmentation to increase the size of it
   train_ds_e22 = train_ds_e22.map(lambda x,y: augmentImage(x,y)) # Here x and y are input images and ground truth

train_ds_e22 = train_ds_e22.prefetch(buffer_size=1000) # Allows prefetching of elements while later elements are prepared


train_ds_FI = train_ds_FI.cache() # cache dataset for it to be used over iterations
train_ds_FI = train_ds_FI.shuffle(buffer_size=1000).batch(batchSize) # Shuffle for random order
train_ds_FI = train_ds_FI.repeat() # Repeats dataset indefinitely to avoid errors
if params['dsAugmentation'] == 1: # We can apply dataset augmentation to 
   train_ds_FI = train_ds_FI.map(lambda x,y: augmentImage(x,y))
train_ds_FI = train_ds_FI.prefetch(buffer_size=1000) # Allows prefetching of elements while later elements are prepared

# Validation dataset preprocessing
val_ds_e22 = val_ds_e22.cache() # cache dataset for it to be used over iterations
val_ds_e22 = val_ds_e22.shuffle(buffer_size=1000).batch(batchSize)
val_ds_e22 = val_ds_e22.prefetch(buffer_size=1000) # Allows prefetching of elements while later elements are prepared

val_ds_FI = val_ds_FI.cache() # cache dataset for it to be used over iterations
val_ds_FI = val_ds_FI.shuffle(buffer_size=1000).batch(batchSize)
val_ds_FI = val_ds_FI.prefetch(buffer_size=1000) # Allows prefetching of elements while later elements are prepared

# The below are just used for validation and shape (TODO: replace validation method with tf tensors)
X_train_e22, X_val_e22, y_train_e22, y_val_e22 = sklearn.model_selection.train_test_split(X, ye22, train_size=trainValRatio)
X_train_FI, X_val_FI, y_train_FI, y_val_FI = sklearn.model_selection.train_test_split(X, yFI, train_size=trainValRatio)

# Get shapes for later use
train_in_shape = X_train_e22.shape
val_in_shape = X_val_e22.shape
train_out_shape = y_train_e22.shape
val_out_shape = y_val_e22.shape


#####################################################################
# CNN Model definition
#####################################################################

def TBDCNet_modelCNN(inputShape, outputShape, params):
  '''
  This function returns a model based on the hyperparameters in the
  sweep definition

  Args
  ----------
  inputShape: the 55x20x3 input image shape
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
  input = tf.keras.layers.Input(shape=inputShape) # Shape (55, 20, 3)
  x = input
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
                
                    x = tf.keras.layers.Conv2DTranspose(filters = 512, kernel_size = 3,  padding='same')(x)
                    if params['skipConnections'] == 1:
                      x = tf.keras.layers.Concatenate()([x, encoder5])


                x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = 3,  padding='same')(x)
                if params['skipConnections'] == 1:
                  x = tf.keras.layers.Concatenate()([x, encoder4])

            x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = 3,  padding='same')(x)
            if params['skipConnections'] == 1:
              x = tf.keras.layers.Concatenate()([x, encoder3])

        x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = 3,  padding='same')(x)
        if params['skipConnections'] == 1:
          x = tf.keras.layers.Concatenate()([x, encoder2])

    x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = 3,  padding='same')(x)
    if params['skipConnections'] == 1:
      x = tf.keras.layers.Concatenate()([x, encoder1])

  x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = 3,  padding='same')(x)

  output = x

  model = tf.keras.Model(inputs=input, outputs=output) # Create model

  # Default initial learning rate is 0.001. If the the decay rate is 1 this will be held constant.
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['initial_lr'],
    decay_steps=steps_per_epoch*epochs,
    decay_rate=params['lr_decay_rate'])

  # Compile model with the optimizer in the sweep definition
  if params['optimizer'] == 'Adadelta':
     model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate = lr_schedule), # Compile
              loss='MeanSquaredError', 
              metrics=['mean_squared_error'])
  elif params['optimizer'] == 'Nadam':
     model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = lr_schedule), # Compile
              loss='MeanSquaredError', 
              metrics=['mean_squared_error'])
  else:
     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule), # Compile
              loss='MeanSquaredError', 
              metrics=['mean_squared_error'])

  return model

#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
checkpoint_path = 'training_checkpoints_{jn}_{num}/cp.ckpt'.format(jn=args.jobname, num = args.parallel)
checkpoint_dir = os.path.dirname(checkpoint_path)

try:
  os.mkdir(checkpoint_dir) # Make checkpoint directory
except:
  pass

# Save best weights to checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only = True,
                                                 monitor = 'val_loss',
                                                 verbose=1)

# Early stopping callback which monitors improvements and stops training if
# it stagnates.
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    min_delta=0, # Minimum improvement to consider an improvement
    patience=40, # Number of epochs with no improvement before stopping training
    verbose=1, # Records message when earlystopping
    mode='auto',
    baseline=None, 
    restore_best_weights=False # Do not restore best weights after early stopping, we do this manually to allow recording of the full training history
)

#####################################################################
# Model instantiation
#####################################################################

tf.keras.backend.clear_session() # Clear the state and frees up memory

# CNN Model creation
modelCNN = TBDCNet_modelCNN(inputShape = train_in_shape[1:], outputShape = train_out_shape[1:], params = params)
modelCNNname = 'CNNModel1'

# Save normalisation (standard scaler) values to allow inverse transformation 
means = scaler.mean_
std = np.sqrt(scaler.var_)
# In the scaler we have the following indeces
# [   0     1     2     3     4     5      6      7     8     9     10 ]
# [ 'e11' 'e22' 'e12' 'S11' 'S22' 'S12' 'SMises' 'FI' 'E11' 'E22' 'E12']

#####################################################################
# Model training
#####################################################################

# Fit model to Failure index
modelCNN_history = modelCNN.fit(train_ds_FI,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_ds_FI,
                                validation_steps = validation_steps,
                                callbacks=[early_stopping_monitor, cp_callback]
                                )


#####################################################################
# Data export
#####################################################################

trainingHist = modelCNN_history.history # save training history
modelCNN.load_weights(checkpoint_path) # load best model weights

predCNN = modelCNN.predict(X) # Make prediction
predCNN_val = modelCNN.predict(X_val_FI) # Prediction of only validation data

# Inverse standardisation and reshaping to evaluate with physical quantities intact
predCNN_invStandard = predCNN[:,:,:,0]*std[7]+means[7]
predCNN_val_invStandard = predCNN_val[:,:,:,0]*std[7]+means[7] 
ground_truth_invStandard = yFI*std[7]+means[7]
ground_truth_val_invStandard = y_val_FI*std[7]+means[7]

# Reminder: the header index is the following
# [   0        1         2       3     4     5     6     7     8      9      10   11    12    13  ]
# ['label' 'x_coord' 'y_coord' 'e11' 'e22' 'e12' 'S11' 'S22' 'S12' 'SMises' 'FI' 'E11' 'E22' 'E12']

# In the scaler we have the following indeces
# [   0     1     2     3     4     5      6      7     8     9     10 ]
# [ 'e11' 'e22' 'e12' 'S11' 'S22' 'S12' 'SMises' 'FI' 'E11' 'E22' 'E12']

inputDat = np.zeros(samples2D.shape)
for i in range(0,3): # First 3 features are not normalised
    inputDat[:,:,:,i] = samples2D[:,:,:,i]
for i in range(3,14): # Remaining features must have normalisation inversed
    inputDat[:,:,:,i] = samples2D[:,:,:,i]*std[i-3]+means[i-3]

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

