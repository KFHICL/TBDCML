# %%

#####################################################################
# Description
#####################################################################
'''
This script allows training and evaluation of Benchmark models locally
using the LFC18, MC24, and MC24_Extended datasets


Inputs:
sweep_definition_{jn}.csv: sweep definition file with same jobname as -j, placed in same directory as this script

Ouputs:
All saved to outputs folder where 
{jn} = jobname
{num} = index of model in sweep definition

trainHist_{jn}_{num}.json: training history (curves)

parameters_{jn}_{num}.json: model hyperparameters
results_{jn}_{num}.json: results for training, validation, and test datasets (RMSE, SSIM etc.)
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
import json
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Add arguments for parallel running and training of several different models 
argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--parallel", help="Index for parallel running on HPC") # parameter to allow parallel running on the HPC
argParser.add_argument("-j", "--jobname", help="Job name") # Name of job passed when calling script
args = argParser.parse_args()
sweepIdx = int(args.parallel) # Index of model in sweep definition file


# Sweep definition containing hyperparameters
sweepPath = 'sweep_definition_{jn}.csv'.format(jn=args.jobname[:-2]) # Name of sweep definition file, one for all repetitions hence [:-2]

print(os.getcwd())
print(sweepPath)
os.listdir(os.getcwd())

sweep_params = pd.read_csv(sweepPath)
sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[sweepIdx]


yNames = ['FI'] # Names of ground truth features in input csv
normalizerLength = 20 # Number of random samples used for computation of mean and variance used in data normalisation 

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
modelOutPath = 'model_{jn}_{num}.keras'.format(jn=args.jobname, num = args.parallel) # Model architecture and weights
modelOutPath = os.path.join('dataout',modelOutPath)
RMSEOutPath = 'RMSE_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # RMSE of model to quickly compare models when many are trained in a sweep
RMSEOutPath = os.path.join('dataout',RMSEOutPath)
RMSEOutPath_val = 'RMSE_val_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel) # RMSE of model to quickly compare models when many are trained in a sweep
RMSEOutPath_val = os.path.join('dataout',RMSEOutPath_val)
resultpath = 'results_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel)
resultpath = os.path.join('dataout',resultpath)


if params['Dataset'] == 'LFC18': # ABAQUS DATA FROM GAUDRON2018
  trainDat_name = 'Gaudron2018' 
  sampleShape = [55,20]
  xNames = ['E11','E22','E12'] # Names of input features in input csv
  samplesPerFile = 1
  winKernel = 5
elif params['Dataset'] == 'MC24': # MECOMPOSITES MODEL FROM 2024 (100 samples)
  trainDat_name = 'MatLabModel2024' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
  winKernel = 7
elif params['Dataset'] == 'MC24_200': # MECOMPOSITES MODEL FROM 2024 (200 samples)
  trainDat_name = 'MatLabModel2024_200' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
  winKernel = 7
elif params['Dataset'] == 'MC24_500': # MECOMPOSITES MODEL FROM 2024 (500 samples)
  trainDat_name = 'MatLabModel2024_500' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
  winKernel = 7
elif params['Dataset'] == 'MC24_1000': # MECOMPOSITES MODEL FROM 2024 (1000 samples)
  trainDat_name = 'MatLabModel2024_1000' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 1
  winKernel = 7

elif params['Dataset'] == 'MC24x': # MC24_extended dataset (4000 samples 224x224 resolution)
  trainDat_name = 'MatLabModel2024_224_4kSamples' 
  sampleShape = [224,224]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 40
  winKernel = 17

# Various settings
trainDat_path = os.path.join('datain',trainDat_name)
yNames = ['FI'] # Names of ground truth features in input csv

numSamples = len(os.listdir(trainDat_path))*samplesPerFile # number of samples is number of files in datain
batchSize = params['batchSize'] # Batch size for training
valSize = math.floor(params['valSize']*numSamples) # Training and validation data split ratio
testSize = math.floor(params['testSize']*numSamples)
train_length = numSamples-valSize-testSize # Number of training samples 
epochs = params['Epochs'] # Max epochs for training
# epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = valSize // batchSize


# Load data  
def loadSampleNew(path):
    # Assuming loadSample uses pandas to read the CSV file
    # Adjust the delimiter and header options as needed
    _, file_extension = os.path.splitext(path)
    match file_extension:
       case '.csv':
        sample = pd.read_csv(path)
        # samples = sample.to_numpy()
        samples = np.array(sample)
       case '.parquet':
        sample = pd.read_parquet(path, engine='auto')
        samples = [y for x, y in sample.groupby('specimen')]
        samples = np.array(list(map(lambda x: x.to_numpy(), samples)))
    headers = np.array(sample.columns.values.tolist())
    samples = samples.reshape(samplesPerFile,sampleShape[0],sampleShape[1],-1)


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
    X = np.asarray(X).astype('float32')
    Y = np.asarray(Y).astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    

    if "coordinates" in headers: 
      coordIdx = np.where(headers == "coordinates")

      headers = np.concatenate(([[headers[0],'x_coord','y_coord'],headers[2:]])) # rectify the headers to include x and y coordinates separately
    return headers, ds


def load_all_samples(trainDat_path, numSamples):
    files = [os.path.join(trainDat_path, file) for file in os.listdir(trainDat_path)]


    def process_file(filepath):
        headers, values = loadSampleNew(filepath)
        return headers, values

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for i, future in enumerate(as_completed(futures)):
            if i == 0:
               headers, samples = future.result()
            else:
               addSamp = future.result()[1]
               samples = samples.concatenate(addSamp)

            print('Now loading file number {num} out of {total}'.format(num=i+1, total=numSamples/samplesPerFile))

    return headers, samples

headers, samples = load_all_samples(trainDat_path, numSamples)


samples = samples.shuffle(buffer_size=len(samples)) # Shuffle set
train_ds = samples.take(train_length)
remaining = samples.skip(train_length)
val_ds = remaining.take(valSize)
test_ds = remaining.skip(valSize)

# Take a copy of the datasets for RMSE evaluation at the end before repeat and shuffling is passed
train_ds_eval = train_ds.batch(batchSize)
val_ds_eval = val_ds.batch(batchSize)
test_ds_eval = test_ds.batch(batchSize)

X_trainShape = (train_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(xNames))
X_valShape = (val_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(xNames))
X_testShape = (test_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(xNames))
y_trainShape = (train_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(yNames))
y_valShape = (val_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(yNames))
y_testShape = (test_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(yNames))
    


# Define mean and variance for normalization based only on training set
feature_ds = train_ds.take(normalizerLength).map(lambda x, y: x) 
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(feature_ds)
# print(samples_array.shape)

# Training preprocessing
train_ds = train_ds.cache() # cache dataset for it to be used over iterations. Any operation before this will not be reapplied each iteration
train_ds = train_ds.shuffle(buffer_size = len(train_ds)) # Shuffle for random order


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=0):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

if params['dsAugmentation'] == 1:
  # train_ds = train_ds.map(
  #   lambda x, y: (augmentDs(x, training=True),augmentDs(y, training=True))) # Apply augmentations to increase the dataset size
  train_ds = train_ds.map(Augment())
train_ds = train_ds.batch(batchSize) # Batch
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
  inputShape: the length x width x features, input image shape
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
  x = normalizer(input)
  # if params['dsAugmentation'] == 1:
  #   x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)(x)


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

  x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',activation='linear')(x)

  if params['type'] == 'dense': # For the dense model we jsut pull the output y
     output = y
  else:
    output = x

  model = tf.keras.Model(inputs=input, outputs=output) # Create model
  return model
#   # Default initial learning rate is 0.001. If the the decay rate is 1 this will be held constant.
#   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=params['initial_lr'],
#     decay_steps=steps_per_epoch*epochs,
#     decay_rate=params['lr_decay_rate'])

#   def custom_loss(y_true,y_pred):
#     SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
#     loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.nn.relu(y_true))))
#     loss = tf.reduce_mean(loss)
#     return loss
  
#   def custom_loss5(y_true,y_pred):
#     SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
#     loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.math.multiply(tf.nn.relu(y_true),5))))
#     loss = tf.reduce_mean(loss)
#     return loss

#   def peak_loss(y_true,y_pred):
#     peakVal = tf.reduce_max(y_true, keepdims=True)
#     cond = tf.equal(y_true, peakVal)
#     # peakLoc = tf.where(cond)
#     # peakLoc_1d = tf.squeeze(peakLoc)
#     errorGrid = tf.math.subtract(y_true,y_pred)
#     zeroGrid = tf.math.subtract(y_true,y_true) # Grid of zeros so we only get loss in peak location
#     # peakPred = y_pred[peakLoc_1d.numpy()[0]]
#     # peakPred = tf.slice(y_pred, peakLoc, [1,1])
#     loss = tf.where(cond, errorGrid, zeroGrid)
#     loss = tf.reduce_mean(loss)

#     # loss = peakPred-peakVal
#     return loss


# #   Loss functions can be swept
#   if params['loss'] == 'MSE':
#     lossfunc = tf.keras.losses.MeanSquaredError()
#   elif params['loss'] == 'MAE':
#     lossfunc = tf.keras.losses.MeanAbsoluteError()
#   elif params['loss'] == 'Custom':
#     lossfunc = custom_loss
#   elif params['loss'] == 'Peak':
#     lossfunc = peak_loss
#   elif params['loss'] == 'Custom5':
#     lossfunc = custom_loss5
    

#   # Additional metrics to computes
#   def SSIM_metric(y_true, y_pred):
#     y_pred = tf.cast(y_pred, tf.float32) # y_pred is in a different type, recast
#     # squared_difference = tf.keras.ops.square(y_true - y_pred)
#     # return tf.keras.ops.mean(squared_difference)  # Note the `axis=-1`
      
#     return tf.reduce_mean(tf.image.ssim(
#     img1 = y_true,
#     img2 = y_pred,
#     max_val = 1,
#     filter_size=winKernel,
#     filter_sigma=1.5,
#     k1=0.01,
#     k2=0.03,
#     return_index_map=False
#     )   )



#   # Compile model with the optimizer in the sweep definition
#   if params['optimizer'] == 'Adadelta':
#      model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
#               loss=lossfunc, 
#               metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])
#   elif params['optimizer'] == 'Nadam':
#      model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
#               loss=lossfunc, 
#               metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])
#   else:
#      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
#               loss=lossfunc, 
#               metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])

#   return model

# %% Additional metrics to computes
def SSIM_metric(y_true, y_pred):
  y_pred = tf.cast(y_pred, tf.float32) # y_pred is in a different type, recast
    
  return tf.reduce_mean(tf.image.ssim(
  img1 = y_true,
  img2 = y_pred,
  max_val = 1,
  filter_size=winKernel,
  filter_sigma=1.5,
  k1=0.01,
  k2=0.03,
  return_index_map=False
  )   )


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
# %% Alternative model architectures
# Architectures are often designed for minmax scaling of features between -1 and 1.
# Data is passed normalised to mean 0 and std 1, which is close enough

def Xception_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.Xception(
      include_top=False,
      weights=None,
      input_tensor=None,
      input_shape=inputShape,
      pooling=None,
  )
  x = base_model(inputs = input)
  return input, x


def mobileNetV2_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.MobileNetV2(
      include_top=False,
      weights=None,
      input_tensor=None,
      input_shape=inputShape,
      pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def VGG16_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def ResNet50_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def ResNet50V2_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def InceptionV3_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def InceptionResNetV2_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def DenseNet121_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x

def NASNetMobile_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.NASNetMobile(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
  )
  x = base_model(inputs = input)
  return input, x


def EfficientNetV2S_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    include_preprocessing=False,
  )
  x = base_model(inputs = input)
  return input, x

def EfficientNetV2M_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.EfficientNetV2M(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    include_preprocessing=False,
  )
  x = base_model(inputs = input)
  return input, x

def EfficientNetV2L_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.EfficientNetV2L(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    include_preprocessing=False,
  )
  x = base_model(inputs = input)
  return input, x


def ConvNeXtTiny_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.ConvNeXtTiny(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    include_preprocessing=False,
  )
  x = base_model(inputs = input)
  return input, x

def ConvNeXtSmall_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.ConvNeXtSmall(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    include_preprocessing=False,
  )
  x = base_model(inputs = input)
  return input, x

def ConvNeXtLarge_Model(inputShape): # Expects scaled inputs in range -1 to 1
  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  
  base_model = tf.keras.applications.ConvNeXtLarge(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    include_preprocessing=False,
  )
  x = base_model(inputs = input)
  return input, x


def applyDecoder(input, x, outputShape, params):
  bnShape = x.shape # bottleneck
  inShape = bnShape[1] # assuming square
  outShape = outputShape[1]
  # strides = range(1,inShape+1) # Acceptable strides
  # kernels = range(1,inShape+1) # Acceptable kernel sizes
  # padding = range(1,inShape+1) # Acceptable padding sizes
  # combinations = []
  # for k in kernels:
  #    for s in strides:
  #       for p in padding:
  #         if s*(inShape-1)+k-2*p == outShape:
  #           combinations += [k,s]
  s = int(outShape/inShape) # Should be 32 (stride)
  pd = "same"
  if inShape == 5: # Need to upsamples to 7x7 for correct inverse conv
    x = tf.keras.layers.Resizing(
    height = 7,
    width = 7,
    interpolation='bilinear',
    crop_to_aspect_ratio=False,
    )(x)

    s = int(outShape/7)

  outputs = tf.keras.layers.Conv2DTranspose(filters = 1, # One filter for one output channel
                                      kernel_size = 3,
                                      strides = s,
                                      padding = pd)(x)
  
  model = tf.keras.Model(input, outputs)
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
   x = normalizer(input)
   if params['dsAugmentation'] == 1:
    x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)(x)

   # Encoder
   skip1, x1 = downSamplingBlock(x, 64, params)
   skip2, x2 = downSamplingBlock(x1, 128, params)
   skip3, x3 = downSamplingBlock(x2, 256, params)
   skip4, x4 = downSamplingBlock(x3, 512, params)

   # Bottlenexk
  
   bottleneck = double_convBlock(x4, 1024, params)

   # Decoder

   u6 = upSamplingBlock(bottleneck, skip4, 512, params)
   u7 = upSamplingBlock(u6, skip3, 256, params)
   u8 = upSamplingBlock(u7, skip2, 128, params)
   u9 = upSamplingBlock(u8, skip1, 64, params)

   # Final layer
   outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation = "linear")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(input, outputs, name="U-Net")

   return unet_model
   
#%%
#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
# checkpoint_path = 'training_checkpoints_{jn}_{num}/cp.ckpt'.format(jn='TESTJOB', num = 1)
# checkpoint_path = 'training_checkpoints_{jn}_{num}/model.weights.h5'.format(jn='TESTJOB', num = 1)
checkpoint_path = 'epoch-{epoch:02d}.weights.h5'
checkpoint_dir = 'training_checkpoints_{jn}_{num}'.format(jn=args.jobname, num = args.parallel)
cpLoadName = 'model.weights.h5' # The name of the checkpoint with the best weights at the end of training

# checkpoint_path = 'training_checkpoints_{jn}_{num}/model.weights.keras'.format(jn='TESTJOB', num = 1)
# checkpoint_dir = os.path.dirname(checkpoint_path)

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


class cp_delete_callback(tf.keras.callbacks.Callback):
  def __init__(self, checkpoint_dir, cpLoadName, nCp = 1):
    super().__init__()
    self.checkpoint_dir = checkpoint_dir # directory of checkpoints
    self.nCp = nCp # number of checkpoints to keep
    self.cpLoadName = cpLoadName

  def on_epoch_end(self, epoch, logs=None): # Delete the checkpoints before the best weights
    if not len(os.listdir(self.checkpoint_dir)) == 0:
      cps = os.listdir(self.checkpoint_dir)
      for file in cps[:-1]:
        os.remove(os.path.join(self.checkpoint_dir,file))

  def on_train_end(self, logs=None): # Rename the best checkpoint to allow easy loading
    bestCp = os.listdir(self.checkpoint_dir)[-1]
    oldPath = os.path.join(self.checkpoint_dir,bestCp)
    newPath = os.path.join(self.checkpoint_dir,self.cpLoadName)
    os.rename(oldPath, newPath)
     
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(tf.timestamp() - self.timetaken) 
        # print(self.times)
    def get_epoch_times(self):
        # Return the list of epoch times as a numpy array
        return np.array(self.times)     


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
# modelCNN = TBDCNet_modelCNN(inputShape = X_trainShape[1:], outputShape = y_trainShape[1:], params = params)

match params['type']:
  case 'Xception':
      input, output = Xception_Model(inputShape = X_trainShape[1:])
  case 'MobileNetV2':
      input, output = mobileNetV2_Model(inputShape = X_trainShape[1:])
  case 'VGG16':
      input, output = VGG16_Model(inputShape = X_trainShape[1:])
  case 'ResNet50':
      input, output = ResNet50_Model(inputShape = X_trainShape[1:])
  case 'ResNet50V2':
      input, output = ResNet50V2_Model(inputShape = X_trainShape[1:])
  case 'InceptionV3':
      input, output = InceptionV3_Model(inputShape = X_trainShape[1:])
  case 'InceptionResNetV2':
      input, output = InceptionResNetV2_Model(inputShape = X_trainShape[1:])
  case 'DenseNet121':
      input, output = DenseNet121_Model(inputShape = X_trainShape[1:])
  case 'NASNetMobile':
      input, output = NASNetMobile_Model(inputShape = X_trainShape[1:])
  case 'EfficientNetV2S':
      input, output = EfficientNetV2S_Model(inputShape = X_trainShape[1:])
  case 'EfficientNetV2M':
      input, output = EfficientNetV2M_Model(inputShape = X_trainShape[1:])
  case 'EfficientNetV2L':
      input, output = EfficientNetV2L_Model(inputShape = X_trainShape[1:])
  case 'ConvNeXtTiny':
      input, output = ConvNeXtTiny_Model(inputShape = X_trainShape[1:])
  case 'ConvNeXtSmall':
      input, output = ConvNeXtSmall_Model(inputShape = X_trainShape[1:])
  case 'ConvNeXtLarge':
      input, output = ConvNeXtLarge_Model(inputShape = X_trainShape[1:])
  case 'UNet':
      CNNModel = TBDCNet_UNet(inputShape = X_trainShape[1:], outputShape = y_trainShape[1:], params = params)
  case 'default':
      CNNModel = TBDCNet_modelCNN(inputShape = X_trainShape[1:], outputShape = y_trainShape[1:], params = params)

def preModel_compile(CNNModel):
  # Compile model with the optimizer in the sweep definition
  if params['optimizer'] == 'Adadelta':
      CNNModel.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])
  elif params['optimizer'] == 'Nadam':
      CNNModel.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])
  else:
      CNNModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])
  return CNNModel

if not params['type'] == 'default':
   if not params['type'] == 'UNet':
    CNNModel = applyDecoder(input, output, outputShape = y_trainShape[1:], params = params)
CNNModel = preModel_compile(CNNModel)


CNNModel.summary()

modelCNNname = 'CNNModel1'

#%%
#####################################################################
# Model training
#####################################################################

# Known issue: sometimes throws error related to the shape of the labels...
# Fit model to Failure index
time_callback_ins = timecallback()
modelCNN_history = CNNModel.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_ds,
                                validation_steps = validation_steps,
                                callbacks=[early_stopping_monitor, cp_callback, cp_delete_callback(checkpoint_dir, cpLoadName), time_callback_ins]
                                )

# Get the recorded epoch times after training is complete
epoch_times = {'trainTime':time_callback_ins.get_epoch_times().tolist()}

#%%
#####################################################################
# Data export
#####################################################################

trainingHist = modelCNN_history.history # save training history
trainingHist.update(epoch_times) # Add training time to history

CNNModel.load_weights(os.path.join(checkpoint_dir,cpLoadName)) # load best model weights


train_results = CNNModel.evaluate(
    x=train_ds_eval,
    y=None,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=True
)

train_results = pd.DataFrame.from_dict(train_results, orient='index',
                       columns=['train']).T

val_results = CNNModel.evaluate(
    x=val_ds_eval,
    y=None,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=True
)
val_results = pd.DataFrame.from_dict(val_results, orient='index',
                       columns=['val']).T


test_results = CNNModel.evaluate(
    x=test_ds_eval,
    y=None,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=True
)

test_results = pd.DataFrame.from_dict(test_results, orient='index',
                       columns=['test']).T

results = pd.concat([train_results, val_results, test_results])


#%% Save outputs

with open(histOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(trainingHist, f, indent=2)

with open(paramOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(params.to_json(), f, indent=2)

with open(resultpath, 'w') as f: # Dump data to json file at specified path
    json.dump(results.to_json(), f, indent=2)

CNNModel.save(modelOutPath)