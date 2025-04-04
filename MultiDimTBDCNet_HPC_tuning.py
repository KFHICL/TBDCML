#####################################################################
# Description
#####################################################################
'''
This script is run on the computing cluster (HPC) and contains the
model definition and training processes. It must be pointed to a 
sweep definition csv wherein the model hyperparameters are tabulated


Inputs:
-j: jobname, note that jobs run using the routine require the appending of "_" to the jobname when calling
-p: parallel, parameter to allow parallelisation on the HPC. Each p corresponds to a model defined in the sweep definition csv file

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
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#####################################################################
# Settings
#####################################################################

yNames = ['FI'] # Names of ground truth features in input csv - label to be predictednormalizerLength = 20 # Number of random samples used for computation of mean and variance used in data normalisation 
normalizerLength = 20 # Number of random samples used for computation of mean and variance used in data normalisation 


# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)

#####################################################################
# Formatting and settings done automatically
#####################################################################

# Add arguments for parallel running and training of several different models 
argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--parallel", help="Index for parallel running on HPC") # parameter to allow parallel running on the HPC
argParser.add_argument("-j", "--jobname", help="Job name") # Name of job passed when calling script
args = argParser.parse_args()
sweepIdx = int(args.parallel) # Index of model in sweep definition file


# Sweep definition containing hyperparameters
sweepPath = 'sweep_definition_{jn}.csv'.format(jn=args.jobname[:-2]) # Name of sweep definition file, one for all repetitions hence [:-2]
sweep_params = pd.read_csv(sweepPath)
sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[sweepIdx]
jobname = args.jobname[:-2]
# Dataset selection
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

elif params['Dataset'] == 'MC24_1000VfConst': # MECOMPOSITES MODEL FROM 2024 (1000 samples where Vf is kept constant for transfer learning study)
  trainDat_name = 'MatLabModel2024_1000SamplesVfConstant' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features


elif params['Dataset'] == 'MC24_10000': # MECOMPOSITES MODEL FROM 2024 (10,000 samples)
  trainDat_name = 'MatLabModel2024_10000' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

elif params['Dataset'] == 'MC24_100000': # MECOMPOSITES MODEL FROM 2024 (100,000 samples)
  trainDat_name = 'MatLabModel2024_100000' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

elif params['Dataset'] == 'MC24_VarVf': # MECOMPOSITES MODEL FROM 2024 (100 samples) variable Vf and random seed equivalent to the const Vf set
  trainDat_name = 'MatLabModel2024_100SamplesVfVariable' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

elif params['Dataset'] == 'MC24_ConstVf': # MECOMPOSITES MODEL FROM 2024 (100 samples) constant Vf 
  trainDat_name = 'MatLabModel2024_100SamplesVfConstant' 
  sampleShape = [60,20]
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

# Various settings
trainDat_path = os.path.join('datain',trainDat_name)
numSamples = len(os.listdir(trainDat_path))*samplesPerFile # number of samples is number of files in datain
batchSize = params['batchSize'] # Batch size for training
valSize = math.floor(params['valSize']*numSamples) # Training and validation data split ratio
testSize = math.floor(params['testSize']*numSamples)
trainValRatio = 1-params['valSize'] # For transfer learning study take no test set
train_length = numSamples-valSize-testSize # Number of training samples 
epochs = params['Epochs'] # Max epochs for training
# epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = valSize // batchSize

# Paths for data output
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
resultpath = 'results_{jn}_{num}.json'.format(jn=jobname, num = args.parallel)
resultpath = os.path.join('dataout',resultpath)

#####################################################################
# Data functions
#####################################################################
def formatCoords(values,coordIdx): # Formatting of coordinate column which is present in LFC18 dataset
    coords = [[x for x in values[:,coordIdx][y].split(' ') if x] for y in range(len(values[:,coordIdx]))] # Split coordinates by delimiter (space)
    coords = [np.char.strip(x, '[') for x in coords] # Coordinate output from abaqus has leading "["
    coords = [[x for x in coords[y] if x] for y in range(len(values[:,coordIdx]))] # remove empty array elements
    coords = np.array([[float(x) for x in coords[y][0:2]] for y in range(len(values[:,coordIdx]))]) # Take 2d coordinates and convert to float
    return coords

def loadSample(path = str): # Load a specimen (old and slow version, but more interpretable)
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



# New sample loading routine which is faster but less interpretable
# def loadSampleNew(path):
#     '''
#     Load single specimen
#     '''
#     # Assuming loadSample uses pandas to read the CSV file
#     # Adjust the delimiter and header options as needed
#     sample = pd.read_csv(path)
#     headers = sample.columns.values.tolist()
#     values = np.array(sample)

#     if "coordinates" in headers: 
#       coordIdx = headers.index("coordinates")
#       if '[' in values[0,1]: # Some coordinates will be formatted with brackets from abaqus export
#         coords = formatCoords(values,coordIdx)
#         values = np.column_stack((values[:,0],coords,values[:,2:])).astype(float) # Create a new values vector which contains the coordinates

#       headers = np.concatenate(([[headers[0],'x_coord','y_coord'],headers[2:]])) # rectify the headers to include x and y coordinates separately
#     headers = np.array(headers)
#     return headers, values


# def load_all_samples(trainDat_path, numSamples):
#     '''
#     Load all specimens in path
#     '''
#     files = [os.path.join(trainDat_path, file) for file in os.listdir(trainDat_path)]
#     headers_list = []
#     samples_list = []

#     def process_file(filepath):
#         headers, values = loadSampleNew(filepath)
#         return headers, values

#     with ThreadPoolExecutor() as executor:
#         futures = {executor.submit(process_file, file): file for file in files}
#         for i, future in enumerate(as_completed(futures)):
#             headers, result = future.result()
#             if i == 0:
#                 headers_list = headers  # Use headers from the first file
#             samples_list.append(result)
#             # print('Now loading file number {num} out of {total}'.format(num=i+1, total=numSamples))

#     samples_array = np.array(samples_list)
#     return headers_list, samples_array


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
      gtMatrix = tf.image.flip_left_right(tf.reshape(gtMatrix,[-1,height,width,1]))
      gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

    if  randAug.normal([]) > 0: # Randomly flip an image vertically 50% of the time
      inputMatrices = tf.image.flip_up_down(inputMatrices)
      gtMatrix = tf.image.flip_up_down(tf.reshape(gtMatrix,[-1,height,width,1]))
      gtMatrix = tf.reshape(gtMatrix,[-1,height,width])
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

# Can be used to show predicitions but generally unused
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

# Format is csv files with columns 
# Try new method of loading samples
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    

    # samples = samples.reshape(samples.shape[0],sampleShape[0],sampleShape[1],-1)
    
    if "coordinates" in headers: 
      coordIdx = np.where(headers == "coordinates")
      # This part only needed to export grid for plotting
      # if '[' in values[0,1]: # Some coordinates will be formatted with brackets from abaqus export
      #   coords = formatCoords(values,coordIdx)
      #   values = np.column_stack((values[:,0],coords,values[:,2:])).astype(float) # Create a new values vector which contains the coordinates

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
            
            
            # if i == 0:
            #     headers_list = headers  # Use headers from the first file
            #     samples_list = result
            #     # samples_list = tf.data.experimental.from_list(result)
            # else:
            #     # tmp = tf.data.experimental.from_list(result)
            #     # tmp = result
            #   samples_list = np.append(samples_list,result,axis = 0)
            print('Now loading file number {num} out of {total}'.format(num=i+1, total=numSamples/samplesPerFile))

    # samples_array = tf.data.experimental.from_list(samples_list)
    # samples_array = np.array(samples_list)
    return headers, samples
#####################################################################
# Data import, normalisation, and reshaping
#####################################################################

# New method
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

# Old method
# for i,file in enumerate(os.listdir(trainDat_path)):
#     filepath = os.path.join(trainDat_path,file)
#     if i==0:
#         headers, samples = loadSample(filepath)
#         samples = samples.reshape(1, np.shape(samples)[0],np.shape(samples)[1])
#     else:
#         addSamp = loadSample(filepath)[1]
#         samples = np.concatenate((samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))

# Non-scaled
# samples_NonScaled = samples
# # Reshape sample variable to have shape (samples, row, column, features)
# samples2D = samples.reshape(numSamples,sampleShape[0],sampleShape[1],samples.shape[-1])

#####################################################################
# ML training Settings and dataset preprocessing
#####################################################################

# Reminder: the header index is the following for the LFC18 dataset
# [   0        1         2       3     4     5     6     7     8      9      10   11    12    13  ]
# ['label' 'x_coord' 'y_coord' 'e11' 'e22' 'e12' 'S11' 'S22' 'S12' 'SMises' 'FI' 'E11' 'E22' 'E12']

# Find indeces of input features 
# featureIdx = []
# for name in xNames:
#    featureIdx += [np.where(headers == name)[0][0]]

# # Find indeces of labels
# gtIdx = []
# for name in yNames:
#    gtIdx += [np.where(headers == name)[0][0]]
   
# X = samples2D[:,:,:,featureIdx]  # Input features

# Y = samples2D[:,:,:,gtIdx] # Labels

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, Y, train_size=trainValRatio, shuffle = True)
# X_trainShape = X_train.shape
# X_valShape = X_val.shape
# y_trainShape = y_train.shape
# y_valShape = y_val.shape

# # Standardisation/normalisation scaling
# if params['standardisation'] == 'MinMax': # Normalistaion
#     Xscaler = preprocessing.MinMaxScaler() # Do a scaler for the in and outputs separately (to be able to inversely standardise predictions)
#     Yscaler = preprocessing.MinMaxScaler() # Note: these shoud be fit to the training data and applied without fitting to the validation data to avoid data leakage
# elif params['standardisation'] == 'Standard': # Standardisation
#     Xscaler = preprocessing.StandardScaler() # Default is scale by mean and divide by std
#     Yscaler = preprocessing.StandardScaler()
# elif params['standardisation'] == 0:
#    print('Warning: no scaling of data applied')

# if params['standardisation'] != 0:
#     # Fit the scaler
#     Xscaler.fit(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1)) # Scaler only takes input of shape (data,features)
#     # Transform training data
#     X_train = Xscaler.transform(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1))
#     X_train = X_train.reshape(X_trainShape) # reshape to 2D samples
#     # Transform validation data
#     X_val = Xscaler.transform(X_val.reshape(X_val.shape[0]*X_val.shape[1]*X_val.shape[2],-1))
#     X_val = X_val.reshape(X_valShape)

#     Yscaler.fit(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
#     y_train = Yscaler.transform(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
#     y_train = y_train.reshape(y_trainShape) # reshape to 2D samples
#     y_val = Yscaler.transform(y_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
#     y_val = y_val.reshape(y_valShape)

# # Create tensor datasets
# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) 
# val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) 

# # Training dataset preprocessing
# train_ds = train_ds.cache() # cache dataset for it to be used over iterations. Any operation before this will not be reapplied each iteration
# train_ds = train_ds.shuffle(buffer_size = len(train_ds)).batch(batchSize) # Shuffle for random order
# train_ds = train_ds.repeat() # Repeats dataset indefinitely to avoid errors
# if params['dsAugmentation'] == 1: # We can apply dataset augmentation to effectively increase the size of the dataset
#    train_ds = train_ds.map(lambda x,y: augmentImage(x,y)) # Here x and y are input images and ground truth
# train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# # Validation dataset preprocessing
# val_ds = val_ds.cache() # cache dataset for it to be used over iterations
# val_ds = val_ds.shuffle(buffer_size = len(val_ds)).batch(batchSize)
# val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Get shapes for later use
# train_in_shape = X_train.shape
# val_in_shape = X_val.shape
# train_out_shape = y_train.shape
# val_out_shape = y_val.shape


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

  output = x

  model = tf.keras.Model(inputs=input, outputs=output) # Create model
  return model
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
  
#   Loss functions can be swept
if params['loss'] == 'MSE':
  lossfunc = tf.keras.losses.MeanSquaredError()
elif params['loss'] == 'MAE':
  lossfunc = tf.keras.losses.MeanAbsoluteError()
elif params['loss'] == 'Custom':
  lossfunc = custom_loss



#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
checkpoint_path = 'epoch-{epoch:02d}.weights.h5'
checkpoint_dir = 'training_checkpoints_{jn}_{num}'.format(jn=args.jobname, num = args.parallel)
cpLoadName = 'model.weights.h5' # The name of the checkpoint with the best weights at the end of training


try:
  os.mkdir(checkpoint_dir) # Make checkpoint directory if it doesn't already exist
except:
  pass # Could be coded better but we don't want the sweep to fail due to issues on the HPC with mkdir

# Save best weights to checkpoint based on only validation loss

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
        print(self.times)
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
    restore_best_weights=False # Do not restore best weights after early stopping, 
                               # we do this manually to allow recording of the full training history and identify overfitting etc.
)

#####################################################################
# Model instantiation
#####################################################################

tf.keras.backend.clear_session() # Clear the state and frees up memory

# CNN Model creation
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

CNNModel = preModel_compile(CNNModel)

CNNModel.summary()
modelCNNname = 'CNNModel1'

#####################################################################
# Model training
#####################################################################

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


#####################################################################
# Data export
#####################################################################

trainingHist = modelCNN_history.history # save training history
trainingHist.update(epoch_times) # Add training time to history

CNNModel.load_weights(os.path.join(checkpoint_dir,cpLoadName)) # load best model weights

# predCNN = modelCNN.predict(X_train) # Make prediction
# predCNN_val = modelCNN.predict(X_val) # Prediction of only validation data
# predCNNShape = predCNN.shape
# predCNN_valShape = predCNN_val.shape

# # Inverse scaling
# if params['standardisation'] == 0:
#    predCNN_invStandard = predCNN
#    predCNN_val_invStandard = predCNN_val
#    ground_truth_invStandard = y_train
#    ground_truth_val_invStandard = y_val
# else:
#    predCNN_invStandard = Yscaler.inverse_transform(predCNN.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
#    predCNN_invStandard = predCNN_invStandard.reshape(predCNNShape)
#    predCNN_val_invStandard = Yscaler.inverse_transform(predCNN_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
#    predCNN_val_invStandard = predCNN_val_invStandard.reshape(predCNN_valShape)
   
#    ground_truth_invStandard =  Yscaler.inverse_transform(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
#    ground_truth_invStandard = ground_truth_invStandard.reshape(y_trainShape)
#    ground_truth_val_invStandard = Yscaler.inverse_transform(y_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
#    ground_truth_val_invStandard = ground_truth_val_invStandard.reshape(y_valShape)


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
# RMSE = tf.keras.metrics.RootMeanSquaredError()
# RMSE.update_state(ground_truth_invStandard,predCNN_invStandard)
# print('RMSE for training set = ' + str(RMSE.result().numpy()))
# if ground_truth_val_invStandard is not None:
#     RMSE_val = tf.keras.metrics.RootMeanSquaredError()
#     RMSE_val.update_state(ground_truth_val_invStandard,predCNN_val_invStandard)
#     print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))


# Save outputs

with open(histOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(trainingHist, f, indent=2)

with open(paramOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(params.to_json(), f, indent=2)

with open(resultpath, 'w') as f: # Dump data to json file at specified path
    json.dump(results.to_json(), f, indent=2)

# inputDat = np.zeros(samples2D.shape)
# for i in range(0,samples2D.shape[-1]): # Un-normalise all input features
#     inputDat[:,:,:,i] = samples2D[:,:,:,i]

# with open(histOutPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(trainingHist, f, indent=2)

# with open(predOutPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(predCNN_invStandard.tolist(), f, indent=2)

# with open(predOutPath_val, 'w') as f: # Dump data to json file at specified path
#     json.dump(predCNN_val_invStandard.tolist(), f, indent=2)

# with open(gtOutPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(ground_truth_invStandard.tolist(), f, indent=2)

# with open(gtOutPath_val, 'w') as f: # Dump data to json file at specified path
#     json.dump(ground_truth_val_invStandard.tolist(), f, indent=2)

# with open(paramOutPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(params.to_json(), f, indent=2)

# with open(inputOutPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(inputDat.tolist(), f, indent=2)

# with open(RMSEOutPath, 'w') as f: # Dump data to json file at specified path
#     json.dump(RMSE.result().numpy().tolist(), f, indent=2)

# with open(RMSEOutPath_val, 'w') as f: # Dump data to json file at specified path
#     json.dump(RMSE_val.result().numpy().tolist(), f, indent=2)


# Save the model
CNNModel.save(modelOutPath)