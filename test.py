#%%

import sys

import os
import random
import time
import math
import datetime
# import shutil
import json
# import scipy
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
TESTING = 0
# os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
# import tf_keras as keras # Legacy keras version which is equal to the one on the HPC


#%% Test model loading 
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\fullSweep1106_repeat1\dataout\model_fullSweep1106_repeat1_1.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\dataoutTESTJOB\model_TESTJOB_1.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24CrossValidation2808_1\dataout\model_MC24CrossValidation2808_1_1.keras"

# loaded_model = keras.models.load_model(modelPath)
# loaded_model.summary()

#%% Settings for test script
sweep_params = pd.read_csv(os.path.join(os.getcwd(),'sweep_definition_test.csv'))


sweepIdx = 1
yNames = ['FI'] # Names of ground truth features in input csv
normalizerLength = 20 # Number of random samples used for computation of mean and variance used in data normalisation 

# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)

sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[1]
jobname = 'TESTJOB'
parallel = 1
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

resultpath = 'results_{jn}_{num}.json'.format(jn=jobname, num = parallel)
resultpath = os.path.join('dataoutTESTJOB',resultpath)
#%% Import data
# trainDat_path = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\datain'
trainDat_path = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'
# trainDat_path = r'C:\Users\kaspe\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'

if params['Dataset'] == 'LFC18': # ABAQUS DATA FROM GAUDRON2018
  trainDat_name = 'Gaudron2018' 
  sampleShape = [55,20]
  xNames = ['E11','E22','E12'] # Names of input features in input csv
  trainDat_path = os.path.join(trainDat_path,'Gaudron2018') # Path for training data samples
  samplesPerFile = 1
  winKernel = 5
elif params['Dataset'] == 'MC24': # MECOMPOSITES MODEL FROM 2024 (100 samples)
  trainDat_name = 'MatLabModel2024' 
  sampleShape = [60,20]
  trainDat_path = os.path.join(trainDat_path,'MatLabModel2024')
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
  trainDat_path = os.path.join(trainDat_path,'MatLabModel2024_200')
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
  trainDat_path = os.path.join(trainDat_path,'MatLabModel2024_500')
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
  trainDat_path = os.path.join(trainDat_path,'MatLabModel2024_1000')
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
  trainDat_path = os.path.join(trainDat_path,'MatLabModel2024_224_4kSamples')
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
  samplesPerFile = 40
  winKernel = 17



# elif params['Dataset'] == 'MC24_10000': # MECOMPOSITES MODEL FROM 2024 (10,000 samples)
#   trainDat_name = 'MatLabModel2024_10000' 
#   sampleShape = [60,20]
#   trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1239_10kSamples'
#   if params['MC24_Features'] == 'Stiffness':
#     xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
#   elif params['MC24_Features'] == 'Vf_c2':
#     xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
#   elif params['MC24_Features'] == 'All':
#      xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

# elif params['Dataset'] == 'MC24_100000': # MECOMPOSITES MODEL FROM 2024 (100,000 samples)
#   trainDat_name = 'MatLabModel2024_100000' 
#   sampleShape = [60,20]
#   trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1439_100kSamples'
#   if params['MC24_Features'] == 'Stiffness':
#     xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
#   elif params['MC24_Features'] == 'Vf_c2':
#     xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
#   elif params['MC24_Features'] == 'All':
#      xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features




numSamples = len(os.listdir(trainDat_path))*samplesPerFile # number of samples is number of files in datain
if TESTING:
  numSamples = 40
batchSize = params['batchSize'] # Batch size for training
valSize = math.floor(params['valSize']*numSamples) # Training and validation data split ratio
testSize = math.floor(params['testSize']*numSamples)
train_length = numSamples-valSize-testSize # Number of training samples 
epochs = params['Epochs'] # Max epochs for training
# epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = valSize // batchSize


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
    



# def augmentImage(inputMatrices,gtMatrix):
#     '''
#   Apply augmentations to increase the dataset size

#   Args
#   ----------
#   inputMatrices: the Batchx55x20x3 input
#   gtMatrix: the Batchx55x30 ground truth

#   Returns
#   ----------
#   inputMatrices,gtMatrix with consistent augmentations applied

#   '''
#     height, width = sampleShape[0], sampleShape[1] # image dimensions

#     if randAug.normal([]) > 0: # Randomly flip an image horizontally 50% of the time
#       inputMatrices = tf.image.flip_left_right(inputMatrices)
#       gtMatrix = tf.image.flip_left_right(gtMatrix)
#       # gtMatrix = tf.image.flip_left_right(tf.reshape(gtMatrix,[-1,height,width,1]))
#       # gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

#     if  randAug.normal([]) > 0: # Randomly flip an image vertically 50% of the time
#       inputMatrices = tf.image.flip_up_down(inputMatrices)
#       gtMatrix = tf.image.flip_up_down(gtMatrix)
#       # gtMatrix = tf.image.flip_up_down(tf.reshape(gtMatrix,[-1,height,width,1]))
#       # gtMatrix = tf.reshape(gtMatrix,[-1,height,width])

#     # We can crop and resize but this messes with the boundary conditions hence not done right now
#     # if randAug.normal([]) > 0.67: # Scale to a random size within the bounding box and fit to a random location within this
#     #   crop_width = randAug.uniform(shape=(), minval=math.floor(0.7 * width), maxval=math.floor(0.9 * width), dtype = tf.int32)
#     #   crop_height = randAug.uniform(shape=(), minval=math.floor(0.7 * height), maxval=math.floor(0.9 * height), dtype = tf.int32)
#     #   offset_x = randAug.uniform(shape=(), minval=0, maxval=(width - crop_width), dtype = tf.int32)
#     #   offset_y = randAug.uniform(shape=(), minval=0, maxval=(height - crop_height), dtype = tf.int32)

#     #   inputMatrices = tf.image.crop_to_bounding_box(inputMatrices, offset_y, offset_x, crop_height, crop_width) # Crop to bounding box
#     #   gtMatrix = tf.image.crop_to_bounding_box(tf.reshape(gtMatrix,[-1,height,width,1]), offset_y, offset_x, crop_height, crop_width)
#     #   newHeight = crop_height
#     #   newWidth = crop_width
#     #   gtMatrix = tf.reshape(gtMatrix,[-1,newHeight,newWidth]) # Reshape ground truth back to not have channels dimensions
    
#     #   inputMatrices = tf.image.resize(inputMatrices, (height, width)) # Resize to original size (we want all images same size) - this distorts the image
#     #   gtMatrix = tf.image.resize(tf.reshape(gtMatrix,[-1,newHeight,newWidth,1]), (height, width), method='nearest')
#     #   gtMatrix = tf.reshape(gtMatrix,[-1,height,width])
#     #   inputMatrices = tf.cast(inputMatrices, tf.float64)
#     #   gtMatrix = tf.cast(gtMatrix, tf.float64)

      
#     return (inputMatrices,gtMatrix)


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
    if TESTING == 1:
      files = files[0:1]

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

headers, samples = load_all_samples(trainDat_path, numSamples)
if TESTING == 1:
  batchSize = 2

samples = samples.shuffle(buffer_size=len(samples), seed=seed) # Shuffle set
train_ds = samples.take(train_length)
remaining = samples.skip(train_length)
val_ds = remaining.take(valSize)
test_ds = remaining.skip(valSize)

# Take a copy of the datasets for RMSE evaluation at the end before repeat and shuffling is passed
train_ds_eval = train_ds.batch(batchSize).cache()
val_ds_eval = val_ds.batch(batchSize).cache()
test_ds_eval = test_ds.batch(batchSize).cache()

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
# val_ds = val_ds.shuffle(buffer_size = len(val_ds)).batch(batchSize) # avoid shuffling to be deterministic
val_ds = val_ds.batch(batchSize) # Batch
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Test preprocessing
test_ds = test_ds.cache() # cache dataset for it to be used over iterations
# test_ds = test_ds.shuffle(buffer_size = len(test_ds)).batch(batchSize)
test_ds = test_ds.batch(batchSize) # Batch
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared


# %%

# for i,file in enumerate(os.listdir(trainDat_path)):
#     print('Now loading file number {num} out of {total}'.format(num = i, total = numSamples))
#     filepath = os.path.join(trainDat_path,file)
#     if i==0:
#         headers, samples = loadSample(filepath)
#         samples = samples.reshape(1, np.shape(samples)[0],np.shape(samples)[1])
#     else:
#         addSamp = loadSample(filepath)[1]
#         samples = np.concatenate((samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
# samples_NonStandard = samples
# samples, scaler = normalise(samples.reshape(samples.shape[0]*samples.shape[1],-1),params) # retain the scaler parameters such that inverse scaling can be done
# means = scaler.mean_ # Will have 1 value for each feature in the data
# std = np.sqrt(scaler.var_)
# Reshape sample variable to have shape (samples, row, column, features)
# samples2D = samples.reshape(numSamples,sampleShape[0],sampleShape[1],samples.shape[-1])

# # Find indeces of input features 
# featureIdx = []
# for name in xNames:
#    featureIdx += [np.where(headers == name)[0][0]]

# # Find indeces of ground truth features 
# gtIdx = []
# for name in yNames:
#    gtIdx += [np.where(headers == name)[0][0]]

# X = samples2D[:,:,:,featureIdx]  # Input features

# Y = samples2D[:,:,:,gtIdx] # Labels

# # The below are just used for validation and shape (TODO: replace validation method with tf tensors)
# X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, Y, train_size=trainValRatio, shuffle = True)
# X_trainShape = X_train.shape
# X_valShape = X_val.shape
# y_trainShape = y_train.shape
# y_valShape = y_val.shape


# # Standardisation/normalisation
# if params['standardisation'] == 'MinMax':
#     Xscaler = preprocessing.MinMaxScaler() # Do a scaler for the in and outputs separately (to be able to inversely standardise predictions)
#     Yscaler = preprocessing.MinMaxScaler() # Note: these shoud be fit to the training data and applied without fitting to the validation data to avoid data leakage
# elif params['standardisation'] == 'Standard':
#     Xscaler = preprocessing.StandardScaler() # Default is scale by mean and divide by std
#     Yscaler = preprocessing.StandardScaler()
# elif params['standardisation'] == 0:
#    print('Warning: no standardisation of data applied')
# if params['standardisation'] != 0:
#     Xscaler.fit(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1)) # Scaler only takes input of shape (data,features)
#     X_train = Xscaler.transform(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1))
#     X_train = X_train.reshape(X_trainShape) # reshape to 2D samples
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



# # Training preprocessing
# train_ds = train_ds.cache() # cache dataset for it to be used over iterations. Any operation before this will not be reapplied each iteration
# train_ds = train_ds.shuffle(buffer_size = len(train_ds)).batch(batchSize) # Shuffle for random order
# train_ds = train_ds.repeat() # Repeats dataset indefinitely to avoid errors
# if params['dsAugmentation'] == 1: # We can apply dataset augmentation to effectively increase the dataset size
#    train_ds = train_ds.map(lambda x,y: augmentImage(x,y))
# train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# # Validation preprocessing
# val_ds = val_ds.cache() # cache dataset for it to be used over iterations
# val_ds = val_ds.shuffle(buffer_size = len(val_ds)).batch(batchSize)
# val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared


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





#%% Plot data distribution to see if following gaussian approximately

# # Exx
# ExxPlot = pd.DataFrame(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1)[:,0])
# ExxPlot.columns = ['Exx']
# ExxPlot['Specimens']='All data'

# temp = pd.DataFrame(X_val.reshape(X_val.shape[0]*X_val.shape[1]*X_val.shape[2],-1)[:,0])
# temp.columns = ['Exx']
# temp['Specimens']='Validation data'

# ExxPlot = pd.concat([ExxPlot, temp])

# # Eyy
# EyyPlot = pd.DataFrame(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1)[:,1])
# EyyPlot.columns = ['Eyy']
# EyyPlot['Specimens']='All data'

# temp = pd.DataFrame(X_val.reshape(X_val.shape[0]*X_val.shape[1]*X_val.shape[2],-1)[:,1])
# temp.columns = ['Eyy']
# temp['Specimens']='Validation data'

# EyyPlot = pd.concat([EyyPlot, temp])

# # Gxy
# GxyPlot = pd.DataFrame(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1)[:,2])
# GxyPlot.columns = ['Gxy']
# GxyPlot['Specimens']='All data'

# temp = pd.DataFrame(X_val.reshape(X_val.shape[0]*X_val.shape[1]*X_val.shape[2],-1)[:,2])
# temp.columns = ['Gxy']
# temp['Specimens']='Validation data'

# GxyPlot = pd.concat([GxyPlot, temp])

# # FI
# FIPlot = pd.DataFrame(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
# FIPlot.columns = ['FI']
# FIPlot['Specimens']='All data'

# temp = pd.DataFrame(y_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
# temp.columns = ['FI']
# temp['Specimens']='Validation data'

# FIPlot = pd.concat([FIPlot, temp])


# px = 1/plt.rcParams['figure.dpi']  # pixel in inches
# fig = plt.figure(figsize=(1000*px, 600*px), layout="constrained")
# plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis
# # E11
# ax = plt.subplot(2,2,1)
# sns.histplot(ExxPlot, x='Exx',hue = 'Specimens')
# plt.grid()
# plt.xlabel('Exx [MPa]')
# plt.ylabel('Count')
# plt.title('Horizontal stiffness')

# # E22
# ax = plt.subplot(2,2,2)
# sns.histplot(EyyPlot, x='Eyy',hue = 'Specimens')
# plt.grid()
# plt.xlabel('Eyy [MPa]')
# plt.ylabel('Count')
# plt.title('Vertical stiffness')

# #G12
# ax = plt.subplot(2,2,3)
# sns.histplot(GxyPlot, x='Gxy',hue = 'Specimens')
# plt.grid()
# plt.xlabel('Gxy [Mpa]')
# plt.ylabel('Count')
# plt.title('Shear stiffness')
# #FI
# ax = plt.subplot(2,2,4)
# sns.histplot(FIPlot, x='FI',hue = 'Specimens')
# plt.grid()
# plt.xlabel('FI')
# plt.ylabel('Count')
# plt.title('Failure index')

# plt.show()

# %% Plot sample to check import

# # Test that import and reshape is correct
# fig, axs = plt.subplots(2, int(len(headers)/2), sharex=True, sharey=True,figsize=[12,7.5]) # Create subplots to fit all variables
# sampleNum = 0
# # Plot  map
# for i in range(len(headers)):
#   ax = plt.subplot(2, int(len(headers)/2), i+1)
#   CS = ax.contourf(grid[0],grid[1],samples2D[sampleNum,:,:,i])
#   plt.xlabel('x')
#   plt.ylabel('y')
#   plt.title(headers[i])
#   fig.colorbar(CS)

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

   # Custom activation function is linear between 0 and 1 and otherwise constant
  # def custom_activation(x):
  #     return tf.math.minimum(K.relu(x), 1)

  x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',activation='linear')(x)

  if params['type'] == 'dense': # For the dense model we jsut pull the output y
     output = y
  else:
    output = x

  model = tf.keras.Model(inputs=input, outputs=output) # Create model

  return model



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
      name="xception",
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
      name="mobilenetV2",
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
    name="vgg16",
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
    name="ResNet50",
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
    name="ResNet50V2",
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
    name="InceptionV3",
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
    name="InceptionResNetV2",
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
    name="DenseNet121",
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
    name="NASNetMobile",
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
    name="efficientnetv2-s",
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
    name="efficientnetv2-m",
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
    name="efficientnetv2-l",
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
    name="ConvNeXtTiny",
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
    name="ConvNeXtSmall",
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
    name="ConvNeXtLarge",
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
    pad_to_aspect_ratio=False,
    fill_mode='constant',
    fill_value=0.0,
    data_format=None,
    )(x)

    s = int(outShape/7)

  outputs = tf.keras.layers.Conv2DTranspose(filters = 1, # One filter for one output channel
                                      kernel_size = 3,
                                      strides = s,
                                      padding = pd)(x)
  
  model = tf.keras.Model(input, outputs)
  return model


# %% UNET

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

# testList = [ 
#     'Xception',
#     'MobileNetV2',
#     'VGG16',
#     'ResNet50',
#     'ResNet50V2',
#     'InceptionV3',
#     'InceptionResNetV2',
#     'DenseNet121',
#     'NASNetMobile',
#     'EfficientNetV2S',
#     'EfficientNetV2M',
#     'EfficientNetV2L',
#     'ConvNeXtTiny',
#     'ConvNeXtSmall',
#     'ConvNeXtLarge'
# ]






# x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (int(params['layer1Kernel']),int(params['layer1Kernel'])),  padding='same',strides = 2,activation=params['conv1Activation'])(x)
  
# outputs = tf.keras.layers.Conv2D(outputShape[-1], 3, activation="linear", padding="same")(x) # Regression layer
# model = tf.keras.Model(input, outputs)
# return model

# Xception = Xception_Model(inputShape = X_trainShape[1:], outputShape=y_trainShape[1:], params = params)






#%%
#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
# checkpoint_path = 'training_checkpoints_{jn}_{num}/cp.ckpt'.format(jn='TESTJOB', num = 1)
# checkpoint_path = 'training_checkpoints_{jn}_{num}/model.weights.h5'.format(jn='TESTJOB', num = 1)
checkpoint_path = 'epoch-{epoch:04d}.weights.h5'
checkpoint_dir = 'training_checkpoints_{jn}_{num}'.format(jn=jobname, num = 1)
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
    restore_best_weights=False # Do not restore best weights after early stopping, we do this manually to allow recording of the full training history
)

#%%
############################################################con#######
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
epochs = 200
time_callback_ins = timecallback()
# Clear existing files in the checkpoint directory
for file in os.listdir(checkpoint_dir):
  file_path = os.path.join(checkpoint_dir, file)
  if os.path.isfile(file_path):
    os.remove(file_path)

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

# predCNN = modelCNN.predict(train_ds_eval)
# predCNN_val = modelCNN.predict(val_ds_eval)
# predCNN_test = modelCNN.predict(test_ds_eval)
# Old method where the normalisation was not part of the model
# predCNN = modelCNN.predict(X_train) # Make prediction
# predCNN_val = modelCNN.predict(X_val) # Prediction of only validation data
# predCNNShape = predCNN.shape
# predCNN_valShape = predCNN_val.shape

# # Inverse standardisation
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
results['RMSE'] = np.sqrt(results['mean_squared_error'])
print(results)



# %%
# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(modelCNN_history.history['loss'], label='Training Loss')
plt.plot(modelCNN_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.hlines(y=0.023481, xmin=0, xmax=epochs, color='orange', linestyle='--', label='Val loss from eval')
plt.hlines(y=0.021117, xmin=0, xmax=epochs, color='blue', linestyle='--', label='Train loss from eval')
plt.hlines(y=np.min(modelCNN_history.history['val_loss']), xmin=0, xmax=epochs, color='g', linestyle='--', label='Min val loss')
# plt.ylim([0.01,0.03])
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation metrics (e.g., MAE)
# plt.figure(figsize=(10, 6))
# plt.plot(modelCNN_history.history['mean_absolute_error'], label='Training MAE')
# plt.plot(modelCNN_history.history['val_mean_absolute_error'], label='Validation MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.title('Training and Validation MAE')
# plt.legend()
# plt.grid(True)
# plt.show()
# # RMSE
# RMSE = tf.keras.metrics.RootMeanSquaredError()
# RMSE.update_state(train_ds_eval,predCNN)
# print('RMSE for training set = ' + str(RMSE.result().numpy()))
# if ground_truth_val_invStandard is not None:
#     RMSE_val = tf.keras.metrics.RootMeanSquaredError()
#     RMSE_val.update_state(ground_truth_val_invStandard,predCNN_val_invStandard)
#     print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))



# RMSE = tf.keras.metrics.RootMeanSquaredError()
# RMSE.update_state(ground_truth_invStandard,predCNN_invStandard)
# print('RMSE for training set = ' + str(RMSE.result().numpy()))
# if ground_truth_val_invStandard is not None:
#     RMSE_val = tf.keras.metrics.RootMeanSquaredError()
#     RMSE_val.update_state(ground_truth_val_invStandard,predCNN_val_invStandard)
#     print('RMSE for validation set  = ' + str(RMSE_val.result().numpy()))


#%% Save outputs

with open(histOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(trainingHist, f, indent=2)

with open(paramOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(params.to_json(), f, indent=2)

with open(resultpath, 'w') as f: # Dump data to json file at specified path
    json.dump(results.to_json(), f, indent=2)


# Old outputs from LFC18 and MC24
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
# %% Various tests
# Predict on a single sample from the test dataset
# modelPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\LFC18BaselineCrossValidation2208\LFC18BaselineCrossValidation2208_1\dataout\model_LFC18BaselineCrossValidation2208_1_1.keras"
# testmodel = tf.keras.models.load_model(modelPath, custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss})
for sample, ground_truth in train_ds.take(1):
  prediction = CNNModel.predict(tf.expand_dims(sample[0], axis=0))  # Predict on the first sample in the batch
  
  prediction = tf.squeeze(prediction)  # Remove batch dimension
  
  # print(ground_truth.shape)
  ground_truth = tf.squeeze(ground_truth[0])  # Remove batch dimension

  sample = sample[0]
  # print(sample.shape)
  # sample = tf.squeeze(sample[0], axis=0)  # Remove batch dimension
RMSE = tf.keras.metrics.RootMeanSquaredError()
RMSE.update_state(ground_truth,prediction)
print(RMSE.result().numpy())

MSE = tf.keras.metrics.MeanSquaredError()
MSE.update_state(ground_truth,prediction)
print(MSE.result().numpy())

# Plot the prediction field vs ground truth
grid_x, grid_y = tf.meshgrid(
  tf.linspace(0.0, 1.0, sample.shape[1]),
  tf.linspace(0.0, 1.0, sample.shape[0])
)

# Plot the prediction and ground truth
plt.figure(figsize=(12, 6))

# Ground truth
plt.subplot(1, 2, 1)
plt.contourf(grid_x, grid_y, ground_truth)
plt.colorbar()
plt.title('Ground Truth')
plt.xlabel('x')
plt.ylabel('y')

# Prediction
plt.subplot(1, 2, 2)
plt.contourf(grid_x, grid_y, prediction)
plt.colorbar()
plt.title('Prediction')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()



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
