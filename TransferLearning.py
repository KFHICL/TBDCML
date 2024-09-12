#%%
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

from concurrent.futures import ThreadPoolExecutor, as_completed

import seaborn as sns
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC


#%% Settings for transfer learning script
sweep_params = pd.read_csv(r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\sweep_definition_TransferLearning.csv')
sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[1]
jobname = 'TLTEST'
parallel = 1
timeStamp = datetime.datetime.now().strftime("%Y%m%d%H%M") # Not currently used
histOutName = 'trainHist_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Training history file
histOutPath = os.path.join('dataoutTLTEST',histOutName)
predOutName = 'predictions_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Predictions
predOutPath = os.path.join('dataoutTLTEST',predOutName)
predOutName_val = 'predictions_val_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Predictions
predOutPath_val = os.path.join('dataoutTLTEST',predOutName_val)
gtOutName = 'groundTruth_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Ground truths
gtOutPath = os.path.join('dataoutTLTEST',gtOutName)
gtOutName_val = 'groundTruth_val_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Ground truths
gtOutPath_val = os.path.join('dataoutTLTEST',gtOutName_val)
paramOutName = 'parameters_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Hyperparameters
paramOutPath = os.path.join('dataoutTLTEST',paramOutName)
inputOutName = 'input_{jn}_{num}.json'.format(jn=jobname, num = parallel) # Model inputs
inputOutPath = os.path.join('dataoutTLTEST',inputOutName)
modelOutPath = 'model_{jn}_{num}.keras'.format(jn=jobname, num = parallel) # Model architecture and weights
modelOutPath = os.path.join('dataoutTLTEST',modelOutPath)
RMSEOutPath = 'RMSE_{jn}_{num}.json'.format(jn=jobname, num = parallel) # RMSE of model to quickly compare models when many are trained in a sweep
RMSEOutPath = os.path.join('dataoutTLTEST',RMSEOutPath)
RMSEOutPath_val = 'RMSE_val_{jn}_{num}.json'.format(jn=jobname, num = parallel) # RMSE of model to quickly compare models when many are trained in a sweep
RMSEOutPath_val = os.path.join('dataoutTLTEST',RMSEOutPath_val)

#%% Import data
if params['Dataset'] == 'LFC18': # ABAQUS DATA FROM GAUDRON2018
  trainDat_name = 'Gaudron2018' 
  sampleShape = [55,20]
  xNames = ['E11','E22','E12'] # Names of input features in input csv
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain' # Path for training data samples
  
elif params['Dataset'] == 'MC24': # MECOMPOSITES MODEL FROM 2024 (100 samples)
  trainDat_name = 'MatLabModel2024' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240703_1417'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

elif params['Dataset'] == 'MC24_200': # MECOMPOSITES MODEL FROM 2024 (1000 samples)
  trainDat_name = 'MatLabModel2024_200' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240726_1456_200Samples'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

elif params['Dataset'] == 'MC24_500': # MECOMPOSITES MODEL FROM 2024 (1000 samples)
  trainDat_name = 'MatLabModel2024_500' 
  sampleShape = [60,20]
  trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240726_1457_500Samples'
  if params['MC24_Features'] == 'Stiffness':
    xNames = ['Ex','Ey','Gxy'] # Use stiffnesses (default)
  elif params['MC24_Features'] == 'Vf_c2':
    xNames = ['Vf','c2'] # Use fibre volume fraction and orientation distribution
  elif params['MC24_Features'] == 'All':
     xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features

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


yNames = ['FI'] # Names of ground truth features in input csv
numSamples = len(os.listdir(trainDat_path)) # number of samples is number of files in datain
batchSize = params['batchSize'] # Batch size for training
trainValRatio = params['trainValRatio'] # Training and validation data split ratio
train_length = round(numSamples * trainValRatio) # Number of training samples 
epochs = params['Epochs'] # Max epochs for training
# epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = math.ceil((numSamples-train_length) / batchSize)

# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)


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


def load_all_samples(trainDat_path, numSamples):
    files = [os.path.join(trainDat_path, file) for file in os.listdir(trainDat_path)]
    headers_list = []
    samples_list = []

    def process_file(filepath):
        headers, values = loadSampleNew(filepath)
        return headers, values

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for i, future in enumerate(as_completed(futures)):
            headers, result = future.result()
            if i == 0:
                headers_list = headers  # Use headers from the first file
            samples_list.append(result)
            print('Now loading file number {num} out of {total}'.format(num=i+1, total=numSamples))

    samples_array = np.array(samples_list)
    return headers_list, samples_array

headers, samples = load_all_samples(trainDat_path, numSamples)
# print(samples_array.shape)



# for i,file in enumerate(os.listdir(trainDat_path)):
#     print('Now loading file number {num} out of {total}'.format(num = i, total = numSamples))
#     filepath = os.path.join(trainDat_path,file)
#     if i==0:
#         headers, samples = loadSample(filepath)
#         samples = samples.reshape(1, np.shape(samples)[0],np.shape(samples)[1])
#     else:
#         addSamp = loadSample(filepath)[1]
#         samples = np.concatenate((samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
samples_NonStandard = samples
# samples, scaler = normalise(samples.reshape(samples.shape[0]*samples.shape[1],-1),params) # retain the scaler parameters such that inverse scaling can be done
# means = scaler.mean_ # Will have 1 value for each feature in the data
# std = np.sqrt(scaler.var_)
# Reshape sample variable to have shape (samples, row, column, features)
samples2D = samples.reshape(numSamples,sampleShape[0],sampleShape[1],samples.shape[-1])

# Find indeces of input features 
featureIdx = []
for name in xNames:
   featureIdx += [np.where(headers == name)[0][0]]

# Find indeces of ground truth features 
gtIdx = []
for name in yNames:
   gtIdx += [np.where(headers == name)[0][0]]

X = samples2D[:,:,:,featureIdx]  # Input features

Y = samples2D[:,:,:,gtIdx] # Labels

# The below are just used for validation and shape (TODO: replace validation method with tf tensors)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, Y, train_size=trainValRatio, shuffle = True)
X_trainShape = X_train.shape
X_valShape = X_val.shape
y_trainShape = y_train.shape
y_valShape = y_val.shape



# RESHAPE TO CONFORM WITH PRE-TRAINED MODELS
# PADS WITH ZEROS TO BE SQUARE


# Standardisation/normalisation
if params['standardisation'] == 'MinMax':
    Xscaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) # Do a scaler for the in and outputs separately (to be able to inversely standardise predictions)
    Yscaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) # Note: these shoud be fit to the training data and applied without fitting to the validation data to avoid data leakage
elif params['standardisation'] == 'Standard':
    Xscaler = preprocessing.StandardScaler() # Default is scale by mean and divide by std
    Yscaler = preprocessing.StandardScaler()
elif params['standardisation'] == 0:
   print('Warning: no standardisation of data applied')
if params['standardisation'] != 0:
    Xscaler.fit(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1)) # Scaler only takes input of shape (data,features)
    X_train = Xscaler.transform(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],-1))
    X_train = X_train.reshape(X_trainShape) # reshape to 2D samples
    X_val = Xscaler.transform(X_val.reshape(X_val.shape[0]*X_val.shape[1]*X_val.shape[2],-1))
    X_val = X_val.reshape(X_valShape)

    Yscaler.fit(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
    y_train = Yscaler.transform(y_train.reshape(y_train.shape[0]*y_train.shape[1]*y_train.shape[2],-1))
    y_train = y_train.reshape(y_trainShape) # reshape to 2D samples
    y_val = Yscaler.transform(y_val.reshape(y_val.shape[0]*y_val.shape[1]*y_val.shape[2],-1))
    y_val = y_val.reshape(y_valShape)


X_train = tf.image.resize(
    X_train,
    (64,64),
    method=tf.image.ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)

X_val = tf.image.resize(
    X_val,
    (64,64),
    method=tf.image.ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)


# X_train = tf.image.resize(
#     X_train,
#     (X_trainShape[1],X_trainShape[1]),
#     method=tf.image.ResizeMethod.BILINEAR,
#     preserve_aspect_ratio=False,
#     antialias=False,
#     name=None
# )

# X_val = tf.image.resize(
#     X_val,
#     (X_valShape[1],X_valShape[1]),
#     method=tf.image.ResizeMethod.BILINEAR,
#     preserve_aspect_ratio=False,
#     antialias=False,
#     name=None
# )

# X_train = tf.image.resize_with_pad(
#     X_train,
#     X_trainShape[1],
#     X_trainShape[1],
#     method=tf.image.ResizeMethod.BILINEAR,
#     antialias=False
# )

# X_val = tf.image.resize_with_pad(
#     X_val,
#     X_valShape[1],
#     X_valShape[1],
#     method=tf.image.ResizeMethod.BILINEAR,
#     antialias=False
# )

# X_train = tf.image.resize_with_pad(
#     X_train,
#     X_trainShape[1],
#     X_valShape[1],
#     method='nearest',
#     antialias=False
# )

# X_val = tf.image.resize_with_pad(
#     X_val,
#     X_valShape[1],
#     X_valShape[1],
#     method='nearest',
#     antialias=False
# )


# Create tensor datasets
# ds = tf.data.Dataset.from_tensor_slices((X, Y)) 
# ds = ds.shuffle(buffer_size = len(ds),reshuffle_each_iteration=False ) # Shuffle dataset to get different val/train datasets each run of the code
# # BE CAREFUL: reshuffle_each_iteration must be false to avoid shuffling each epoch before taking out the validation data. If set to True we get data leakage from the validation set

# # Split into training and validation datasets
# train_ds = ds.take(train_length)
# val_ds = ds.skip(train_length)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) 
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) 




# Training preprocessing
train_ds = train_ds.cache() # cache dataset for it to be used over iterations. Any operation before this will not be reapplied each iteration
train_ds = train_ds.shuffle(buffer_size = len(train_ds)).batch(batchSize) # Shuffle for random order
train_ds = train_ds.repeat() # Repeats dataset indefinitely to avoid errors
if params['dsAugmentation'] == 1: # We can apply dataset augmentation to effectively increase the dataset size
   train_ds = train_ds.map(lambda x,y: augmentImage(x,y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Validation preprocessing
val_ds = val_ds.cache() # cache dataset for it to be used over iterations
val_ds = val_ds.shuffle(buffer_size = len(val_ds)).batch(batchSize)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Get shapes for later use
train_in_shape = X_train.shape
val_in_shape = X_val.shape
train_out_shape = y_train.shape
val_out_shape = y_val.shape





#%%
#####################################################################
# Transfer learning model definitions
#####################################################################

tf.keras.backend.clear_session()
inputShape = train_in_shape[1:]

# Instantiate base model

baseModel = tf.keras.applications.MobileNetV2(
    input_shape=inputShape,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
)

# baseModel = tf.keras.applications.ResNet50(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=inputShape,
#     pooling=None,
# )

# baseModel = tf.keras.applications.EfficientNetV2B0(
#     include_top = False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     include_preprocessing=False,
# )

# baseModel = tf.keras.applications.EfficientNetB1(
#     input_shape=inputShape,
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     pooling=None,
# )

baseModel.trainable = False

input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
x = input

# x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
# x = tf.keras.applications.resnet.preprocess_input(x)

# x = keras.applications.mobilenet_v2.preprocess_input(x)


# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 30,30,96
    'block_3_expand_relu',   # 15,15,144
    'block_6_expand_relu',   # 8,8,192
    'block_13_expand_relu',  # 4,4,576
    'block_16_project',      # 2,2,960
]
base_model_outputs = [baseModel.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=baseModel.input, outputs=base_model_outputs)

down_stack.trainable = False

def deconv_block(filters, size, norm_type='batchnorm', dropout_rate = 0.5):
  '''
  Conv2DTranspose => Batchnorm => Dropout => Relu
  '''
  initializer = tf.random_normal_initializer(0., 0.02) # Initialises value randomly from normal distribution
  result = tf.keras.Sequential() # groups layers to stack (creates stack here)
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                      use_bias=False)) # Adds deconvolutional layer

  result.add(tf.keras.layers.BatchNormalization()) # adds batch normalisation layer
  result.add(tf.keras.layers.Dropout(dropout_rate)) # Dropout
  result.add(tf.keras.layers.ReLU()) # Add activation function
  return result # Spit out upsampling block


up_stack = [
    deconv_block(512, 3, dropout_rate = 0.5),  # 4x4 -> 8x8
    deconv_block(256, 3, dropout_rate = 0.4),  # 8x8 -> 16x16
    deconv_block(128, 3, dropout_rate = 0.3),  # 16x16 -> 32x32
    deconv_block(64, 3, dropout_rate = 0.2),   # 32x32 -> 64x64
] # Create decoder from 4 upsampling blocks of each the layers in devonv_block

skips = down_stack(input)
x = skips[-1]
skips = reversed(skips[:-1])

# Upsampling and establishing the skip connections
for up, skip in zip(up_stack, skips):
  x = up(x)
  concat = tf.keras.layers.Concatenate()
  x = concat([x, skip])

# This is the last layer of the model
last = tf.keras.layers.Conv2DTranspose(
    filters=1, kernel_size=3, strides=2,
    padding='same',name = 'Final')  #32x23 -> 64x64

x = last(x)

# Re-establish dimensions using convolutional math
x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (5,7),strides = (1,3),  padding='same',activation='linear')(x)

# x = baseModel(x)

# x = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)

# x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3,3),strides = (2,2),  padding='valid',activation='linear')(x)


# Many deconv layers
# x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3,3),strides = (3,1),  padding='same',activation='relu')(x)
# x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3,3),strides = (2,2),  padding='same',activation='relu')(x)
# x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (3,3),strides = (1,1),  padding='same',activation='relu')(x)
# x = tf.keras.layers.Conv2DTranspose(filters = 16, kernel_size = (3,3),strides = (5,5),  padding='same',activation='relu')(x)
# x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (3,3),strides = (1,1),  padding='same',activation='linear')(x)

# Single deconv layer
# x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (3,3),strides = (30,10),  padding='same',activation='linear')(x)

# Fully connected output
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# imagenet_utils.validate_activation(classifier_activation, weights)
# x = tf.keras.layers.Dense(units = train_out_shape[1]*train_out_shape[2], activation='sigmoid', name="dense")(x)
# x = tf.keras.layers.Dense(units = train_out_shape[1]*train_out_shape[2], activation='linear', name="predictions")(x)


# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(units = train_out_shape[1]*train_out_shape[2], activation="linear", name="Linear_output")(x)
# x = tf.keras.layers.Reshape((train_out_shape[1],train_out_shape[2],1))(x)




# x = tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (5,5),strides = (30,10),  padding='same',activation='linear')(x)

output = x

model = tf.keras.Model(inputs=input, outputs=output) # Create model
keras.utils.plot_model(baseModel, show_shapes=True)
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001,epsilon = 0.0000001), # Compile
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=['mean_absolute_error','mean_squared_error'])
# %%
#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
checkpoint_path = 'training_checkpoints_{jn}_{num}/cp.ckpt'.format(jn='TLTEST', num = 1)
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
    patience=60, # Number of epochs with no improvement before stopping training
    verbose=1, # Records message when earlystopping
    mode='auto',
    baseline=None, 
    restore_best_weights=False # Do not restore best weights after early stopping, we do this manually to allow recording of the full training history
)


#%%
#####################################################################
# Model training
#####################################################################
# for testing
epochs = 25
# Fit model to Failure index
# model_history = model.fit(x = X_train, y = y_train,
#                                 epochs=epochs,
#                                 steps_per_epoch=steps_per_epoch,
#                                 validation_data=val_ds,
#                                 validation_steps = validation_steps,
#                                 callbacks=[early_stopping_monitor, cp_callback]
#                                 )

model_history = model.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_ds,
                                validation_steps = validation_steps,
                                callbacks=[early_stopping_monitor, cp_callback]
                                )


# %% Fine tune base model

# Unfreeze the base model
baseModel.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=['mean_absolute_error','mean_squared_error'])

# Train end-to-end. Be careful to stop before you overfit!
model_history = model.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_ds,
                                validation_steps = validation_steps,
                                callbacks=[early_stopping_monitor, cp_callback]
                                )

baseModel.trainable = False


# %% model evaluation on training data
model.evaluate(
    x=X_train,
    y=y_train,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=False,
)

# %%
model.get_metrics_result()

# %% compute losses manually
pred = model.predict(X_train)

RMSEManual = np.sqrt(np.mean(np.square(y_train-pred)))

model.loss(
    y_train, pred, sample_weight=None
)

# %%
singlePred = model.predict(np.expand_dims(X_train[0], axis=0))
#%%
lastL = model.get_layer(
    name=None, index=-4
)

# %%
#####################################################################
# Data export
#####################################################################

trainingHist = model_history.history # save training history
model.load_weights(checkpoint_path) # load best model weights

predCNN = model.predict(X_train) # Make prediction
predCNN_val = model.predict(X_val) # Prediction of only validation data
predCNNShape = predCNN.shape
predCNN_valShape = predCNN_val.shape

# RMSE
RMSE = tf.keras.metrics.RootMeanSquaredError()
RMSE.update_state(y_train,predCNN)
print('RMSE for training set before reverse scaling = ' + str(RMSE.result().numpy()))
if y_val is not None:
    RMSE_val = tf.keras.metrics.RootMeanSquaredError()
    RMSE_val.update_state(y_val,predCNN_val)
    print('RMSE for validation set before reverse scaling = ' + str(RMSE_val.result().numpy()))


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
model.save(modelOutPath)
# %%
