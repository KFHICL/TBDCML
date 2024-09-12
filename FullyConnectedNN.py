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

import seaborn as sns
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC


#%% Test model loading 
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\Dropout1806_1\dataout\model_Dropout1806_1_1.keras" # Baseline model
# # modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\dataoutTESTJOB\model_TESTJOB_1.keras"
# loaded_model = keras.models.load_model(modelPath)
# loaded_model.summary()

#%% Settings for test script
sweep_params = pd.read_csv(r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\sweep_definition_FFNN.csv')
sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[1]
jobname = 'FFNN'
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

#%% Import data
trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain' # Path for training data samples
# trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240703_1417' # Path for training data samples
numSamples = len(os.listdir(trainDat_path)) # number of samples is number of files in datain
sampleShape = [55,20]
# sampleShape = [60,20]
xNames = ['E11','E22','E12'] # Names of input features in input csv
# xNames = ['Ex','Ey','Gxy'] # Names of input features in input csv
yNames = ['FI'] # Names of ground truth features in input csv
batchSize = params['batchSize'] # Batch size for training
trainValRatio = params['trainValRatio'] # Training and validation data split ratio
train_length = round(numSamples * trainValRatio) # Number of training samples 
epochs = 500 # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = math.ceil((100-train_length) / batchSize)

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
    

# Format is csv files with columns 
for i,file in enumerate(os.listdir(trainDat_path)):
    filepath = os.path.join(trainDat_path,file)
    if i==0:
        headers, samples = loadSample(filepath)
        samples = samples.reshape(1, np.shape(samples)[0],np.shape(samples)[1])
    else:
        addSamp = loadSample(filepath)[1]
        samples = np.concatenate((samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
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


# Standardisation/normalisation
if params['standardisation'] == 'MinMax':
    Xscaler = preprocessing.MinMaxScaler() # Do a scaler for the in and outputs separately (to be able to inversely standardise predictions)
    Yscaler = preprocessing.MinMaxScaler() # Note: these shoud be fit to the training data and applied without fitting to the validation data to avoid data leakage
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








def FCNNModel(inputShape, outputShape, params):
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

  # Define the model architecture. Each convolutional layer has settings related to the kernel size, 
  # activation function, and regularizer. After each convolutional layer there may be a batch-
  # normalization layer, a max pooling layer, and a dropout layer. The number of convolutional layers
  # is given in the sweep definition


  input = tf.keras.layers.Input(shape=inputShape) # Shape (Long, short, inputs)
  x = input
  # Flatten to connect all pixels in an image to the hidden layers
  x = keras.layers.Flatten()(x)

  # First hidden layer with variable number of nodes and activation function
  x = tf.keras.layers.Dense(units=40,activation='tanh')(x)
  x = tf.keras.layers.Dropout(rate = params['dropout'])(x)
  x = tf.keras.layers.Dense(units=40,activation='tanh')(x)
  x = tf.keras.layers.Dropout(rate = 0.1)(x)
  x = tf.keras.layers.Dense(units=40,activation='tanh')(x)
  x = tf.keras.layers.Dropout(rate = 0.1)(x)
  
  # Make last layer a linear connection with 1100 units and reshape output
  x = tf.keras.layers.Dense(units = inputShape[0]*inputShape[1], activation="linear", name="Linear_output")(x)
  x = keras.layers.Reshape((inputShape[0],inputShape[1],-1))(x)
  output = x

  model = tf.keras.Model(inputs=input, outputs=output) # Create model

  # Default initial learning rate is 0.001. If the the decay rate is 1 this will be held constant.
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['initial_lr'],
    decay_steps=steps_per_epoch*epochs,
    decay_rate=params['lr_decay_rate'])
  

#   def custom_loss(y_true,y_pred):
#     SE_base = tf.math.square(y_true-y_pred)
#     loss = SE_base*(1+tf.nn.relu(y_true))
#     loss = tf.reduce_mean(loss)
#     return loss
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
    



  # Compile model with the optimizer in the sweep definition
  if params['optimizer'] == 'Adadelta':
     model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate = lr_schedule,epsilon = 0.001), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])
  elif params['optimizer'] == 'Nadam':
     model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = lr_schedule,epsilon = 0.001), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])
  else:
     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule,epsilon = 0.001), # Compile
              loss=lossfunc, 
              metrics=['mean_absolute_error','mean_squared_error'])

  return model


#%%
#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
checkpoint_path = 'training_checkpoints_{jn}_{num}/cp.ckpt'.format(jn='TESTJOB', num = 1)
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
# Model instantiation
#####################################################################

tf.keras.backend.clear_session() # Clear the state and frees up memory

# CNN Model creation
modelFCNN = FCNNModel(inputShape = train_in_shape[1:], outputShape = train_out_shape[1:], params = params)
modelFCNNname = 'FCNNModel1'
modelFCNN.summary()

#%%
#####################################################################
# Model training
#####################################################################

# Fit model to Failure index
modelFCNN_history = modelFCNN.fit(train_ds,
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

trainingHist = modelFCNN_history.history # save training history
modelFCNN.load_weights(checkpoint_path) # load best model weights

predCNN = modelFCNN.predict(X_train) # Make prediction
predCNN_val = modelFCNN.predict(X_val) # Prediction of only validation data
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
modelFCNN.save(modelOutPath)
# %%
