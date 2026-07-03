#####################################################################
# Description
#####################################################################
'''
This script is run on the computing cluster (HPC) and contains the
model definition and training processes. It must be pointed to a 
sweep definition csv wherein the model hyperparameters are tabulated

A 10-fold cross validation of the model in the first row of the sweep definition is done
The whole dataset is split into 10 subsets, and each fold uses one of them for validation


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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tbdcml_workflow.architectures as shared_architectures
import tbdcml_workflow.custom_models as shared_custom_models

from tbdcml_workflow import (
  build_model_from_params,
  drop_sample_ids,
  get_sample_ids,
  load_all_samples,
  make_ssim_metric,
  resolve_dataset_spec,
  resolve_loss,
  seed_everything,
)

#####################################################################
# Settings
#####################################################################

yNames = ['FI'] # Names of ground truth features in input csv
normalizerLength = 20 # Number of random samples used for computation of mean and variance used in data normalisation 
k = 10 # Number of folds in cross validation

# For reproducible results set a seed
seed = 0
seed_everything(seed)

#####################################################################
# Formatting and settings done automatically
#####################################################################
# Add arguments for parallel running and training of several different models
argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--parallel", help="Index for parallel running on HPC") # parameter to allow parallel running on the HPC
argParser.add_argument("-j", "--jobname", help="Job name") # Name of job passed when calling script
argParser.add_argument("--params-file", help=(
    "Path to a JSON hyperparameter dict to confirm via k-fold CV, e.g. "
    "best_params.json from optuna_local.py or a parameters_<jn>_t<trial>.json "
    "from BenchMarks_Optuna.py. If omitted, falls back to the old behaviour "
    "of reading row 1 of sweep_definition_<jobname>.csv."
))
args = argParser.parse_args()
sweepIdx = 1
kFold = int(args.parallel)

if args.params_file:
    # Optuna-driven workflow: confirm one already-chosen configuration.
    print(f"Loading hyperparameters from {args.params_file}")
    with open(args.params_file) as f:
        params = json.load(f)
else:
    # Legacy workflow: row 1 of the sweep definition CSV is always the
    # configuration being cross-validated (see module docstring).
    sweepPath = 'sweep_definition_{jn}.csv'.format(jn=args.jobname[:-2]) # Name of sweep definition file, one for all repetitions hence [:-2]
    print(os.getcwd())
    print(sweepPath)
    os.listdir(os.getcwd())

    sweep_params = pd.read_csv(sweepPath)
    sweep_params = sweep_params.set_index('Index')
    params = sweep_params.loc[sweepIdx]

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
samplepath = 'samples_{jn}_{num}.json'.format(jn=args.jobname, num = args.parallel)
samplepath = os.path.join('dataout',samplepath)

# Dataset selection
dataset_spec = resolve_dataset_spec(params['Dataset'], params.get('MC24_Features'))
trainDat_name = dataset_spec.train_dat_name
sampleShape = list(dataset_spec.sample_shape)
xNames = dataset_spec.x_names
samplesPerFile = dataset_spec.samples_per_file
winKernel = dataset_spec.win_kernel

# Various settings
trainDat_path = os.path.join('datain',trainDat_name)
numSamples = len(os.listdir(trainDat_path))*samplesPerFile # Number of data samples (i.e. TBDC specimens)
batchSize = params['batchSize'] # Batch size for training
valSize = math.floor(params['valSize']*numSamples) # Training and validation data split ratio
testSize = math.floor(params['testSize']*numSamples)
train_length = numSamples-valSize-testSize # Number of training samples 
epochs = params['Epochs'] # Max epochs for training
steps_per_epoch = train_length // batchSize
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate=params['initial_lr'],
  decay_steps=steps_per_epoch*epochs,
  decay_rate=params['lr_decay_rate'])

lossfunc = resolve_loss(params['loss'], variant='hpc')
SSIM_metric = make_ssim_metric(winKernel)

validation_steps = valSize // batchSize # Not used anymore, I want to run full validation set in one go
headers, samples = load_all_samples(
    trainDat_path,
    numSamples,
    xNames,
    yNames,
    tuple(sampleShape),
    samplesPerFile,
)


# Split into training and validation datasets using k-fold
valIdx = int((kFold-1)*(numSamples/k)) # The k fold variable is 1-indexed, valIdx marks the start of the validation set in this fold
valIndeces = [*range(valIdx,valIdx+int(numSamples/k),1)]

trainIndeces =[*range(0,valIdx,1),*range(valIdx+int(numSamples/k),numSamples,1)]

if valIdx>0:
  train1 = samples.take(valIdx) # First samples in training set
remaining = samples.skip(valIdx)
val_ds = remaining.take(valSize)
train2 = remaining.skip(valSize)

if valIdx>0:
  train_ds = train1.concatenate(train2)
else:
   train_ds = train2
if testSize > 0: # Does not work for cross validation
  test_ds = remaining.skip(valSize)

# Extract sample IDs before dropping them, for later comparison
train_sample_ids = get_sample_ids(train_ds)
val_sample_ids = get_sample_ids(val_ds)
test_sample_ids = get_sample_ids(test_ds) if testSize > 0 else []

train_ds = drop_sample_ids(train_ds)
val_ds = drop_sample_ids(val_ds)
if testSize > 0:
  test_ds = drop_sample_ids(test_ds)

# Take a copy of the datasets for RMSE evaluation at the end before repeat and shuffling is passed
train_ds_eval = train_ds.batch(batchSize).cache()
val_ds_eval = val_ds.batch(batchSize).cache()
if testSize > 0:
  test_ds_eval = test_ds.batch(batchSize).cache()

X_trainShape = (train_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(xNames))
X_valShape = (val_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(xNames))
if testSize > 0:
  X_testShape = (test_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(xNames))
y_trainShape = (train_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(yNames))
y_valShape = (val_ds.cardinality().numpy().item(),sampleShape[0],sampleShape[1],len(yNames))
if testSize > 0:
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
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Validation preprocessing
val_ds = val_ds.cache() # cache dataset for it to be used over iterations
val_ds = val_ds.batch(batchSize) # Batch
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

# Test preprocessing
if testSize > 0:
  test_ds = test_ds.cache() # cache dataset for it to be used over iterations
  # test_ds = test_ds.shuffle(buffer_size = len(test_ds)).batch(batchSize)
  test_ds = test_ds.batch(batchSize) # Batch
  test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared


#####################################################################
# CNN Model definition
#####################################################################

def get_padding_shape(height, width, multiple=32): # Can do up to 4 levels of downsampling with 64x32 images
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return ((pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2))

def TBDCNet_modelCNN(inputShape, outputShape, params):
  return shared_custom_models.build_tbdcnet_model_cnn(inputShape, outputShape, params, normalizer)

def TBDCNet_UNet(inputShape, outputShape, params):
  return shared_custom_models.build_tbdcnet_unet(inputShape, outputShape, params, normalizer, seed)

#####################################################################
# Training callbacks
#####################################################################

# Checkpoints to allow saving best model at various points
checkpoint_path = 'epoch-{epoch:04d}.weights.h5'
checkpoint_dir = 'training_checkpoints_{jn}_{num}'.format(jn=args.jobname, num = args.parallel)
cpLoadName = 'model.weights.h5' # The name of the checkpoint with the best weights at the end of training


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
#####################################################################
# Model instantiation
#####################################################################

tf.keras.backend.clear_session() # Clear the state and frees up memory

# CNN Model creation

CNNModel = build_model_from_params(
    model_map={
      "Xception": shared_architectures.Xception_Model,
      "MobileNetV2": shared_architectures.mobileNetV2_Model,
      "VGG16": shared_architectures.VGG16_Model,
      "ResNet50": shared_architectures.ResNet50_Model,
      "ResNet50V2": shared_architectures.ResNet50V2_Model,
      "InceptionV3": shared_architectures.InceptionV3_Model,
      "InceptionResNetV2": shared_architectures.InceptionResNetV2_Model,
      "DenseNet121": shared_architectures.DenseNet121_Model,
      "NASNetMobile": shared_architectures.NASNetMobile_Model,
      "EfficientNetV2S": shared_architectures.EfficientNetV2S_Model,
      "EfficientNetV2M": shared_architectures.EfficientNetV2M_Model,
      "EfficientNetV2L": shared_architectures.EfficientNetV2L_Model,
      "ConvNeXtTiny": shared_architectures.ConvNeXtTiny_Model,
      "ConvNeXtSmall": shared_architectures.ConvNeXtSmall_Model,
      "ConvNeXtLarge": shared_architectures.ConvNeXtLarge_Model,
      "UNet": lambda inputShape, outputShape, params: shared_custom_models.build_tbdcnet_unet(inputShape, outputShape, params, normalizer, seed),
      "default": lambda inputShape, outputShape, params: shared_custom_models.build_tbdcnet_model_cnn(inputShape, outputShape, params, normalizer),
      "dense": lambda inputShape, outputShape, params: shared_custom_models.build_tbdcnet_model_cnn(inputShape, outputShape, params, normalizer),
      "applyDecoder": shared_architectures.applyDecoder,
    },
  params=params,
  input_shape=X_trainShape[1:],
  output_shape=y_trainShape[1:],
  normalizer=normalizer,
  lr_schedule=lr_schedule,
  lossfunc=lossfunc,
  ssim_metric=SSIM_metric,
)
CNNModel.summary()

modelCNNname = 'CNNModel1'

#####################################################################
# Model training
#####################################################################

# Fit model to Failure index
time_callback_ins = timecallback()
modelCNN_history = CNNModel.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=int(steps_per_epoch),
                                validation_data=val_ds,
                                callbacks=[early_stopping_monitor, cp_callback, cp_delete_callback(checkpoint_dir, cpLoadName), time_callback_ins]
                                )

# NOT USING validation_steps = validation_steps, as this is not needed for the validation dataset

# Get the recorded epoch times after training is complete
epoch_times = {'trainTime':time_callback_ins.get_epoch_times().tolist()}


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
    return_dict=True,
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
    return_dict=True,
)
val_results = pd.DataFrame.from_dict(val_results, orient='index',
                       columns=['val']).T

if testSize > 0:
  test_results = CNNModel.evaluate(
      x=test_ds_eval,
      y=None,
      batch_size=None,
      verbose='auto',
      sample_weight=None,
      steps=None,
      callbacks=None,
      return_dict=True,
  )

  test_results = pd.DataFrame.from_dict(test_results, orient='index',
                        columns=['test']).T
  
  results = pd.concat([train_results, val_results, test_results])
else:
  results = pd.concat([train_results, val_results])
results['RMSE'] = np.sqrt(results['mean_squared_error'])
print(results)

sample_ids = {
  "train": train_sample_ids,
  "val": val_sample_ids,
  "test": test_sample_ids
}
with open(samplepath, 'w') as f:
  json.dump(sample_ids, f, indent=2)

with open(histOutPath, 'w') as f: # Dump data to json file at specified path
    json.dump(trainingHist, f, indent=2)

with open(paramOutPath, 'w') as f: # Dump data to json file at specified path
    # `params` is a pandas Series when read from the legacy sweep CSV, or a
    # plain dict when loaded via --params-file (Optuna workflow).
    json.dump(params.to_dict() if hasattr(params, "to_dict") else params, f, indent=2)

with open(resultpath, 'w') as f: # Dump data to json file at specified path
    json.dump(results.to_json(), f, indent=2)

CNNModel.save(modelOutPath)
