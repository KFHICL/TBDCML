# %%
# update
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
import tensorflow_probability as tfp

from tensorflow.python.client import device_lib
from scipy.stats import ks_2samp
print(device_lib.list_local_devices())
TESTING = 0

import tbdcml_workflow.architectures as shared_architectures
import tbdcml_workflow.custom_models as shared_custom_models
# os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
# import tf_keras as keras # Legacy keras version which is equal to the one on the HPC


#%% Model loading for transfer learning
# tf.keras.backend.clear_session() # Clear the state and frees up memory
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250814_MC24_1000_constVf_Opti\model_20250814_MC24_1000_constVf_Opti_1.keras"
# # import tf_keras as keras # Legacy keras version which is equal to the one on the HPC
# def load_model(model_path):
#     # os.environ["TF_USE_LEGACY_KERAS"] = "1"


#     # Define SSIM_metric and custom_loss for loading
#     def SSIM_metric(y_true, y_pred):
#         y_pred = tf.cast(y_pred, tf.float32) # y_pred is in a different type, recast
            
#         return tf.reduce_mean(tf.image.ssim(
#         img1 = y_true,
#         img2 = y_pred,
#         max_val = 1,
#         filter_size=winKernel,
#         filter_sigma=1.5,
#         k1=0.01,
#         k2=0.03,
#         return_index_map=False
#         )   )

#     def custom_loss(y_true, y_pred):
#         SE_base = tf.math.square(tf.math.subtract(y_true, y_pred))
#         loss = tf.math.multiply(SE_base, (tf.math.add(tf.constant(1, dtype=tf.float32), tf.nn.relu(y_true))))
#         loss = tf.reduce_mean(loss)
#         return loss

#     # model = keras.models.load_model(
#     #     model_path,
#     #     compile=True,
#     #     custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss}
#     # )
#     model = tf.keras.models.load_model(
#     model_path,
#     compile=True,
#     custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss}
#     )
#     print("Model Summary:")
#     model.summary()
#     print("\nModel Layers:")
#     for i, layer in enumerate(model.layers):
#         print(f"{i}: {layer.name} ({layer.__class__.__name__}) - {layer.output_shape}")
#     if hasattr(model, 'optimizer') and model.optimizer is not None:
#         print("\nOptimizer:", type(model.optimizer).__name__)
#         print("Optimizer config:", model.optimizer.get_config())
#     if hasattr(model, 'loss') and model.loss is not None:
#         print("\nLoss function:", model.loss)
#     if hasattr(model, 'metrics') and model.metrics:
#         print("\nMetrics:", [m.name if hasattr(m, 'name') else m for m in model.metrics])
  
#     return model

# # Example usage:
# loadedModel = load_model(modelPath)

# %%

#####################################################################
# Settings
#####################################################################

from tbdcml_workflow import (
  build_model_from_params,
  check_overlapping_samples,
  drop_sample_ids,
  get_sample_ids,
  load_all_samples,
  make_ssim_metric,
  resolve_dataset_spec,
  resolve_loss,
  seed_everything,
)
sweepDefFolder = os.path.join(os.getcwd(), "sweep_definitions")
# sweep_params = pd.read_csv(os.path.join(os.getcwd(),'sweep_definition_literatureModels.csv'))
sweep_params = pd.read_csv(os.path.join(sweepDefFolder,'sweep_definition_test.csv'))
sweepIdx = 1
yNames = ['FI'] # Names of ground truth features in input csv
normalizerLength = 40 # Number of random samples used for computation of mean and variance used in data normalisation 

# For reproducible results set a seed
seed = 0
seed_everything(seed)

sweep_params = sweep_params.set_index('Index')
params = sweep_params.loc[sweepIdx]

# Build a local output folder and file paths for this run, following the
# same {jobname}_{num} naming convention used by the HPC scripts.
name = params['type'] # Job name
jobname = f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

folderName = f"{jobname}_{sweepIdx}"
result_folder = os.path.join(os.getcwd(), "localResults", folderName)
os.makedirs(result_folder, exist_ok=True)

parallel = 1
histOutPath = os.path.join(result_folder, f'trainHist_{jobname}_{parallel}.json')
paramOutPath = os.path.join(result_folder, f'parameters_{jobname}_{parallel}.json')
resultpath = os.path.join(result_folder, f'results_{jobname}_{parallel}.json')
samplepath = os.path.join(result_folder, f'samples_{jobname}_{parallel}.json')
modelOutPath = os.path.join(result_folder, f'model_{jobname}_{parallel}.keras')

# Dataset selection (same lookup used by BenchMarks.py / CrossValidation.py,
# just pointed at the local copy of the data instead of the HPC "datain" folder)
localDatain_root = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'
dataset_spec = resolve_dataset_spec(params['Dataset'], params.get('MC24_Features'))
trainDat_name = dataset_spec.train_dat_name
sampleShape = list(dataset_spec.sample_shape)
xNames = dataset_spec.x_names
samplesPerFile = dataset_spec.samples_per_file
winKernel = dataset_spec.win_kernel
trainDat_path = os.path.join(localDatain_root, trainDat_name)

# Various settings
numSamples = len(os.listdir(trainDat_path))*samplesPerFile # Number of data samples (i.e. TBDC specimens)
if TESTING:
  numSamples = 400
batchSize = params['batchSize'] # Batch size for training
valSize = math.floor(params['valSize']*numSamples) # Training and validation data split ratio
testSize = math.floor(params['testSize']*numSamples)
train_length = numSamples-valSize-testSize # Number of training samples
epochs = params['Epochs'] # Max epochs for training
steps_per_epoch = train_length // batchSize
validation_steps = valSize // batchSize # Not used anymore, I want to run full validation set in one go

headers, samples = load_all_samples(
  trainDat_path,
  numSamples,
  xNames,
  yNames,
  tuple(sampleShape),
  samplesPerFile,
  testing=bool(TESTING),
)

# Split into training, validation, and test sets (same method as BenchMarks.py)
samples = samples.shuffle(buffer_size=len(samples), seed=seed, reshuffle_each_iteration=False)
train_ds = samples.take(train_length)
remaining = samples.skip(train_length)
val_ds = remaining.take(valSize)
if testSize > 0:
  test_ds = remaining.skip(valSize)

# %%
# Shared loss helpers
SSIM_metric = make_ssim_metric(winKernel)

# Default initial learning rate is 0.001. If the the decay rate is 1 this will be held constant.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate=params['initial_lr'],
  decay_steps=steps_per_epoch*epochs,
  decay_rate=params['lr_decay_rate'])

lossfunc = resolve_loss(params['loss'], variant='local')

# Extract sample IDs before dropping them, for later comparison
train_sample_ids = get_sample_ids(train_ds)
val_sample_ids = get_sample_ids(val_ds)
test_sample_ids = get_sample_ids(test_ds) if testSize > 0 else []

train_ds = drop_sample_ids(train_ds)
val_ds = drop_sample_ids(val_ds)
if testSize > 0:
  test_ds = drop_sample_ids(test_ds)
######### Cross-validation method 
# Split into training and validation datasets using k-fold
# kFold = 1
# k = 10
# valIdx = int((kFold-1)*(numSamples/k)) # The k fold variable is 1-indexed, valIdx marks the start of the validation set in this fold
# valIndeces = [*range(valIdx,valIdx+int(numSamples/k),1)]

# trainIndeces =[*range(0,valIdx,1),*range(valIdx+int(numSamples/k),numSamples,1)]

# samples = samples.shuffle(buffer_size=numSamples, seed=seed, reshuffle_each_iteration=False)

# if valIdx>0:
#   train1 = samples.take(valIdx) # First samples in training set
# remaining = samples.skip(valIdx)
# val_ds = remaining.take(valSize)
# train2 = remaining.skip(valSize)

# if valIdx>0:
#   train_ds = train1.concatenate(train2)
# else:
#    train_ds = train2
# if testSize > 0: # Does not work for cross validation
#   test_ds = remaining.skip(valSize)

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

check_overlapping_samples(train_sample_ids, val_sample_ids)

# %%
def plot_feature_distribution(ds, feature_idx=0, title=""):
  # Collect a sample of features
  feature_samples = []
  label_samples = []
  for x, y in ds.take(10):  # take 1 batch
    x_norm = normalizer(x)  # Apply normalizer
    feature_samples.append(x_norm[:, :, :, feature_idx].numpy().ravel())
    label_samples.append(y[:, :, :, 0].numpy().ravel())  # Assuming single label channel

  feature_samples = np.concatenate(feature_samples)
  label_samples = np.concatenate(label_samples)

  plt.hist(feature_samples, bins=50, alpha=0.5, label=f"{title} Feature")
  # plt.hist(label_samples, bins=50, alpha=0.5, label=f"{title} Label")

plot_feature_distribution(train_ds, feature_idx=0, title="Train normalized")
plot_feature_distribution(val_ds, feature_idx=0, title="Val normalized")
plt.legend()
plt.show()
# %%
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

#%% Checkpoints to allow saving best model at various points
checkpoint_path = 'epoch-{epoch:04d}.weights.h5'
checkpoint_dir = os.path.join('trainingCheckpoints','training_checkpoints_{jn}_{num}'.format(jn=jobname, num = 1))
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
    # patience=1000, # Number of epochs with no improvement before stopping training
    patience=60, # Number of epochs with no improvement before stopping training
    verbose=1, # Records message when earlystopping
    mode='auto',
    baseline=None, 
    restore_best_weights=False # Do not restore best weights after early stopping, we do this manually to allow recording of the full training history
)

# %%
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
# Plot the model architecture
tf.keras.utils.plot_model(
  CNNModel,
  to_file=os.path.join(folderName, f"{modelCNNname}_architecture.png"),
  show_shapes=True,
  show_layer_names=True,
  expand_nested=True,
  dpi=100
)

# %% Toubleshooting

class GradientAndLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, train_data, log_every=1):
        super().__init__()
        self.train_data = iter(train_data)  # make it iterable
        self.log_every = log_every
        self.batch_losses = []
        self.grad_norms = []

    def on_train_batch_end(self, batch, logs=None):
        # Store loss
        loss = logs.get('loss')
        self.batch_losses.append(loss)

        if batch % self.log_every == 0:
            # Get the same batch again (next in iterator)
            try:
                inputs, targets = next(self.train_data)
            except StopIteration:
                self.train_data = iter(self.model.train_data)  # reset if needed
                inputs, targets = next(self.train_data)

            with tf.GradientTape() as tape:
                preds = self.model(inputs, training=True)
                loss_value = self.model.compiled_loss(targets, preds)

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            grad_norm = np.sqrt(sum([tf.reduce_sum(tf.square(g)).numpy()
                                     for g in grads if g is not None]))
            self.grad_norms.append(grad_norm)

    def on_epoch_end(self, epoch, logs=None):
        if self.grad_norms:
            print(f"\nEpoch {epoch+1} — Avg grad norm: {np.mean(self.grad_norms):.4f}, "
                  f"Max grad norm: {np.max(self.grad_norms):.4f}")

# %% Inspection of datasets:
# Find the index of the string in train_sample_ids that comes first alphabetically
first_alphabetical_index = train_sample_ids.index(min(train_sample_ids))
print("Index of the first alphabetically sorted string:", first_alphabetical_index)

def get_first_batch(dataset, n=1):
    for batch in dataset.take(n):
        return batch

# Fetch batches
raw_x, raw_y = get_first_batch(train_ds_eval)
aug_x, aug_y = get_first_batch(train_ds)

# raw_x, raw_y = get_first_batch(val_ds_eval)
# aug_x, aug_y = get_first_batch(val_ds)

print("Raw batch shape:", raw_x.shape)
print("Aug batch shape:", aug_x.shape)
    
import matplotlib.pyplot as plt

i = first_alphabetical_index//batchSize  # index in batch
plt.subplot(1,2,1)
plt.title("Raw")
plt.imshow(raw_x[i,...,0])

plt.subplot(1,2,2)
plt.title("After Training Pipeline")
plt.imshow(aug_x[i,...,0])
plt.show()
# %%
#####################################################################
# Model training
#####################################################################
logger = GradientAndLossLogger(train_ds, log_every=1)
# Fit model to Failure index
# epochs = 10
time_callback_ins = timecallback()
# Clear existing files in the checkpoint directory
for file in os.listdir(checkpoint_dir):
  file_path = os.path.join(checkpoint_dir, file)
  if os.path.isfile(file_path):
    os.remove(file_path)

modelCNN_history = CNNModel.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=int(steps_per_epoch),
                                validation_data=val_ds,
                                callbacks=[early_stopping_monitor, cp_callback, cp_delete_callback(checkpoint_dir, cpLoadName), time_callback_ins,
                                          #  logger,
                                           ]
                                )

# Re-adapt model to the new dataset (where Vf isn't constant)

# const_feature_idx = 3  # your formerly constant feature index
# initializer = tf.keras.initializers.GlorotUniform()

# for layer in loadedModel.layers:
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         weights, biases = layer.get_weights()
#         # Only for first conv layer where in_channels = original feature count
#         if weights.shape[2] == loadedModel.input_shape[-1]:  
#             # Reset only that feature's kernel slice
#             new_slice = initializer(shape=(weights.shape[0], weights.shape[1], 1, weights.shape[3]))
#             weights[:, :, const_feature_idx:const_feature_idx+1, :] = new_slice
#             layer.set_weights([weights, biases])
#         break  # stop after first Conv2D

# normalizer = loadedModel.get_layer(name="normalization")
# normalizer.adapt(feature_ds)

# Assign a new schedule
# new_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-4,
#     decay_steps=steps_per_epoch*epochs,
#     decay_rate=1
# )

# new_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate = 1e-4,
#     decay_steps = steps_per_epoch*epochs,
#     alpha=0.0,
#     name='CosineDecay',
#     warmup_target=1e-2,
#     warmup_steps=steps_per_epoch*100
# )
# loadedModel.optimizer.learning_rate = new_schedule



# modelCNN_history = loadedModel.fit(train_ds,
#                                 epochs=epochs,
#                                 steps_per_epoch=int(steps_per_epoch),
#                                 validation_data=val_ds,
#                                 callbacks=[early_stopping_monitor, cp_callback, cp_delete_callback(checkpoint_dir, cpLoadName), time_callback_ins,
#                                           #  logger,
#                                            ]
#                                 )

# NOT USING validation_steps = validation_steps, as this is not needed for the validation dataset

# Get the recorded epoch times after training is complete
epoch_times = {'trainTime':time_callback_ins.get_epoch_times().tolist()}

# %%
# After model is built (and compiled), inspect model.losses
print("Number of regularization losses:", len(CNNModel.losses))
print("Sum of regularization losses:", float(tf.math.add_n(CNNModel.losses)) if CNNModel.losses else 0.0)

mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

def dataset_loss(model, dataset):
    losses = []
    for x_batch, y_batch in dataset:
        # get predictions in inference mode
        preds = model(x_batch, training=False)
        losses.append(mse(y_batch, preds).numpy())
    return np.mean(losses)

train_loss_no_reg = dataset_loss(CNNModel, train_ds_eval)   # train_dataset batched, deterministic order
val_loss_no_reg   = dataset_loss(CNNModel, val_ds)

print("Train (no reg, inference mode):", train_loss_no_reg)
print("Val   (no reg, inference mode):", val_loss_no_reg)


def per_sample_losses(model, dataset):
    losses = []
    for x,y in dataset:
        preds = model(x, training=False)
        # flatten per sample
        batch_losses = tf.reduce_mean(tf.square(y - preds), axis=[1,2,3]).numpy()
        losses.extend(batch_losses)
    return np.array(losses)

train_sample_losses = per_sample_losses(CNNModel, train_ds_eval)  # ensure deterministic order
val_sample_losses   = per_sample_losses(CNNModel, val_ds)

import matplotlib.pyplot as plt
plt.hist(train_sample_losses, bins=100, alpha=0.6, label='train')
plt.hist(val_sample_losses, bins=100, alpha=0.6, label='val')
plt.legend(); plt.show()

print("train median, 95%ile:", np.median(train_sample_losses), np.percentile(train_sample_losses,95))
print("val   median, 95%ile:", np.median(val_sample_losses),   np.percentile(val_sample_losses,95))
# %%

# LOADED MODEL
######################################################################################
# trainingHist = modelCNN_history.history # save training history
# trainingHist.update(epoch_times) # Add training time to history

# loadedModel.load_weights(os.path.join(checkpoint_dir,cpLoadName)) # load best model weights


# train_results = loadedModel.evaluate(
#     x=train_ds_eval,
#     y=None,
#     batch_size=None,
#     verbose='auto',
#     sample_weight=None,
#     steps=None,
#     callbacks=None,
#     return_dict=True,
# )

# train_results = pd.DataFrame.from_dict(train_results, orient='index',
#                        columns=['train']).T

# val_results = loadedModel.evaluate(
#     x=val_ds_eval,
#     y=None,
#     batch_size=None,
#     verbose='auto',
#     sample_weight=None,
#     steps=None,
#     callbacks=None,
#     return_dict=True,
# )
# val_results = pd.DataFrame.from_dict(val_results, orient='index',
#                        columns=['val']).T

# if testSize > 0:
#   test_results = loadedModel.evaluate(
#       x=test_ds_eval,
#       y=None,
#       batch_size=None,
#       verbose='auto',
#       sample_weight=None,
#       steps=None,
#       callbacks=None,
#       return_dict=True,
#   )

#   test_results = pd.DataFrame.from_dict(test_results, orient='index',
#                         columns=['test']).T
  
#   results = pd.concat([train_results, val_results, test_results])
# else:
#   results = pd.concat([train_results, val_results])
# results['RMSE'] = np.sqrt(results['mean_squared_error'])
# print(results)

# %%




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

#%% Save outputs
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
    json.dump(params.to_json(), f, indent=2)

with open(resultpath, 'w') as f: # Dump data to json file at specified path
    json.dump(results.to_json(), f, indent=2)

CNNModel.save(modelOutPath)

# %%
# For every sample, find predicted FI at all locations where true FI == 1 (train and val separately)

preds_train = CNNModel.predict(train_ds_eval)
preds_val = CNNModel.predict(val_ds_eval)
# For every sample in train and val, find locations where true FI == 1 and compare to predicted values

def extract_pred_at_label_one(preds, labels):
  """
  For each sample, find all locations where label == 1 and return the predicted values at those locations.
  Returns a list of arrays (one per sample), each containing the predicted values at label==1 locations.
  """
  results = []
  for i in range(labels.shape[0]):
    label_sample = labels[i, ...]
    pred_sample = preds[i, ...]
    # Remove last dim if singleton
    if label_sample.ndim == 3 and label_sample.shape[-1] == 1:
      label_sample = label_sample[..., 0]
    if pred_sample.ndim == 3 and pred_sample.shape[-1] == 1:
      pred_sample = pred_sample[..., 0]
    mask = (label_sample == 1)
    pred_at_ones = pred_sample[mask]
    results.append(pred_at_ones)
  return results

# Get true labels for train and val sets
y_train = []
for _, y in train_ds_eval:
  y_train.append(y.numpy())
y_train = np.concatenate(y_train, axis=0)

y_val = []
for _, y in val_ds_eval:
  y_val.append(y.numpy())
y_val = np.concatenate(y_val, axis=0)

# Find predicted values at label==1 locations
preds_at_ones_train = extract_pred_at_label_one(preds_train, y_train)
preds_at_ones_val = extract_pred_at_label_one(preds_val, y_val)

# Optionally, flatten all for histogram or error analysis
all_preds_at_ones_train = np.concatenate(preds_at_ones_train)
all_preds_at_ones_val = np.concatenate(preds_at_ones_val)

# Example: plot histogram of predicted values at label==1 locations
plt.figure(figsize=(10,5))
plt.hist(all_preds_at_ones_train, bins=50, alpha=0.6, label='Train')
plt.hist(all_preds_at_ones_val, bins=50, alpha=0.6, label='Val')
plt.xlabel('Predicted Value at FI==1 locations')
plt.ylabel('Count')
plt.legend()
plt.title('Predicted values at locations where label == 1')
plt.show()

# %%

# Flatten datasets for KS test (inputs only)

# def dataset_to_numpy_X(dataset):
#     """Convert tf.data.Dataset -> numpy array for X only."""
#     all_X = []
#     for X_batch, _ in dataset:
#         all_X.append(X_batch.numpy())  # shape: (batch, 60, 20, 5)
#     return np.concatenate(all_X, axis=0)  # shape: (n_samples, 60, 20, 5)

# def ks_test_featurewise(train_ds, val_ds):
#     # Convert datasets to numpy arrays
#     X_train = dataset_to_numpy_X(train_ds)
#     X_val = dataset_to_numpy_X(val_ds)

#     # Results storage
#     results = []

#     # Loop over features
#     for f in range(X_train.shape[-1]):  # last dim = feature
#         # Flatten over all time & spatial dims
#         train_feature = X_train[..., f].ravel()
#         val_feature = X_val[..., f].ravel()

#         # KS test
#         ks_stat, p_value = ks_2samp(train_feature, val_feature)
#         results.append((f, ks_stat, p_value))

#     # Print nicely
#     print("Feature | KS statistic | p-value")
#     print("-" * 35)
#     for f, ks_stat, p_value in results:
#         print(f"{f:7d} | {ks_stat:12.4f} | {p_value:.4e}")

#     return results

# # Example usage:
# results = ks_test_featurewise(train_ds_eval, val_ds_eval)
# %%
# Plot the distribution of each feature in training vs validation datasets
# X_train = dataset_to_numpy_X(train_ds_eval)
# X_val = dataset_to_numpy_X(val_ds_eval)

# num_features = X_train.shape[-1]
# fig, axes = plt.subplots(num_features, 1, figsize=(10, 3 * num_features), sharex=False)

# for f in range(num_features):
#   ax = axes[f] if num_features > 1 else axes
#   train_feature = X_train[..., f].ravel()
#   val_feature = X_val[..., f].ravel()
#   ax.hist(train_feature, bins=50, alpha=0.6, label='Train', color='tab:blue', density=True)
#   ax.hist(val_feature, bins=50, alpha=0.6, label='Val', color='tab:orange', density=True)
#   ax.set_title(f'Feature {f} Distribution')
#   ax.set_ylabel('Density')
#   ax.legend()
# plt.xlabel('Feature Value')
# plt.tight_layout()
# plt.show()
# %%
# Summarise metrics and training curves in a single plot

fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot training and validation loss
ax1.plot(modelCNN_history.history['loss'], label='Training Loss', color='tab:blue')
ax1.plot(modelCNN_history.history['val_loss'], label='Validation Loss', color='tab:orange')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.set_title('Training & Validation Loss with Metrics Summary')
ax1.grid(True, which='both', axis='y')
ax1.legend(loc='upper right')

# Prepare metrics summary text
summary_lines = []
for metric in results.columns:
  summary_lines.append(f"{metric}:")
  for split in results.index:
    summary_lines.append(f"  {split}: {results.loc[split, metric]:.5f}")
summary_text = "\n".join(summary_lines)

# Place metrics summary as annotation box outside the plot
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.gcf().text(1.02, 0.98, summary_text, fontsize=11,
         verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout(rect=[0, 0, 1, 1])  # Make space on the right for the box
plt.show()

# %%
# Heatmap: prediction vs true value for training and validation sets

def get_preds_and_true(model, dataset):
    preds = []
    trues = []
    for x_batch, y_batch in dataset:
        pred_batch = model(x_batch, training=False)
        preds.append(pred_batch.numpy().ravel())
        trues.append(y_batch.numpy().ravel())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues

def plot_heatmap(true, pred, ax, title, bins=100, cmap='viridis'):
  h = ax.hist2d(true, pred, bins=bins, cmap=cmap, norm=plt.matplotlib.colors.LogNorm())
  ax.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', lw=2, label='Perfect Prediction')
  ax.set_xlabel('True Value')
  ax.set_ylabel('Prediction')
  ax.set_title(title)
  ax.grid(True)
  plt.colorbar(h[3], ax=ax, label='Count')

train_preds, train_true = get_preds_and_true(CNNModel, train_ds_eval)
val_preds, val_true = get_preds_and_true(CNNModel, val_ds_eval)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_heatmap(train_true, train_preds, axes[0], 'Train: Prediction vs True')
plot_heatmap(val_true, val_preds, axes[1], 'Validation: Prediction vs True')

axes[0].legend()
axes[1].legend()
axes[0].set_xlim([0, 1])
axes[1].set_xlim([0, 1])
axes[0].set_aspect('equal', adjustable='box')
axes[1].set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
# %%
# Compute sqrt(variance(y_train)) for the training set
y_train_values = []
for _, y_batch in train_ds_eval:
  y_train_values.append(y_batch.numpy())
y_train_array = np.concatenate(y_train_values, axis=0)
sqrt_var_y_train = np.sqrt(np.var(y_train_array))
print("sqrt(variance(y_train)):", sqrt_var_y_train)
# %%

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(logger.batch_losses)
plt.yscale('log')
plt.title("Batch Loss (log scale)")

plt.subplot(1,2,2)
plt.plot(logger.grad_norms)
plt.yscale('log')
plt.title("Gradient Norm (log scale)")

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
  # print(prediction.shape)
  prediction = tf.squeeze(prediction)  # Remove batch dimension
  # print(ground_truth.shape)
  # print(ground_truth.shape)
  ground_truth = tf.squeeze(ground_truth[0])  # Remove batch dimension
  print(ground_truth.shape)
  print(prediction.shape)
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

# %%

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
headers, eval_samples = load_all_samples(trainDat_path, numSamples)

eval_samples_sample_ids = get_sample_ids(eval_samples)

# Remove sample IDs from datasets (keep only X, Y)
def drop_sample_ids(dataset):
  return dataset.map(lambda x, y, _: (x, y))

eval_samples = drop_sample_ids(eval_samples)
# %%
# Load the model
tf.keras.backend.clear_session() # Clear the state and frees up memory
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

class LayerScale(tf.keras.layers.Layer):
  """Layer scale module.

  References:

  - https://arxiv.org/abs/2103.17239

  Args:
      init_values (float): Initial value for layer scale. Should be within
          [0, 1].
      projection_dim (int): Projection dimensionality.

  Returns:
      Tensor multiplied to the scale.
  """

  def __init__(self, init_values, projection_dim, **kwargs):
      super().__init__(**kwargs)
      self.init_values = init_values
      self.projection_dim = projection_dim

  def build(self, _):
      self.gamma = self.add_weight(
          shape=(self.projection_dim,),
          initializer=initializers.Constant(self.init_values),
          trainable=True,
      )

  def call(self, x):
      return x * self.gamma

  def get_config(self):
      config = super().get_config()
      config.update(
          {
              "init_values": self.init_values,
              "projection_dim": self.projection_dim,
          }
      )
      return config


def load_model(model_path):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"


    # Define SSIM_metric and custom_loss for loading
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

    def custom_loss(y_true, y_pred):
        SE_base = tf.math.square(tf.math.subtract(y_true, y_pred))
        loss = tf.math.multiply(SE_base, (tf.math.add(tf.constant(1, dtype=tf.float32), tf.nn.relu(y_true))))
        loss = tf.reduce_mean(loss)
        return loss

    model = keras.models.load_model(
        model_path,
        compile=True,
        custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss, 'LayerScale': LayerScale}
    )
    # model = tf.keras.models.load_model(
    # model_path,
    # compile=True,
    # custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss}
    # )
    print("Model Summary:")
    model.summary()
    print("\nModel Layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} ({layer.__class__.__name__}) - {layer.output_shape}")
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        print("\nOptimizer:", type(model.optimizer).__name__)
        print("Optimizer config:", model.optimizer.get_config())
    if hasattr(model, 'loss') and model.loss is not None:
        print("\nLoss function:", model.loss)
    if hasattr(model, 'metrics') and model.metrics:
        print("\nMetrics:", [m.name if hasattr(m, 'name') else m for m in model.metrics])
  
    return model

# Example usage:

model_path = r"C:\Users\kfh23\Desktop\model_20250909_MC24x_run16_2_1.keras"
loaded_model = load_model(model_path)

# Load sample IDs
sample_ids_path = r"C:\Users\kfh23\Desktop\samples_20250909_MC24x_run16_2_1.json"
with open(sample_ids_path, 'r') as f:
  sample_ids = json.load(f)

# Split eval_samples into train, val, test datasets based on sample IDs
train_ids = set(sample_ids["train"])
val_ids = set(sample_ids["val"])
test_ids = set(sample_ids["test"])

def filter_dataset_by_ids(dataset, dataset_ids, target_ids):
  target_ids_tensor = tf.constant(list(target_ids), dtype=tf.string)
  dataset_ids_tensor = tf.constant(dataset_ids, dtype=tf.string)

  filtered_dataset = dataset.enumerate().filter(
    lambda idx, _: tf.reduce_any(tf.equal(dataset_ids_tensor[idx], target_ids_tensor))
  ).map(lambda _, data: data)
  return filtered_dataset

eval_samples_ids = np.array(eval_samples_sample_ids)  # Convert to numpy array for indexing
train_ds_eval = filter_dataset_by_ids(eval_samples, eval_samples_ids, train_ids)
val_ds_eval = filter_dataset_by_ids(eval_samples, eval_samples_ids, val_ids)
test_ds_eval = filter_dataset_by_ids(eval_samples, eval_samples_ids, test_ids)

# Test the dataset pipeline without shuffling
# Take a copy of the datasets for RMSE evaluation at the end before repeat and shuffling is passed 
train_ds_eval_cache = train_ds_eval.batch(batchSize).cache() 
val_ds_eval_cache = val_ds_eval.batch(batchSize).cache() 
test_ds_eval_cache = test_ds_eval.batch(batchSize).cache()

# feature_ds = train_ds.take(normalizerLength).map(lambda x, y: x) 
# normalizer = tf.keras.layers.Normalization() 
# normalizer.adapt(feature_ds)

train_ds_eval = train_ds_eval.cache() # cache dataset for it to be used over iterations. Any operation before this will not be reapplied each iteration 
# train_ds = train_ds.shuffle(buffer_size = len(train_ds)) # Shuffle for random order

if params['dsAugmentation'] == 1: 
  train_ds_eval = train_ds_eval.map(Augment())

train_ds_eval = train_ds_eval.batch(batchSize) # Batch
train_ds_eval = train_ds_eval.repeat() # Repeats dataset indefinitely to avoid errors
train_ds_eval = train_ds_eval.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared 

val_ds_eval = val_ds_eval.cache() # cache dataset for it to be used over iterations 
val_ds_eval = val_ds_eval.batch(batchSize) # Batch
val_ds_eval = val_ds_eval.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared

test_ds_eval = test_ds_eval.cache() # cache dataset for it to be used over iterations 
test_ds_eval = test_ds_eval.batch(batchSize) # Batch
test_ds_eval = test_ds_eval.prefetch(buffer_size=tf.data.AUTOTUNE) # Allows prefetching of elements while later elements are prepared


# %%
# Define the specimen index to plot
def get_first_batch(dataset, n=1):
    for batch in dataset.take(n):
        return batch

# Fetch batches
# raw_x, raw_y = get_first_batch(train_ds_eval)
# aug_x, aug_y = get_first_batch(train_ds_eval_cache)
raw_x, raw_y = get_first_batch(val_ds_eval)
aug_x, aug_y = get_first_batch(val_ds_eval_cache)

print("Raw batch shape:", raw_x.shape)
print("Aug batch shape:", aug_x.shape)


i = 2  # index in batch
plt.subplot(1,2,1)
plt.title("Raw")
plt.imshow(raw_x[i,...,0])

plt.subplot(1,2,2)
plt.title("After Training Pipeline")
plt.imshow(aug_x[i,...,0])
plt.show()

# Print mean and std values of raw_x and aug_x specimens
print("Raw Specimen Mean:", np.mean(raw_x[i, ..., 0]))
print("Raw Specimen Std:", np.std(raw_x[i, ..., 0]))
print("Augmented Specimen Mean:", np.mean(aug_x[i, ..., 0]))
print("Augmented Specimen Std:", np.std(aug_x[i, ..., 0]))

  # %%
# Evaluate the model on the train, val, and test datasets
train_results = loaded_model.evaluate(train_ds_eval, steps = len(train_ids)/batchSize, return_dict=True)
train_results_cached = loaded_model.evaluate(train_ds_eval_cache, return_dict=True)
val_results = loaded_model.evaluate(val_ds_eval, steps = len(val_ids)/batchSize, return_dict=True)
val_results_cached = loaded_model.evaluate(val_ds_eval_cache, return_dict=True)
test_results = loaded_model.evaluate(test_ds_eval, steps = len(test_ids)/batchSize, return_dict=True)
test_results_cached = loaded_model.evaluate(test_ds_eval_cache, return_dict=True) 

# Print the results
print("Train Results:", train_results)
print("Train Results (cached):", train_results_cached)
print("Validation Results:", val_results)
print("Validation Results (cached):", val_results_cached)
print("Test Results:", test_results)
print("Test Results (cached):", test_results_cached)

# %%
# Load the training history
train_hist_path = r"C:\Users\kfh23\Desktop\trainHist_20250909_MC24x_run16_2_1.json"
with open(train_hist_path, 'r') as f:
  train_history = json.load(f)

# Plot the training and validation loss curves
plt.figure(figsize=(14, 6))

# Subplot 1: Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(train_history['loss'], label='Training Loss', color='tab:blue')
plt.plot(train_history['val_loss'], label='Validation Loss', color='tab:orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.ylim(top=5e-3)
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Subplot 2: SSIM Metric
plt.subplot(1, 2, 2)
plt.plot(train_history['SSIM_metric'], label='Training SSIM', color='tab:green')
plt.plot(train_history['val_SSIM_metric'], label='Validation SSIM', color='tab:red')
plt.xlabel('Epochs')
plt.ylabel('SSIM Metric')
plt.title('Training and Validation SSIM Metric')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# %% Make predictions plot


headers, eval_samples = load_all_samples(trainDat_path, numSamples)

eval_samples_sample_ids = get_sample_ids(eval_samples)

# Remove sample IDs from datasets (keep only X, Y)
def drop_sample_ids(dataset):
  return dataset.map(lambda x, y, _: (x, y))

eval_samples = drop_sample_ids(eval_samples)

# %%
import glob
# folderPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout"
folderPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout"
# folderPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout"

# Find all files in folderPath that start with "model"
model_files = glob.glob(os.path.join(folderPath, "model*.keras"))
tf.keras.backend.clear_session() # Clear the state and frees up memory
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import random
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.layers.layer import Layer
from keras.src.models import Functional
from keras.src.models import Sequential
from keras.src.ops import operation_utils
from keras.src.utils import file_utils



def load_model(model_path):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"


    # Define SSIM_metric and custom_loss for loading
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

    def custom_loss(y_true, y_pred):
        SE_base = tf.math.square(tf.math.subtract(y_true, y_pred))
        loss = tf.math.multiply(SE_base, (tf.math.add(tf.constant(1, dtype=tf.float32), tf.nn.relu(y_true))))
        loss = tf.reduce_mean(loss)
        return loss

    # model = keras.models.load_model(
    #     model_path,
    #     compile=True,
    #     custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss, 'LayerScale': LayerScale}
    # )
    model = keras.models.load_model(
        model_path,
        compile=True,
        custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss}
    )
    # model = tf.keras.models.load_model(
    # model_path,
    # compile=True,
    # custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss}
    # )
    print("Model Summary:")
    model.summary()
    print("\nModel Layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} ({layer.__class__.__name__}) - {layer.output_shape}")
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        print("\nOptimizer:", type(model.optimizer).__name__)
        print("Optimizer config:", model.optimizer.get_config())
    if hasattr(model, 'loss') and model.loss is not None:
        print("\nLoss function:", model.loss)
    if hasattr(model, 'metrics') and model.metrics:
        print("\nMetrics:", [m.name if hasattr(m, 'name') else m for m in model.metrics])
  
    return model



# model_files = [model_files[3]]

# Loop over the model files and load the corresponding sample_ids
for modelIdx, model_path in enumerate(model_files):
  sample_ids = {}
  print(f"Loading model: {model_path}")
  try:
    loaded_model = load_model(model_path)
  except Exception as e:
    print(f"Error loading model {model_path}: {e}")
    continue

  # Construct the corresponding sample_ids path
  sample_ids_path = model_path.replace(".keras", ".json")
  sample_ids_path = sample_ids_path.replace("model", "samples")
  if os.path.exists(sample_ids_path):
    print(f"Loading sample IDs: {sample_ids_path}")
    with open(sample_ids_path, 'r') as f:
      sample_ids = json.load(f)
  else:
    print(f"Sample IDs file not found for model: {model_path}")
    print(f"Train IDs are first 90%, val IDs are last 10% of total samples.")
    sample_ids["train"] = eval_samples_sample_ids[:int(0.9*len(eval_samples_sample_ids))]
    sample_ids["val"] = eval_samples_sample_ids[int(0.9*len(eval_samples_sample_ids)):]
    sample_ids["test"] = []




  # Split eval_samples into train, val, test datasets based on sample IDs
  train_ids = set(sample_ids["train"])
  val_ids = set(sample_ids["val"])
  test_ids = set(sample_ids["test"])

  def filter_dataset_by_ids(dataset, dataset_ids, target_ids):
    target_ids_tensor = tf.constant(list(target_ids), dtype=tf.string)
    dataset_ids_tensor = tf.constant(dataset_ids, dtype=tf.string)

    filtered_dataset = dataset.enumerate().filter(
      lambda idx, _: tf.reduce_any(tf.equal(dataset_ids_tensor[idx], target_ids_tensor))
    ).map(lambda _, data: data)
    return filtered_dataset

  eval_samples_ids = np.array(eval_samples_sample_ids)  # Convert to numpy array for indexing
  train_ds_eval = filter_dataset_by_ids(eval_samples, eval_samples_ids, train_ids)
  val_ds_eval = filter_dataset_by_ids(eval_samples, eval_samples_ids, val_ids)
  test_ds_eval = filter_dataset_by_ids(eval_samples, eval_samples_ids, test_ids)

  # Test the dataset pipeline without shuffling
  # Take a copy of the datasets for RMSE evaluation at the end before repeat and shuffling is passed 
  train_ds_eval_cache = train_ds_eval.batch(batchSize).cache() 
  val_ds_eval_cache = val_ds_eval.batch(batchSize).cache() 
  test_ds_eval_cache = test_ds_eval.batch(batchSize).cache()

  # Predict and visualize the first sample in train, val, and test datasets
  def predict_and_visualize_combined(train_ds, val_ds, test_ds, model, output_path):
    datasets = [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(8, 3 * len(datasets)))

    for i, (dataset_name, dataset) in enumerate(datasets):
      for sample, ground_truth in dataset.take(1):
        prediction = model.predict(tf.expand_dims(sample[0], axis=0))  # Predict on the first sample in the batch
        prediction = tf.squeeze(prediction)  # Remove batch dimension
        ground_truth = tf.squeeze(ground_truth[0])  # Remove batch dimension
        squared_error = tf.square(ground_truth - prediction)  # Compute squared error

        # Plot the prediction, ground truth, and squared error
        vmin_value = np.min([np.min(ground_truth), np.min(prediction)])
        im = axes[i, 0].imshow(ground_truth, cmap='viridis', vmin=vmin_value)
        axes[i, 0].set_title(f"{dataset_name} - Ground Truth")
        axes[i, 0].axis('off')
        axes[i, 0].colorbar = plt.colorbar(axes[i, 0].imshow(ground_truth, cmap='viridis', vmin=vmin_value), ax=axes[i, 0])

        axes[i, 1].imshow(prediction, cmap='viridis', vmin=im.get_clim()[0], vmax=im.get_clim()[1])
        axes[i, 1].set_title(f"{dataset_name} - Prediction")
        axes[i, 1].axis('off')
        axes[i, 1].colorbar = plt.colorbar(axes[i, 1].imshow(prediction, cmap='viridis', vmin=im.get_clim()[0], vmax=im.get_clim()[1]), ax=axes[i, 1])

        axes[i, 2].imshow(squared_error, cmap='viridis')
        axes[i, 2].set_title(f"{dataset_name} - Squared Error")
        axes[i, 2].axis('off')
        axes[i, 2].colorbar = plt.colorbar(axes[i, 2].imshow(squared_error, cmap='viridis'), ax=axes[i, 2])

        break

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.show()

  # Define the output path for the high-resolution PDF
  fileName = os.path.basename(model_path).replace(".keras", "")
  # output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\MC24x_run16_predictions\{fileName}_dataset_predictions.pdf"
  output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\LFC18_postOp_predictions\{fileName}_dataset_predictions.pdf"

  # Visualize and save the combined figure
  predict_and_visualize_combined(train_ds_eval_cache, val_ds_eval_cache, test_ds_eval_cache, loaded_model, output_pdf_path)

# %%

input, output = ConvNeXtTiny_Model(inputShape = X_trainShape[1:])#
convNextModel = applyDecoder(input, output, outputShape = y_trainShape[1:], params = params)


convNextModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule,epsilon = params['epsilon']), # Compile
        loss=lossfunc, 
        metrics=['mean_absolute_error','mean_squared_error', SSIM_metric])

convNextModel.summary()

kerasFilePath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\model_20250909_MC24x_run16_1_14.h5"

convNextModel.load_weights(kerasFilePath, skip_mismatch=True)


import zipfile

with zipfile.ZipFile(kerasFilePath) as z:
    z.extract("variables/variables.data-00000-of-00001", path="./recovered/")
    z.extract("variables/variables.index", path="./recovered/")

import h5py
f = h5py.File(kerasFilePath, 'r')
print(list(f.keys()))

for layer in convNextModel.layers:
    if "convnext_stage_0_block_0" in layer.name:
        print(layer.name, layer.weights)



# %% Load n models, n training curves, and n samples. Plot the ground truth, prediction, squared error and training curves for each model
import matplotlib
# modelPaths = [
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\model_20250808_LFC18_crossVal_Opti_1_1.keras",
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\model_20250814_MC24_1000_crossVal_Opti_1_1.keras",
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\model_20250909_MC24x_run16_1_1.keras",
# ]

# historyPaths = [
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\trainHist_20250808_LFC18_crossVal_Opti_1_1.json",
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\trainHist_20250814_MC24_1000_crossVal_Opti_1_1.json",
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_1.json",
# ]

# samplePaths = [
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain\Unnotched_TBDC_2022_3.csv",
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1233_1kSamples\sample_648.csv", # sample_4.csv
#   #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_961to1000.parquet", #"samples_961to1000_parquet_5"
#   #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_1161to1200.parquet", #    Very good prediction
#    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_441to480.parquet", #    "samples_161to200_parquet_1",
# ]

# modelPaths = [
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_3\dataout\model_20250808_LFC18_crossVal_Opti_3_9.keras",
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\model_20250814_MC24_1000_crossVal_Opti_1_1.keras",
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\model_20250909_MC24x_run16_1_1.keras",
# ]

# # historyPaths = [
# #    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\trainHist_20250808_LFC18_crossVal_Opti_1_1.json",
# #    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\trainHist_20250814_MC24_1000_crossVal_Opti_1_1.json",
# #    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_1.json",
# # ]

# historyPaths = [
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250806_LFC18_crossVal_Baseline_1\dataout\trainHist_20250806_LFC18_crossVal_Baseline_1_1.json",
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250806_crossVal\20250806_MC24_crossVal_Baseline_1\dataout\trainHist_20250806_MC24_crossVal_Baseline_1_1.json",
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Baseline_1\dataout\trainHist_20250814_MC24_1000_crossVal_Baseline_1_1.json",
#   #  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_1.json",
# ]

# samplePaths = [
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain\Unnotched_TBDC_2022_3.csv",
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1233_1kSamples\sample_648.csv", # sample_4.csv
#   #  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_961to1000.parquet", #"samples_961to1000_parquet_5"
#   #  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_1161to1200.parquet", #    Very good prediction
#    r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_441to480.parquet", #    "samples_161to200_parquet_1",
# ]

modelPaths = [
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_3\dataout\model_20250808_LFC18_crossVal_Opti_3_9.keras",
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\model_20250814_MC24_1000_crossVal_Opti_1_1.keras",
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\model_20250909_MC24x_run16_1_1.keras",
]

historyPaths = [
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\trainHist_20250808_LFC18_crossVal_Opti_1_1.json",
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\trainHist_20250814_MC24_1000_crossVal_Opti_1_1.json",
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_1.json",
]

# historyPaths = [
  #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250806_LFC18_crossVal_Baseline_1\dataout\trainHist_20250806_LFC18_crossVal_Baseline_1_1.json",
  #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250806_crossVal\20250806_MC24_crossVal_Baseline_1\dataout\trainHist_20250806_MC24_crossVal_Baseline_1_1.json",
  #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Baseline_1\dataout\trainHist_20250814_MC24_1000_crossVal_Baseline_1_1.json",
  #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_1.json",
# ]

samplePaths = [
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain\Unnotched_TBDC_2022_3.csv",
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240725_1233_1kSamples\sample_648.csv", # sample_4.csv
  #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_961to1000.parquet", #"samples_961to1000_parquet_5"
  #  r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_1161to1200.parquet", #    Very good prediction
   r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples\samples_441to480.parquet", #    "samples_161to200_parquet_1",
]
sampleTypes = [
  'MeC-Macro',
  'MeC-Meso-M',
  'MeC-Meso-L',
]
# sampleTypes = [
#   'MeC-Macro',
#   'MeC-Meso-S',
#   'MeC-Meso-M',
# ]


# Load samples for prediction
samples = []

for i, samplePath in enumerate(samplePaths):
  sampleType = sampleTypes[i]
  if sampleType == 'MeC-Macro':
     sampleShape = [55,20]
     samplesPerFile = 1
     xNames = ['E11','E22','E12']
  elif sampleType == 'MeC-Meso-M':
     sampleShape = [60,20]
     samplesPerFile = 1
     xNames = ['Ex','Ey','Gxy','Vf','c2']
  elif sampleType == 'MeC-Meso-L':
     sampleShape = [224,224]
     samplesPerFile = 40
     xNames = ['Ex','Ey','Gxy','Vf','c2']
  headers, values = loadSampleNew(samplePath)
  print(f"Samples ID: {get_sample_ids(values)}")
  values = drop_sample_ids(values)
  values = values.batch(1)
  samples.append(values)

# Load models and make predictions
models = []
ground_truths = []
predictions = []
squared_errors = []

for i, modelPath in enumerate(modelPaths):
  loaded_model = load_model(modelPath)

  models.append(loaded_model)

  prediction = []
  ground_truth = []
  for x, y in samples[i].take(1):  # Extract ground truth from the first sample in the dataset
      ground_truth.append(y.numpy())
      prediction.append(loaded_model.predict(x))
  prediction = tf.squeeze(prediction)  # Remove batch dimension
  ground_truth = tf.squeeze(ground_truth)  # Remove batch dimension
  predictions.append(prediction.numpy())
  ground_truths.append(ground_truth.numpy())

  squared_error = (ground_truth - prediction)**2  # Compute squared error
  squared_errors.append(squared_error.numpy())

# Load training curves
training_histories = []
for historyPath in historyPaths:
  with open(historyPath, 'r') as f:
    train_history = json.load(f)
  training_histories.append(train_history)


#%% Plotting training curves


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figscale = 1
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio


 # Share y-axis within each row
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight*10)
# fig.set_figwidth(figWidth*10)

RMSE_palette = {
'train': '#66D3B3',  # lighter green
'val': '#029E73',  # base green
# 'test': '#01523D'  # darker green
}

NRMSE_palette = {
'train': '#ffc505',  # lighter orange
'val': '#d55e00',  # base orange
# 'test': '#01523D'  # darker green
}

SSIM_palette = {
'train': '#66ADD6',  # lighter blue
'val': '#0173B2',  # base blue
# 'test': '#01436A'  # darker blue
}
# fig, axes = plt.subplots(5, 3, figsize=(12, 16))
fig, axes = plt.subplots(1, 3, figsize=(6, 2),sharey=False)
# Ensure y-tick labels are shown on all subplots
for ax in axes.flat:
  ax.tick_params(labelleft=True)

for i in range(len(training_histories)):
  # Calculate row and column for 5x3 grid

  
  # Plot training and validation loss
  axes[i].plot(training_histories[i]['loss'], linestyle='-', color=SSIM_palette['val'], linewidth=1, alpha=1)
  axes[i].plot(training_histories[i]['val_loss'], linestyle='--', color=NRMSE_palette['val'], linewidth=0.75, alpha=0.75)

  axes[i].set_ylabel('MSE')
  axes[i].set_xlabel('Epoch')
  axes[i].set_yscale('log')
  
  # Add legend and grid
  axes[i].plot([], [], linestyle='-', color=SSIM_palette['val'], label='Train', linewidth=1, alpha=1)
  axes[i].plot([], [], linestyle='--', color=NRMSE_palette['val'], label='Val', linewidth=0.75, alpha=0.75)
  axes[i].legend(loc='upper right')
  axes[i].grid(True, axis='y', linestyle='--', alpha=0.75, which='major')
  axes[i].minorticks_on()
  axes[i].grid(True, axis='y', linestyle=':', alpha=0.5, which='minor')
axes[0].set_ylabel('$\mathcal{L}_\mathrm{custom}$', fontsize=8)
axes[1].set_ylim(bottom = 1e-3, top=0.03)
axes[2].set_ylim(bottom = 1e-3, top=0.03)
  # Add model type as title
  # axes[row, col].set_title(type_order[i])
# axes[0,0].set_ylim(top=0.1)
# Hide empty subplots (positions 13 and 14 in 5x3 grid)
# for i in range(len(training_histories), 15):
#   row = i // 3
#   col = i % 3
#   axes[row, col].axis('off')

plt.tight_layout()

# output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\TBDCNet_traincurves.pdf"
# output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\TBDCNet_traincurves_v2.pdf"
output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\TBDCNet_traincurves_v2.pdf"
plt.savefig(output_pdf_path, format='pdf', dpi=300)
plt.show()

# %%
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figscale = 1
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio


fig, axes = plt.subplots(len(modelPaths), 4)  # Share y-axis within each row
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight*10)
# fig.set_figwidth(figWidth*10)

RMSE_palette = {
'train': '#66D3B3',  # lighter green
'val': '#029E73',  # base green
# 'test': '#01523D'  # darker green
}

NRMSE_palette = {
'train': '#ffc505',  # lighter orange
'val': '#d55e00',  # base orange
# 'test': '#01523D'  # darker green
}

SSIM_palette = {
'train': '#66ADD6',  # lighter blue
'val': '#0173B2',  # base blue
# 'test': '#01436A'  # darker blue
}
for i in range(len(modelPaths)):
  # Plot training and validation loss
  l_train, = axes[i, 0].plot(training_histories[i]['loss'], linestyle='-', color=SSIM_palette['val'], linewidth=1, alpha=1)
  axes[i, 0].plot(training_histories[i]['val_loss'], linestyle='--', color=NRMSE_palette['val'], linewidth=0.75, alpha=0.75)

  # axes[i, 0].set_xlabel('Epochs')
  axes[i, 0].set_ylabel('MSE')
  axes[i, 0].set_xlabel('Epoch')
  axes[i, 0].set_yscale('log')
  # axes[i, 0].set_title(f'Model {i+1}: Training and Validation Loss')
  # if i == 0:  # Add legend only for the first row
  axes[i, 0].plot([], [], linestyle='-', color=SSIM_palette['val'], label='Train', linewidth=1, alpha=1)
  axes[i, 0].plot([], [], linestyle='--', color=NRMSE_palette['val'], label='Val', linewidth=0.75, alpha=0.75)
  axes[i, 0].legend(loc='upper right')
  axes[i, 0].grid(True, axis='y', linestyle='--', alpha=0.75, which='major')
  axes[i, 0].minorticks_on()
  axes[i, 0].grid(True, axis='y', linestyle=':', alpha=0.5, which='minor')
  # Add figure denoter in the top left corner of each axes[i, 0]
  axes[i, 0].text(
      -0.6, 1.15, f"({chr(97 + i)}): {sampleTypes[i]}",  # 'a', 'b', 'c', etc.
      transform=axes[i, 0].transAxes,
      fontsize=8,
      fontweight='bold',
      va='top',
      ha='left'
  )


  # Plot ground truth
  im1 = axes[i, 1].imshow(ground_truths[i], cmap='viridis')
  cbar = plt.colorbar(im1, ax=axes[i, 1])
  cbar.set_label('Ground Truth FI')
  axes[i, 1].axis('off')


  # Plot prediction
  im2 = axes[i, 2].imshow(predictions[i], cmap='viridis', vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])

  axes[i, 2].axis('off')
  cbar = plt.colorbar(im2, ax=axes[i, 2])
  cbar.set_label('Prediction FI')

  # Plot squared error
  im3 = axes[i, 3].imshow(squared_errors[i], cmap='viridis')
  cbar = plt.colorbar(im3, ax=axes[i, 3])
  cbar.set_label('Squared Error FI')
  axes[i, 3].axis('off')

# axes[-1, 0].set_xlabel('Epoch')
axes[0, 0].set_ylim()
axes[1, 0].set_ylim()
axes[2, 0].set_ylim(top=0.01)
# ax0.set_title('Training and Validation MSE Curves')
# Add custom legend for models (outside, right)


# Add a legend entry for line styles (train/val), also outside
# style_lines = [
#     plt.Line2D([0], [0], color='black', linestyle='-', label='Train', linewidth=0.7, alpha=1),
#     plt.Line2D([0], [0], color='black', linestyle='--', label='Val', linewidth=0.7, alpha=0.7)
# ]
# # legend2 = plt.legend(handles=style_lines, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.12), title="Data")
# legend2 = fig.legend(handles=style_lines, title="Data", loc="lower center",ncol=2, bbox_to_anchor=(0.29, -0.1))
# # fig.legend(handles=monochrome_legend, loc="lower center", ncol=2, bbox_to_anchor=(0.8, -0.1), title="Data")
# legend1 = ax0.legend(lines, model_labels[:len(lines)], title="MeC-Meso \nspecimens", loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
# ax0.add_artist(legend2)  # Add the model legend back
# ax0.set_title('')


plt.tight_layout(w_pad=1, h_pad=0.5)


output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\PostOp_dataset_predictions.pdf"
plt.savefig(output_pdf_path, format='pdf', dpi=300)
plt.show()
# %% Different try where we make 2 separate plots for training curves and predictions
# Predictions
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "8"
latexWidth = 315
figscale = 1
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio


fig, axes = plt.subplots(len(modelPaths), 3)  
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight*10)
# fig.set_figwidth(figWidth*10)

RMSE_palette = {
'train': '#66D3B3',  # lighter green
'val': '#029E73',  # base green
# 'test': '#01523D'  # darker green
}

NRMSE_palette = {
'train': '#ffc505',  # lighter orange
'val': '#d55e00',  # base orange
# 'test': '#01523D'  # darker green
}

SSIM_palette = {
'train': '#66ADD6',  # lighter blue
'val': '#0173B2',  # base blue
# 'test': '#01436A'  # darker blue
}
sampleWidths= [50, 50, 224]
for i in range(len(modelPaths)):
  



  # Plot ground truth
  im1 = axes[i, 0].imshow(ground_truths[i], cmap='viridis')
  divider = make_axes_locatable(axes[i, 0])
  cax = divider.append_axes("right", size=0.07, pad=0.1)
  cbar = plt.colorbar(im1, cax=cax)
  # cbar.set_label('Ground Truth FI')
  axes[i, 0].axis('off')
  axes[i, 0].set_anchor('C')  # Center the image in the axes
  axes[i, 0].set_title('Ground Truth')
  dx = sampleWidths[i]/ground_truths[i].shape[1]  # Assuming 50 units correspond to the full width of the image

  scalebar = ScaleBar(dx, "mm", fixed_units="mm", fixed_value = 25,width_fraction=0.015*dx, box_alpha = 0,rotation ="vertical-only",bbox_to_anchor=(0, 1),bbox_transform=axes[i, 0].transAxes)
  axes[i, 0].add_artist(scalebar)

  # Plot prediction
  im2 = axes[i, 1].imshow(predictions[i], cmap='viridis', vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])

  divider = make_axes_locatable(axes[i, 1])
  cax = divider.append_axes("right", size=0.07, pad=0.1)
  cbar = plt.colorbar(im2, cax=cax)
  # cbar.set_label('Prediction FI')
  axes[i, 1].axis('off')
  axes[i, 1].set_anchor('C')  # Center the image in the axes
  axes[i, 1].set_title('Prediction')

  # Plot squared error
  im3 = axes[i, 2].imshow(squared_errors[i], cmap='viridis')
  divider = make_axes_locatable(axes[i, 2])
  cax = divider.append_axes("right", size=0.07, pad=0.1)
  cbar = plt.colorbar(im3, cax=cax)
  cbar.locator = plt.matplotlib.ticker.MaxNLocator(nbins=3)
  cbar.update_ticks()
  # cbar.set_label('Squared Error FI')
  axes[i, 2].axis('off')
  axes[i, 2].set_anchor('C')  # Center the image in the axes
  axes[i, 2].set_title('Squared Error')

  # axes[i, 0].text(
  #     -0.6, 0.5, f"({chr(97 + i)}): {sampleTypes[i]}",  # 'a', 'b', 'c', etc.
  #     transform=axes[i, 0].transAxes,
  #     fontsize=10,
  #     fontweight='bold',
  #     va='center',
  #     ha='center',
  #     rotation='vertical'
  # )
  # axes[i, 1].text(
  #     0.5, -0.1, f"({chr(97 + i)}): {sampleTypes[i]}",  # 'a', 'b', 'c', etc.
  #     transform=axes[i, 1].transAxes,
  #     fontsize=10,
  #     fontweight='bold',
  #     va='center',
  #     ha='center',
  # )
  fig.text(
    0, 1-0.33333/2 - i * (1 / len(modelPaths)),  # Adjust vertical position based on the row index
    " ",  # 'a', 'b', 'c', etc.
    fontsize=10,
    fontweight='bold',
    va='center',
    ha='left',
  )

# axes[-1, 0].set_xlabel('Epoch')
# ax0.set_title('Training and Validation MSE Curves')
# Add custom legend for models (outside, right)


# Add a legend entry for line styles (train/val), also outside
# style_lines = [
#     plt.Line2D([0], [0], color='black', linestyle='-', label='Train', linewidth=0.7, alpha=1),
#     plt.Line2D([0], [0], color='black', linestyle='--', label='Val', linewidth=0.7, alpha=0.7)
# ]
# # legend2 = plt.legend(handles=style_lines, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.12), title="Data")
# legend2 = fig.legend(handles=style_lines, title="Data", loc="lower center",ncol=2, bbox_to_anchor=(0.29, -0.1))
# # fig.legend(handles=monochrome_legend, loc="lower center", ncol=2, bbox_to_anchor=(0.8, -0.1), title="Data")
# legend1 = ax0.legend(lines, model_labels[:len(lines)], title="MeC-Meso \nspecimens", loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
# ax0.add_artist(legend2)  # Add the model legend back
# ax0.set_title('')




plt.tight_layout(w_pad = -3)


# output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVEAR 4\Individual Project\Paper\Figures\PostOp_predictions.pdf"
output_pdf_path = rf"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\PostOp_prediction_v2.pdf"
plt.savefig(output_pdf_path, format='pdf', dpi=300)
plt.show()


# %% Plot training curves for Imagenet models
historyPaths = [
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_1.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_2.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_3.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_4.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_5.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_6.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_7.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_8.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_9.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_10.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_11.json",
  # r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_12.json",
  # r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_13.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_14.json",
  # r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_15.json",
  # r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_16.json",
  r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_1\dataout\trainHist_20250909_MC24x_run16_1_17.json",
]

type_order = [
    "VGG16",
    "UNet",
    "ResNet50",
    "InceptionV3",
    "ResNet50V2",
    "InceptionResNetV2",
    "Xception",
    "MobileNetV2",
    "DenseNet121",
    "NASNetMobile",
    "EfficientNetV2S",
    "ConvNeXtTiny",
    "TBDCNet-Meso-L"
]

original_order = [
  "TBDCNet-Meso-L",
  "Xception",
  "MobileNetV2",
  "VGG16",
  "ResNet50",
  "ResNet50V2",
  "InceptionV3",
  "InceptionResNetV2",
  "DenseNet121",
  "NASNetMobile",
  "EfficientNetV2S",
  "ConvNeXtTiny",
  "UNet"
]
# Create a mapping from original_order to historyPaths
order_mapping = dict(zip(original_order, historyPaths))

# Reorder historyPaths according to type_order
historyPaths = [order_mapping[model_type] for model_type in type_order if model_type in order_mapping]

training_histories = []
for historyPath in historyPaths:
  print('loading', historyPath)
  with open(historyPath, 'r') as f:
    train_history = json.load(f)
  training_histories.append(train_history)

# Plotting
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figscale = 1
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio


 # Share y-axis within each row
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight*10)
# fig.set_figwidth(figWidth*10)

RMSE_palette = {
'train': '#66D3B3',  # lighter green
'val': '#029E73',  # base green
# 'test': '#01523D'  # darker green
}

NRMSE_palette = {
'train': '#ffc505',  # lighter orange
'val': '#d55e00',  # base orange
# 'test': '#01523D'  # darker green
}

SSIM_palette = {
'train': '#66ADD6',  # lighter blue
'val': '#0173B2',  # base blue
# 'test': '#01436A'  # darker blue
}
# fig, axes = plt.subplots(5, 3, figsize=(12, 16))
fig, axes = plt.subplots(5, 3, figsize=(6, 8), sharey=True)
# Ensure y-tick labels are shown on all subplots
for ax in axes.flat:
  ax.tick_params(labelleft=True)

for i in range(len(training_histories)):
  # Calculate row and column for 5x3 grid
  row = i // 3
  col = i % 3
  
  # Plot training and validation loss
  axes[row, col].plot(training_histories[i]['loss'], linestyle='-', color=SSIM_palette['val'], linewidth=1, alpha=1)
  axes[row, col].plot(training_histories[i]['val_loss'], linestyle='--', color=NRMSE_palette['val'], linewidth=0.75, alpha=0.75)

  axes[row, col].set_ylabel('MSE')
  axes[row, col].set_xlabel('Epoch')
  axes[row, col].set_yscale('log')
  
  # Add legend and grid
  axes[row, col].plot([], [], linestyle='-', color=SSIM_palette['val'], label='Train', linewidth=1, alpha=1)
  axes[row, col].plot([], [], linestyle='--', color=NRMSE_palette['val'], label='Val', linewidth=0.75, alpha=0.75)
  axes[row, col].legend(loc='upper right')
  axes[row, col].grid(True, axis='y', linestyle='--', alpha=0.75, which='major')
  axes[row, col].minorticks_on()
  axes[row, col].grid(True, axis='y', linestyle=':', alpha=0.5, which='minor')
  
  # Add model type as title
  axes[row, col].set_title(type_order[i])
axes[0,0].set_ylim(top=0.1)
# Hide empty subplots (positions 13 and 14 in 5x3 grid)
for i in range(len(training_histories), 15):
  row = i // 3
  col = i % 3
  axes[row, col].axis('off')

plt.tight_layout()

output_pdf_path = rf"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\ImageNet_traincurves.pdf"
plt.savefig(output_pdf_path, format='pdf', dpi=300)
plt.show()
