# %%
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

os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

# %% Load input data 
trainDat_path = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'

# MC24 standard
# trainDat_name = 'MatLabModel2024' 
# sampleShape = [60,20]
# trainDat_path = os.path.join(trainDat_path,'MatLabModel2024')
# xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
# samplesPerFile = 1
# winKernel = 7

# LFC18 standard
trainDat_name = 'Gaudron2018' 
sampleShape = [55,20]
xNames = ['E11','E22','E12'] # Names of input features in input csv
trainDat_path = os.path.join(trainDat_path,'Gaudron2018') # Path for training data samples
samplesPerFile = 1
winKernel = 5

yNames = ['FI'] # Names of ground truth features in input csv
numSamples = len(os.listdir(trainDat_path))*samplesPerFile # number of samples is number of files in datain
# For reproducible results set a seed
seed = 0
tf.random.set_seed(seed)

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
    files = files[1:2]

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

# %% Load the model to visualise
# Pass the custom objects dictionary to a custom object scope and place
# the `keras.models.load_model()` call within the scope.
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

custom_objects = {"SSIM_metric": SSIM_metric}

# modelPath = r"C:\Users\kfh23\Desktop\tempFiles\model_20250401_MC24_CrossValidation_PostOpti_1_1.keras"
# modelPath = r"C:\Users\kfh23\Desktop\tempFiles\model_20250311_LFC18_CrossValidation_PreOpti_1_1.keras"
modelPath = r"C:\Users\kfh23\Desktop\tempFiles\model_20250401_LFC18_CrossValidation_PostOpti_1_1.keras"
with keras.saving.custom_object_scope(custom_objects):
    model = keras.models.load_model(modelPath)


# %%  Calculate model outputs
inputSpecimen = samples.batch(1) # Only 1 specimen

# intermediate representations for all layers except the first layer.
layer_outputs = [layer.output for layer in model.layers]
visual_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)

# flipModel = keras.models.Model(inputs = model.input, outputs = layer_outputs[2])

# run your image through the network; make a prediction
feature_maps = visual_model.predict(inputSpecimen)

# %% Visualise the model outputs

#%% Visualise the input specimen
input_sample = next(iter(inputSpecimen))[0].numpy()  # Extract the first input specimen as a NumPy array

# Assuming the input has multiple channels, visualize each channel separately
num_channels = input_sample.shape[-1]
fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))

for i in range(num_channels):
    ax = axes[i] if num_channels > 1 else axes
    ax.imshow(input_sample[0, :, :, i], cmap='viridis')
    ax.set_title(f'Channel {i + 1}')
    ax.axis('off')

plt.tight_layout()
# plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
plt.show()

# %%
# Collect the names of each layer except the first one for plotting
layer_names = [layer.name for layer in model.layers]

# Plotting intermediate representation images layer by layer
c = 0
for layer_name, feature_map in zip(layer_names, feature_maps):
    # layer_name = layer_names[0]
    # feature_map = feature_maps[0]
    # number of features in an individual feature map
    n_features = feature_map.shape[-1]  
    # The feature map is in shape of (1, size, size, n_features)
    sizey, sizex = feature_map.shape[1], feature_map.shape[2] 
    # Tile our feature images in matrix `display_grid
    display_grid = np.zeros((sizey, sizex * n_features))
    # Fill out the matrix by looping over all the feature images of your image
    for i in range(n_features):
        # Postprocess each feature of the layer to make it pleasible to your eyes
        x = feature_map[0, :, :, i]
        # x -= x.mean()
        # x /= x.std()
        # x *= 64
        # x += 128
        # x = np.clip(x, 0, 255).astype('uint8')
        # We'll tile each filter into this big horizontal grid
        # if c == 0:
        #    plt.figure(figsize=(scale * n_features, scale))
        #    CS = plt.imshow(x*2)
        #    fig.colorbar(CS, shrink = 0.85)
        #    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
        display_grid[:, i * sizex : (i + 1) * sizex] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    
    CS = plt.imshow(display_grid, aspect='equal', cmap='viridis')
    fig.colorbar(CS)
    c = c+1
# %% Show predictions during training

for i, layer_name in enumerate(layer_names):
    feature_map_Training =visual_model(input_sample)[i].numpy()
    n_features = feature_map_Training.shape[-1]  
    # The feature map is in shape of (1, size, size, n_features)
    sizey, sizex = feature_map_Training.shape[1], feature_map_Training.shape[2] 
    # Tile our feature images in matrix `display_grid
    display_grid = np.zeros((sizey, sizex * n_features))
    # Fill out the matrix by looping over all the feature images of your image
    for j in range(n_features):
        # Postprocess each feature of the layer to make it pleasible to your eyes
        x = feature_map_Training[0, :, :, j]
        # x -= x.mean()
        # x /= x.std()
        # x *= 64
        # x += 128
        # x = np.clip(x, 0, 255).astype('uint8')
        # We'll tile each filter into this big horizontal grid
        # if c == 0:
        #    plt.figure(figsize=(scale * n_features, scale))
        #    CS = plt.imshow(x*2)
        #    fig.colorbar(CS, shrink = 0.85)
        #    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
        display_grid[:, j * sizex : (j + 1) * sizex] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)

    CS = plt.imshow(display_grid, aspect='equal', cmap='viridis')
    fig.colorbar(CS)



# %%
#Select a convolutional layer
layer = model.layers[0]

#Get weights
kernels, biases = layer.get_weights()

#Normalize kernels into [0, 1] range for proper visualization
kernels = (kernels - np.min(kernels, axis=3)) / (np.max(kernels, axis=3) - np.min(kernels, axis=3))

#Weights are usually (width, height, channels, num_filters)
#Save weight images
import cv2

for i in range(kernels.shape[3]):
    filter = kernels[:, :, :, i]
    cv2.imwrite('filter-{}.png'.format(i), filter)
# %%
