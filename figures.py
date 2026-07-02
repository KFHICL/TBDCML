#####################################################################
# Description
# Update 
#####################################################################
'''
This script is used to manually generate figures from the results in 
the study

It is recommended to work in python interactive windows (using e.g 
VS code)

Inputs:
None - Fix paths etc. manually in code

Ouputs:
Figures visualising model performance and behaviour
'''

# %%
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
import skimage as ski

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
# import cv2
# import plotly
import warnings
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

import argparse

# %% Settings
plt.style.use("seaborn-v0_8-colorblind")
sampleNum = 0
loadAllSamp = False
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != "Individual Project":
    current_dir = os.path.dirname(current_dir)

# %% Import all datasets

# MC24 extended needs slightly different loading method
gridPath = os.path.join(current_dir, "Data", "MatLabModelFiles", "sampleGrid_224.json")
# gridPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\sampleGrid_224.json"
with open(gridPath) as json_file: # load into dict
    MC24x_grid = np.array(json.load(json_file)) # grid for plotting

# MC24x_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20241017_1550_224_4kSamples"
MC24x_path = os.path.join(current_dir, "Data", "MatLabModelFiles", "20241017_1550_224_4kSamples")
MC24x_numSamples = len(os.listdir(MC24x_path)) # Number of data samples (i.e. TBDC specimens)
MC24x_sampleShape = [224,224]
MC24x_xNames = ['Ex','Ey','Gxy','Vf','c2'] # Names of input features in input csv
MC24x_yNames = ['FI'] # Names of ground truth features in input csv
samplesPerFile = 40
numSamples = len(os.listdir(MC24x_path))*samplesPerFile


def loadSamplex(path = str,xNames = list, yNames = list, sampleShape=list):
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
for i,file in enumerate(os.listdir(MC24x_path)):
    print('Now loading file number {num} out of {total}'.format(num = i+1, total = numSamples/samplesPerFile))
    filepath = os.path.join(MC24x_path,file)
    if i==0:
        headers, samples = loadSamplex(filepath, xNames = MC24x_xNames, yNames = MC24x_yNames, sampleShape=MC24x_sampleShape)
    else:
        addSamp = loadSamplex(filepath, xNames = MC24x_xNames, yNames = MC24x_yNames, sampleShape=MC24x_sampleShape)[1]
        MC24x_samples = samples.concatenate(addSamp)
    if not loadAllSamp:
        if i>=4: # Only load first 5 files (200 samples) to save time
            break



# %%



# gridPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\sampleGrid.json"
gridPath = os.path.join(current_dir, "Data", "MatLabModelFiles", "sampleGrid.json")
with open(gridPath) as json_file: # load into dict
    MC24_grid = np.array(json.load(json_file)) # grid for plotting

# MC24_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240809_1357_100SamplesVfVariable'
MC24_path = os.path.join(current_dir, "Data", "MatLabModelFiles", "20240809_1357_100SamplesVfVariable")
MC24_numSamples = len(os.listdir(MC24_path)) # Number of data samples (i.e. TBDC specimens)
MC24_sampleShape = [60,20]
MC24_xNames = ['Ex','Ey','Gxy','Vf','c2'] # Names of input features in input csv
MC24_yNames = ['FI'] # Names of ground truth features in input csv

# MC24 dataset with constant Vf
# MC24_path_constVf = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240809_1402_100SamplesVfConstant'
MC24_path_constVf = os.path.join(current_dir, "Data", "MatLabModelFiles", "20240809_1402_100SamplesVfConstant")

# MC24 dataset not seen by any model
# MC24_path_Unseen = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240819_0954_100UnseenSamples"
MC24_path_Unseen = os.path.join(current_dir, "Data", "MatLabModelFiles", "20240819_0954_100UnseenSamples")

# LFC18 dataset
# gridPathOld = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json"
gridPathOld = os.path.join(current_dir, "Data", "FlorianAbaqusFiles", "sampleGrid.json")
with open(gridPathOld) as json_file: # load into dict
    LFC18_grid = np.array(json.load(json_file)) # grid for plotting

# LFC18_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain' # The data format after extracting from Abaqus
LFC18_path = os.path.join(current_dir, "Data", "FlorianAbaqusFiles", "datain") # The data format after extracting from Abaqus
LFC18_numSamples = len(os.listdir(LFC18_path)) # Number of data samples (i.e. TBDC specimens)
LFC18_sampleShape = [55,20]
LFC18_xNames = ['E11','E22','E12'] # Names of input features in input csv
LFC18_yNames = ['FI'] # Names of ground truth features in input csv

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

# Import MC24
for i,file in enumerate(os.listdir(MC24_path)):
    filepath = os.path.join(MC24_path,file)
    if i==0:
        MC24_headers, MC24_samples = loadSample(filepath)
        MC24_samples = MC24_samples.reshape(1, np.shape(MC24_samples)[0],np.shape(MC24_samples)[1])
    else:
        addSamp = loadSample(filepath)[1]
        MC24_samples = np.concatenate((MC24_samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
MC24_samples_NonStandard = MC24_samples
# Reshape sample variable to have shape (samples, row, column, features)
MC24_samples2D = MC24_samples.reshape(MC24_numSamples,MC24_sampleShape[0],MC24_sampleShape[1],MC24_samples.shape[-1])

# Find indeces of input features 
MC24_featureIdx = []
for name in MC24_xNames:
   MC24_featureIdx += [np.where(MC24_headers == name)[0][0]]

# Find indeces of ground truth features 
MC24_gtIdx = []
for name in MC24_yNames:
   MC24_gtIdx += [np.where(MC24_headers == name)[0][0]]
   
MC24_X = MC24_samples2D[:,:,:,MC24_featureIdx]  # Input features

MC24_Y = MC24_samples2D[:,:,:,MC24_gtIdx] # Labels


# Import LFC18
for i,file in enumerate(os.listdir(LFC18_path)):
    filepath = os.path.join(LFC18_path,file)
    if i==0:
        LFC18_headers, LFC18_samples = loadSample(filepath)
        LFC18_samples = LFC18_samples.reshape(1, np.shape(LFC18_samples)[0],np.shape(LFC18_samples)[1])
    else:
        addSamp = loadSample(filepath)[1]
        LFC18_samples = np.concatenate((LFC18_samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
LFC18_samples_NonStandard = LFC18_samples
# Reshape sample variable to have shape (samples, row, column, features)
LFC18_samples2D = LFC18_samples.reshape(LFC18_numSamples,LFC18_sampleShape[0],LFC18_sampleShape[1],LFC18_samples.shape[-1])

# Find indeces of input features 
LFC18_featureIdx = []
for name in LFC18_xNames:
   LFC18_featureIdx += [np.where(LFC18_headers == name)[0][0]]

# RENAME HEADERS IN LFC18 TO MATCH MC24
LFC18_headers[np.where(LFC18_headers == 'E11')] = 'Ex'
LFC18_headers[np.where(LFC18_headers == 'E22')] = 'Ey'
LFC18_headers[np.where(LFC18_headers == 'E12')] = 'Gxy'

# Find indeces of ground truth features 
LFC18_gtIdx = []
for name in LFC18_yNames:
   LFC18_gtIdx += [np.where(LFC18_headers == name)[0][0]]
   
LFC18_X = LFC18_samples2D[:,:,:,LFC18_featureIdx]  # Input features

LFC18_Y = LFC18_samples2D[:,:,:,LFC18_gtIdx] # Labels

# Import MC24 with constant Vf
for i,file in enumerate(os.listdir(MC24_path_constVf)):
    filepath = os.path.join(MC24_path_constVf,file)
    if i==0:
        MC24_headers_constVf, MC24_samples_constVf = loadSample(filepath)
        MC24_samples_constVf = MC24_samples_constVf.reshape(1, np.shape(MC24_samples_constVf)[0],np.shape(MC24_samples_constVf)[1])
    else:
        addSamp = loadSample(filepath)[1]
        MC24_samples_constVf = np.concatenate((MC24_samples_constVf,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
MC24_samples_NonStandard = MC24_samples
# Reshape sample variable to have shape (samples, row, column, features)
MC24_samples2D_constVf = MC24_samples_constVf.reshape(MC24_numSamples,MC24_sampleShape[0],MC24_sampleShape[1],MC24_samples.shape[-1])

# Find indeces of input features 
MC24_featureIdx_constVf = []
for name in MC24_xNames:
   MC24_featureIdx_constVf += [np.where(MC24_headers == name)[0][0]]

# Find indeces of ground truth features 
MC24_gtIdx_constVf = []
for name in MC24_yNames:
   MC24_gtIdx_constVf += [np.where(MC24_headers == name)[0][0]]
   
MC24_X_constVf = MC24_samples2D_constVf[:,:,:,MC24_featureIdx_constVf]  # Input features

MC24_Y_constVf = MC24_samples2D_constVf[:,:,:,MC24_gtIdx_constVf] # Labels


# Import MC24 Unseen Sanmples
for i,file in enumerate(os.listdir(MC24_path_Unseen)):
    filepath = os.path.join(MC24_path_Unseen,file)
    if i==0:
        MC24_headers_Unseen, MC24_samples_Unseen = loadSample(filepath)
        MC24_samples_Unseen = MC24_samples_Unseen.reshape(1, np.shape(MC24_samples_Unseen)[0],np.shape(MC24_samples_Unseen)[1])
    else:
        addSamp = loadSample(filepath)[1]
        MC24_samples_Unseen = np.concatenate((MC24_samples_Unseen,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
# Reshape sample variable to have shape (samples, row, column, features)
MC24_samples2D_Unseen = MC24_samples_Unseen.reshape(MC24_numSamples,MC24_sampleShape[0],MC24_sampleShape[1],MC24_samples.shape[-1])

# Find indeces of input features 
MC24_featureIdx_Unseen = []
for name in MC24_xNames:
   MC24_featureIdx_Unseen += [np.where(MC24_headers_Unseen == name)[0][0]]

# Find indeces of ground truth features 
MC24_gtIdx_Unseen = []
for name in MC24_yNames:
   MC24_gtIdx_Unseen += [np.where(MC24_headers_Unseen == name)[0][0]]
   
MC24_X_Unseen = MC24_samples2D_Unseen[:,:,:,MC24_featureIdx_Unseen]  # Input features

MC24_Y_Unseen = MC24_samples2D_Unseen[:,:,:,MC24_gtIdx_Unseen] # Labels


# %% Do predictions on unseen samples

# Load model
# modelPaths = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24CrossValidation1408"
modelPaths = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "MC24CrossValidation1408")
repeats = 1
numModels = 1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=1)

for i in range(repeats): # For each repeat (1=indexed)
    for j in range(numModels): # For each model (1-indexed)
        print('Now loading repeat {rp} model number {num}'.format(rp = i+1, num = j+1))
        # RMSEpath = modelPaths+'_{rp}\dataout\RMSE_val_MC24CrossValidation1408_{rp}_{m}.json'.format(rp = i+1, m = j+1)
        modelPath = modelPaths+'_{rp}\dataout\model_MC24CrossValidation1408_{rp}_{m}.keras'.format(rp = i+1, m = j+1)
        gtPath = modelPaths+'_{rp}\dataout\groundTruth_MC24CrossValidation1408_{rp}_{m}.json'.format(rp = i+1, m = j+1)
        gtPath_val = modelPaths+'_{rp}\dataout\groundTruth_val_MC24CrossValidation1408_{rp}_{m}.json'.format(rp = i+1, m = j+1)
        predPath = modelPaths+'_{rp}\dataout\predictions_MC24CrossValidation1408_{rp}_{m}.json'.format(rp = i+1, m = j+1)
        predPath_val = modelPaths+'_{rp}\dataout\predictions_val_MC24CrossValidation1408_{rp}_{m}.json'.format(rp = i+1, m = j+1)
        
        loaded_model = keras.models.load_model(modelPath)

        with open(gtPath) as json_file: # load into dict
            groundTruth = np.array(json.load(json_file))
        
        with open(gtPath_val) as json_file: # load into dict
            groundTruth_val = np.array(json.load(json_file))

        with open(predPath) as json_file: # load into dict
            prediction = np.array(json.load(json_file))
        
        with open(predPath_val) as json_file: # load into dict
            prediction_val = np.array(json.load(json_file))
            
        



# loaded_model = keras.models.load_model(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24CrossValidation1408_1\dataout\model_MC24CrossValidation1408_1_1.keras")
# loaded_model = keras.models.load_model(modelPath)
# %% FUnction for creating contour plot


#   # Plot ground truth
# im1 = ax.imshow(MC24x_plotSamp[:,:,tmp]/1000, cmap='viridis')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size=0.07, pad=0.1)
# cbar = plt.colorbar(im1, cax=cax)
#   # cbar.set_label('Ground Truth FI')
# ax.axis('off')
# ax.set_anchor('C')  # Center the image in the axes
# dx = scaleBarDx  # Assuming 50 units correspond to the full width of the image
# scalebar = ScaleBar(dx, "mm", fixed_units="mm", fixed_value = 25,width_fraction=0.015*dx, box_alpha = 0,rotation ="vertical-only",bbox_to_anchor=(0, 1),bbox_transform=ax.transAxes)
# ax.add_artist(scalebar)



from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_contour(grid, samples2D,  ax, xlab = None, ylab = None, cbarlab = None, cBarBins = 5,scaleBarDx = None):
    if np.min(samples2D) == np.max(samples2D):
         CS = ax.contourf(grid[0],grid[1],samples2D)
         cbar = fig.colorbar(CS,ticks=[], shrink = 0.85)
         cbar.ax.text(0.1, -0.03, round(np.min(samples2D),2), transform=cbar.ax.transAxes, 
            va='top', ha='left')
         cbar.set_label(cbarlab, rotation=270,labelpad=9)
    else:
        im1 = ax.imshow(samples2D, cmap='viridis')
        # CS = ax.contourf(grid[0],grid[1],samples2D,levels=np.linspace(np.min(samples2D), np.max(samples2D), 10))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.1, pad=0.1)
        cbar = plt.colorbar(im1, ticks=[], cax=cax, shrink=0.5)
        # cbar = fig.colorbar(CS, ticks=[], cax=cax, shrink=0.5)
        # cbar.ax.text(0.1, -0.03, round(np.min(samples2D),2), transform=cbar.ax.transAxes, 
        #     va='top', ha='left')
        # cbar.ax.text(0.1, 1.03, round(np.max(samples2D),2), transform=cbar.ax.transAxes, 
        #     va='bottom', ha='left')
        cbar.ax.text(0.5, -0.03, round(np.min(samples2D),2), transform=cbar.ax.transAxes, 
            va='top', ha='center')
        cbar.ax.text(0.5, 1.03, round(np.max(samples2D),2), transform=cbar.ax.transAxes, 
            va='bottom', ha='center')
        # fig.subplots_adjust(right=0.88)
        cbar.ax.text(1, 0.5, cbarlab, transform=cbar.ax.transAxes, 
            rotation=270, va='center', ha='left')
        # cbar.set_label(cbarlab, rotation=270,labelpad=14)
        # cbar.ax.yaxis.set_label_position('right')
    
    plt.ylabel(ylab)
    plt.title(xlab)
    # plt.xlabel(xlab)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.set_aspect('equal', 'box')
    # Add a scale bar
    if scaleBarDx is not None:
        # scalebarLoc = [np.min(grid[0])+10, np.max(grid[1])+5]  # Top left corner
        scalebar_length = 25  # Adjust this value to match the scale of your data
        # Create scale bar

        # dx = sampleWidths[i]/ground_truths[i].shape[1]  # Assuming 50 units correspond to the full width of the image
        # dx = 1
        # scalebar = ScaleBar(dx, "mm", fixed_units="mm", fixed_value = 25,width_fraction=0.015*dx, box_alpha = 0,rotation ="vertical-only",bbox_to_anchor=(0, 1),bbox_transform=axes[i, 0].transAxes)
        scalebar = ScaleBar(scaleBarDx, "mm", fixed_units="mm", fixed_value = 25,width_fraction=0.015*scaleBarDx, box_alpha = 0,rotation ="vertical-only",bbox_to_anchor=(0, 1),bbox_transform=ax.transAxes)
        ax.add_artist(scalebar)

        # # Draw a white rectangle underneath the scale bar and text
        # rect = matplotlib.patches.Rectangle(
        #     (scalebarLoc[0] - 10, scalebarLoc[1] +2),
        #     scalebarLoc[0]+scalebar_length + 10, - 10,
        #     transform=ax.transData,
        #     color='white',
        #     alpha=0.5,
        #     zorder=1
        # )
        # ax.add_patch(rect)

        # Draw the scale bar and text in black
        # ax.plot([scalebarLoc[0], scalebarLoc[0] + scalebar_length], [scalebarLoc[1], scalebarLoc[1]], color='black', lw=2, transform=ax.transData, clip_on=False, zorder=2)
        # ax.text(scalebarLoc[0] + scalebar_length / 2, scalebarLoc[1] + 2, f'{scalebar_length} mm', ha='center', va='bottom', fontsize=8, transform=ax.transData, color="black", zorder=2)
    # cbar = fig.colorbar(CS,ticks=[np.min(samples2D), np.max(samples2D)], shrink = 0.8)
    
    # cbar.ax.locator_params(nbins=cBarBins)

# %% Plot unaltered MC24 and LFC18 samples

# PLOT Ex, Ey, Gxy, FI
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixel
# matplotlib.rcParams["mathtext.fontset"] = 'stix'
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
# matplotlib.rcParams["mathtext.fontset"] = 'stixsans'
plt.rcParams["font.size"] = "8"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
# figHeight = figWidth*Ratio*(2/4) # 2 subplots high, 4 subplots wide 
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="tight", dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight)
# fig.set_figwidth(figWidth)
sampleNum = 0


# Prep MC24x sample for plotting
MC24x_samp = MC24x_samples.take(1)
MC24x_X = np.zeros((MC24x_sampleShape[0],MC24x_sampleShape[1], len(MC24x_xNames)))
MC24x_Y = np.zeros((MC24x_sampleShape[0],MC24x_sampleShape[1], len(MC24x_yNames)))

for images, labels in MC24x_samp:
    for n in range(len(MC24x_xNames)):
        MC24x_X[:,:,n] = images[:,:,n].numpy()
    MC24x_Y = labels.numpy()

MC24x_plotSamp = np.concatenate((MC24x_Y,MC24x_X), axis = -1)
# Ex
# ax = plt.subplot(1, 1, 1)
# tmp = np.where(MC24_headers == 'Ex')[0][0]
# plot_contour(grid = MC24x_grid, samples2D = MC24x_X[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = 'MC24_extended', cbarlab = 'Stiffness [GPA]', cBarBins = 3)
# ax.set_aspect('equal', 'box')
    # ax = plt.subplot(4, 8, i + 1)
    # plot_contour(grid = LFC18_grid, samples2D = images[i].numpy(),  ax = ax, xlab = None, ylab = 'LFC18', cbarlab = 'Stiffness [GPA]', cBarBins = 3)
    # plt.imshow(images[i].numpy().astype("uint8"))





# Ex
ax = plt.subplot(3, 6, 1)
tmp = np.where(LFC18_headers == 'Ex')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = 'MeC-Macro', cbarlab = '$E_{\mathrm{x}}$ [GPa]', cBarBins = 3)
ax.set_ylabel('(a)', fontweight='bold', rotation=0, labelpad=15, fontsize=10)

ax = plt.subplot(3, 6, 6+1)
tmp = np.where(MC24_headers == 'Ex')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = None, ylab = 'MeC-Meso-S/M', cbarlab = '$E_{\mathrm{x}}$ [GPa]', cBarBins = 3)
ax.set_ylabel('(b)', fontweight='bold', rotation=0, labelpad=15, fontsize=10)

ax = plt.subplot(3, 6, 12+1)
tmp = np.where(MC24_headers == 'Ex')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = 'MeC-Meso-L', cbarlab = '$E_{\mathrm{x}}$ [GPa]', cBarBins = 3)
ax.set_aspect('equal', 'box')
ax.set_ylabel('(c)', fontweight='bold', rotation=0, labelpad=15, fontsize=10)

# Ey
ax = plt.subplot(3, 6, 2)
tmp = np.where(LFC18_headers == 'Ey')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$E_{\mathrm{y}}$ [GPa]', cBarBins = 3)

ax = plt.subplot(3, 6, 6+2)
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$E_{\mathrm{y}}$ [GPa]', cBarBins = 3)
ax = plt.subplot(3, 6, 12+2)
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$E_{\mathrm{y}}$ [GPa]', cBarBins = 3)
ax.set_aspect('equal', 'box')

# Gxy
ax = plt.subplot(3, 6, 3)
tmp = np.where(LFC18_headers == 'Gxy')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$G_{\mathrm{xy}}$ [GPa]', cBarBins = 3)

ax = plt.subplot(3, 6, 6+3)
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$G_{\mathrm{xy}}$ [GPa]', cBarBins = 3)

ax = plt.subplot(3, 6, 12+3)
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$G_{\mathrm{xy}}$ [GPa]', cBarBins = 3)
ax.set_aspect('equal', 'box')

# Vf
ax = plt.subplot(3, 6, 4)
# plt.title('')
ax.axis("off")

ax = plt.subplot(3, 6, 6+4)
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$V_{\mathrm{f}}$', cBarBins = 3)

ax = plt.subplot(3, 6, 12+4)
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$V_{\mathrm{f}}$', cBarBins = 3)
ax.set_aspect('equal', 'box')

# Cos squared
ax = plt.subplot(3, 6, 5)
# plt.title('')
ax.axis("off")

ax = plt.subplot(3, 6, 6+5)
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$c_{2}$', cBarBins = 3)

ax = plt.subplot(3, 6, 12+5)
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$c_{2}$', cBarBins = 3)
ax.set_aspect('equal', 'box')

# FI
ax = plt.subplot(3, 6, 6)
tmp = np.where(LFC18_headers == 'FI')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = '', ylab = None, cbarlab = '$FI$', cBarBins = 3)

ax = plt.subplot(3, 6, 6+6)
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$FI$', cBarBins = 3)

ax = plt.subplot(3, 6, 12+6)
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$FI$', cBarBins = 3)
ax.set_aspect('equal', 'box')

# Uncomment to save
# plt.savefig('DatasetSamples.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
plt.savefig('C:\\Users\\kfh23\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\DatasetSamples.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
plt.show()

# %% Plot Meso-L separately
# PLOT Ex, Ey, Gxy, FI
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixel
# matplotlib.rcParams["mathtext.fontset"] = 'stix'
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
# matplotlib.rcParams["mathtext.fontset"] = 'stixsans'
plt.rcParams["font.size"] = "10"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
# figHeight = figWidth*Ratio*(2/4) # 2 subplots high, 4 subplots wide 
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="tight", dpi = resolution_scaling*100) # 100 is default size
# fig.set_figheight(figHeight)
# fig.set_figwidth(figWidth)
sampleNum = 0


# Prep MC24x sample for plotting
MC24x_samp = MC24x_samples.take(1)
MC24x_X = np.zeros((MC24x_sampleShape[0],MC24x_sampleShape[1], len(MC24x_xNames)))
MC24x_Y = np.zeros((MC24x_sampleShape[0],MC24x_sampleShape[1], len(MC24x_yNames)))

for images, labels in MC24x_samp:
    for n in range(len(MC24x_xNames)):
        MC24x_X[:,:,n] = images[:,:,n].numpy()
    MC24x_Y = labels.numpy()

MC24x_plotSamp = np.concatenate((MC24x_Y,MC24x_X), axis = -1)
# Ex
# ax = plt.subplot(1, 1, 1)
# tmp = np.where(MC24_headers == 'Ex')[0][0]
# plot_contour(grid = MC24x_grid, samples2D = MC24x_X[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = 'MC24_extended', cbarlab = 'Stiffness [GPA]', cBarBins = 3)
# ax.set_aspect('equal', 'box')
    # ax = plt.subplot(4, 8, i + 1)
    # plot_contour(grid = LFC18_grid, samples2D = images[i].numpy(),  ax = ax, xlab = None, ylab = 'LFC18', cbarlab = 'Stiffness [GPA]', cBarBins = 3)
    # plt.imshow(images[i].numpy().astype("uint8"))

# Create a figure with uneven subplot grid
fig = plt.figure(layout="tight", dpi=resolution_scaling * 100)

# Define the grid layout
grid = fig.add_gridspec(1, 6)
# fig.set_figheight(fig.get_figheight() * 1.5)

# MeC-Macro:
ax = fig.add_subplot(grid[0, 0])
tmp = np.where(LFC18_headers == 'Ex')[0][0]
scaleBarDx = 50/LFC18_sampleShape[1]
# scaleBarDx = 1
# scaleBarDx = 50/LFC18_samples2D[sampleNum,:,:,tmp].shape[1] 
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$E_{\mathrm{x}}$ [GPa]', cBarBins = 3, scaleBarDx = scaleBarDx)


ax = fig.add_subplot(grid[0, 1])
tmp = np.where(LFC18_headers == 'Ey')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$E_{\mathrm{y}}$ [GPa]', cBarBins = 3)

ax = fig.add_subplot(grid[0, 2])
tmp = np.where(LFC18_headers == 'Gxy')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$G_{\mathrm{xy}}$ [GPa]', cBarBins = 3)

ax = fig.add_subplot(grid[0, 5])
tmp = np.where(LFC18_headers == 'FI')[0][0]
plot_contour(grid = LFC18_grid, samples2D = LFC18_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = '', ylab = None, cbarlab = '$F$', cBarBins = 3)
# fig.text(0, 0.19, '(a)', fontweight='bold', fontsize=12)
# fig.text(0, 0.19, '(a): MeC-Macro', fontsize=10)
fig.text(
  0, 0.5,  # Adjust vertical position based on the row index
  " ",  # 'a', 'b', 'c', etc.
  fontsize=10,
  fontweight='bold',
  va='center',
  ha='left',
)
plt.savefig('C:\\Users\\kfh23\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\DatasetSamples_Macro_v2.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.05)
plt.show()



# Add a dotted line between the rows of subplots

# divider_ax = fig.add_subplot(grid[:, :], frame_on=False)
# divider_ax.axis("off")
# divider_ax.plot([0, 1], [0.75, 0.75], transform=fig.transFigure, color="grey", linestyle="--", linewidth=0.8)


# divider_ax = fig.add_subplot(grid[:, :], frame_on=False)
# divider_ax.axis("off")
# divider_ax.plot([0, 1], [0.5, 0.5], transform=fig.transFigure, color="grey", linestyle="--", linewidth=0.8)

#MeC-Meso-S/M:

# Create a figure with uneven subplot grid
fig = plt.figure(layout="tight", dpi=resolution_scaling * 100)

# Define the grid layout
grid = fig.add_gridspec(1, 6)
ax = fig.add_subplot(grid[0, 0])
tmp = np.where(MC24_headers == 'Ex')[0][0]
scaleBarDx = 50/MC24_sampleShape[1]
# scaleBarDx = 1
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$E_{\mathrm{x}}$ [GPa]', cBarBins = 3, scaleBarDx = scaleBarDx)
# fig.text(-0.03, 0.625, '(b)', fontweight='bold', fontsize=12)

ax = fig.add_subplot(grid[0, 1])
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$E_{\mathrm{y}}$ [GPa]', cBarBins = 3)

ax = fig.add_subplot(grid[0, 2])
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = '', ylab = None, cbarlab = '$G_{\mathrm{xy}}$ [GPa]', cBarBins = 3)
ax = fig.add_subplot(grid[0, 3])
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$V_{\mathrm{f}}$', cBarBins = 3)

ax = fig.add_subplot(grid[0, 4])
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$c_{2}$', cBarBins = 3)

ax = fig.add_subplot(grid[0, 5])
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = '', ylab = None, cbarlab = '$F$', cBarBins = 3)
# fig.text(0, 0.19, '(b): MeC-Meso-S/M', fontsize=10)
fig.text(
  0, 0.5,  # Adjust vertical position based on the row index
  " ",  # 'a', 'b', 'c', etc.
  fontsize=10,
  fontweight='bold',
  va='center',
  ha='left',
)

plt.savefig('C:\\Users\\kfh23\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\DatasetSamples_Meso-SM_v2.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.05)
plt.show()



# MeC-Meso-L:
# Create a figure with uneven subplot grid
fig = plt.figure(layout="tight", dpi=resolution_scaling * 100)

# Define the grid layout
grid = fig.add_gridspec(2, 6)


ax = fig.add_subplot(grid[0:1, 0 * 2:(0 + 1) * 2])  # Adjusted to span two rows
tmp = np.where(MC24_headers == 'Ex')[0][0]
scaleBarDx = 224/MC24x_sampleShape[1]



# scaleBarDx = 1
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$E_{\mathrm{x}}$ [GPa]', cBarBins = 3, scaleBarDx = scaleBarDx)
# ax.set_aspect('equal', 'box')
# fig.text(-0.03, 0.25, '(c)', fontweight='bold', fontsize=12)

ax = fig.add_subplot(grid[0:1, 1 * 2:(1 + 1) * 2])  # Adjusted to span two rows
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$E_{\mathrm{y}}$ [GPa]', cBarBins = 3)
ax.set_aspect('equal', 'box')

ax = fig.add_subplot(grid[0:1, 2 * 2:(2 + 1) * 2])  # Adjusted to span two rows
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = '$G_{\mathrm{xy}}$ [GPa]', cBarBins = 3)
ax.set_aspect('equal', 'box')

ax = fig.add_subplot(grid[1:2, 0 * 2:(0 + 1) * 2])  # Adjusted to span two rows
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$V_{\mathrm{f}}$', cBarBins = 3)
ax.set_aspect('equal', 'box')

ax = fig.add_subplot(grid[1:2, 1 * 2:(1 + 1) * 2])  # Adjusted to span two rows
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$c_{2}$', cBarBins = 3)
ax.set_aspect('equal', 'box')

ax = fig.add_subplot(grid[1:2, 2 * 2:(2 + 1) * 2])  # Adjusted to span two rows
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24x_grid, samples2D = MC24x_plotSamp[:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = '$F$', cBarBins = 3)
ax.set_aspect('equal', 'box')
fig.tight_layout(h_pad=-3)

fig.text(
  0, 0.5,  # Adjust vertical position based on the row index
  " ",  # 'a', 'b', 'c', etc.
  fontsize=10,
  fontweight='bold',
  va='center',
  ha='left',
)

# fig.text(0, 0, '(c): MeC-Meso-L', fontsize=10)
plt.savefig('C:\\Users\\kfh23\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\DatasetSamples_Meso-L_v2.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.05)
plt.show()


# Uncomment to save
# plt.savefig('DatasetSamples.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
# plt.savefig('C:\\Users\\kfh23\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\DatasetSamples_v2.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
# plt.show()


# %%
# Create a grid for plotting the MC24x samples


# x = np.linspace(MC24x_sampleShape[0], 1, 224)  # 224 points from 48.75 to 1.25
# y = np.linspace(MC24x_sampleShape[0], 1, 224)  # 224 points from 148.75 to 1.25
# MC24x_grid = np.meshgrid(x, y)
# MC24x_grid = np.stack((MC24x_grid[0],MC24x_grid[1]))

# # Save the grid to the specified path
# grid_save_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\sampleGrid_224.json"
# with open(grid_save_path, 'w') as json_file:
#     json.dump(MC24x_grid.tolist(), json_file)

# %% Compare MC24 with and without constant Vf

# PLOT Ex, Ey, Gxy, FI
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixel
# matplotlib.rcParams["mathtext.fontset"] = 'stix'
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
# matplotlib.rcParams["mathtext.fontset"] = 'stixsans'
plt.rcParams["font.size"] = "6"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
# figHeight = figWidth*Ratio*(2/4) # 2 subplots high, 4 subplots wide 
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
sampleNum = 2

# Ex
ax = plt.subplot(2, 6, 1)
tmp = np.where(MC24_headers == 'Ex')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = None, ylab = 'MC24 Variable Vf', cbarlab = 'Stiffness [GPA]', cBarBins = 3)

ax = plt.subplot(2, 6, 6+1)
tmp = np.where(MC24_headers == 'Ex')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D_constVf[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Ex', ylab = 'MC24 Constant Vf', cbarlab = 'Stiffness [GPA]', cBarBins = 3)


# Ey
ax = plt.subplot(2, 6, 2)
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = 'Stiffness [GPA]', cBarBins = 3)

ax = plt.subplot(2, 6, 6+2)
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D_constVf[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Ey', ylab = None, cbarlab = 'Stiffness [GPA]', cBarBins = 3)


# Gxy
ax = plt.subplot(2, 6, 3)
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = None, ylab = None, cbarlab = 'Stiffness [GPA]', cBarBins = 3)

ax = plt.subplot(2, 6, 6+3)
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D_constVf[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Gxy', ylab = None, cbarlab = 'Stiffness [GPA]', cBarBins = 3)

# Vf
ax = plt.subplot(2, 6, 4)
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = 'Volume fraction', cBarBins = 3)

ax = plt.subplot(2, 6, 6+4)
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D_constVf[sampleNum,:,:,tmp],  ax = ax, xlab = 'Vf', ylab = None, cbarlab = 'Volume fraction', cBarBins = 3)

# Cos squared
ax = plt.subplot(2, 6, 5)
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = 'cos^2(q)', cBarBins = 3)

ax = plt.subplot(2, 6, 6+5)
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D_constVf[sampleNum,:,:,tmp],  ax = ax, xlab = 'c2', ylab = None, cbarlab = 'cos^2(q)', cBarBins = 3)


# FI
ax = plt.subplot(2, 6, 6)
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp],  ax = ax, xlab = None, ylab = None, cbarlab = 'Failure Index', cBarBins = 3)

ax = plt.subplot(2, 6, 6+6)
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D_constVf[sampleNum,:,:,tmp],  ax = ax, xlab = 'FI', ylab = None, cbarlab = 'Failure Index', cBarBins = 3)

# Uncomment to save
plt.savefig('DatasetSamples_ConstantVf.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
plt.show()

# %% Full fields for the MC24 dataset
# PLOT Ex, Ey, Gxy, FI
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixel
# matplotlib.rcParams["mathtext.fontset"] = 'stix'
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
# matplotlib.rcParams["mathtext.fontset"] = 'stixsans'
plt.rcParams["font.size"] = "6"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
# figHeight = figWidth*Ratio*(2/4) # 2 subplots high, 4 subplots wide 
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
sampleNum = 0

ax = plt.subplot(2, 3, 1)
tmp = np.where(MC24_headers == 'Ex')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Ex', ylab = 'MC24', cbarlab = 'Stiffness [GPA]', cBarBins = 3)

ax = plt.subplot(2, 3, 2)
tmp = np.where(MC24_headers == 'Ey')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Ey', ylab = 'MC24', cbarlab = 'Stiffness [GPA]', cBarBins = 3)

ax = plt.subplot(2, 3, 3)
tmp = np.where(MC24_headers == 'Gxy')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Gxy', ylab = 'MC24', cbarlab = 'Stiffness [GPA]', cBarBins = 3)

ax = plt.subplot(2, 3, 4)
tmp = np.where(MC24_headers == 'Vf')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'Vf', ylab = 'MC24', cbarlab = 'Volume fraction', cBarBins = 3)

ax = plt.subplot(2, 3, 5)
tmp = np.where(MC24_headers == 'c2')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'c2', ylab = 'MC24', cbarlab = 'cos^2(Orientation)', cBarBins = 3)

ax = plt.subplot(2, 3, 6)
tmp = np.where(MC24_headers == 'FI')[0][0]
plot_contour(grid = MC24_grid, samples2D = MC24_samples2D[sampleNum,:,:,tmp]/1000,  ax = ax, xlab = 'FI', ylab = 'MC24', cbarlab = 'Failure Index', cBarBins = 3)



# %% 

# %% Verification of Exx extraction of Eyy

# The mean young's modulus should be ~1/3 of the tow modulus
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixel
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
plt.style.use("seaborn-v0_8-colorblind")
resolution_scaling = 1 # Manually scale DPI and text accordingly
# Flatten and put in dataframe
ExDf = pd.DataFrame((MC24_samples2D[:,:,:,np.where(MC24_headers == 'Ex')[0][0]]/1000).reshape(-1))
EyDf = pd.DataFrame((MC24_samples2D[:,:,:,np.where(MC24_headers == 'Ey')[0][0]]/1000).reshape(-1))

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

ax = plt.subplot(2,2,1)
# bins = np.arange(-5, 150, 10)
# sns.histplot(ErrorDistDf, ax=ax, bins = bins)
sns.histplot(ExDf, ax=ax)
plt.grid()
plt.xlabel('Ex [GPa]')
plt.title('Ex')
meanEx = round(np.mean(ExDf),2)
ax.axvline(x=meanEx, c='k', ls='-', lw=2.5)
ax.annotate('Mean Ex = ' + str(meanEx), xy = (0.5,0.9), xycoords = 'axes fraction')
plt.legend([],[], frameon=False)

ax = plt.subplot(2,2,2)
# bins = np.arange(-5, 150, 10)
# sns.histplot(ErrorDistDf, ax=ax, bins = bins)
sns.histplot(EyDf, ax=ax)
plt.grid()
plt.xlabel('Ey [GPa]')
plt.title('Ey')
meanEy = round(np.mean(EyDf),2)
ax.axvline(x=meanEy, c='k', ls='-', lw=2.5)
ax.annotate('Mean Ey = ' + str(meanEy), xy = (0.5,0.9), xycoords = 'axes fraction')
plt.legend([],[], frameon=False)

# plt.show()

# %% Comparative data analysis LFC18 to MC24

# Compare data in terms of the following measures:
# 1: Plot distributions before and after standardisation
# 2: Pearson and Spearman's correlation with Failure index
# 2.1 Do heatmap plot 
# 3: Principal component analysis
# 4: Number of principal components needed to capture variance well
# 5: Do Scipy statistics


# First put data into pandas dataframes
LFC18_sampleIdx = np.linspace(1,LFC18_numSamples,LFC18_numSamples) # Index of each sample
MC24_sampleIdx = np.linspace(1,MC24_numSamples,MC24_numSamples) # 
LFC18_coordIdx = np.linspace(1,LFC18_sampleShape[0]*LFC18_sampleShape[1],LFC18_sampleShape[0]*LFC18_sampleShape[1]) # Index of each coordinate within a simple sample
MC24_coordIdx = np.linspace(1,MC24_sampleShape[0]*MC24_sampleShape[1],MC24_sampleShape[0]*MC24_sampleShape[1])
LFC18_midx = pd.MultiIndex.from_product([LFC18_sampleIdx,LFC18_coordIdx]) # Index for dataframe
MC24_midx = pd.MultiIndex.from_product([MC24_sampleIdx,MC24_coordIdx])
# Name multiindeces
LFC18_midx = LFC18_midx.set_names(['Sample', 'Datapoint'])
MC24_midx = MC24_midx.set_names(['Sample', 'Datapoint'])

# Flatten to 2D arrays where each column is a feature
LFC18_samplesFlat = LFC18_samples2D.reshape(LFC18_numSamples*LFC18_sampleShape[0]*LFC18_sampleShape[1],len(LFC18_headers))
MC24_samplesFlat = MC24_samples2D.reshape(MC24_numSamples*MC24_sampleShape[0]*MC24_sampleShape[1],len(MC24_headers))
MC24_samplesFlat_constVf = MC24_samples2D_constVf.reshape(MC24_numSamples*MC24_sampleShape[0]*MC24_sampleShape[1],len(MC24_headers))


# Put data in dataframes
LFC18Df = pd.DataFrame(LFC18_samplesFlat, index = LFC18_midx, columns=LFC18_headers)
LFC18Df['Dataset'] = 'LFC18'
MC24Df = pd.DataFrame(MC24_samplesFlat, index = MC24_midx, columns=MC24_headers)
MC24Df['Dataset'] = 'MC24'
MC24Df_constVf = pd.DataFrame(MC24_samplesFlat_constVf, index = MC24_midx, columns=MC24_headers)
MC24Df_constVf['Dataset'] = 'MC24_constVf'
AllDf = pd.concat([LFC18Df, MC24Df])

VfVariationDf = pd.concat([MC24Df, MC24Df_constVf])

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figscale = 1.5
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
# figHeight = figWidth*Ratio*(2/4) # 2 subplots high, 4 subplots wide 
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

#%% 1: Distributions before standardisation
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixel
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "8"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
plt.style.use("seaborn-v0_8-colorblind")


from matplotlib.patches import Patch
def distPlot(df, features):
    fig = plt.figure(layout="tight", dpi = resolution_scaling*100) # 100 is default size
    # fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
    # fig.set_figheight(figHeight)
    # fig.set_figwidth(figWidth)
    numplts = len(features)
    dims = [2,int(np.ceil(numplts/2))]
    for i in range(numplts):
        ax = plt.subplot(dims[0],dims[1],i+1)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # sns.histplot(df, x = features[i], kde = True, kde_kws = {"bw_adjust": 2}, hue = 'Dataset',element = 'step',linewidth=0,alpha=0.6)
        palette = sns.color_palette("colorblind")
        sns.kdeplot(df,ax = ax, x=features[i], bw_adjust=2, hue='Dataset', fill=True, common_norm=True, palette={"LFC18": palette[0], "MC24": palette[1]})
        # if features[i] == 'Vf':
            #  plt.ylim((0,3.3))
        plt.grid()
        plt.xlabel('')
        plt.ylabel('')
        plt.title(features[i])

        lab = None
        match features[i]:
            case 'Ex':
                lab = '$E_{\mathrm{x}}$'
            case 'Ey':
                lab = '$E_{\mathrm{y}}$'
            case 'Gxy':
                lab = '$G_{\mathrm{xy}}$'
            case 'Vf':
                lab = '$V_{\mathrm{f}}$'
            case 'c2':
                lab = '$c_{2}$'
            case 'FI':
                lab = '$F$'
        ax.set_xlabel(lab)
        ax.set_title(f"({chr(97 + i)})")
    # h,l = ax.get_legend_handles_labels()
    # h,l = fig.axes
    # fig.legend(title='Dataset',labels=np.flip(df['Dataset'].unique()), 
    #        loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.1))
    handles = [
    Patch(facecolor=matplotlib.colors.to_rgba(palette[0], alpha=0.4), edgecolor=palette[0]),  # MeC-Macro (left)
    Patch(facecolor=matplotlib.colors.to_rgba(palette[1], alpha=0.4), edgecolor=palette[1])   # MeC-Meso-S  (right)
    ]
    print(handles)
    fig.legend(title='Dataset',handles=handles, labels=['MeC-Macro','MeC-Meso-S/M/L'], 
        loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.07))
    for ax in fig.axes[:]:
        ax.get_legend().remove()

plt.tight_layout()
# All data points
distPlot(AllDf, features = MC24_headers[MC24_featureIdx+MC24_gtIdx])
plt.savefig('C:\\Users\\kaspe\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\DataDistributions.pdf', bbox_inches='tight', pad_inches = 0)
# distPlot(VfVariationDf, features = MC24_headers[MC24_featureIdx+MC24_gtIdx])
# plt.savefig('DataDistributions_constVf.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
plt.show()

# %% Correlations with FI
# Pearson correlation coefficient describing linear correlation
pearsonLFC18 = np.zeros((LFC18_numSamples,len(LFC18_featureIdx))) # Pearson correlation with Failure index
pearsonLFC18_Pval = np.copy(pearsonLFC18)
pearsonMC24 = np.zeros((MC24_numSamples,len(MC24_featureIdx))) # Pearson correlation with Failure index
pearsonMC24_Pval = np.copy(pearsonMC24)
pearsonMC24_constVf = np.zeros((MC24_numSamples,len(MC24_featureIdx))) # Pearson correlation with Failure index
pearsonMC24_Pval_constVf = np.copy(pearsonMC24_constVf)

# Spearman's correlation coefficient is the correlation of rank (i.e. if features monotonically increase together independent of proportionality)
spearmanLFC18 = np.zeros((LFC18_numSamples,len(LFC18_featureIdx)))
spearmanLFC18_Pval = np.copy(pearsonLFC18)
spearmanMC24 = np.zeros((MC24_numSamples,len(MC24_featureIdx)))
spearmanMC24_Pval = np.copy(pearsonMC24)
spearmanMC24_constVf = np.zeros((MC24_numSamples,len(MC24_featureIdx)))
spearmanMC24_Pval_constVf = np.copy(pearsonMC24_constVf)

for i in range(LFC18_numSamples):
    for j,k in enumerate(LFC18_featureIdx): # Calculate correlation coefficients
        pearsonLFC18[i,j], pearsonLFC18_Pval[i,j] = scipy.stats.pearsonr(LFC18_samples[i,:,LFC18_gtIdx[0]],LFC18_samples[i,:,k])
        spearmanLFC18[i,j], spearmanLFC18_Pval[i,j] = scipy.stats.spearmanr(LFC18_samples[i,:,LFC18_gtIdx[0]],LFC18_samples[i,:,k])

for i in range(MC24_numSamples):
    for j,k in enumerate(MC24_featureIdx): # Calculate correlation coefficients
        pearsonMC24[i,j], pearsonMC24_Pval[i,j] = scipy.stats.pearsonr(MC24_samples[i,:,MC24_gtIdx[0]],MC24_samples[i,:,k])
        spearmanMC24[i,j], spearmanMC24_Pval[i,j] = scipy.stats.spearmanr(MC24_samples[i,:,MC24_gtIdx[0]],MC24_samples[i,:,k])

for i in range(MC24_numSamples):
    for j,k in enumerate(MC24_featureIdx): # Calculate correlation coefficients
        pearsonMC24_constVf[i,j], pearsonMC24_Pval_constVf[i,j] = scipy.stats.pearsonr(MC24_samples_constVf[i,:,MC24_gtIdx[0]],MC24_samples_constVf[i,:,k])
        spearmanMC24_constVf[i,j], spearmanMC24_Pval_constVf[i,j] = scipy.stats.spearmanr(MC24_samples_constVf[i,:,MC24_gtIdx[0]],MC24_samples_constVf[i,:,k])


# Format in dataframes
pearsonLFC18Df = pd.DataFrame(pearsonLFC18, columns=LFC18_headers[LFC18_featureIdx])
pearsonLFC18Df['Dataset'] = 'LFC18'
pearsonMC24Df = pd.DataFrame(pearsonMC24, columns=MC24_headers[MC24_featureIdx])
pearsonMC24Df['Dataset'] = 'MC24'
pearsonMC24Df_constVf = pd.DataFrame(pearsonMC24_constVf, columns=MC24_headers[MC24_featureIdx])
pearsonMC24Df_constVf['Dataset'] = 'MC24_constVf'
pearsonDf = pd.concat([pearsonLFC18Df, pearsonMC24Df])
pearsonDf = pd.melt(pearsonDf, id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])
pearsonDf_constVf = pd.concat([pearsonMC24Df, pearsonMC24Df_constVf])
pearsonDf_constVf = pd.melt(pearsonDf_constVf, id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])


spearmanLFC18Df = pd.DataFrame(spearmanLFC18, columns=LFC18_headers[LFC18_featureIdx])
spearmanLFC18Df['Dataset'] = 'LFC18'
spearmanMC24Df = pd.DataFrame(spearmanMC24, columns=MC24_headers[MC24_featureIdx])
spearmanMC24Df['Dataset'] = 'MC24'
spearmanMC24Df_constVf = pd.DataFrame(spearmanMC24_constVf, columns=MC24_headers[MC24_featureIdx])
spearmanMC24Df_constVf['Dataset'] = 'MC24_constVf'
spearmanDf = pd.concat([spearmanLFC18Df, spearmanMC24Df])
spearmanDf = pd.melt(spearmanDf, id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])
spearmanDf_constVf  = pd.concat([spearmanMC24Df, spearmanMC24Df_constVf ])
spearmanDf_constVf  = pd.melt(spearmanDf_constVf , id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])


# %% Visualisation with violin plots
fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
# Plot correlation coefficients
ax = plt.subplot(1,2,1)
plt.grid()
g = sns.violinplot(ax=ax, data=pearsonDf,x = 'variable',y = 'value',hue="Dataset", split=True, gap=.2, inner="quart",linewidth=1,saturation = 1)
plt.title('Pearson correlation with failure index')
plt.ylabel('Correlation coefficient')
plt.xlabel('')
g.set_xticks(range(len(pearsonMC24[0])))
g.set_xticklabels(MC24_headers[MC24_featureIdx])
# ax.get_legend().remove()
for collection in ax.collections:
    if isinstance(collection, matplotlib.collections.PolyCollection):
        collection.set_edgecolor(matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=1))
        collection.set_facecolor(matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=0.5))
        edgecol = matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=1)
        facecol = matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=0.5)
for han in ax.get_legend_handles_labels():
    for hand in han:
        if isinstance(hand, matplotlib.patches.Rectangle):
            hand.set_edgecolor(hand.get_facecolor())
            hand.set_facecolor(matplotlib.colors.to_rgba(hand.get_facecolor()[:-1], alpha=0.5))
            # h.set_linewidth(1.5)

h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()


ax = plt.subplot(1,2,2)
plt.grid()
g = sns.violinplot(ax=ax, data=spearmanDf,x = 'variable',y = 'value',hue="Dataset", split=True, gap=.2, inner="quart",linewidth=1,saturation = 1)
plt.title("Spearman's correlation with failure index")
plt.ylabel('')
plt.xlabel('')
g.set_xticks(range(len(spearmanMC24[0])))
g.set_xticklabels(MC24_headers[MC24_featureIdx])
# h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()
# children = plt.gca().get_children()
for collection in ax.collections:
    if isinstance(collection, matplotlib.collections.PolyCollection):
        collection.set_edgecolor(matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=1))
        collection.set_facecolor(matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=0.5))
        edgecol = matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=1)
        facecol = matplotlib.colors.to_rgba(collection.get_facecolor()[0][:-1], alpha=0.5)

fig.legend(title='Dataset',handles = h,labels=l, 
           loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.15))


# plt.savefig('Correlation.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
# plt.show()


# %% Visualisation with density plot

plt.rcParams["font.size"] = "8"
def densPlot(df, features, groundTruth):
    pearsonLFC18 = np.zeros(len(LFC18_featureIdx)) # Pearson correlation with Failure index
    pearsonLFC18_Pval = np.copy(pearsonLFC18)
    pearsonLFC18_conf = np.zeros((len(LFC18_featureIdx),2))
    pearsonMC24 = np.zeros(len(MC24_featureIdx)) # Pearson correlation with Failure index
    pearsonMC24_Pval = np.copy(pearsonMC24)
    pearsonMC24_conf = np.zeros((len(MC24_featureIdx),2))

    for f in range(len(features)):
        if not (features[f] == 'Vf' or features[f] == 'c2'):
            res = scipy.stats.pearsonr(df[df['Dataset']=='LFC18'][groundTruth[0]],df[df['Dataset']=='LFC18'][features[f]])
            pearsonLFC18[f] = res.statistic
            pearsonLFC18_Pval[f] = res.pvalue
            pearsonLFC18_conf[f,:] = res.confidence_interval(confidence_level = 0.95)
            pearsonLFC18_conf[f,:] = np.abs(pearsonLFC18_conf[f,:]-pearsonLFC18[f])

        res = scipy.stats.pearsonr(df[df['Dataset']=='MC24'][groundTruth[0]],df[df['Dataset']=='MC24'][features[f]])
        pearsonMC24[f] = res.statistic
        pearsonMC24_Pval[f] = res.pvalue
        pearsonMC24_conf[f] = res.confidence_interval(confidence_level = 0.95)
        pearsonMC24_conf[f,:] = np.abs(pearsonMC24_conf[f,:]-pearsonMC24[f])
        
    corrDf1 = pd.DataFrame([pearsonLFC18], columns=LFC18_headers[LFC18_featureIdx])
    corrDf1['Dataset'] = 'LFC18'
    corrDf2 = pd.DataFrame([pearsonMC24], columns=MC24_headers[MC24_featureIdx])
    corrDf2['Dataset'] = 'MC24'
    corrDf = pd.concat([corrDf1, corrDf2])
    corrDf = pd.melt(corrDf, id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])

    print(pearsonMC24)
    print(pearsonLFC18)
    print(pearsonMC24_Pval)
    print(pearsonLFC18_Pval)
    print(pearsonLFC18_conf)



    fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
    # fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
    # fig.set_figheight(figHeight)
    # fig.set_figwidth(figWidth)
    numplts = len(features)
    palette = sns.color_palette("colorblind")
    dims = [2,int(np.ceil(numplts/2))]
    for i in range(numplts):
        ax = plt.subplot(dims[0],dims[1],i+1)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # sns.histplot(df, x = features[i], kde = True, kde_kws = {"bw_adjust": 2}, hue = 'Dataset',element = 'step',linewidth=0,alpha=0.6)
        if not (features[i] == 'Vf' or features[i] == 'c2'):
            sns.histplot(df, x = groundTruth[0], y = features[i],thresh=1, hue = 'Dataset',binwidth = [np.max(df[groundTruth[0]])/32,np.max(df[features[i]])/32],palette = sns.color_palette('colorblind'))
            # sns.kdeplot(df, x = groundTruth[0], y = features[i], hue = 'Dataset',binwidth = [np.max(df[groundTruth[0]])/40,np.max(df[features[i]])/40])
            # random_indices = np.random.choice(len(df[df['Dataset']=='LFC18'][groundTruth[0]]), size=1000, replace=False)

            # xdat = np.array(df[df['Dataset']=='LFC18'][groundTruth[0]])[random_indices]
            # ydat = np.array(df[df['Dataset']=='LFC18'][features[i]])[random_indices]
            # sns.scatterplot(
            #     x=xdat, 
            #     y=ydat, 
            #     s=1, 
            #     color=palette[0], 
            #     # label="Data Points"
            # )
            # hist = sns.histplot(
            #     x=data['Ground Truth'], 
            #     y=data['Prediction'], 
            #     bins=100, 
            #     cbar=True, 
            #     pthresh=0.1,
            #     cbar_kws={'label': 'Density'}, 
            #     cmap="viridis",
            #     label="Point concentration"
            # )


            cmapOrange = ax.collections[1].get_cmap()
            lgdnhand =  ax.get_legend().legend_handles.copy()
            # for collection in ax.collections:
            #     print(collection)
            #     if isinstance(collection, matplotlib.collections.QuadMesh):
            #         facCol = ax.collections[1].get_cmap()
        else:
            sns.histplot(df[df['Dataset']=='MC24'],thresh=1,pthresh=0.00, x = groundTruth[0], y = features[i], hue = 'Dataset', binwidth = [np.max(df[df['Dataset']=='MC24'][groundTruth[0]])/32,np.max(df[df['Dataset']=='MC24'][features[i]])/32],palette = sns.color_palette('colorblind'))
            ax.collections[0].set_cmap(cmapOrange)
            # for han in ax.get_legend_handles_labels():
            #     for hand in han:
            #         if isinstance(hand, matplotlib.patches.Rectangle):
            #             hand.set_facecolor(cmapOrange)
        ax.get_legend().remove()
        plt.grid()
        
        plt.xlabel('$F$', labelpad=-0.5)
        lab = None
        match features[i]:
            case 'Ex':
                lab = '$E_{\mathrm{x}}$'
            case 'Ey':
                lab = '$E_{\mathrm{y}}$'
            case 'Gxy':
                lab = '$G_{\mathrm{xy}}$'
            case 'Vf':
                lab = '$V_{\mathrm{f}}$'
            case 'c2':
                lab = '$c_{2}$'
            case 'FI':
                lab = '$F$'
        # ax.set_title(lab)
        plt.ylabel(lab, labelpad=-0.5)
        ax.set_title(f"({chr(97 + i)})")
        # plt.title(features[i])
    palette = sns.color_palette("colorblind")
    handles = [
    Patch(facecolor=matplotlib.colors.to_rgba(palette[0], alpha=0.4), edgecolor=palette[0]),  # MeC-Macro (left)
    Patch(facecolor=matplotlib.colors.to_rgba(palette[1], alpha=0.4), edgecolor=palette[1])   # MeC-Meso-S  (right)
    ]
    fig.legend(title='Dataset', handles=handles, labels=['MeC-Macro','MeC-Meso-S/M/L'],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    ax6 = plt.subplot(dims[0],dims[1],6) # For plotting correlation coeffs
    bp = sns.barplot(data=corrDf,x = 'variable',y = 'value', hue = 'Dataset', palette={"LFC18": palette[0], "MC24": palette[1]})
    ticklabels = []
    for ticklab in ax6.get_xticklabels():
        match ticklab.get_text():
            case 'Ex':
                lab = '$E_{\mathrm{x}}$'
            case 'Ey':
                lab = '$E_{\mathrm{y}}$'
            case 'Gxy':
                lab = '$G_{\mathrm{xy}}$'
            case 'Vf':
                lab = '$V_{\mathrm{f}}$'
            case 'c2':
                lab = '$c_{2}$'
            case 'FI':
                lab = '$F$'
        ticklabels.append(lab)
    ax6.set_xticklabels(ticklabels)
        # ax.set_title(lab)
    
    plt.grid()
    plt.xlabel('')
    plt.ylabel('Correlation', labelpad=-2)
    plt.title('(f)')
    ax6.get_legend().remove()

# All data points
densPlot(AllDf, features = MC24_headers[MC24_featureIdx], groundTruth = MC24_headers[MC24_gtIdx])
plt.tight_layout()
plt.savefig('C:\\Users\\kaspe\\OneDrive\\UNIVERSITY\\YEAR 4\\Individual Project\\Paper\\Figures\\CorrelationScatter.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()

# %% Plot results from HP optimisation in single barchart
# resToPlot = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_metrics.csv")
# resToPlot = pd.read_excel(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results\Pre and post optimisation results all dataset.xlsx")
resToPlot = pd.read_excel(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results\Pre and post optimisation results all dataset.xlsx")

# Reshape the data into long form
data_long = pd.melt(
    resToPlot,
    id_vars=["Dataset", "Stage"],
    value_vars=[
        "RMSE Train", "RMSE Train std", "RMSE Val", "RMSE Val std",
        "NRMSE Train", "NRMSE Train std", "NRMSE Val", "NRMSE Val std",
        "SSIM Train", "SSIM Train std", "SSIM Val", "SSIM Val std"
    ],
    var_name="Metric_Stage",
    value_name="Value"
)

# Extract 'Metric' and 'Type' (Mean or Std) from 'Metric_Stage'
data_long["Metric"] = data_long["Metric_Stage"].apply(lambda x: x.split(" ")[0])
data_long["Type"] = data_long["Metric_Stage"].apply(lambda x: "std" if "std" in x else "mean")
data_long["Data"] = data_long["Metric_Stage"].apply(lambda x: "train" if "Train" in x else "val")

# Pivot the data to separate 'mean' and 'std' into columns
data_long = data_long.pivot_table(
    index=["Dataset", "Stage", "Metric", "Data"],
    columns="Type",
    values="Value",
    aggfunc="first"
).reset_index()

# Rename columns for clarity
data_long = data_long.rename(columns={"mean": "Mean", "std": "Std"})
# Add rows for "Optimised" values in the MeC-Meso-S dataset and set Mean and Std to NaN
# for metric in ["NRMSE", "RMSE", "SSIM"]:
#     data_long = pd.concat([
#         data_long,
#         pd.DataFrame({
#             "Dataset": ["MeC-Meso-S"],
#             "Stage": ["Optimised"],
#             "Metric": [metric],
#             "Data": ["train"],
#             "Mean": [np.nan],
#             "Std": [np.nan]
#         })
#     ], ignore_index=True)
#     data_long = pd.concat([
#         data_long,
#         pd.DataFrame({
#             "Dataset": ["MeC-Meso-S"],
#             "Stage": ["Optimised"],
#             "Metric": [metric],
#             "Data": ["val"],
#             "Mean": [np.nan],
#             "Std": [np.nan]
#         })
#     ], ignore_index=True)


RMSEs = data_long[data_long['Metric'] == 'RMSE']
NRMSEs = data_long[data_long['Metric'] == 'NRMSE']
SSIMs = data_long[data_long['Metric'] == 'SSIM']

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

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "8"
latexWidth = 315
figscale = 1
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = 3 * figWidth/1.618 # Golden ratio
# figHeight = figWidth*Ratio*(2/4) # 2 subplots high, 4 subplots wide 
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="tight", dpi = resolution_scaling*100) # 100 is default size
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
# Create a barplot with error bars
# plt.figure(figsize=(10, 9))
# plt.subplot(3, 1, 1)

# ax = sns.catplot(
#     data=RMSEs,
#     kind="bar",
#     x="Stage", 
#     y="Mean", 
#     col="Dataset",
#     hue='Data',
#     hue_order=['train', 'val'],  # Specify the order of the hue categories
#     palette=RMSE_palette,
#     # height=4, aspect=.5,
#     )

dataset_list = ['MeC-Macro', 'MeC-Meso-S', 'MeC-Meso-M']
modelList = ['TBDCNet-Macro', 'TBDCNet-Meso-S', 'TBDCNet-Meso-M']
all_stages = ['Baseline', 'Optimised']  
fig, axs = plt.subplots(3, 3, sharey='row')  # Share y-axis within each row

# fig.set_figheight(figHeight)
# fig.set_figwidth(figWidth)
for plt_idx in range(3):
    ax = axs[0,plt_idx]
    sns.barplot(
        data=RMSEs[RMSEs['Dataset'] == dataset_list[plt_idx]],
        x="Stage", 
        y="Mean", 
        hue='Data',
        hue_order=['train', 'val'],  # Specify the order of the hue categories
        palette=RMSE_palette,
        ax=ax,
        # order=['Baseline', 'Optimised']  # Ensure Baseline is always on the left
    )
    ax.set_xlim(-0.5, len(all_stages) - 0.5) # Force space for the optimised bar even if missing
    # x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:4]]
    # y_coords = [p.get_height() for p in ax.patches[:4]]
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:-2]]
    y_coords = [p.get_height() for p in ax.patches[:-2]]
    errorvals = RMSEs[RMSEs['Dataset'] == dataset_list[plt_idx]]['Std'].dropna()
    ax.errorbar(x=x_coords, y=y_coords, yerr=errorvals, fmt="none", c="k")

    ax.set_ylabel('RMSE')
    ax.set_xlabel('')
    # ax.set_title(dataset_list[plt_idx], fontsize=10)
    # ax.set_title(f"({chr(97 + plt_idx)})", fontweight='bold')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_ylim(np.min(RMSEs['Mean']*0.8), np.max(RMSEs['Mean']*1.1)) 
    ax.tick_params(axis='y')
    ax.grid(axis='y', linestyle='--')
    ax.set_xticklabels(['Baseline\nCNN', modelList[plt_idx]])
    if plt_idx == 1:
        ax.legend(title='Data')
    else:
        ax.get_legend().remove()

for plt_idx in range(3):
    ax = axs[1,plt_idx]
    sns.barplot(
        data=NRMSEs[NRMSEs['Dataset'] == dataset_list[plt_idx]],
        x="Stage", 
        y="Mean", 
        hue='Data',
        hue_order=['train', 'val'],  # Specify the order of the hue categories
        palette=NRMSE_palette,
        ax=ax
    )
    ax.set_xlim(-0.5, len(all_stages) - 0.5) # Force space for the optimised bar even if missing
    # x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:4]]
    # y_coords = [p.get_height() for p in ax.patches[:4]]
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:-2]]
    y_coords = [p.get_height() for p in ax.patches[:-2]]
    errorvals = NRMSEs[NRMSEs['Dataset'] == dataset_list[plt_idx]]['Std'].dropna()
    ax.errorbar(x=x_coords, y=y_coords, yerr=errorvals, fmt="none", c="k")

    ax.set_ylabel('NRMSE')
    ax.set_xlabel('')
    # ax.set_title(dataset_list[plt_idx], fontsize=6)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_ylim(np.min(NRMSEs['Mean']*0.8), np.max(NRMSEs['Mean']*1.15)) 
    ax.tick_params(axis='y')
    ax.grid(axis='y', linestyle='--')
    ax.set_xticklabels(['Baseline\nCNN', modelList[plt_idx]])
    if plt_idx == 1:
        ax.legend(title='Data')
    else:
        ax.get_legend().remove()

for plt_idx in range(3):
    ax = axs[2,plt_idx]
    sns.barplot(
        data=SSIMs[SSIMs['Dataset'] == dataset_list[plt_idx]],
        x="Stage", 
        y="Mean", 
        hue='Data',
        hue_order=['train', 'val'],  # Specify the order of the hue categories
        palette=SSIM_palette,
        ax=ax
    )
    ax.set_xlim(-0.5, len(all_stages) - 0.5) # Force space for the optimised bar even if missing
    # x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:4]]
    # y_coords = [p.get_height() for p in ax.patches[:4]]
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:-2]]
    y_coords = [p.get_height() for p in ax.patches[:-2]]
    errorvals = SSIMs[SSIMs['Dataset'] == dataset_list[plt_idx]]['Std'].dropna()
    ax.errorbar(x=x_coords, y=y_coords, yerr=errorvals, fmt="none", c="k")

    ax.set_ylabel('SSIM')
    ax.set_xlabel('')
    # ax.set_title(dataset_list[plt_idx], fontsize=6)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_ylim(np.min(SSIMs['Mean']*0.95), 1) 
    ax.tick_params(axis='y')
    ax.grid(axis='y', linestyle='--')
    ax.set_xticklabels(['Baseline\nCNN', modelList[plt_idx]])
    if plt_idx == 1:
        ax.legend(title='Data')
    else:
        ax.get_legend().remove()


plt.tight_layout()
# plt.show()
# Save the figure as a high-resolution PDF
# output_pdf_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\PrePost_HPOpti_results.pdf"
# output_pdf_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\PrePost_HPOpti_results_v2.pdf"
output_pdf_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\PrePost_HPOpti_results_v3.pdf"
plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved to {output_pdf_path}")
plt.show()

# ax = sns.barplot(
#     # data=RMSEs[RMSEs['Data']=='test'],
#     data=RMSEs,
#     x='type',
#     y='Mean',
#     hue='Data',
#     hue_order=['train', 'val', 'test'],  # Specify the order of the hue categories
#     palette=RMSE_palette
# )

# x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
# y_coords = [p.get_height() for p in ax.patches]



# plt.subplot(2, 1, 2)
# ax2 = sns.barplot(
#     data=SSIMs,
#     x='type',
#     y='Mean',
#     hue='Data',
#     hue_order=['train', 'val', 'test'],  # Specify the order of the hue categories
#     palette=SSIM_palette
# )
# x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax2.patches[:-3]]
# y_coords = [p.get_height() for p in ax2.patches[:-3]]
# # ax.errorbar(x=x_coords, y=y_coords, yerr=RMSEs[RMSEs['Data']=='test']['Std'], fmt="none", c="k")
# ax2.errorbar(x=x_coords, y=y_coords, yerr=SSIMs['Std'], fmt="none", c="k")
# ax2.set_ylim(0.5, None)  # Start y-axis at 0
# ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90, fontsize=12, ha='center',va = 'top')
# ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=10)
# ax2.set_ylabel('SSIM', fontsize=12)
# ax2.set_xlabel('', fontsize=12)
# ax2.legend(
#     title='Data',
#     loc='upper left',
#     # bbox_to_anchor=(1, 1),
#     fontsize=10,
#     title_fontsize=10
# )

# # ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=10)
# ax.set_xticklabels([''] * len(ax.get_xticks()), fontsize=1)  # Remove xtick labels on ax
# ax2.grid(axis='y', linestyle='--')
# plt.tight_layout()

# # Save the figure as a high-resolution PDF
# output_pdf_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\high_resolution_figure.pdf"
# plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
# print(f"Figure saved to {output_pdf_path}")

# plt.show()

# scatPlot(AllDf, features = MC24_headers[-2:], groundTruth = MC24_headers[MC24_gtIdx])
# %% Visualisation with density plot of constant Vf


def densPlot(df, features, groundTruth):
    pearsonMC24 = np.zeros(len(MC24_featureIdx)) # Pearson correlation with Failure index
    pearsonMC24_Pval = np.copy(pearsonMC24)
    pearsonMC24_conf = np.zeros((len(MC24_featureIdx),2))
    pearsonMC24_constVf = np.zeros(len(MC24_featureIdx)) # Pearson correlation with Failure index
    pearsonMC24_Pval_constVf = np.copy(pearsonMC24_constVf)
    pearsonMC24_conf_constVf = np.zeros((len(MC24_featureIdx),2))
    for f in range(len(features)):
        res = scipy.stats.pearsonr(df[groundTruth[0]],df[features[f]])
        pearsonMC24[f] = res.statistic
        pearsonMC24_Pval[f] = res.pvalue
        pearsonMC24_conf[f] = res.confidence_interval(confidence_level = 0.95)
        pearsonMC24_conf[f,:] = np.abs(pearsonMC24_conf[f,:]-pearsonMC24[f])

    corrDf2 = pd.DataFrame([pearsonMC24], columns=MC24_headers[MC24_featureIdx])
    corrDf2['Dataset'] = 'MC24'
    corrDf = corrDf2
    corrDf = pd.melt(corrDf, id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])

    # print(pearsonMC24)
    # print(pearsonLFC18)
    # print(pearsonMC24_Pval)
    # print(pearsonLFC18_Pval)
    # print(pearsonLFC18_conf)



    fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
    # fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
    fig.set_figheight(figHeight)
    fig.set_figwidth(figWidth)
    numplts = len(features)
    dims = [2,int(np.ceil(numplts/2))]
    for i in range(numplts):
        ax = plt.subplot(dims[0],dims[1],i+1)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # sns.histplot(df, x = features[i], kde = True, kde_kws = {"bw_adjust": 2}, hue = 'Dataset',element = 'step',linewidth=0,alpha=0.6)
        sns.histplot(df, x = groundTruth[0], y = features[i], hue = 'Dataset',binwidth = [np.max(df[groundTruth[0]])/30,np.max(df[features[i]])/30])
        # cmapOrange = ax.collections[1].get_cmap()
        # lgdnhand =  ax.get_legend().legend_handles
        ax.get_legend().remove()
        plt.grid()
        plt.xlabel(groundTruth[0], labelpad=-0.5)
        plt.ylabel(features[i], labelpad=-0.5)
        plt.title(features[i])

    # fig.legend(title = 'Dataset',handles = lgdnhand, labels=np.flip(df['Dataset'].unique()),
    #        loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.15))
    ax6 = plt.subplot(dims[0],dims[1],6) # For plotting correlation coeffs
    bp = sns.barplot(data=corrDf,x = 'variable',y = 'value', hue = 'Dataset')
    plt.grid()
    plt.xlabel('')
    plt.ylabel('Correlation', labelpad=-2)
    plt.title('Pearson correlation with FI')
    ax6.get_legend().remove()

# Plot corr against each other
pearsonMC24 = np.zeros(len(MC24_featureIdx)) # Pearson correlation with Failure index
pearsonMC24_Pval = np.copy(pearsonMC24)
pearsonMC24_conf = np.zeros((len(MC24_featureIdx),2))
pearsonMC24_constVf = np.zeros(len(MC24_featureIdx)) # Pearson correlation with Failure index
pearsonMC24_Pval_constVf = np.copy(pearsonMC24_constVf)
pearsonMC24_conf_constVf = np.zeros((len(MC24_featureIdx),2))
for f in range(len(MC24_headers[MC24_featureIdx])):
    res = scipy.stats.pearsonr(VfVariationDf[VfVariationDf['Dataset']=='MC24'][MC24_headers[MC24_gtIdx][0]],VfVariationDf[VfVariationDf['Dataset']=='MC24'][MC24_headers[MC24_featureIdx][f]])
    pearsonMC24[f] = res.statistic
    pearsonMC24_Pval[f] = res.pvalue
    pearsonMC24_conf[f] = res.confidence_interval(confidence_level = 0.95)
    pearsonMC24_conf[f,:] = np.abs(pearsonMC24_conf[f,:]-pearsonMC24[f])

for f in range(len(MC24_headers[MC24_featureIdx])):
    res = scipy.stats.pearsonr(VfVariationDf[VfVariationDf['Dataset']=='MC24_constVf'][MC24_headers[MC24_gtIdx][0]],VfVariationDf[VfVariationDf['Dataset']=='MC24_constVf'][MC24_headers[MC24_featureIdx][f]])
    pearsonMC24_constVf[f] = res.statistic
    pearsonMC24_Pval_constVf[f] = res.pvalue
    pearsonMC24_conf_constVf[f] = res.confidence_interval(confidence_level = 0.95)
    pearsonMC24_conf_constVf[f,:] = np.abs(pearsonMC24_conf[f,:]-pearsonMC24[f])


corrDf2 = pd.DataFrame([pearsonMC24], columns=MC24_headers[MC24_featureIdx])
corrDf2['Dataset'] = 'MC24'
corrDf3 = pd.DataFrame([pearsonMC24_constVf], columns=MC24_headers[MC24_featureIdx])
corrDf3['Dataset'] = 'MC24_ConstVf'

corrDf = pd.concat([corrDf2,corrDf3])
corrDf = pd.melt(corrDf, id_vars=['Dataset'], value_vars=MC24_headers[MC24_featureIdx])
plt.style.use("seaborn-v0_8-colorblind")

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "10"
latexWidth = 315
figWidth = latexWidth*px
# Ratio = (138/50)# Specimen ratio
figHeight = 1.1*figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
ax = plt.subplot(1,1,1)
bp = sns.barplot(data=corrDf,x = 'variable',y = 'value', hue = 'Dataset')
plt.grid()
plt.xlabel('')
plt.ylabel('Correlation', labelpad=-2)
# plt.title('Pearson correlation with FI')
h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()
# ax.set_yscale('log')
fig.legend(title='Dataset',handles = h,labels=l, 
           loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.3))


plt.savefig('CorrelationScatter_ConstVf.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()


# densPlot(AllDf, features = MC24_headers[MC24_featureIdx], groundTruth = MC24_headers[MC24_gtIdx])
# plt.savefig('CorrelationScatter.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
densPlot(VfVariationDf[VfVariationDf['Dataset']=='MC24'], features = MC24_headers[MC24_featureIdx], groundTruth = MC24_headers[MC24_gtIdx])
plt.savefig('CorrelationScatter_VariableVf.pdf', bbox_inches='tight', pad_inches = 0)
densPlot(VfVariationDf[VfVariationDf['Dataset']=='MC24_constVf'], features = MC24_headers[MC24_featureIdx], groundTruth = MC24_headers[MC24_gtIdx])
plt.savefig('CorrelationScatter_ConstVf.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()
#%% PCA
from sklearn.decomposition import PCA

# First 2 a 2D PCA of both datasets
pcaLFC18 = PCA(n_components=2) # number of components is maximum the dimensionality of the data
pcaMC24 = PCA(n_components=2) # number of components is maximum the dimensionality of the data
# pca.fit(samples[0,:,3:])
LFC18_PC = pcaLFC18.fit_transform(LFC18_samples[:,:,LFC18_featureIdx].reshape(-1, len(LFC18_featureIdx)))
MC24_PC = pcaMC24.fit_transform(MC24_samples[:,:,MC24_featureIdx].reshape(-1, len(MC24_featureIdx)))
# components = pca.components_ # has the vector components for the principal components


plotIdxLFC18 = random.sample(range(LFC18_PC.shape[0]),3000)
plotIdxMC24 = random.sample(range(MC24_PC.shape[0]),3000)

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
ax = plt.subplot(1,2,1)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
CS = plt.scatter(x = LFC18_PC[plotIdxLFC18,0],y = LFC18_PC[plotIdxLFC18,1], c=LFC18_samples.reshape(LFC18_PC.shape[0],-1)[plotIdxLFC18,LFC18_gtIdx], alpha = 0.75,s = 2)
plt.title('First 2 principal components of LFC18')
plt.xlabel('PC1')
plt.ylabel('PC2')
fig.colorbar(CS, label = 'Failure index')
plt.grid()

ax = plt.subplot(1,2,2)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
CS = plt.scatter(x = MC24_PC[plotIdxLFC18,0],y = MC24_PC[plotIdxLFC18,1], c=MC24_samples.reshape(MC24_PC.shape[0],-1)[plotIdxMC24,MC24_gtIdx], alpha = 0.75,s = 2)
plt.title('First 2 principal components of MC24')
plt.xlabel('PC1')
plt.ylabel('PC2')
fig.colorbar(CS, label = 'Failure index')
plt.grid()


# %% 3D PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
# First 2 a 2D PCA of both datasets
pcaLFC18 = PCA(n_components=3) # number of components is maximum the dimensionality of the data
pcaMC24 = PCA(n_components=3) # number of components is maximum the dimensionality of the data
# pca.fit(samples[0,:,3:])
LFC18_PC = pcaLFC18.fit_transform(LFC18_samples[:,:,LFC18_featureIdx].reshape(-1, len(LFC18_featureIdx)))
MC24_PC = pcaMC24.fit_transform(MC24_samples[:,:,MC24_featureIdx].reshape(-1, len(MC24_featureIdx)))
# components = pca.components_ # has the vector components for the principal components
plotIdxLFC18 = random.sample(range(LFC18_PC.shape[0]),2000)
plotIdxMC24 = random.sample(range(MC24_PC.shape[0]),2000)


# marker_data = go.Scatter3d(
#     x=LFC18_PC[plotIdxLFC18,0],
#     y=LFC18_PC[plotIdxLFC18,1],
#     z=LFC18_PC[plotIdxLFC18,2],
#     marker=go.scatter3d.Marker(size=3, showscale=True,colorscale='Viridis'),
#     marker_color=LFC18_samples.reshape(LFC18_PC.shape[0],-1)[plotIdxLFC18,LFC18_gtIdx],
#     opacity=0.75,
#     mode='markers'
# )


# fig=go.Figure(data=marker_data)
# fig.update_layout(scene = dict(
#                     xaxis_title='PC1',
#                     yaxis_title='PC2',
#                     zaxis_title='PC3'),
#                     width=700,
#                     margin=dict(r=20, b=10, l=10, t=10))

# fig.layout.coloraxis.colorbar.title = 'Failure Index'
# fig.write_image('PCA3D.pdf')
# # plt.savefig('PCA3D.pdf', bbox_inches='tight', pad_inches = 0)
# fig.show()

fig = plt.figure(layout="tight", dpi = resolution_scaling*100) # 100 is default size
# fig = plt.figure(dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
ax = fig.add_subplot(1,2,1, projection='3d')
# ax = plt.subplot(1,2,1, projection='3d')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
CS = ax.scatter(LFC18_PC[plotIdxLFC18,0],LFC18_PC[plotIdxLFC18,1],LFC18_PC[plotIdxLFC18,2], c=LFC18_samples.reshape(LFC18_PC.shape[0],-1)[plotIdxLFC18,LFC18_gtIdx], alpha = 0.75,s = 2)
# CS = plt.scatter(LFC18_PC[plotIdxLFC18,0],LFC18_PC[plotIdxLFC18,1],LFC18_PC[plotIdxLFC18,2], c=LFC18_samples.reshape(LFC18_PC.shape[0],-1)[plotIdxLFC18,LFC18_gtIdx], alpha = 0.75,s = 2)
plt.title('First 3 principal components of LFC18')
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_zlabel('PC3')
ax.zaxis.set_ticks_position('lower')
ax.zaxis.set_label_position('lower')
fig.colorbar(CS, label = 'Failure index', shrink=0.4, location='bottom',pad=0.2)
plt.grid()

# ax = plt.subplot(1,2,2, projection='3d')
ax = fig.add_subplot(1,2,2, projection='3d')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
CS = ax.scatter(MC24_PC[plotIdxLFC18,0],MC24_PC[plotIdxLFC18,1],MC24_PC[plotIdxLFC18,2], c=MC24_samples.reshape(MC24_PC.shape[0],-1)[plotIdxMC24,MC24_gtIdx], alpha = 0.75,s = 2)
plt.title('First 3 principal components of MC24')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.xaxis.set_label_position('upper')
ax.yaxis.set_ticks_position('lower')
ax.xaxis.set_ticks_position('upper')

fig.colorbar(CS, label = 'Failure index', shrink=0.4, location='bottom',pad=0.2)
plt.grid()


# plt.savefig('PCA3D.pdf', bbox_inches='tight', pad_inches = 0)
# plt.show()

# %% Number of PC to capture variance

nums18 = np.arange(len(LFC18_featureIdx)) # Number of principal components 1 to 10
nums24 = np.arange(len(MC24_featureIdx))
var_ratio_LFC18 = []
var_ratio_MC24 = []

for num in nums18:
  pcatest_LFC18 = PCA(n_components=num)
  pcatest_LFC18.fit(LFC18_samples[:,:,LFC18_featureIdx].reshape(-1, len(LFC18_featureIdx)))
  var_ratio_LFC18.append(np.sum(pcatest_LFC18.explained_variance_ratio_))
  
for num in nums24:
  pcatest_MC24 = PCA(n_components=num)
  pcatest_MC24.fit(MC24_samples[:,:,MC24_featureIdx].reshape(-1, len(MC24_featureIdx)))
  var_ratio_MC24.append(np.sum(pcatest_MC24.explained_variance_ratio_))


plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figscale = 1.5
figWidth = latexWidth*px*figscale
Ratio = (138/50)# Specimen ratio
figHeight = figWidth/1.618 # Golden ratio
fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)
# Plot correlation coefficients
ax = plt.subplot(1,1,1)
plt.grid()
# plt.plot(nums18,var_ratio_LFC18,marker='o', label='LFC18')
# plt.plot(nums24,var_ratio_MC24,marker='o', label='MC24')
g = sns.lineplot(x = nums18,y = var_ratio_LFC18,marker='o', label='LFC18',markeredgecolor=None)
h = sns.lineplot(x = nums24,y = var_ratio_MC24,marker='s', label='MC24',markeredgecolor=None)
# g.set_markeredgecolor('black')
# h.set_markeredgecolor('black')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance ratio')
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.legend(title = 'Dataset',ncol=2,loc="lower center",bbox_to_anchor=(0.5, -0.35))
# lgdnhand =  ax.get_legend().legend_handles
# fig.legend(title = 'Dataset',handles = lgdnhand, labels=['LFC18','MC24'],
#            loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.15))

plt.savefig('CapturedVariance.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()
# %% Optimizer comparison

# Generate 2D function to minimize value of with optimizer


def adam(lr = 0.5,b_1 = 0.9,b_2 = 0.999,e = 10**(-7),steps = 1000):
    t = [0]
    m = [0]
    v = [0]
    

    param = np.array([8])

    def adam_func(param):
        f = np.tanh(param)**2+param**2/500+(np.tanh(param-5)**2)/10 + random.normalvariate(-0.025,0.025)
        return f

    def adam_grad(param):
        delta = 0.01
        over = adam_func(param+delta)
        under = adam_func(param-delta)
        grad = (over-under)/2*delta
        # grad = param/250 + 2*np.tanh(param)*1/(np.cosh(param))**2 + np.tanh(param-5)*1/(np.cosh(param-5))**2
        return grad


    for i in range(steps-1):
        t += [t[-1]+1]
        alpha = lr*np.sqrt(1-b_2**t[-1])/(1-b_1**t[-1])
        if i == 0:
            g = [adam_grad(param[-1])]
        else:
            g += [adam_grad(param[-1])]

        m += [m[-1]+((g[-1]-m[-1])*(1-b_1))]
        v += [v[-1]+((g[-1]**2-v[-1])*(1-b_2))]

        # m += [b_1*m[-1] + (1-b_1)*g[-1]]
        # v += [b_2*v[-1] + (1-b_2)*g[-1]*g[-1]]

        if i == 0:
            mhat = [m[-1]/(1-b_1**t[-1])]
            vhat = [v[-1]/(1-b_2**t[-1])]
        else:
            mhat += [m[-1]/(1-b_1**t[-1])]
            vhat += [v[-1]/(1-b_2**t[-1])]

        # param = np.append(param,param[-1] - alpha * mhat[-1]/(np.sqrt(vhat[-1]) + e))
        param = np.append(param,param[-1] - alpha * m[-1]/(np.sqrt(v[-1]) + e))
    
    f = np.tanh(param)**2+(param**2)/500+(np.tanh(param-5))**2/10
    return np.array(t),f.reshape(-1), param.reshape(-1)


def NAdam(lr = 1,b_1 = 0.9,b_2 = 0.999,e = 10**(-7),steps = 1000):
    t = [0]
    m = [0]
    v = [0]
    uprod = [0]

    param = np.array([8])

    def adam_func(param):
        f = (np.tanh(param)**2)/2+param**2/500+(np.tanh(param-5)**2)/10 + random.normalvariate(-0.0075,0.0075)
        # f = np.tanh(param)**2+param**2/500+(np.tanh(param-5)**2)/10
        return f

    def adam_grad(param):
        delta = 0.01
        over = adam_func(param+delta)
        under = adam_func(param-delta)
        grad = (over-under)/2*delta
        # grad = param/250 + 2*np.tanh(param)*1/(np.cosh(param))**2 + np.tanh(param-5)*1/(np.cosh(param-5))**2
        return grad

    decay = 0.96
    for i in range(steps-1):
        t += [t[-1]+1]
        tnext = t[-1]+2
        uprod += [uprod[-1] * b_1 * (1 - 0.5 * 0.96**(t[-1]))]

        alpha = lr*np.sqrt(1-b_2**t[-1])/(1-b_1**t[-1])
        if i == 0:
            g = [adam_grad(param[-1])]
            u_t = [b_1 * (1 - 0.5 * (decay**t[-1]))]
            u_t1 = [b_1 * (1 - 0.5 * (decay**tnext))]
        else:
            g += [adam_grad(param[-1])]
            u_t += [b_1 * (1 - 0.5 * (decay**t[-1]))]
            u_t1 += [b_1 * (1 - 0.5 * (decay**tnext))]
        uprod_t1 = uprod[-1] * u_t1[-1]
       
        m += [m[-1]+((g[-1]-m[-1])*(1-b_1))]
        v += [v[-1]+((g[-1]**2-v[-1])*(1-b_2))]

        # m += [b_1*m[-1] + (1-b_1)*g[-1]]
        # v += [b_2*v[-1] + (1-b_2)*g[-1]*g[-1]]

        if i == 0:
            mhat = [((u_t[-1]*m[-1])/(1-uprod_t1)) + (((1-u_t[-1])*g[-1])/(1-uprod[-1]))]
            vhat = [v[-1]/(1-b_2**t[-1])]
        else:
            mhat += [((u_t[-1]*m[-1])/(1-uprod_t1)) + (((1-u_t[-1])*g[-1])/(1-uprod[-1]))]
            vhat += [v[-1]/(1-b_2**t[-1])]

        # param = np.append(param,param[-1] - alpha * mhat[-1]/(np.sqrt(vhat[-1]) + e))
        param = np.append(param,param[-1] - (lr * mhat[-1])/(np.sqrt(v[-1]) + e))
    
    f = (np.tanh(param)**2)/2+(param**2)/500+(np.tanh(param-5))**2/10
    return np.array(t),f.reshape(-1), param.reshape(-1)


xlin = np.linspace(-10,10,1000)
ylin = (np.tanh(xlin)**2)/2+xlin**2/500+(np.tanh(xlin-5)**2)/10

steps = 10000
t,UNUSED, UNUSED = adam(steps = steps)



fig = plt.figure(layout="constrained")
ax = plt.subplot(2,2,1)
ax.plot(xlin,ylin)
# ax.vlines([0,5],[0.3,0.3],[1.7,1.7],)
plt.title('Function to be minimised')
ax.scatter([0],[0.10008186852556392],color=['#56B4E9'])
ax.scatter([4.894894894894895],[1.04879],color='#D55E00')
ax.legend(['Function','Global minimum','Local minimum'])
plt.xlabel('x')
plt.grid()

eps_vals = [10e-7,10e-6,10e-5]
Aval = np.empty((steps,len(eps_vals)))
Aparam = np.empty((steps,len(eps_vals)))

cm_subsection = np.linspace(0, 1, len(eps_vals)) 

colors = [ matplotlib.cm.viridis(x) for x in cm_subsection ]


ax = plt.subplot(2,2,2)
for idx,eps in enumerate(eps_vals):
    UNUSED,Aval[:,idx], Aparam[:,idx] = NAdam(e = eps,steps = steps)
    ax.plot(t,Aparam[:,idx], color = colors[idx])
ax.hlines([0],[0],[steps],colors=['#56B4E9'])
ax.hlines([4.894894894894895],[0],[steps],colors='#D55E00')
ax.legend(eps_vals+['Global minimum','Local minimum'])
plt.xlim([0,steps])
plt.title('x in function to be minimized')
plt.grid()

ax = plt.subplot(2,2,3)
for idx,eps in enumerate(eps_vals):
    ax.plot(t,Aval[:,idx], color = colors[idx])
plt.title('Function value')
ax.hlines([0.10008186852556392],[0],[steps],colors=['#56B4E9'])
ax.hlines([1.04879],[0],[steps],colors='#D55E00')
plt.xlim([0,steps])
ax.legend(eps_vals+['Global minimum','Local minimum'])
plt.grid()

plt.show()

# %%
# class Arrow3D(FancyArrowPatch):

#     def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._xyz = (x, y, z)
#         self._dxdydz = (dx, dy, dz)

#     def draw(self, renderer):
#         x1, y1, z1 = self._xyz
#         dx, dy, dz = self._dxdydz
#         x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

#         xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         super().draw(renderer)
        
#     def do_3d_projection(self, renderer=None):
#         x1, y1, z1 = self._xyz
#         dx, dy, dz = self._dxdydz
#         x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

#         xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

#         return np.min(zs) 

# def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
#     '''Add an 3d arrow to an `Axes3D` instance.'''

#     arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
#     ax.add_artist(arrow)


# setattr(Axes3D, 'arrow3D', _arrow3D)

#####################################################################
# Settings
#####################################################################
# plt.style.use("seaborn-v0_8-colorblind") # Consistent colour scheme
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# r = 0.05
# u, v = np.mgrid[0:np.pi/2:30j, 0:np.pi/2:20j]
# x = np.cos(u) * np.sin(v)
# y = np.sin(u) * np.sin(v)
# z = np.cos(v)
# ax.plot_surface(x, y, z, cmap=plt.cm.viridis,alpha=0.6)

# ax.view_init(20, 20) 

# ax.arrow3D(0,0,0,
#            1,1,1,
#            mutation_scale=20,
#            arrowstyle="-|>")

# ax.set_xticks([0.5],labels = ['X'])
# ax.set_yticks([0.5],labels = ['S'])
# ax.set_zticks([0.5],labels = ['Y'])
# plt.show()
# %% RMSE comparison of selected models
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# r1 = pd.Series([0.13932267,	0.1312578,	0.13298206], name='Baseline')  # Baseline model
# r2 = pd.Series([0.12348962,	0.1269816,	0.12398586], name='Smaller batch size') # Batch size of 4
# r3 = pd.Series([0.12679584,	0.13527173,	0.12746832], name='Larger convolution window') # Kernel of size 4
# r4 = pd.Series([0.10533698,	0.11521004,	0.11750075], name='Longer training') # 4-layer model, longer training with early stopping
# r5 = pd.Series([0.112298615,0.115193091,0.112744972], name='Silu activation function') # Silu activation
# r6 = pd.Series([0.09844742,	0.10513075,	0.11276722], name = 'Final model')

# # Final model with 10-fold cross validation:
# r7 = pd.Series([0.10891952,	0.10808814,	0.10763301,	0.09643763,	0.09844742,	0.11448937,	0.10834616,	0.10191214,	0.11064257,	0.10917973,	0.10927559,	0.10266547,	0.11111066,	0.10206658,	0.10513075,	0.09653068,	0.10023911,	0.0969057,	0.11271869,	0.11323597,	0.11356502,	0.09728512,	0.09374001,	0.10241187,	0.11276722,	0.10735505,	0.11108778,	0.09843335,	0.11136408,	0.09723821], name='Final Model Crossvalidation')




# RMSEPoints = pd.concat([r1,r2,r3,r4,r5,r6,r7], axis=1)
# # RMSEPoints = pd.concat([r1,r2,r3,r4,r5,r6], axis=1)
# # print(RMSEPoints)
# px = 1/plt.rcParams['figure.dpi']  # pixel in inches
# fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
# # plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis
# axis = plt.subplot(1,1,1)
# sns.boxplot(ax=axis, data=RMSEPoints,palette = sns.color_palette('colorblind', 1))
# highlightBox = axis.patches[0]
# highlightBox.set_facecolor('#DE8F05')

# highlightBox = axis.patches[-2]
# highlightBox.set_facecolor('#CC78BC')
# highlightBox = axis.patches[-1]
# highlightBox.set_facecolor('#029E73')
# # highlightBox.set_edgecolor('black')
# # highlightBox.set_linewidth(3)

# plt.grid()
# plt.ylabel('Root mean squared error')
# plt.title('Performance of selected models during hyperparameter optimisation')

# plt.show()
# # sns.pointplot(data=RMSEPoints, x="name", y="body_mass_g")



# # %% Plot of longitudinal stiffness and FI of specimen from new matlab model
# Qxx = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\Qxx.csv",header=None)
# FI = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\FI.csv",header=None)
# gridX = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridx.csv",header=None)
# gridY = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridy.csv",header=None)


# # Qxx = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\Qxx.csv", delimiter=',')
# # FI = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\FI.csv", delimiter=',')
# # gridX = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridx.csv", delimiter=',')
# # gridY = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridy.csv", delimiter=',')





# # print(FI)
# # FI = np.pad(FI, pad_width = ((0, 1),(0,1)), mode='constant', constant_values = 1)
# # Qxx = np.pad(Qxx, pad_width = ((0, 1),(0,1)), mode='constant', constant_values = 1)
# # print(FI)
# FI = np.pad(FI, pad_width = ((0, 1),(0,1)), mode='edge')
# Qxx = np.pad(Qxx, pad_width = ((0, 1),(0,1)), mode='edge')


# px = 1/plt.rcParams['figure.dpi']  # pixel in inches
# fig = plt.figure(figsize=(300*px, 300*px), layout="constrained")
# plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis

# ax = plt.subplot(1,2,1) # E_xx
# CS = ax.contourf(gridX,gridY,Qxx/1000)
# cbar = fig.colorbar(CS)
# cbar.ax.set_ylabel('Stiffness [GPa]')
# plt.title('E_xx')
# ax.set_xticks([])
# ax.set_yticks([])


# ax = plt.subplot(1,2,2) # E_xx
# CS2 = ax.contourf(gridX,gridY,FI)
# cbar = fig.colorbar(CS2)
# cbar.ax.set_ylabel('Failure Index')
# plt.title('Failure Index')
# ax.set_xticks([])
# ax.set_yticks([])

# plt.show()


# # %% Activation functions
# x1 = np.linspace(-2.5,2.5)
# ytanh = tf.keras.activations.tanh(x1).numpy()
# yrelu = tf.keras.activations.relu(x1).numpy()
# ysoftplus = tf.keras.activations.softplus(x1).numpy()
# yelu = tf.keras.activations.elu(x1).numpy()
# ylrelu = tf.keras.activations.leaky_relu(x1).numpy()
# ygelu = tf.keras.activations.gelu(x1).numpy()
# ysilu = tf.keras.activations.silu(x1).numpy()

# plt.style.use("seaborn-v0_8-colorblind")
# plt.figure(figsize=(5,3), layout = "constrained", dpi = 300)
# plt.subplot(1,2,1)
# plt.plot(x1, ytanh)
# plt.plot(x1, yrelu)
# plt.plot(x1, ysoftplus)
# plt.plot(x1, ygelu)
# plt.legend(['Tanh', 'Relu', 'SoftPlus','Gelu'])
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('Activation function (x)')
# plt.subplot(1,2,2)
# plt.plot(x1, yelu)
# plt.plot(x1, ylrelu)

# plt.plot(x1, ysilu)
# plt.legend(['Elu', 'Lrelu',  'Silu'])
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('Activation function (x)')
# # plt.legend(['Tanh', 'Relu', 'SoftPlus', 'Elu', 'Lrelu', 'Gelu', 'Silu'])

# plt.show()



# %% Loss functions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib

plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "10"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly



y_true_const = np.zeros((1000,1))
y_true = np.linspace(-1,1,1000).reshape(1000,1)
y_pred = np.linspace(-1,1,1000).reshape(1000,1)
Y,P = np.meshgrid(y_true,y_pred) # Y is true P is prediction

MSE_Mesh = np.square(Y-P)
MAE_Mesh = np.abs(Y-P)
MSLE_Mesh = np.square(np.log10(Y + 1) - np.log10(P + 1))
Custom_Mesh = np.square(Y-P)*(1+np.maximum(Y, 0))

plt.style.use("seaborn-v0_8-colorblind")
fig = plt.figure(figsize=(12,7.5), layout = "constrained", dpi = 600)

# ax = plt.subplot(2,2,1)
# CS = plt.contourf(Y,P, MAE_Mesh,levels = 100)
# cbar = fig.colorbar(CS)
# cbar.ax.set_ylabel('Loss')
# plt.title('Mean Absolute Error Loss')
# plt.xlabel('True value')
# plt.ylabel('Predicted value')

# ax = plt.subplot(2,2,2)
# CS2 = plt.contourf(Y,P, MSE_Mesh,levels = 100)
# cbar = fig.colorbar(CS2)
# cbar.ax.set_ylabel('Loss')
# plt.title('Mean Squared Error Loss')
# plt.xlabel('True value')
# plt.ylabel('Predicted value')

# ax = plt.subplot(2,2,3)
# CS3 = plt.contourf(Y,P, MSLE_Mesh,levels = 100)
# cbar = fig.colorbar(CS3)
# cbar.ax.set_ylabel('Loss')
# plt.title('Mean Squared Logarithmic Error Loss')
# plt.xlabel('True value')
# plt.ylabel('Predicted value')

# ax = plt.subplot(2,2,4)
# CS4 = plt.contourf(Y,P, Custom_Mesh,levels = 100)
# cbar = fig.colorbar(CS4)
# cbar.ax.set_ylabel('Loss')
# plt.title('Custom loss function')
# plt.xlabel('True value')
# plt.ylabel('Predicted value')

# # Also do 3D plots:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS1 = ax.plot_surface(Y, P, MAE_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS1)
cbar.ax.set_ylabel('Loss')
plt.title('MAE Loss')
plt.xlabel('True value')
plt.ylabel('Prediction')
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticklabels([])
plt.savefig('MAE.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS2 = ax.plot_surface(Y, P, MSE_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS2)
cbar.ax.set_ylabel('Loss')
plt.title('MSE Loss')
plt.xlabel('True value')
plt.ylabel('Prediction')
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticklabels([])
plt.savefig('MSE.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS3 = ax.plot_surface(Y, P, MSLE_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS3)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Squared Logarithmic Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticklabels([])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS4 = ax.plot_surface(Y, P, Custom_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS4)
cbar.ax.set_ylabel('Loss')
plt.title('Custom loss function')
plt.xlabel('True value')
plt.ylabel('Prediction')
# ax.w_zaxis.line.set_lw(0.)
ax.set_zticklabels([])
plt.savefig('CustomLoss.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)

plt.show()

####### Uncomment for 1D plots of loss functions
# def MSEFun(y_true,y_pred):
#    return tf.keras.losses.MeanSquaredError().call(y_true,y_pred)
# #    return tf.math.square(y_true-y_pred)

# def MAEFun(y_true,y_pred):
#    return tf.keras.losses.MeanAbsoluteError().call(y_true,y_pred)
# #    return tf.math.abs(y_true-y_pred)

# def MSLEFun(y_true,y_pred):
#    return tf.keras.losses.MeanSquaredLogarithmicError().call(y_true,y_pred)
# #    return tf.math.square(tf.math.log(y_true + 1) - tf.math.log(y_pred + 1))


# def customFun(y_true,y_pred):
#    SE_base = tf.math.square(y_true-y_pred)
# #    SE_base = y_true-y_pred
#    loss = SE_base*(1+tf.nn.relu(y_true))
#    return loss

# MSE = []
# MAE = []
# MSLE = []
# customLoss = []
# trueVals = [-1,-0.5,0,0.5,1]
# for i in trueVals: # For different true values 
#     MSE += [MSEFun(y_true_const+i,y_pred)]
#     MAE += [MAEFun(y_true_const+i,y_pred)]
#     MSLE += [MSLEFun(y_true_const+i,y_pred)]
#     customLoss += [customFun(y_true_const+i,y_pred)]


# plt.style.use("seaborn-v0_8-colorblind")
# fig = plt.figure(figsize=(16,10), layout = "constrained", dpi = 600)

# for i in range(5): # 5 different true values
#     plt.subplot(3,2,i+1)
#     plt.plot(y_pred, MSE[i])
#     plt.plot(y_pred, MAE[i])
#     plt.plot(y_pred, MSLE[i])
#     plt.plot(y_pred, customLoss[i],'--')
#     plt.legend(['Squared Error','Absolute Error','Squared Logarithmic Error','Custom Loss'])
#     plt.grid()
#     plt.title('Loss from prediction at true value of {val}'.format(val=trueVals[i]))
#     plt.xlabel('Prediction')
#     plt.ylabel('Loss')
#     plt.ylim(0,2)

#  %%


# gridPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\sampleGrid.json"
# with open(gridPath) as json_file: # load into dict
#     grid = np.array(json.load(json_file)) # grid for plotting

# gridPathOld = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\sampleGrid.json"
# with open(gridPathOld) as json_file: # load into dict
#     gridOld = np.array(json.load(json_file)) # grid for plotting

# trainDat_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240702_1740' # The data format after extracting from Abaqus
# numSamples = len(os.listdir(trainDat_path)) # Number of data samples (i.e. TBDC specimens)
# sampleShape = [60,20]
# xNames = ['Ex','Ey','Gxy'] # Names of input features in input csv
# yNames = ['FI'] # Names of ground truth features in input csv

# def loadSample(path = str):
#   '''
#   Imports data in csv and formats into a tensor
#   Data from Abaqus comes in a slightly bothersome format, this 
#   function manually reformats it
#   '''
#   # Read sample csv data
#   sample = pd.read_csv(path)
#   headers = np.array(sample.columns.values.tolist())
#   values = np.array(sample)
# #   headers = np.concatenate(([[headers[0],'x_coord','y_coord'],headers[2:]])) # rectify the headers to include x and y coordinates separately
#   return headers, values

# # Import all data samples
# for i,file in enumerate(os.listdir(trainDat_path)):
#     filepath = os.path.join(trainDat_path,file)
#     if i==0:
#         headers, samples = loadSample(filepath)
#         samples = samples.reshape(1, np.shape(samples)[0],np.shape(samples)[1])
#     else:
#         addSamp = loadSample(filepath)[1]
#         samples = np.concatenate((samples,addSamp.reshape(1, np.shape(addSamp)[0],np.shape(addSamp)[1])))
# samples_NonStandard = samples
# # Reshape sample variable to have shape (samples, row, column, features)
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



# plt.style.use("seaborn-v0_8-colorblind")
# # fig = plt.figure(figsize=(12,7.5), layout = "constrained", dpi = 600)

# # ax = plt.subplot(2,2,1)
# # CS = plt.contourf(grid[0],grid[1],Y[0].reshape(Y.shape[1],-1))
# # cbar = fig.colorbar(CS)
# # cbar.ax.set_ylabel('Failure Index')
# # plt.title('Failure Index')


# Test that import and reshape is correct
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
fig, axs = plt.subplots(2, int(len(LFC18_headers)/2), sharex=True, sharey=True,figsize=[12,7.5]) # Create subplots to fit all variables
sampleNum = 0
# Plot  map
for i in range(len(LFC18_headers)):
  ax = plt.subplot(2, int(len(LFC18_headers)/2), i+1)
  CS = ax.contourf(LFC18_grid[0],LFC18_grid[1],LFC18_samples2D[sampleNum,:,:,i])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(LFC18_headers[i])
  fig.colorbar(CS)



# %% SSIM


def plot_contour(grid, samples2D,  ax, xlab = None, ylab = None, cbarlab = None, cBarBins = 5, scale = None):
    if scale == None:
        CS = ax.contourf(grid[0],grid[1],samples2D,levels=np.linspace(np.min(samples2D), np.max(samples2D), 10))
    else:
        CS = ax.contourf(grid[0],grid[1],samples2D,levels=scale.levels) 
    plt.ylabel(ylab, rotation=0, labelpad=20)
    plt.xlabel(xlab)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # cbar = fig.colorbar(CS,ticks=[np.min(samples2D), np.max(samples2D)], shrink = 0.8)
    print(CS.levels)
    cbar = fig.colorbar(CS,ticks=[], shrink = 0.85)
    # cbar.ax.text(0.1, -0.03, round(np.min(samples2D),1), transform=cbar.ax.transAxes, 
    #     va='top', ha='left')
    # cbar.ax.text(0.1, 1.03, round(np.max(samples2D),1), transform=cbar.ax.transAxes, 
    #     va='bottom', ha='left')
    cbar.ax.text(0.1, -0.03, round(np.min(CS.levels),1), transform=cbar.ax.transAxes, 
        va='top', ha='left')
    cbar.ax.text(0.1, 1.03, round(np.max(CS.levels),1), transform=cbar.ax.transAxes, 
        va='bottom', ha='left')
    # cbar.ax.locator_params(nbins=cBarBins)
    cbar.set_label(cbarlab, rotation=270,labelpad=10)
    return CS



plt.style.use("seaborn-v0_8-colorblind")

samples = [0,1,2]
sampleShape = [55,20]
dataset = 'LFC18'
groundTruths = np.empty(shape = (len(samples), sampleShape[0],sampleShape[1]))* np.nan  # Array of ground truths
predictions = np.empty(shape = (len(samples), sampleShape[0],sampleShape[1]))* np.nan  # Array of ground truths

current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != "Individual Project":
    current_dir = os.path.dirname(current_dir)

# Results imported manually (1) (model depth = 1):
# predPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\ModelDepth1806_1\dataout\predictions_ModelDepth1806_1_2.json"
# gtPathPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\ModelDepth1806_1\dataout\groundTruth_ModelDepth1806_1_2.json"

predPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "ModelDepth1806_1", "dataout", "predictions_ModelDepth1806_1_2.json")
gtPathPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "ModelDepth1806_1", "dataout", "groundTruth_ModelDepth1806_1_2.json")


with open(predPath) as json_file: # load into dict
            prediction = np.array(json.load(json_file))

with open(gtPathPath) as json_file: # load into dict
            groundTruth = np.array(json.load(json_file))

groundTruths[0] = groundTruth[0].reshape(sampleShape)
predictions[0] = prediction[0].reshape(sampleShape)

# model depth 2:
# predPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\ModelDepth1806_1\dataout\predictions_ModelDepth1806_1_3.json"
# gtPathPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\ModelDepth1806_1\dataout\groundTruth_ModelDepth1806_1_3.json"
predPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "ModelDepth1806_1", "dataout", "predictions_ModelDepth1806_1_3.json")
gtPathPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "ModelDepth1806_1", "dataout", "groundTruth_ModelDepth1806_1_3.json")
with open(predPath) as json_file: # load into dict
            prediction = np.array(json.load(json_file))

with open(gtPathPath) as json_file: # load into dict
            groundTruth = np.array(json.load(json_file))

groundTruths[1] = groundTruth[2].reshape(sampleShape)
predictions[1] = prediction[2].reshape(sampleShape)

# Model depth 3:
# predPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\ModelDepth1806_1\dataout\predictions_ModelDepth1806_1_1.json"
# gtPathPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\ModelDepth1806_1\dataout\groundTruth_ModelDepth1806_1_1.json"
predPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "ModelDepth1806_1", "dataout", "predictions_ModelDepth1806_1_1.json")
gtPathPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "ModelDepth1806_1", "dataout", "groundTruth_ModelDepth1806_1_1.json")
with open(predPath) as json_file: # load into dict
            prediction = np.array(json.load(json_file))

with open(gtPathPath) as json_file: # load into dict
            groundTruth = np.array(json.load(json_file))

groundTruths[2] = groundTruth[4].reshape(sampleShape)
predictions[2] = prediction[4].reshape(sampleShape)


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "10"
latexWidth = 315
figWidth = latexWidth*px
Ratio = (138/50)# Specimen ratio
figHeight = Ratio*figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

# harMean = 2/(1/8 + 1/50)
# pixelLength = 2.5

# charLengthPixels = harMean/pixelLength


rows = len(samples)

figLabels = ['(a)','(b)','(c)']
for i in samples:
    im1 = groundTruths[i].reshape(sampleShape)
    im2 = predictions[i].reshape(sampleShape)

    RMSEObj = tf.keras.metrics.RootMeanSquaredError()
    RMSEObj.update_state(im1,im2)
    RMSE = RMSEObj.result().numpy()

    # PSNR = cv2.PSNR(im1, im2)

    # winKernel = int(np.floor(charLengthPixels)) # Takwe as material characteristic length
    if dataset == 'LFC18':
        winKernel = 5
    elif dataset == 'MC24':
        winKernel = 7
    elif dataset == 'MC24x':
        winKernel = 13
    
    
    range = np.max(im1) - np.min(im1)
    SSIM, simIm = ski.metrics.structural_similarity(im1, im2, win_size=winKernel, gradient=False, data_range=range, channel_axis=None, gaussian_weights=False, full=True)

    # Ground truth
    ax = plt.subplot(rows, 2, 1+i*2)
    cScale = plot_contour(grid = LFC18_grid, samples2D = im1,  ax = ax, xlab = None, ylab = figLabels[i], cbarlab = 'Failure Index', cBarBins = 3)
    if i == 0:
         plt.title('Ground truth')
    # Prediction
    ax = plt.subplot(rows, 2, 2+i*2)
    cScale2 = plot_contour(grid = LFC18_grid, samples2D = im2,  ax = ax, xlab  = None, ylab = None, cbarlab = 'Failure Index', cBarBins = 3, scale = cScale)
    if i == 0:
         plt.title('Prediction')
    # ax.xaxis.set_label_coords(0.5,0.01)
    # # Error
    # ax = plt.subplot(rows, 4, 3+i*4)
    # plot_contour(grid = LFC18_grid, samples2D = im2-im1,  ax = ax, xlab = None, ylab = None, cbarlab = 'Error', cBarBins = 3)


    # ax = plt.subplot(rows, 3, 3+i*3)
    # ax.set_axis_off()
    ax.annotate(
                xy=(-22,-62),
                text='RMSE = ' + str(round(RMSE,2)) + '\n' + 'SSIM = ' + str(round(SSIM,2)),
                ha='left',
                size=8,
                color = 'black',
                weight = 'medium',
                bbox = dict(facecolor='white', alpha=0.6, boxstyle = 'Round', edgecolor = 'white')
            )
    
    # ax.annotate(
    #             xy=(0,0.7),
    #             text='SSIM = ' + str(round(SSIM,2)),
    #             ha='left',
    #             size=6
    #         )

    # ax.annotate(
    #             xy=(0,0.6),
    #             text='PSNR = ' + str(round(PSNR,2)),
    #             ha='left',
    #             size=6
    #         )


    # # Similarity
    # ax = plt.subplot(rows, 4, 4+i*4)
    # plot_contour(grid = LFC18_grid, samples2D = simIm,  ax = ax, xlab = None, ylab = None, cbarlab = 'Similarity', cBarBins = 3)



plt.savefig('PerformanceMeasures.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
plt.show()



# %% FIgure sfor showing number of models to train to do gridsearch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator
plt.style.use("seaborn-v0_8-colorblind")


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "10"
latexWidth = 315
figWidth = latexWidth*px
# Ratio = (138/50)# Specimen ratio
figHeight = 1.1*figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

HP_Nums = pd.DataFrame(data = pd.Series(data={'T/V':3,'Batch':7,'Kernel':7,'Optimizer':3,'Act':7,'Loss':3,'Dropout':7,'LR':5,'LR decay':4,'Pooling':2,'B Norm':2,'Depth':5,'D Aug':2,'Skip':2,'Dec act':2,'Epsilon':7, 'Downsamp':2, 'Filter dim':4}))
# HP_Nums = pd.DataFrame(data = pd.Series(data={'Train-val split':3,'Batch Size':6,'Kernel size':7,'Optimizer':3,'Activation func.':7,'Loss':3,'Dropout':7,'Initial LR':5,'LR decay':4,'Max pooling':2,'Batch norm.':2,'Model depth':6,'Data aug':2,'Skip connections':2,'Scaling':2,'Decoder activation':2,'Epsilon':7}))
HP_Nums = HP_Nums.rename(columns={0: "Number of values"})
HP_Nums['Cumulative prod.'] = np.cumprod(HP_Nums)
HP_Nums['Hyperparameter'] = HP_Nums.index



ax = plt.subplot(1,1,1)
bp = sns.barplot(data=HP_Nums,ax = ax, x = 'Hyperparameter',y = "Number of values" ,orient = 'v',color="#0173B2")

ax.set_xticklabels( 
    labels=ax.get_xticklabels(), rotation=90) 
ax.tick_params(axis='x', labelsize=8)
# plt.grid()

# ax2 = ax.twinx()
# lp = sns.lineplot(data=HP_Nums,ax = ax2, x = 'Hyperparameter',y = "Cumulative prod." ,color="#DE8F05")
# ax2.set_yscale('log')

# ax.set_ylabel('Number of options',color = 'black',bbox=dict(facecolor="#0173B2", edgecolor="#0173B2", pad=0.2, alpha=0.5, boxstyle = 'Round'))
ax.set_ylabel('Number of options',color = 'black')
# ax2.set_ylabel('Cumulative prod.',bbox=dict(facecolor="#DE8F05", edgecolor="#DE8F05", pad=0.2, alpha=0.5, boxstyle = 'Round'))
# ax2.grid(None)
ax.grid(True)
# Left y-axis: ticks every 1
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Right y-axis: ticks every 5
# ax2.set_yticks([1e3, 1e6, 1e9])
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    

plt.savefig('GridSearch_NumModels_simple.png', dpi=400, bbox_inches='tight', pad_inches = 0.2)
plt.show()


# %% Test how long it takes to train a model locally (and how long to predict 100 specimenst0 = time.time()
t0 = time.time()
loaded_model.predict(MC24_X_Unseen)

t1 = time.time()

total = t1-t0
print(total)





# %% Qualitative results of both final models


def plot_contour(grid, samples2D,  ax, xlab = None, ylab = None, cbarlab = None, cBarBins = 5, scale = None):
    if type(scale) == 'NoneType':
        CS = ax.contourf(grid[0],grid[1],samples2D,levels=np.linspace(np.min(samples2D), np.max(samples2D), 10))
    else:
        CS = ax.contourf(grid[0],grid[1],samples2D,levels=scale) 
    plt.ylabel(ylab, rotation=0, labelpad=20)
    plt.xlabel(xlab)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # cbar = fig.colorbar(CS,ticks=[np.min(samples2D), np.max(samples2D)], shrink = 0.8)
    print(CS.levels)
    cbar = fig.colorbar(CS,ticks=[], shrink = 0.85)
    # cbar.ax.text(0.1, -0.03, round(np.min(samples2D),1), transform=cbar.ax.transAxes, 
    #     va='top', ha='left')
    # cbar.ax.text(0.1, 1.03, round(np.max(samples2D),1), transform=cbar.ax.transAxes, 
    #     va='bottom', ha='left')
    cbar.ax.text(0.1, -0.03, round(np.min(CS.levels),1), transform=cbar.ax.transAxes, 
        va='top', ha='left')
    cbar.ax.text(0.1, 1.03, round(np.max(CS.levels),1), transform=cbar.ax.transAxes, 
        va='bottom', ha='left')
    # cbar.ax.locator_params(nbins=cBarBins)
    cbar.set_label(cbarlab, rotation=270,labelpad=10)
    return CS



plt.style.use("seaborn-v0_8-colorblind")

samples = [0,1]
sampleShapeLFC18 = [55,20]
sampleShapeMC24 = [60,20]


# LFC18_GTPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\CrossValidation2808_1\dataout\groundTruth_val_CrossValidation2808_1_1.json"
# LFC18_PredPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\CrossValidation2808_1\dataout\predictions_val_CrossValidation2808_1_1.json"
LFC18_GTPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "CrossValidation2808_1", "dataout", "groundTruth_val_CrossValidation2808_1_1.json")
LFC18_PredPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "CrossValidation2808_1", "dataout", "predictions_val_CrossValidation2808_1_1.json")


# MC24_GTPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24CrossValidation2808_1\dataout\groundTruth_val_MC24CrossValidation2808_1_1.json"
# MC24_PredPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\CNNTrainingSweepsResults\MC24CrossValidation2808_1\dataout\predictions_val_MC24CrossValidation2808_1_1.json"
MC24_GTPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "MC24CrossValidation2808_1", "dataout", "groundTruth_val_MC24CrossValidation2808_1_1.json")
MC24_PredPath = os.path.join(current_dir, "Data", "CNNTrainingSweepsResults", "MC24CrossValidation2808_1", "dataout", "predictions_val_MC24CrossValidation2808_1_1.json")

with open(LFC18_PredPath) as json_file: # load into dict
            predictionLFC18 = np.array(json.load(json_file))

with open(LFC18_GTPath) as json_file: # load into dict
            groundTruthLFC18 = np.array(json.load(json_file))

with open(MC24_PredPath) as json_file: # load into dict
            predictionMC24 = np.array(json.load(json_file))

with open(MC24_GTPath) as json_file: # load into dict
            groundTruthMC24 = np.array(json.load(json_file))



# groundTruths[0] = groundTruth[0].reshape(sampleShape)
# predictions[0] = prediction[0].reshape(sampleShape)


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figWidth = latexWidth*px
# Ratio = (138/50)# Specimen ratio
Ratio = 1.41428571429 # A4 paper ratio
# figHeight = Ratio*figWidth/1.618 # Golden ratio
figHeight = Ratio*figWidth
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)

harMean = 2/(1/8 + 1/50)
pixelLength = 2.5

charLengthPixels = harMean/pixelLength
charLengthPixelsLFC18 = 5
charLengthPixelsMC24 = 7

rows = len(samples)*2
cols = 4

figLabels = ['LFC18','LFC18','MC24','MC24']
for i in samples:
    # LFC18
    im1 = groundTruthLFC18[i+5].reshape(sampleShapeLFC18)
    im2 = predictionLFC18[i+5].reshape(sampleShapeLFC18)
    winKernel = int(np.floor(charLengthPixelsLFC18))
    grid = LFC18_grid


    RMSEObj = tf.keras.metrics.RootMeanSquaredError()
    RMSEObj.update_state(im1,im2)
    RMSE = RMSEObj.result().numpy()

     # Takwe as material characteristic length
    range = np.max(im1) - np.min(im1)
    SSIM, simIm = ski.metrics.structural_similarity(im1, im2, win_size=winKernel, gradient=False, data_range=range, channel_axis=None, gaussian_weights=False, full=True)

    # Ground truth
    ax1 = plt.subplot(rows, cols, 1+i*cols)
    cScale = plot_contour(grid = grid, samples2D = im1,  ax = ax1, xlab = None, ylab = figLabels[i], cbarlab = 'FI', cBarBins = 3)
    if i == 0:
         plt.title('Ground truth')
    # ax1.set_ylabel('LFC18')
    # Prediction
    ax = plt.subplot(rows, cols, 2+i*cols)
    cScale2 = plot_contour(grid = grid, samples2D = im2,  ax = ax, xlab  = None, ylab = None, cbarlab = 'FI', cBarBins = 3, scale = cScale.levels)
    if i == 0:
         plt.title('Prediction')
    # Error
    ax = plt.subplot(rows, cols, 3+i*cols)
    cScale2 = plot_contour(grid = grid, samples2D = np.abs(im2-im1),  ax = ax, xlab  = None, ylab = None, cbarlab = 'Error', cBarBins = 3)
    if i == 0:
         plt.title('Abs Error')
    # SSIM over whole image
    ax = plt.subplot(rows, cols, 4+i*cols)
    cScale2 = plot_contour(grid = grid, samples2D = simIm,  ax = ax, xlab  = None, ylab = None, cbarlab = 'SSIM', cBarBins = 3, scale = np.linspace(0,1,15))
    if i == 0:
         plt.title('SSIM')




    # ax.xaxis.set_label_coords(0.5,0.01)
    # # Error
    # ax = plt.subplot(rows, 4, 3+i*4)
    # plot_contour(grid = LFC18_grid, samples2D = im2-im1,  ax = ax, xlab = None, ylab = None, cbarlab = 'Error', cBarBins = 3)


    # ax = plt.subplot(rows, 3, 3+i*3)
    # ax.set_axis_off()
    ax1.annotate(
                xy=(-1.8,0.25),
                xycoords = 'axes fraction',
                text='RMSE = ' + str(round(RMSE,3)) + '\n' + 'SSIM = ' + str(round(SSIM,3)),
                ha='left',
                size=6,
                color = 'black',
                weight = 'medium')
    # bbox = dict(facecolor='white', alpha=0.6, boxstyle = 'Round', edgecolor = 'white')
    
    # MC24
    im1 = groundTruthMC24[i+3].reshape(sampleShapeMC24)
    im2 = predictionMC24[i+3].reshape(sampleShapeMC24)
    winKernel = int(np.floor(charLengthPixelsMC24))
    grid = MC24_grid

    RMSEObj = tf.keras.metrics.RootMeanSquaredError()
    RMSEObj.update_state(im1,im2)
    RMSE = RMSEObj.result().numpy()

     # Takwe as material characteristic length
    range = np.max(im1) - np.min(im1)
    SSIM, simIm = ski.metrics.structural_similarity(im1, im2, win_size=winKernel, gradient=False, data_range=range, channel_axis=None, gaussian_weights=False, full=True)

    # Ground truth
    ax1 = plt.subplot(rows, cols, 1+i*cols+8)
    cScale = plot_contour(grid = grid, samples2D = im1,  ax = ax1, xlab = None, ylab = figLabels[i+2], cbarlab = 'FI', cBarBins = 3)
    # ax1.set_ylabel('LFC18')
    # Prediction
    ax = plt.subplot(rows, cols, 2+i*cols+8)
    cScale2 = plot_contour(grid = grid, samples2D = im2,  ax = ax, xlab  = None, ylab = None, cbarlab = 'FI', cBarBins = 3, scale = cScale.levels)

    # Error
    ax = plt.subplot(rows, cols, 3+i*cols+8)
    cScale2 = plot_contour(grid = grid, samples2D = np.abs(im2-im1),  ax = ax, xlab  = None, ylab = None, cbarlab = 'Error', cBarBins = 3)

    # SSIM over whole image
    ax = plt.subplot(rows, cols, 4+i*cols+8)
    cScale2 = plot_contour(grid = grid, samples2D = simIm,  ax = ax, xlab  = None, ylab = None, cbarlab = 'SSIM', cBarBins = 3, scale = np.linspace(0,1,15))





    # ax.xaxis.set_label_coords(0.5,0.01)
    # # Error
    # ax = plt.subplot(rows, 4, 3+i*4)
    # plot_contour(grid = LFC18_grid, samples2D = im2-im1,  ax = ax, xlab = None, ylab = None, cbarlab = 'Error', cBarBins = 3)


    # ax = plt.subplot(rows, 3, 3+i*3)
    # ax.set_axis_off()
    ax1.annotate(
                xy=(-1.8,0.25),
                xycoords = 'axes fraction',
                text='RMSE = ' + str(round(RMSE,3)) + '\n' + 'SSIM = ' + str(round(SSIM,3)),
                ha='left',
                size=6,
                color = 'black',
                weight = 'medium')
    
    # ax.annotate(
    #             xy=(0,0.7),
    #             text='SSIM = ' + str(round(SSIM,2)),
    #             ha='left',
    #             size=6
    #         )

    # ax.annotate(
    #             xy=(0,0.6),
    #             text='PSNR = ' + str(round(PSNR,2)),
    #             ha='left',
    #             size=6
    #         )


    # # Similarity
    # ax = plt.subplot(rows, 4, 4+i*4)
    # plot_contour(grid = LFC18_grid, samples2D = simIm,  ax = ax, xlab = None, ylab = None, cbarlab = 'Similarity', cBarBins = 3)



plt.savefig('Final_model_examples_abs.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.2)
plt.show()

# %% Comparison of SSIM RMSE and std of 2 cross-validations
# Define data structures for baseline and optimised models
import numpy as np
import matplotlib.pyplot as plt
baseline_metrics = {
    "Train SSIM": {"mean": 0.8348, "std": 0.0316},
    "Val SSIM": {"mean": 0.7372, "std": 0.0201},
    "Train RMSE": {"mean": 0.1057, "std": 0.0137},
    "Val RMSE": {"mean": 0.1651, "std": 0.0062}
}

optimised_metrics = {
    "Train SSIM": {"mean": 0.8742, "std": 0.0141},
    "Val SSIM": {"mean": 0.8526, "std": 0.0127},
    "Train RMSE": {"mean": 0.1168, "std": 0.0057},
    "Val RMSE": {"mean": 0.1328, "std": 0.0046}
}

# Prepare data for plotting
metrics = ["Train SSIM", "Val SSIM", "Train RMSE", "Val RMSE"]
baseline_means = [baseline_metrics[m]["mean"] for m in metrics]
baseline_stds = [baseline_metrics[m]["std"] for m in metrics]
optimised_means = [optimised_metrics[m]["mean"] for m in metrics]
optimised_stds = [optimised_metrics[m]["std"] for m in metrics]


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "6"
latexWidth = 315
figWidth = latexWidth*px
# Ratio = (138/50)# Specimen ratio
Ratio = 1.41428571429 # A4 paper ratio
# figHeight = Ratio*figWidth/1.618 # Golden ratio
figHeight = Ratio*figWidth
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly

fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)
fig.set_figwidth(figWidth)


x = np.arange(len(metrics))
width = 0.35

plt.style.use("seaborn-v0_8-colorblind")
fig, ax = plt.subplots(figsize=(7,4), dpi=150)
rects1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline', capsize=5, color="#0173B2")
rects2 = ax.bar(x + width/2, optimised_means, width, yerr=optimised_stds, label='Optimised', capsize=5, color="#DE8F05")

ax.set_ylabel('Metric Value')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20)
ax.set_title('Model Performance: Baseline vs Optimised')
ax.legend()

# Annotate improvement
for i in range(len(metrics)):
    improvement = optimised_means[i] - baseline_means[i]
    if "SSIM" in metrics[i]:
        txt = f"+{improvement:.3f}"
    else:
        txt = f"{improvement:.3f}"
    ax.text(x[i], max(baseline_means[i], optimised_means[i]) + 0.01, txt, ha='center', va='bottom', fontsize=9, color='black')

plt.tight_layout()
plt.show()

