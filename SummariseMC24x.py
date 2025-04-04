# %%

import sys

import os
import glob
import random
import time
import math
import datetime
import shutil
import json
import scipy
import tensorflow as tf # TODO If we want to load models saved on the HPC we need to use legacy keras 2. See test.py for import of this 

os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import argparse

# %% find results and training history files and load these + 

# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20241213_MC24x_Baseline'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20241223_AltModels_Baseline'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250311_LFC18_CrossValidation_PreOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250401_LFC18_CrossValidation_PostOpti'
jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250401_MC24_CrossValidation_PreOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250401_MC24_CrossValidation_PostOpti'
crossVal = True # Cross validation or not
trainEpochs = 1000


results_files = []
history_files = []
repeats = 0
for root, dirs, files in os.walk(jobPath):
    # Check if the current directory is named "dataout"
    if os.path.basename(root) == "dataout":
        repeats += 1
        rFil = np.empty((0)) # current repeat
        hFil = np.empty((0))
        # Find files containing "results" in their names
        for file_name in files:
            
            if "results" in file_name:
                print(file_name)
                file_path = os.path.join(root, file_name)
                results_files.append(file_path)
            if "trainHist" in file_name:
                file_path = os.path.join(root, file_name)
                history_files.append(file_path)
results_files = np.array(results_files).reshape(repeats,-1)
history_files = np.array(history_files).reshape(repeats,-1)
nModels = results_files.shape[-1]

if crossVal:
    results = np.empty((repeats, nModels, 2, 4))  # Shape is (repeat, model number, [train val], [loss, MAE, MSE, SSIM])
else:
    results = np.empty((repeats, nModels, 3, 4)) # Shape is (repeat, model number, [train val test], [loss, MAE, MSE, SSIM])
histories = np.empty((repeats, nModels, trainEpochs, 9)) # ['loss', 'mean_absolute_error', 'mean_squared_error', 'SSIM_metric','val_loss', 'val_mean_absolute_error', 'val_mean_squared_error','val_SSIM_metric', 'trainTime']
for r in range(results.shape[0]): # repeats
    for m in range(results.shape[1]): # models
        with open(results_files[r,m], 'r') as file:
            data = eval(json.load(file))
        df = pd.DataFrame.from_dict(data, orient='columns', dtype=None, columns=None)
        resRows = df.index.to_numpy()
        resCols = df.columns.to_numpy()
        res = df.to_numpy()
        results[r,m] = res

        with open(history_files[r,m], 'r') as file:
            # print(type(json.load(file)))
            data = json.load(file)
        df = pd.DataFrame.from_dict(data, orient='columns', dtype=None, columns=None)
        histRows = df.index.to_numpy()
        histCols = df.columns.to_numpy()
        res = df.to_numpy()
        if res.shape[0]<trainEpochs: # Typically will see early stopping
            res = np.pad(res,((0,trainEpochs-res.shape[0]),(0,0)),'constant', constant_values=np.nan)
        histories[r,m] = res

# %% Load sweep definition
pathList = os.listdir(jobPath)
sweepDef = pd.read_csv(os.path.join(jobPath,[path for path in pathList if 'sweep_definition' in path][0]))
lossIdx = 0
timeIdx = -1

def formatHistory(histories, histCols):
    
    p = histories.shape[-1] # parameters
    r = histories.shape[0] # repeat
    m = histories.shape[1] # models
    e = histories.shape[2] # epochs
    reshaped_array = histories.reshape(-1, p)
    repeat_ids = np.repeat(np.arange(r), m * e)
    model_ids = np.tile(np.repeat(np.arange( stop = m), e), r)
    epoch_ids = np.tile(np.arange(e), r * m)
    
    df = pd.DataFrame(
    reshaped_array,
    columns=[i for i in histCols]  # Name the parameter columns
    )
    df["repeat"] = repeat_ids+1
    df["model"] = model_ids+1
    df["epoch"] = epoch_ids

    df = df[["repeat", "model", "epoch"] + [i for i in histCols]]
    return df

def formatResults(results, resCols):
    
    p = results.shape[-1] # parameters
    r = results.shape[0] # repeat
    m = results.shape[1] # models
    if crossVal:
        d = ['train','val']
    else:
        d = ['train','val','test'] # data
    reshaped_array = results.reshape(-1, p)
    repeat_ids = np.repeat(np.arange(r), m * len(d))
    model_ids = np.tile(np.repeat(np.arange(stop = m), len(d)), r)
    data_ids = np.tile(d, r * m)
    
    df = pd.DataFrame(
    reshaped_array,
    columns=[i for i in resCols]  # Name the parameter columns
    )
    df["repeat"] = repeat_ids+1
    df["model"] = model_ids+1
    df["data"] = data_ids

    df = df[["repeat", "model", "data"] + [i for i in resCols]]
    return df

def plotHist(histDf, idx):
    histDf = histDf[histDf['model'].isin(idx)]
    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams['axes.linewidth'] = 0.25
    plt.rc('axes', axisbelow=True)
    plt.rcParams["font.size"] = "6"
    plt.rcParams['grid.linewidth'] = 0.2
    latexWidth = 315
    figWidth = latexWidth*px
    figHeight = figWidth/1.618 # Golden ratio
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
    cBarBins = 3
    resolution_scaling = 1 # Manually scale DPI and text accordingly
    fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
    fig.set_figheight(figHeight)
    fig.set_figwidth(figWidth)
    ax = plt.subplot(1,2,1)

    lp = sns.lineplot(data = histDf,x="epoch", y="loss",
              hue="model",palette = 'colorblind',linewidth = 0.5)
    lp = sns.lineplot(data = histDf,x="epoch", y="val_loss",
              hue="model",palette = 'colorblind',linewidth = 0.5, linestyle='--')
    ax.set_yscale('log')
    ax.grid(axis = 'y',which = 'minor')
    plt.grid()
    ax.minorticks_on()
    ax.set_ylabel('Loss')
    h,l = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    ax = plt.subplot(2,4,4)
    lp = sns.lineplot(data = histDf,x="epoch", y="trainTime",
              hue="model",palette = 'colorblind',linewidth = 0.5)
    ax.grid(axis = 'y',which = 'minor')
    plt.grid()
    ax.minorticks_on()
    ax.set_ylabel('Time [s]')
    h,l = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    fig.legend(title='Model index',handles = h,labels=l, 
            loc="lower center", ncol=2,bbox_to_anchor=(0.8, 0.2))
    plt.savefig('tC.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
    plt.show()
    
def plotResults(resDf, idx):
    resDf = resDf[resDf['model'].isin(idx)]
    plt.style.use("seaborn-v0_8-colorblind")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams['axes.linewidth'] = 0.25
    plt.rc('axes', axisbelow=True)
    plt.rcParams["font.size"] = "6"
    plt.rcParams['grid.linewidth'] = 0.2
    latexWidth = 315
    figWidth = latexWidth*px
    figHeight = figWidth/1.618 # Golden ratio
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
    cBarBins = 3
    resolution_scaling = 1 # Manually scale DPI and text accordingly
    fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
    fig.set_figheight(figHeight)
    fig.set_figwidth(figWidth)


    ax = plt.subplot(1,1,1)
    ax2 = plt.twinx()
    g = sns.boxplot(ax=ax, data=resDf, x = 'model', y = "mean_squared_error",hue="data", orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#029E73", whis=(0, 100)) # Remember the models are 1-indexed
    h = sns.boxplot(ax=ax2, data=resDf, x = 'model', y = "SSIM_metric", hue="data", orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), color="#0173B2", whis=(0, 100)) # Remember the models are 1-indexed
    # g2 = sns.pointplot(ax = ax,data=resDf, estimator = 'median', ci=None, scale=0.3, color="#029E73", marker='D',linewidth = 0.25)
    # h2 = sns.pointplot(ax = ax2,data=resDf, estimator = 'median', ci=None, scale=0.3, color="#0173B2", marker='o',linewidth = 0.25)
    ax.set_ylim([0.99*np.min(resDf["mean_squared_error"]),np.max(resDf["mean_squared_error"])+np.max(resDf["mean_squared_error"])-0.9*np.min(resDf["mean_squared_error"])])
    ax2.set_ylim([np.min(resDf["SSIM_metric"])-np.max(resDf["SSIM_metric"])+np.min(resDf["SSIM_metric"]),1.01*np.max(resDf["SSIM_metric"])])
    ax.grid(axis = 'y')
    # ax2.grid(False)
    ax.minorticks_on()
    ax.set_ylabel('RMSE',color = 'black',bbox=dict(facecolor="#029E73", edgecolor="#029E73", pad=0.2, alpha=0.5, boxstyle = 'Round'))
    ax2.set_ylabel('SSIM',bbox=dict(facecolor="#0173B2", edgecolor="#0173B2", pad=0.2, alpha=0.5, boxstyle = 'Round'))
    
    h,l = ax.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax.get_legend().remove()
    ax2.get_legend().remove()


    fig.legend(title='Data',handles = h,labels=l, 
            loc="lower left", ncol=1,bbox_to_anchor=(0, -.2))
    fig.legend(title='Data',handles = h2,labels=l2, 
            loc="lower right", ncol=1,bbox_to_anchor=(1, -.2))
    plt.savefig('res.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
    plt.show()





# %%
histDf = formatHistory(histories,histCols = histCols)
resDf = formatResults(results,resCols = resCols)
# %% Make plots
idx = [1,2] # Activation func
# idx = list(range(1, 11))

plotHist(histDf, idx)
plotResults(resDf, idx)

plt.show()
# %%
