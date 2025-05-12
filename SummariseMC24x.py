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
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250404_LFC18_CrossValidation_PostOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250401_MC24_CrossValidation_PreOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250404_MC24_CrossValidation_PostOpti'

#14 04 2025 Comparison between pre and post- opti
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250414_LFC18_CrossValidation_PreOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250414_LFC18_CrossValidation_PostOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250414_MC24_CrossValidation_PreOpti'
# jobPath = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250414_MC24_CrossValidation_PostOpti'


# 29042025 LFC18 optimisation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_trainValSplit"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_batchSize" # FAILED NEED TO RE-RUN
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_kernelSize"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_optimizer"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_activationFunction"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_loss"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_dropout"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_initialLr"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_lrDecay"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_maxPool"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_batchNorm"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_modelDepth"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_dataAug"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_skinConnections"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_decoderAct"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250429_LFC18_epsilon"

# 30042025 LFC18 optimisation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_trainValSplit"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_batchSize" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_kernelSize"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_optimizer"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_activationFunction"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_loss"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_dropout" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_initialLr"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_lrDecay"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_maxPool"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_batchNorm"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_modelDepth"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_dataAug"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_skinConnections"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_decoderAct"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_LFC18_epsilon"

# 30042025 MC24 optimisation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_trainValSplit"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_batchSize" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_kernelSize"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_optimizer"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_activationFunction"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_loss"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_dropout" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_initialLr"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_lrDecay"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_maxPool"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_batchNorm"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_modelDepth"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_dataAug"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_skinConnections"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_decoderAct"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24_epsilon"

# 30042025 MC24x optimisation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_trainValSplit"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_batchSize" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_kernelSize"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_optimizer"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_activationFunction"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_loss"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_dropout" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_initialLr"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_lrDecay"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_maxPool"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_batchNorm"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_modelDepth"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_dataAug"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_skinConnections"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_decoderAct"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250430_MC24x_epsilon"

crossVal = False # Cross validation or not
testDat = False # Test data or not

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
                # print(file_name)
                file_path = os.path.join(root, file_name)
                results_files.append(file_path)
            if "trainHist" in file_name:
                file_path = os.path.join(root, file_name)
                history_files.append(file_path)
# Sort files such that we have repeat 1 model 1, 2, 3,...10, repeat 2 model 1, 2, 3, etc.
results_files = sorted(results_files, key=lambda x: (int(x.split(os.sep)[-3].split('_')[-1]), int(x.split('_')[-1].split('.')[0])))
history_files = sorted(history_files, key=lambda x: (int(x.split(os.sep)[-3].split('_')[-1]), int(x.split('_')[-1].split('.')[0])))
results_files = np.array(results_files).reshape(repeats,-1)
history_files = np.array(history_files).reshape(repeats,-1)
nModels = results_files.shape[-1]

if crossVal or not testDat:
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
    if crossVal or not testDat:
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
    df['RMSE'] = np.sqrt(df['mean_squared_error'])
    return df

def plotHist(histDf, idx, sweepName, sweepVals, output_dir, job_name):
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
    ax = plt.subplot(1,1,1)

    # Fix colour palette for loss and val_loss
    base_palette = sns.color_palette('colorblind', n_colors=len(histDf['model'].unique()))
    def lighten_color(color, amount=0.5):
        return matplotlib.colors.to_rgba(matplotlib.colors.to_hex(color), alpha=1.0 - amount)

    color_dict = dict(zip(histDf['model'].unique(), base_palette))
    light_color_dict = {model: lighten_color(color, amount=0.25) for model, color in color_dict.items()}



    if crossVal: # This method is way faster than the other one
        mean_loss = histDf.groupby("epoch")["loss"].mean()
        std_loss = histDf.groupby("epoch")["loss"].std()
        mean_val_loss = histDf.groupby("epoch")["val_loss"].mean()
        std_val_loss = histDf.groupby("epoch")["val_loss"].std()

        lp1 = sns.lineplot(x=mean_loss.index, y=mean_loss, color="blue", linewidth=0.5, label="Mean Loss")
        plt.fill_between(mean_loss.index, mean_loss - std_loss, mean_loss + std_loss, color="blue", alpha=0.3, label="Loss Std Dev")

        lp2 = sns.lineplot(x=mean_val_loss.index, y=mean_val_loss, color="orange", linewidth=0.5, linestyle='--', label="Mean Val Loss")
        plt.fill_between(mean_val_loss.index, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color="orange", alpha=0.3, label="Val Loss Std Dev")
    else:
        lp1 = sns.lineplot(data=histDf, x="epoch", y="loss",
                          hue="model",  palette=color_dict, linewidth=0.5,errorbar='sd')
        lp2 = sns.lineplot(data=histDf, x="epoch", y="val_loss",
                          hue="model", palette=light_color_dict,linewidth=0.5, linestyle='--',errorbar='sd')
    

    ax.set_yscale('log')
    ax.set_ylim([ax.get_ylim()[0],np.max(histDf.loc[histDf['epoch']>5]['loss'])])
    ax.grid(axis='y', which='minor')
    plt.grid()
    ax.minorticks_on()
    ax.set_ylabel('Loss')
    ax.get_legend().remove()

    # Create custom legend entries for sweep values and line styles
    sweep_legend = [matplotlib.lines.Line2D([0], [0], color=base_palette[i], lw=1, label=f'{val}') for i, val in enumerate(sweepVals)]
    line_style_legend = [
        matplotlib.lines.Line2D([0], [0], color='black', lw=1, linestyle='-', label='Training Data'),
        matplotlib.lines.Line2D([0], [0], color='black', lw=1, linestyle='--', alpha = 0.3, label='Validation Data')
    ]

    # Combine legends
    combined_legend = sweep_legend + line_style_legend

    # Place the legend outside the plot area
    fig.legend(title=sweepName, handles=combined_legend, 
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    
    # Save trainingCurve.pdf
    training_curve_output_path = os.path.join(output_dir, f"{job_name}_trainingCurve.pdf")
    plt.savefig(training_curve_output_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)

    # plt.savefig('trainingCurve.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
    plt.show()

    fig2 = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
    fig2.set_figheight(figHeight)
    fig2.set_figwidth(figWidth)
    ax = plt.subplot(1,1,1)
    if crossVal:
        mean_train_time = histDf.groupby("epoch")["trainTime"].mean()
        std_train_time = histDf.groupby("epoch")["trainTime"].std()

        lp = sns.lineplot(x=mean_train_time.index, y=mean_train_time, color="green", linewidth=0.5, label="Mean Train Time")
        plt.fill_between(mean_train_time.index, mean_train_time - std_train_time, mean_train_time + std_train_time, color="green", alpha=0.3)
    else:
        lp = sns.lineplot(data=histDf, x="epoch", y="trainTime",
                          hue="model", palette='colorblind', linewidth=0.5,errorbar='sd')
    ax.grid(axis = 'y',which = 'minor')
    plt.grid()
    ax.minorticks_on()
    ax.set_ylabel('Time [s]')
    h,l = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    sweep_legend = [matplotlib.lines.Line2D([0], [0], color=base_palette[i], lw=1, label=f'{val}') for i, val in enumerate(sweepVals)]
    fig2.legend(title=sweepName, handles=sweep_legend, 
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    # fig2.legend(title='Model index',handles = h,labels=l, 
    #         loc="lower center", ncol=2,bbox_to_anchor=(0.8, 0.2))


    # Save trainingTime.pdf
    training_time_output_path = os.path.join(output_dir, f"{job_name}_trainingTime.pdf")
    plt.savefig(training_time_output_path, dpi=fig2.dpi, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig('trainingTime.pdf', dpi=fig2.dpi, bbox_inches='tight', pad_inches = 0.1)
    plt.show()
    
def plotResults(resDf, idx, sweepName, sweepVals, output_dir, job_name):
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

    RMSE_palette = {
    'train': '#66D3B3',  # lighter green
    'val': '#029E73',  # base green
    'test': '#01523D'  # darker green
    }

    SSIM_palette = {
    'train': '#66ADD6',  # lighter blue
    'val': '#0173B2',  # base blue
    'test': '#01436A'  # darker blue
    }


    ax = plt.subplot(1,1,1)
    ax2 = plt.twinx()
    # g = sns.boxplot(ax=ax, data=resDf, x = 'model', y = "RMSE",hue="data", orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), palette=RMSE_palette, whis=(0, 100)) # Remember the models are 1-indexed
    g = sns.stripplot(ax=ax, data=resDf, x='model', y="RMSE", hue="data", orient="v", dodge=True, palette=RMSE_palette, alpha = 0.4)  # Remember the models are 1-indexed
    

    # h = sns.boxplot(ax=ax2, data=resDf, x = 'model', y = "SSIM_metric", hue="data", orient="v", width=0.5,linewidth = 0.25, medianprops=dict(alpha=1,linewidth = 0.25), palette=SSIM_palette, whis=(0, 100)) # Remember the models are 1-indexed
    h = sns.stripplot(ax=ax2, data=resDf, x = 'model', y = "SSIM_metric", hue="data", orient="v", dodge=True, palette=SSIM_palette, alpha = 0.4) # Remember the models are 1-indexed
    
    
    # g2 = sns.pointplot(ax = ax,data=resDf, estimator = 'median', ci=None, scale=0.3, color="#029E73", marker='D',linewidth = 0.25)
    # h2 = sns.pointplot(ax = ax2,data=resDf, estimator = 'median', ci=None, scale=0.3, color="#0173B2", marker='o',linewidth = 0.25)
    ax.set_ylim([0.99*np.min(resDf["RMSE"]),np.max(resDf["RMSE"])+np.max(resDf["RMSE"])-0.9*np.min(resDf["RMSE"])])
    ax2.set_ylim([np.min(resDf["SSIM_metric"])-np.max(resDf["SSIM_metric"])+np.min(resDf["SSIM_metric"]),1.01*np.max(resDf["SSIM_metric"])])
    ax.grid(axis = 'y')
    # ax2.grid(False)
    ax.minorticks_on()
    ax.set_ylabel('RMSE',color = 'black',bbox=dict(facecolor="#029E73", edgecolor="#029E73", pad=0.2, alpha=0.5, boxstyle = 'Round'))
    ax2.set_ylabel('SSIM',bbox=dict(facecolor="#0173B2", edgecolor="#0173B2", pad=0.2, alpha=0.5, boxstyle = 'Round'))
    
    
    
    RMSEmeans = resDf.groupby(['model', 'data'])['RMSE'].mean().reset_index()
    sns.pointplot(ax=ax, data=RMSEmeans, x='model', y='RMSE', hue='data', palette=RMSE_palette, dodge=.4, linestyle="none", errorbar=None,
    marker="_", markersize=12, markeredgewidth=3)

    SSIMmeans = resDf.groupby(['model', 'data'])['SSIM_metric'].mean().reset_index()
    sns.pointplot(ax=ax2, data=SSIMmeans, x='model', y='SSIM_metric', hue='data', palette=SSIM_palette, dodge=.4, linestyle="none", errorbar=None,
    marker="_", markersize=12, markeredgewidth=3)

    # Count the number of unique data types in resDf
    unique_data_types = resDf['data'].nunique()
    print(f"Unique data types: {unique_data_types}")

    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h, l = h[-unique_data_types:], l[-unique_data_types:]
    h2, l2 = h2[-unique_data_types:], l2[-unique_data_types:]


    ax.get_legend().remove()
    ax2.get_legend().remove()


    # Set sweepName as x-label and sweepVals as x-ticker labels
    ax.set_xlabel(sweepName)
    ax.set_xticks(range(len(sweepVals)))
    ax.set_xticklabels(sweepVals)

    fig.legend(title='Data',handles = h,labels=l, 
            loc="lower left", ncol=1,bbox_to_anchor=(0, -.2))
    fig.legend(title='Data',handles = h2,labels=l2, 
            loc="lower right", ncol=1,bbox_to_anchor=(1, -.2))
    # plt.savefig('res.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
    # Calculate mean and std for RMSE and SSIM for train and val datasets
    # RMSE_mean = resDf['RMSE'].mean()
    # RMSE_std = resDf['RMSE'].std()
    # SSIM_mean = resDf['SSIM_metric'].mean()
    # SSIM_std = resDf['SSIM_metric'].std()




    # Save res.pdf
    res_output_path = os.path.join(output_dir, f"{job_name}_RMSE_SSIM.pdf")
    plt.savefig(res_output_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)
    # Save the nicely printed table to a CSV file
    table_output_path = os.path.join(output_dir, f"{job_name}_metrics.csv")
    summary_data = []

    if crossVal:
        for data_type in resDf['data'].unique():
            filtered_df = resDf[resDf['data'] == data_type]
            RMSE_mean = filtered_df['RMSE'].mean()
            RMSE_std = filtered_df['RMSE'].std()
            SSIM_mean = filtered_df['SSIM_metric'].mean()
            SSIM_std = filtered_df['SSIM_metric'].std()
            summary_data.append([data_type, 'RMSE', RMSE_mean, RMSE_std])
            summary_data.append([data_type, 'SSIM', SSIM_mean, SSIM_std])
    else:
        for model in resDf['model'].unique():
            filtered_df = resDf[resDf['model'] == model]
            for data_type in filtered_df['data'].unique():
                data_filtered_df = filtered_df[filtered_df['data'] == data_type]
                RMSE_mean = data_filtered_df['RMSE'].mean()
                RMSE_std = data_filtered_df['RMSE'].std()
                SSIM_mean = data_filtered_df['SSIM_metric'].mean()
                SSIM_std = data_filtered_df['SSIM_metric'].std()
                summary_data.append([model, sweepVals[model - 1], data_type, 'RMSE', RMSE_mean, RMSE_std])
                summary_data.append([model, sweepVals[model - 1], data_type, 'SSIM', SSIM_mean, SSIM_std])

        summary_df = pd.DataFrame(summary_data, columns=['Model', sweepName, 'Data', 'Metric', 'Mean', 'Std'])
    print(summary_df)
    summary_df.to_csv(table_output_path, index=False)
    print(f"Metrics summary saved to {table_output_path}")

    plt.show()





# %%
histDf = formatHistory(histories,histCols = histCols)
resDf = formatResults(results,resCols = resCols)
# %% Make plots
# idx = [1] # Activation func
output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x"
job_name = os.path.basename(jobPath)

numModels = len(histDf['model'].unique())
# sweepName = 'valSize' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'batchSize' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'kernelSize' # Name of the sweep parameter
# sweepVals = list(range(1, 8))
# sweepName = 'optimizer' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'Activation function' # Name of the sweep parameter
# sweepVals = sweepDef['conv1Activation'].to_numpy()
# sweepName = 'loss' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'dropout' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'initial_lr' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'lr_decay_rate' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'Max Pooling' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'Batch Normalisation' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'Model depth' # Name of the sweep parameter
# sweepVals = list(range(1, 7)) # Values of the sweep parameter
# sweepName = 'Data augmentation' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'Skip Connections' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
sweepName = 'Decoder activations' # Name of the sweep parameter
sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'epsilon' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()




# sweepName = 'batchSize' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepVals = list(range(1, 7))
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter

idx = list(range(1, 11))
idxHist = [3]
# idx = [0,1]

plotHist(histDf, idx, sweepName, sweepVals, output_dir, job_name)
plotResults(resDf, idx, sweepName, sweepVals, output_dir, job_name)
# Save figures to the specified directory with the job name appended


print(f"Figures saved to {output_dir} with job name appended.")

plt.show()
# %% Show an example prediction and ground truth

samplePath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\datain\MatLabModel2024_224_4kSamples\samples_1to40.parquet"
modelPath = r"C:\Users\kfh23\Desktop\model_20250430_MC24x_trainValSplit_1_1.keras"
xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
yNames = ['FI']
specIdx = 1 # specimen to plot (we don't know what was val/test/train)
sampleShape = (224,224) # Shape of the input data
winKernel = 17
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

sample = pd.read_parquet(samplePath, engine='auto')
samples = [y for x, y in sample.groupby('specimen')]
sample = samples[specIdx] # Take the first sample for testing
headers = np.array(sample.columns.values.tolist())
sample = np.array(sample)


sample = sample.reshape(1,sampleShape[0],sampleShape[1],-1)

# Find indeces of input features 
featureIdx = []
for name in xNames:
    featureIdx += [np.where(headers == name)[0][0]]

# Find indeces of ground truth features 
gtIdx = []
for name in yNames:
    gtIdx += [np.where(headers == name)[0][0]]


X = sample[:,:,:,featureIdx] # Input features
Y = sample[:,:,:,gtIdx] # Labels
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
ds = tf.data.Dataset.from_tensor_slices((X, Y))

# Load the model
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

def custom_loss(y_true,y_pred):
  SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
  loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.nn.relu(y_true))))
  loss = tf.reduce_mean(loss)
  return loss

model = keras.models.load_model(modelPath,custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss})

# Make a prediction
prediction = model.predict(ds.batch(1))

# Reshape the ground truth and prediction to 224x224
ground_truth = Y[0, :, :, 0].reshape(sampleShape)
predicted_field = prediction[0, :, :, 0].reshape(sampleShape)

# Calculate the error
error_field = np.abs(ground_truth - predicted_field)
errorSquared_field = np.abs(ground_truth - predicted_field)**2

# Get the colorbar limits from the ground truth
vmin, vmax = 0, ground_truth.max()

# Plot the ground truth, prediction, and error fields side by side
plt.figure(figsize=(15, 5))

# Ground truth
plt.subplot(1, 3, 1)
plt.imshow(ground_truth, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('Ground Truth')
plt.colorbar()
plt.axis('off')

# Prediction
plt.subplot(1, 3, 2)
plt.imshow(predicted_field, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('Prediction')
plt.colorbar()
plt.axis('off')

# Error
plt.subplot(1, 3, 3)
plt.imshow(errorSquared_field, cmap='viridis')
plt.title('Squared error')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot the distribution of ground truth, predictions, and error
plt.figure(figsize=(10, 6))
sns.kdeplot(ground_truth.flatten(), label='Ground Truth', color='blue', fill=True, alpha=0.5)
sns.kdeplot(predicted_field.flatten(), label='Prediction', color='orange', fill=True, alpha=0.5)
sns.kdeplot(error_field.flatten(), label='Error', color='red', fill=True, alpha=0.5)
plt.title('Distribution of Ground Truth, Predictions, and Error')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# Find the 5 areas with the highest values in the ground truth
flat_ground_truth = ground_truth.flatten()
flat_predicted_field = predicted_field.flatten()
top_5_indices = np.argpartition(flat_ground_truth, -5)[-5:]
top_5_indices = top_5_indices[np.argsort(flat_ground_truth[top_5_indices])[::-1]]

print("Top 5 highest values in the ground truth and corresponding predictions:")
for idx in top_5_indices:
    row, col = divmod(idx, sampleShape[1])
    gt_value = ground_truth[row, col]
    pred_value = predicted_field[row, col]
    print(f"Location: ({row}, {col}), Ground Truth: {gt_value:.4f}, Prediction: {pred_value:.4f}")





# %%

# %%
# Identify models and repeats with a sudden spike in loss
def find_spike_in_loss(histDf, threshold=2):
    spike_info = []
    for model in histDf['model'].unique():
        model_data = histDf[histDf['model'] == model]
        for repeat in model_data['repeat'].unique():
            repeat_data = model_data[model_data['repeat'] == repeat]
            loss_diff = repeat_data['loss'].diff()
            if (loss_diff > threshold).any():
                spike_info.append((model, repeat))
    return spike_info

# Plot loss curve for models and repeats with spikes
def plot_spike_loss_curve(histDf, spike_info):
    for model, repeat in spike_info:
        repeat_data = histDf[(histDf['model'] == model) & (histDf['repeat'] == repeat)]
        plt.figure()
        plt.plot(repeat_data['epoch'], repeat_data['loss'], label=f'Model {model}, Repeat {repeat}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for Model {model}, Repeat {repeat}')
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
spike_info = find_spike_in_loss(histDf, threshold=1000)
print(f"Models and repeats with sudden spike in loss: {spike_info}")
plot_spike_loss_curve(histDf, spike_info)
# %%
