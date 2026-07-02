# %%
# update
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

# %%
# Define the folder containing the CSV files
# folder_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run1\20250430_ValidationStepsOFF"
# dataset = 'LFC18'
# folder_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run1"
# dataset = 'MC24x'
# folder_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run1"
# dataset = 'MC24'

# folder_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run7"
# dataset = 'LFC18'
# folder_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run7"
# dataset = 'MC24'
folder_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_run12"
dataset = 'MC24_1000'



# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Initialize a list to store the dataframes
dataframes = []
file_names = []
# Loop over each CSV file and load it into a pandas dataframe
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)
    file_names.append(os.path.basename(file))
    print(f"Loaded {file} successfully.")

# Initialize a list to store the summary results
summary_results = []

# MC24_1000
# Define the list of baseline model indices (one for each dataframe)
baseline_indices = [1,              # activationFunction relu
                    2,              # batchNorm off
                    5,              # batchSize 64 DIFFERENT TO MC24
                    1,              # dataAug on DIFFERENT TO MC24
                    1,              # decoderAct on DIFFERENT TO MC24

                    1,              # downsampling on DIFFERENT TO MC24

                    2,              # dropout 0.1
                    1,              # epsilon 1e-7

                    3,              # filterScale 0.5 DIFFERENT TO MC24

                    5,              # initialLr 0.001
                    5,              # kernelSize 5 DIFFERENT TO MC24
                    1,              # loss MSE
                    4,              # lrDecay 0.001 DIFFERENT TO MC24
                    1,              # maxPool on 
                    3,              # modelDepth 3
                    1,              # optimizer Adam
                    1,              # skinConnections on DIFFERENT TO MC24
                    2,              # trainValSplit 0.2 DIFFERENT TO MC24
                    ]

# Chosen indices for MC24_1000 (don't matter so are wrong)
# Define the list of baseline model indices (one for each dataframe)
opti_indices = [5,              # activationFunction leaky_relu
                    2,              # batchNorm off
                    5,              # batchSize 64 DIFFERENT TO MC24
                    1,              # dataAug on DIFFERENT TO MC24
                    1,              # decoderAct on DIFFERENT TO MC24

                    1,              # downsampling on DIFFERENT TO MC24

                    2,              # dropout 0.1
                    1,              # epsilon 1e-7

                    3,              # filterScale 0.5 DIFFERENT TO MC24

                    5,              # initialLr 0.001
                    3,              # kernelSize 5 DIFFERENT TO MC24
                    1,              # loss MSE
                    4,              # lrDecay 0.001 DIFFERENT TO MC24
                    1,              # maxPool on 
                    3,              # modelDepth 3
                    1,              # optimizer Adam
                    1,              # skinConnections on DIFFERENT TO MC24
                    2,              # trainValSplit 0.2 DIFFERENT TO MC24
                    ]

# # MC24
# # Define the list of baseline model indices (one for each dataframe)
# baseline_indices = [1,              # activationFunction
#                     2,              # batchNorm
#                     4,              # batchSize
#                     2,              # dataAug
#                     2,              # decoderAct
#                     2,              # dropout
#                     1,              # epsilon
#                     3,              # initialLr
#                     3,              # kernelSize
#                     1,              # loss
#                     1,              # lrDecay
#                     1,              # maxPool
#                     3,              # modelDepth
#                     1,              # optimizer
#                     2,              # skinConnections
#                     1,              # trainValSplit
#                     ]

# # Chosen indices for MC24
# # Define the list of baseline model indices (one for each dataframe)
# opti_indices = [5,              # activationFunction leaky_relu
#                     1,              # batchNorm on
#                     4,              # batchSize 8
#                     1,              # dataAug on
#                     1,              # decoderAct on
#                     3,              # dropout 0.15
#                     3,              # epsilon 1e-5
#                     3,              # initialLr 0.001
#                     3,              # kernelSize 3
#                     3,              # loss Custom
#                     1,              # lrDecay 1
#                     1,              # maxPool on
#                     4,              # modelDepth 4
#                     2,              # optimizer NAdam
#                     1,              # skinConnections on
#                     1,              # trainValSplit 0.1
#                     ]


# Define the list of baseline model indices LFC18
# baseline_indices = [1,              # activationFunction relu
#                     2,              # batchNorm off
#                     4,              # batchSize 8
#                     2,              # dataAug off
#                     2,              # decoderAct off
#                     2,              # dropout 0.1
#                     1,              # epsilon 1e-7
#                     3,              # initialLr 0.001
#                     3,              # kernelSize 3
#                     1,              # loss MSE
#                     1,              # lrDecay 1
#                     1,              # maxPool on
#                     3,              # modelDepth 3
#                     1,              # optimizer Adam
#                     2,              # skinConnections off
#                     1,              # trainValSplit 0.1
#                     ]

# # Chosen indices for LFC18
# # Define the list of baseline model indices (one for each dataframe)
# opti_indices = [5,              # activationFunction leaky_relu
#                     1,              # batchNorm on
#                     4,              # batchSize 8
#                     1,              # dataAug on
#                     1,              # decoderAct on
#                     3,              # dropout 0.15
#                     3,              # epsilon 1e-5
#                     3,              # initialLr 0.001
#                     3,              # kernelSize 3
#                     3,              # loss Custom
#                     1,              # lrDecay 1
#                     1,              # maxPool on
#                     4,              # modelDepth 4
#                     2,              # optimizer NAdam
#                     1,              # skinConnections on
#                     1,              # trainValSplit 0.1
#                     ]

for df in dataframes:
    print(df.columns)

# Loop through each dataframe and its corresponding baseline index
for df, baseline_index,opti_index, file_name in zip(dataframes, baseline_indices, opti_indices,file_names):
    # Identify the baseline model using the provided index
    baseline = df[df['Model'] == baseline_index]
    opti = df[df['Model'] == opti_index]
    worst_model = df[(df['Data'] == 'val') & (df['Metric'] == 'RMSE')].sort_values(by='Mean').iloc[-1]
    # Identify the best model based on the lowest RMSE on validation data
    best_model = df[(df['Data'] == 'val') & (df['Metric'] == 'RMSE')].sort_values(by='Mean').iloc[0]

    # Extract relevant metrics for the baseline
    baseline_rmse = baseline[(baseline['Data'] == 'val') & (baseline['Metric'] == 'RMSE')]['Mean'].values[0]
    baseline_ssim = baseline[(baseline['Data'] == 'val') & (baseline['Metric'] == 'SSIM')]['Mean'].values[0]
    baseline_train_rmse = baseline[(baseline['Data'] == 'train') & (baseline['Metric'] == 'RMSE')]['Mean'].values[0]
    baseline_train_ssim = baseline[(baseline['Data'] == 'train') & (baseline['Metric'] == 'SSIM')]['Mean'].values[0]

    opti_rmse = opti[(opti['Data'] == 'val') & (opti['Metric'] == 'RMSE')]['Mean'].values[0]
    opti_ssim = opti[(opti['Data'] == 'val') & (opti['Metric'] == 'SSIM')]['Mean'].values[0]
    opti_train_rmse = opti[(opti['Data'] == 'train') & (opti['Metric'] == 'RMSE')]['Mean'].values[0]
    opti_train_ssim = opti[(opti['Data'] == 'train') & (opti['Metric'] == 'SSIM')]['Mean'].values[0]

    # Extract relevant metrics for the best model
    best_model_rmse = best_model['Mean']
    best_model_ssim = df[(df['Model'] == best_model['Model']) & (df['Data'] == 'val') & (df['Metric'] == 'SSIM')]['Mean'].values[0]
    best_model_train_rmse = df[(df['Model'] == best_model['Model']) & (df['Data'] == 'train') & (df['Metric'] == 'RMSE')]['Mean'].values[0]
    best_model_train_ssim = df[(df['Model'] == best_model['Model']) & (df['Data'] == 'train') & (df['Metric'] == 'SSIM')]['Mean'].values[0]

    # Extract relevant metrics for the worst model
    worst_model_rmse = worst_model['Mean']
    worst_model_ssim = df[(df['Model'] == worst_model['Model']) & (df['Data'] == 'val') & (df['Metric'] == 'SSIM')]['Mean'].values[0]
    worst_model_train_rmse = df[(df['Model'] == worst_model['Model']) & (df['Data'] == 'train') & (df['Metric'] == 'RMSE')]['Mean'].values[0]
    worst_model_train_ssim = df[(df['Model'] == worst_model['Model']) & (df['Data'] == 'train') & (df['Metric'] == 'SSIM')]['Mean'].values[0]

    # Extract the hyperparameter being trialed (assuming it's stored in the second column of the dataframe)
    hyperparameter = df.columns[1] if len(df.columns) > 1 else 'Unknown'

    # Append the results to the summary
    summary_results.append({
        'File': os.path.basename(file_name),
        'Hyperparameter': hyperparameter,
        'BaseLine_HP_Value': baseline.iloc[0][hyperparameter],
        'Baseline_RMSE': baseline_rmse,
        'Baseline_SSIM': baseline_ssim,
        'Baseline_Train_RMSE': baseline_train_rmse,
        'Baseline_Train_SSIM': baseline_train_ssim,
        'Best_HP_Value': best_model[hyperparameter],
        'Best_Model': best_model['Model'],
        'Best_Model_RMSE': best_model_rmse,
        'Best_Model_SSIM': best_model_ssim,
        'Best_Model_Train_RMSE': best_model_train_rmse,
        'Best_Model_Train_SSIM': best_model_train_ssim,
        'Chosen_HP_Value': opti.iloc[0][hyperparameter],
        'Chosen_Model_RMSE': opti_rmse,
        'Chosen_Model_SSIM': opti_ssim,
        'Chosen_Model_Train_RMSE': opti_train_rmse,
        'Chosen_Model_Train_SSIM': opti_train_ssim,
        'Worst_Model': worst_model['Model'],
        'Worst_HP_Value': worst_model[hyperparameter],
        'Worst_Model_RMSE': worst_model_rmse,
        'Worst_Model_SSIM': worst_model_ssim,
        'Worst_Model_Train_RMSE': worst_model_train_rmse,
        'Worst_Model_Train_SSIM': worst_model_train_ssim,
    })




# Convert the summary results into a dataframe
summary_df = pd.DataFrame(summary_results)
# Save the summary to a CSV file
# output_csv_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results\summary_results.csv"
# summary_df.to_csv(output_csv_path, index=False)
# print(f"Summary results saved to {output_csv_path}")


# %% Load a summery results CSV file
# summary_csv_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_HP_Op_results\summary_results_LFC18.csv"
# summary_csv_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_HP_Op_results\summary_results_MC24.csv"
summary_csv_path = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results\summary_results_MC24_1000.csv"
summary_df = pd.read_csv(summary_csv_path)
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

# Create a new dataframe for the table
table_data = []

# Loop through each hyperparameter in summary_df
for _, row in summary_df.iterrows():
    hyperparameter = row['Hyperparameter']
    
    # Add Baseline row
    table_data.append({
        'Hyperparameter': hyperparameter,
        'Type': 'Baseline',
        'HP_Value': row['BaseLine_HP_Value'],
        # 'Train_RMSE': None,  # Assuming train metrics are not available in summary_df
        # 'Train_SSIM': None,
        'Val_RMSE': row['Baseline_RMSE'],
        'Val_SSIM': row['Baseline_SSIM']
    })
    
    # Add Best row
    table_data.append({
        'Hyperparameter': hyperparameter,
        'Type': 'Best',
        'HP_Value': row['Best_HP_Value'],
        # 'Train_RMSE': None,
        # 'Train_SSIM': None,
        'Val_RMSE': row['Best_Model_RMSE'],
        'Val_SSIM': row['Best_Model_SSIM']
    })
    
    # Add Worst row
    table_data.append({
        'Hyperparameter': hyperparameter,
        'Type': 'Worst',
        'HP_Value': row['Worst_HP_Value'],
        # 'Train_RMSE': None,
        # 'Train_SSIM': None,
        'Val_RMSE': row['Worst_Model_RMSE'],
        'Val_SSIM': row['Worst_Model_SSIM']
    })

    # Add Chosen row
    table_data.append({
        'Hyperparameter': hyperparameter,
        'Type': 'Chosen',
        'HP_Value': row['Chosen_HP_Value'],
        # 'Train_RMSE': None,
        # 'Train_SSIM': None,
        'Val_RMSE': row['Chosen_Model_RMSE'],
        'Val_SSIM': row['Chosen_Model_SSIM']
    })

# Convert the table data into a dataframe
table_df = pd.DataFrame(table_data)

# Display the table
print(table_df)

# Optionally, save the table to a CSV file
folderPath = os.path.dirname(summary_csv_path)
table_csv_path = os.path.join(folderPath, "hyperparameter_table.csv")
table_df.to_csv(table_csv_path, index=False)
print(f"Table saved to {table_csv_path}")

#%% Create a bar chart for baseline and best model RMSE
LFC18_table = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_HP_Op_results\hyperparameter_table_LFC18.csv")
MC24_table = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_HP_Op_results\hyperparameter_table_MC24.csv")
MC24_1000_table = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results\hyperparameter_table_MC24_1000.csv")

# LFC18_table = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_HP_Op_results\hyperparameter_table_LFC18.csv")
# MC24_table = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_HP_Op_results\hyperparameter_table_MC24.csv")
# MC24_1000_table = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results\hyperparameter_table_MC24_1000.csv")


# Quantify variance in baseline sweep
# Calculate mean and std of Val_RMSE for Baseline rows in each table
# lfc18_baseline_rmse = LFC18_table[LFC18_table['Type'] == 'Baseline']['Val_RMSE'].values
# mc24_baseline_rmse = MC24_table[MC24_table['Type'] == 'Baseline']['Val_RMSE'].values
# mc24_1000_baseline_rmse = MC24_1000_table[MC24_1000_table['Type'] == 'Baseline']['Val_RMSE'].values

# lfc18_best_rmse = LFC18_table[LFC18_table['Type'] == 'Best']['Val_RMSE'].values
# mc24_best_rmse = MC24_table[MC24_table['Type'] == 'Best']['Val_RMSE'].values
# mc24_1000_best_rmse = MC24_1000_table[MC24_1000_table['Type'] == 'Best']['Val_RMSE'].values

# lfc18_worst_rmse = LFC18_table[LFC18_table['Type'] == 'Worst']['Val_RMSE'].values
# mc24_worst_rmse = MC24_table[MC24_table['Type'] == 'Worst']['Val_RMSE'].values
# mc24_1000_worst_rmse = MC24_1000_table[MC24_1000_table['Type'] == 'Worst']['Val_RMSE'].values

# # Sweep range in performance of RMSE
# lfc18_RI = lfc18_worst_rmse - lfc18_best_rmse
# mc24_RI = mc24_worst_rmse - mc24_best_rmse
# mc24_1000_RI = mc24_1000_worst_rmse - mc24_1000_best_rmse
# # lfc18_RI = lfc18_baseline_rmse - lfc18_best_rmse
# # mc24_RI = mc24_baseline_rmse - mc24_best_rmse
# # mc24_1000_RI = mc24_1000_baseline_rmse - mc24_1000_best_rmse

# # Baseline stnardard deviation
# lfc18_baseline_std = np.std(lfc18_baseline_rmse)
# mc24_baseline_std = np.std(mc24_baseline_rmse)
# mc24_1000_baseline_std = np.std(mc24_1000_baseline_rmse)

# # Effect dominance ratio
# lfc18_EDR = lfc18_baseline_std/np.average(lfc18_RI)
# mc24_EDR = mc24_baseline_std/np.average(mc24_RI)   
# mc24_1000_EDR = mc24_1000_baseline_std/np.average(mc24_1000_RI)

# # Signal to noise ratio
# snr_lfc18 = lfc18_RI.mean() / lfc18_baseline_std
# snr_mc24 = mc24_RI.mean() / mc24_baseline_std

# MC24_1000_Baseline_Val_RMSE = [
#     0.085337866, 0.088553273, 0.083401878, 0.086940614, 0.083299193,
#     0.084158996, 0.08518527, 0.088397588, 0.090685673, 0.092126725,
#     0.084503254, 0.088032375, 0.082909175, 0.088029125, 0.080974954,
#     0.089127876, 0.087256656, 0.089822175, 0.089653644, 0.092322725,
#     0.085953117, 0.087342996, 0.084036794, 0.087344014, 0.084246805,
#     0.087061941, 0.087740131, 0.091966007, 0.089157192, 0.091351209
# ]
# mc24_1000_crossval_baseline_std = np.std(MC24_1000_Baseline_Val_RMSE)
# snr_mc24_1000 = mc24_1000_RI.mean() / mc24_1000_crossval_baseline_std

# snr_mc24_1000_crossval = mc24_1000_RI.mean() / np.std(mc24_baseline_rmse, ddof=1)

# # Print SNR statistics
# print("Signal-to-Noise Ratio (SNR) Statistics:")
# print(f"LFC18 SNR - Mean: {np.mean(snr_lfc18):.3f}, Median: {np.median(snr_lfc18):.3f}")
# print(f"MC24 SNR - Mean: {np.mean(snr_mc24):.3f}, Median: {np.median(snr_mc24):.3f}")
# print(f"MC24_1000 SNR - Mean: {np.mean(snr_mc24_1000):.3f}, Median: {np.median(snr_mc24_1000):.3f}")

# baseline = lfc18_baseline_rmse
# delta = lfc18_RI
# baseline = mc24_baseline_rmse
# delta = mc24_RI
# rng = np.random.default_rng(1)

# snr_boot = []
# for _ in range(10_000):
#     b = rng.choice(baseline, size=len(baseline), replace=True)
#     d = rng.choice(delta, size=len(delta), replace=True)
#     snr_boot.append(np.mean(d) / np.std(b, ddof=1))

# ci = np.percentile(snr_boot, [2.5, 97.5])
# ci
# plt.hist(snr_boot, bins=50)
# plt.show()


# # Create a figure with 3 subplots for the bar plots
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# # Get hyperparameter names for x-axis labels
# hp_names = LFC18_table[LFC18_table['Type'] == 'Baseline']['Hyperparameter'].values

# # Plot LFC18 data
# ax1.bar(range(len(lfc18_RI)), lfc18_RI, alpha=0.7, color=palette[0])
# ax1.axhline(y=lfc18_baseline_std, color='red', linestyle='--', linewidth=2, label=f'Baseline Std: {lfc18_baseline_std:.3f}')
# ax1.set_title('LFC18 - Range of Improvement vs Baseline Std')
# ax1.set_xlabel('Hyperparameters')
# ax1.set_ylabel('RMSE Range of Improvement')
# ax1.set_xticks(range(len(hp_names)))
# ax1.set_xticklabels(hp_names, rotation=45, ha='right')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# # Plot MC24 data
# ax2.bar(range(len(mc24_RI)), mc24_RI, alpha=0.7, color=palette[1])
# ax2.axhline(y=mc24_baseline_std, color='red', linestyle='--', linewidth=2, label=f'Baseline Std: {mc24_baseline_std:.3f}')
# ax2.set_title('MC24 - Range of Improvement vs Baseline Std')
# ax2.set_xlabel('Hyperparameters')
# ax2.set_ylabel('RMSE Range of Improvement')
# ax2.set_xticks(range(len(hp_names)))
# ax2.set_xticklabels(hp_names, rotation=45, ha='right')
# ax2.legend()
# ax2.grid(True, alpha=0.3)

# # Plot MC24_1000 data (excluding Downsampling and Filter scale)
# hp_names_filtered = MC24_1000_table[MC24_1000_table['Type'] == 'Baseline']['Hyperparameter'].values
# ax3.bar(range(len(mc24_1000_RI)), mc24_1000_RI, alpha=0.7, color=palette[2])
# ax3.axhline(y=mc24_1000_baseline_std, color='red', linestyle='--', linewidth=2, label=f'Baseline Std: {mc24_1000_baseline_std:.3f}')
# ax3.set_title('MC24_1000 - Range of Improvement vs Baseline Std')
# ax3.set_xlabel('Hyperparameters')
# ax3.set_ylabel('RMSE Range of Improvement')
# ax3.set_xticks(range(len(hp_names_filtered)))
# ax3.set_xticklabels(hp_names_filtered, rotation=45, ha='right')
# ax3.legend()
# ax3.grid(True, alpha=0.3)

# plt.tight_layout()

# # Print Effect Dominance Ratios
# print(f"Effect Dominance Ratios:")
# print(f"LFC18 EDR: {lfc18_EDR:.3f}")
# print(f"MC24 EDR: {mc24_EDR:.3f}")
# print(f"MC24_1000 EDR: {mc24_1000_EDR:.3f}")

# plt.show()


# print("LFC18 Baseline Val_RMSE:")
# print(f"Mean: {lfc18_baseline_rmse.mean():.4f}")
# print(f"Std: {lfc18_baseline_rmse.std():.4f}")

# print("\nMC24 Baseline Val_RMSE:")
# print(f"Mean: {mc24_baseline_rmse.mean():.4f}")
# print(f"Std: {mc24_baseline_rmse.std():.4f}")

# print("\nMC24_1000 Baseline Val_RMSE:")
# print(f"Mean: {mc24_1000_baseline_rmse.mean():.4f}")
# print(f"Std: {mc24_1000_baseline_rmse.std():.4f}")



# Drop rows in MC24_1000_table where Hyperparameter is Downsampling or Filter scale
# MC24_1000_table = MC24_1000_table[~MC24_1000_table['Hyperparameter'].isin(['Downsampling', 'Filter scale'])]
plotMC24_1000 = True
if plotMC24_1000:
    ncols = 3
    ticknums = 2
else:
    ncols = 2
    ticknums = 3

plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25

plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "8"
plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = 2*figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly
# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(figWidth * 2, figHeight), gridspec_kw={'width_ratios': [1, 1]}, sharey=True)

# This approach plots both RMSE and SSIM on a single subplot
fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
axLFC18 = fig.add_subplot(1, ncols, 1)
ax2LFC18 = axLFC18.twiny()
axMC24 = fig.add_subplot(1, ncols, 2)
ax2MC24 = axMC24.twiny()
if plotMC24_1000:
    axMC24_1000 = fig.add_subplot(1, ncols, 3)
    ax2MC24_1000 = axMC24_1000.twiny()

fig.set_figheight(figHeight)

fig.set_figwidth(figWidth *2)

# Extract RMSE data for plotting from table_df
rmse_data_LFC18 = LFC18_table[LFC18_table['Type'].isin(['Baseline', 'Best'])].pivot(index='Hyperparameter', columns='Type', values='Val_RMSE').reset_index()
rmse_data_LFC18.rename(columns={'Baseline': 'Baseline_RMSE', 'Best': 'Best_Model_RMSE'}, inplace=True)
rmse_data_MC24 = MC24_table[MC24_table['Type'].isin(['Baseline', 'Best'])].pivot(index='Hyperparameter', columns='Type', values='Val_RMSE').reset_index()
rmse_data_MC24.rename(columns={'Baseline': 'Baseline_RMSE', 'Best': 'Best_Model_RMSE'}, inplace=True)
rmse_data_MC24_1000 = MC24_1000_table[MC24_1000_table['Type'].isin(['Baseline', 'Best'])].pivot(index='Hyperparameter', columns='Type', values='Val_RMSE').reset_index()
rmse_data_MC24_1000.rename(columns={'Baseline': 'Baseline_RMSE', 'Best': 'Best_Model_RMSE'}, inplace=True)


# Extract SSIM data for plotting from table_df
ssim_data_LFC18 = LFC18_table[LFC18_table['Type'].isin(['Baseline', 'Best'])].pivot(index='Hyperparameter', columns='Type', values='Val_SSIM').reset_index()
ssim_data_LFC18.rename(columns={'Baseline': 'Baseline_SSIM', 'Best': 'Best_Model_SSIM'}, inplace=True)
ssim_data_MC24 = MC24_table[MC24_table['Type'].isin(['Baseline', 'Best'])].pivot(index='Hyperparameter', columns='Type', values='Val_SSIM').reset_index()
ssim_data_MC24.rename(columns={'Baseline': 'Baseline_SSIM', 'Best': 'Best_Model_SSIM'}, inplace=True)
ssim_data_MC24_1000 = MC24_1000_table[MC24_1000_table['Type'].isin(['Baseline', 'Best'])].pivot(index='Hyperparameter', columns='Type', values='Val_SSIM').reset_index()
ssim_data_MC24_1000.rename(columns={'Baseline': 'Baseline_SSIM', 'Best': 'Best_Model_SSIM'}, inplace=True)


# Bar width
bar_width = 0.8
front_bar_width = bar_width 
if plotMC24_1000:
    x = np.arange(len(rmse_data_MC24_1000['Hyperparameter']))

    # Add a row for 'Downsampling' hyperparameter with NaN values for LFC18
    downsampling_row = pd.DataFrame({
        'Hyperparameter': ['Downsampling'],
        'Baseline_RMSE': [np.nan],
        'Best_Model_RMSE': [np.nan]
    })
    # For downsampling run 6 has downsampling on and run 7 has downsampling off. 
    #LFC18 use trainValSplit sweep to represent downsampling
    downSamp_on_RMSE_LFC18 = 0.221158014549408
    downSamp_on_SSIM_LFC18 = 0.451950867966666
    downSamp_off_RMSE_LFC18 = 0.17109192141300833
    downSamp_off_SSIM_LFC18 = 0.7257346113666667

    downSamp_on_RMSE_MC24 = 0.0858172575804654
    downSamp_on_SSIM_MC24 = 0.5502111315333332
    downSamp_off_RMSE_MC24 = 0.0870363667941912
    downSamp_off_SSIM_MC24 = 0.7337707678666666

    downsampling_RMSE_LFC18 = pd.DataFrame({
        'Hyperparameter': ['Downsampling'],
        'Baseline_RMSE': [downSamp_off_RMSE_LFC18],
        'Best_Model_RMSE': [downSamp_off_RMSE_LFC18]
    }) 
    downsampling_SSIM_LFC18 = pd.DataFrame({
        'Hyperparameter': ['Downsampling'],
        'Baseline_SSIM': [downSamp_off_SSIM_LFC18],
        'Best_Model_SSIM': [downSamp_off_SSIM_LFC18]
    }) 
    downsampling_RMSE_MC24 = pd.DataFrame({
        'Hyperparameter': ['Downsampling'],
        'Baseline_RMSE': [downSamp_off_RMSE_MC24],
        'Best_Model_RMSE': [downSamp_on_RMSE_MC24]
    }) 
    downsampling_SSIM_MC24 = pd.DataFrame({
        'Hyperparameter': ['Downsampling'],
        'Baseline_SSIM': [downSamp_off_SSIM_MC24],
        'Best_Model_SSIM': [downSamp_on_SSIM_MC24]
    }) 




    filterscale_row = pd.DataFrame({
        'Hyperparameter': ['Filter scale'],
        'Baseline_RMSE': [np.nan],
        'Best_Model_RMSE': [np.nan]
    })
    rmse_data_LFC18 = pd.concat([rmse_data_LFC18, downsampling_RMSE_LFC18, filterscale_row], ignore_index=True)
    rmse_data_MC24 = pd.concat([rmse_data_MC24, downsampling_RMSE_MC24, filterscale_row], ignore_index=True)
    ssim_data_LFC18 = pd.concat([ssim_data_LFC18, downsampling_SSIM_LFC18, filterscale_row.rename(columns={'Baseline_RMSE': 'Baseline_SSIM', 'Best_Model_RMSE': 'Best_Model_SSIM'})], ignore_index=True)
    ssim_data_MC24 = pd.concat([ssim_data_MC24, downsampling_SSIM_MC24, filterscale_row.rename(columns={'Baseline_RMSE': 'Baseline_SSIM', 'Best_Model_RMSE': 'Best_Model_SSIM'})], ignore_index=True)
else:
    x = np.arange(len(rmse_data_LFC18['Hyperparameter']))



# Rename
batch_size_row = rmse_data_LFC18['Hyperparameter'] == 'Batch size'
rmse_data_LFC18.loc[batch_size_row, 'Hyperparameter'] = 'Mini-batch size'
conv_kernel_row = rmse_data_LFC18['Hyperparameter'] == 'Convolutional kernel width/height'
rmse_data_LFC18.loc[conv_kernel_row, 'Hyperparameter'] = 'Conv. kernel size'
lr_decay_row = rmse_data_LFC18['Hyperparameter'] == 'Learning rate decay per 1000 epochs'
rmse_data_LFC18.loc[lr_decay_row, 'Hyperparameter'] = 'Learning rate decay factor'
filter_scale_row = rmse_data_LFC18['Hyperparameter'] == 'Filter scale'
rmse_data_LFC18.loc[filter_scale_row, 'Hyperparameter'] = 'Filter scaling'
ssim_data_LFC18['Hyperparameter'] = rmse_data_LFC18['Hyperparameter']


batch_size_row = rmse_data_MC24['Hyperparameter'] == 'Batch size'
rmse_data_MC24.loc[batch_size_row, 'Hyperparameter'] = 'Mini-batch size'
conv_kernel_row = rmse_data_MC24['Hyperparameter'] == 'Convolutional kernel width/height'
rmse_data_MC24.loc[conv_kernel_row, 'Hyperparameter'] = 'Conv. kernel size'
lr_decay_row = rmse_data_MC24['Hyperparameter'] == 'Learning rate decay per 1000 epochs'
rmse_data_MC24.loc[lr_decay_row, 'Hyperparameter'] = 'Learning rate decay factor'
filter_scale_row = rmse_data_MC24['Hyperparameter'] == 'Filter scale'
rmse_data_MC24.loc[filter_scale_row, 'Hyperparameter'] = 'Filter scaling'
ssim_data_MC24['Hyperparameter'] = rmse_data_MC24['Hyperparameter']

batch_size_row = rmse_data_MC24_1000['Hyperparameter'] == 'Batch size'
rmse_data_MC24_1000.loc[batch_size_row, 'Hyperparameter'] = 'Mini-batch size'
conv_kernel_row = rmse_data_MC24_1000['Hyperparameter'] == 'Convolutional kernel width/height'
rmse_data_MC24_1000.loc[conv_kernel_row, 'Hyperparameter'] = 'Conv. kernel size'
lr_decay_row = rmse_data_MC24_1000['Hyperparameter'] == 'Learning rate decay per 1000 epochs'
rmse_data_MC24_1000.loc[lr_decay_row, 'Hyperparameter'] = 'Learning rate decay factor'
filter_scale_row = rmse_data_MC24_1000['Hyperparameter'] == 'Filter scale'
rmse_data_MC24_1000.loc[filter_scale_row, 'Hyperparameter'] = 'Filter scaling'
ssim_data_MC24_1000['Hyperparameter'] = rmse_data_MC24_1000['Hyperparameter']

# Re-order the dataframes alphabetically by the 'Hyperparameter' column
rmse_data_LFC18 = rmse_data_LFC18.sort_values(by='Hyperparameter', ascending=False)
rmse_data_MC24 = rmse_data_MC24.sort_values(by='Hyperparameter', ascending=False)
rmse_data_MC24_1000 = rmse_data_MC24_1000.sort_values(by='Hyperparameter', ascending=False)

ssim_data_LFC18 = ssim_data_LFC18.sort_values(by='Hyperparameter', ascending=False)
ssim_data_MC24 = ssim_data_MC24.sort_values(by='Hyperparameter', ascending=False)
ssim_data_MC24_1000 = ssim_data_MC24_1000.sort_values(by='Hyperparameter', ascending=False)



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

back_palette = {
# 'baseline': '#3c3c3c',
# 'modified_baseline': '#7c7c7c',
'baseline': "#de8f05",
'modified_baseline': "#d55e00",

}



# Create the colours from the seaborn colourblind palette
# palette = sns.color_palette("colorblind")
# front_color = palette[2]  # First color in the palette
# back_color = palette[1]  # Eighth color in the palette
# modified_back_color = palette[3]  # Second color in the palette

front_color_bar = RMSE_palette['val']  # First color in the palette
front_color_scatter = RMSE_palette['val']  # First color in the palette
back_color = back_palette['baseline']  # Eighth color in the palette
modified_back_color = back_palette['modified_baseline']  # Second color in the palette

    
# Plot baseline RMSE bars horizontally
baseline_bars_LFC18 = axLFC18.barh(x, rmse_data_LFC18['Baseline_RMSE'], height=bar_width, color=back_color, alpha=1, label='Baseline', zorder=3)
baseline_bars_MC24 = axMC24.barh(x, rmse_data_MC24['Baseline_RMSE'], height=bar_width, color=back_color, alpha=1, label='Baseline', zorder=3)
# Plot best model RMSE bars horizontally in front
best_bars_LFC18 = axLFC18.barh(x, rmse_data_LFC18['Best_Model_RMSE'], height=front_bar_width, color=front_color_bar, label='Best', zorder=3)
best_bars_MC24 = axMC24.barh(x, rmse_data_MC24['Best_Model_RMSE'], height=front_bar_width, color=front_color_bar, label='Best', zorder=3)
# Plot SSIM values as stars on the second subplot
# baseline_bars = ax2.barh(x, ssim_data['Baseline_SSIM'], height=bar_width, color=back_color, alpha=0.6, label='Baseline', zorder=3)
# # Plot best model RMSE bars horizontally in front
# best_bars = ax2.barh(x, ssim_data['Best_Model_SSIM'], height=bar_width, color=front_color, label='Best', zorder=3)
if plotMC24_1000:
    baseline_bars_MC24_1000 = axMC24_1000.barh(x, rmse_data_MC24_1000['Baseline_RMSE'], height=bar_width, color=modified_back_color, alpha=1, label='Modified Baseline', zorder=3)
    best_bars_MC24_1000 = axMC24_1000.barh(x, rmse_data_MC24_1000['Best_Model_RMSE'], height=front_bar_width, color=front_color_bar, label='Best', zorder=3)


baseline_stars_LFC18 = ax2LFC18.scatter(ssim_data_LFC18['Baseline_SSIM'], x,marker = 'o', facecolors='none', color=back_color, s=50, zorder=4,alpha = 1, label='Baseline SSIM')
baseline_stars_MC24 = ax2MC24.scatter(ssim_data_MC24['Baseline_SSIM'], x,marker = 'o', facecolors='none', color=back_color, s=50, zorder=4,alpha = 1, label='Baseline SSIM')

best_stars_LFC18 = ax2LFC18.scatter(ssim_data_LFC18['Best_Model_SSIM'], x, marker='*', color=front_color_scatter, s=50, zorder=4, label='Best SSIM')
best_stars_MC24 = ax2MC24.scatter(ssim_data_MC24['Best_Model_SSIM'], x, marker='*', color=front_color_scatter, s=50, zorder=4, label='Best SSIM')

if plotMC24_1000:
    baseline_stars_MC24_1000 = ax2MC24_1000.scatter(ssim_data_MC24_1000['Baseline_SSIM'], x,marker = 'o', facecolors='none', color=modified_back_color, s=50, zorder=4,alpha = 1, label='Modified Baseline SSIM')
    best_stars_MC24_1000 = ax2MC24_1000.scatter(ssim_data_MC24_1000['Best_Model_SSIM'], x, marker='*', color=front_color_scatter, s=50, zorder=4, label='Best SSIM')

RMSEpos = 0.21
SSIMpos = 0.78



from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle
import matplotlib.lines as mlines

# def add_axis_break(ax, x_center=0.55, width=0.04, height=0.12,
#                    line_width=1.6, color='black', zorder=1000):
#     """
#     Draws a break (white box + diagonal slashes) ON TOP of everything
#     using axis-fraction coordinates.
#     """
#     x0 = x_center - width/2
#     y0 = 0.5 - height/2
    
#     # White cover rectangle
#     ax.add_patch(Rectangle(
#         (x0, y0), width, height,
#         transform=ax.transAxes,
#         color='white',
#         zorder=zorder,
#         clip_on=False
#     ))

#     # Left diagonal
#     ax.add_line(mlines.Line2D(
#         [x0 + width*0.05, x0 + width*0.25],
#         [0.5 - height/2, 0.5 + height/2],
#         # [0.5 - height*0.25, 0.5 + height*0.25],
#         transform=ax.transAxes,
#         color=color, lw=line_width,
#         zorder=zorder+1, clip_on=False
#     ))

#     # Right diagonal
#     ax.add_line(mlines.Line2D(
#         [x0 + width*0.75, x0 + width*0.95],
#         [0.5 - height/2, 0.5 + height/2],
#         # [0.5 - height*0.25, 0.5 + height*0.25],
#         transform=ax.transAxes,
#         color=color, lw=line_width,
#         zorder=zorder+1, clip_on=False
#     ))

def add_axis_break(ax, x_center=0.0,y_center=0, width=0.04, height=0.05,
                   line_width=1.6, color='black', zorder=1000):
    """
    Draws a break (white box + diagonal slashes) ON TOP of everything
    using axis-fraction coordinates.
    """
    x0 = x_center - width/2
    y0 = y_center - height/2

    
    # White cover rectangle
    ax.add_patch(Rectangle(
        (x0, y0), width, height,
        transform=ax.transAxes,
        color='white',
        zorder=zorder,
        clip_on=False
    ))

    # Left diagonal
    ax.add_line(mlines.Line2D(
        [x0 + width*0.05, x0 + width*0.25],
        [y0, y0 + height],
        # [0.5 - height*0.25, 0.5 + height*0.25],
        transform=ax.transAxes,
        color=color, lw=line_width,
        zorder=zorder+1, clip_on=False
    ))

    # Right diagonal
    ax.add_line(mlines.Line2D(
        [x0 + width*0.75, x0 + width*0.95],
        [y0, y0 + height],
        # [0.5 - height*0.25, 0.5 + height*0.25],
        transform=ax.transAxes,
        color=color, lw=line_width,
        zorder=zorder+1, clip_on=False
    ))


# zigzag = 0.005
# zigzagy = np.linspace(x[0]-1.5, x[-1]+1.5, num=4)
# zigzag = Path([
#     (xmaxLFC18 - zigzag, zigzagy[0]),
#     (xmaxLFC18 + zigzag, zigzagy[1]),
#     (xmaxLFC18 - zigzag, zigzagy[2]),
#     (xmaxLFC18 + zigzag, zigzagy[3]),
# ], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])

# patch = PathPatch(
#     zigzag,
#     transform=axLFC18.transData,
#     lw=1,
#     alpha=1,
#     edgecolor='black',
#     facecolor='white',
#     fill=True,
#     clip_on=False,
#     zorder=10  # Ensure it is drawn on top
# )

# axLFC18.add_patch(patch)

# Customize the first subplot (LFC18))
axLFC18.set_ylim(-1, len(rmse_data_LFC18['Hyperparameter']))
axLFC18.set_yticks(x)
axLFC18.set_yticklabels(rmse_data_LFC18['Hyperparameter'])
axLFC18.set_xlabel('RMSE', x=RMSEpos)
xminLFC18 = np.floor(rmse_data_LFC18['Best_Model_RMSE'].min()*100)/100
xmaxLFC18 = np.ceil(rmse_data_LFC18['Baseline_RMSE'].max()*100)/100
axLFC18.set_xlim([xminLFC18, xmaxLFC18 * 2 - xminLFC18])
# ax.legend(loc='lower left', bbox_to_anchor=(1, 0))  # outside the plot
axLFC18.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
# Manually set xticks for axLFC18 at reasonable intervals with least possible digits
xticks = np.linspace(xminLFC18, xmaxLFC18-0.01, num=ticknums)
# xticks = np.linspace(np.round(rmse_data_LFC18['Best_Model_RMSE'].min(),2), np.round(rmse_data_LFC18['Baseline_RMSE'].max(),2), num=3)
# xticks = np.round(np.linspace(rmse_data_LFC18['Best_Model_RMSE'].min(), rmse_data_LFC18['Baseline_RMSE'].max(), num=3), 2)
axLFC18.set_xticks(xticks)
# axLFC18.set_title('MeC-Macro')
# axLFC18.set_title("(a)", fontweight='bold')

ax2LFC18.set_xlabel('SSIM', x=SSIMpos)
minSSIM = np.min([ssim_data_LFC18['Baseline_SSIM'].min(), ssim_data_LFC18['Best_Model_SSIM'].min()])
xminLFC18 =  np.floor(minSSIM*100)/100
xmaxLFC18 = np.ceil(ssim_data_LFC18['Best_Model_SSIM'].max()*100)/100
ax2LFC18.set_xlim([(2*xminLFC18 - xmaxLFC18)*0.9 , xmaxLFC18 * 1.01])
ax2LFC18.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
ax2LFC18.tick_params(axis='y', left=False, labelleft=False)
xticks2 = np.linspace(xminLFC18+0.01, xmaxLFC18, num=ticknums)
# xticks2 = np.linspace(np.round(minSSIM,2), np.round(ssim_data_LFC18['Best_Model_SSIM'].max(),2), num=3)
ax2LFC18.set_xticks(xticks2)

add_axis_break(ax2LFC18, x_center=0.5,y_center=1, width=0.05, height=0.03,
                   line_width=1, color='black', zorder=1000)
add_axis_break(ax2LFC18, x_center=0.5,y_center=0, width=0.05, height=0.03,
                   line_width=1, color='black', zorder=1000)

# Customize the second subplot (MC24))
axMC24.set_ylim(-1, len(rmse_data_MC24['Hyperparameter']))
axMC24.set_yticks(x)
axMC24.set_yticklabels([])
# axMC24.set_yticklabels(rmse_data_MC24['Hyperparameter'])
axMC24.set_xlabel('RMSE', x=RMSEpos)
xminMC24 = np.floor(rmse_data_MC24['Best_Model_RMSE'].min()*100)/100
xmaxMC24 = np.ceil(rmse_data_MC24['Baseline_RMSE'].max()*100)/100
axMC24.set_xlim([xminMC24, (xmaxMC24 * 2 - xminMC24)*0.95])
# ax.legend(loc='lower left', bbox_to_anchor=(1, 0))  # outside the plot
axMC24.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
xticks = np.linspace(xminMC24, xmaxMC24-0.01, num=ticknums)

axMC24.set_xticks(xticks)
# axMC24.set_title('MeC-Meso-S')
# axMC24.set_title("(b)", fontweight='bold')


ax2MC24.set_xlabel('SSIM', x=SSIMpos)
minSSIMMC24 = np.min([ssim_data_MC24['Baseline_SSIM'].min(), ssim_data_MC24['Best_Model_SSIM'].min()])
xminMC24 =  np.floor(minSSIMMC24*100)/100
xmaxMC24 = np.ceil(ssim_data_MC24['Best_Model_SSIM'].max()*100)/100
ax2MC24.set_xlim([(2*minSSIMMC24 - xmaxMC24)*0.75 , xmaxMC24 * 1.03])
ax2MC24.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
ax2MC24.tick_params(axis='y', left=False, labelleft=False)
xticks2 = np.linspace(xminMC24, xmaxMC24, num=ticknums)
ax2MC24.set_xticks(xticks2)

add_axis_break(ax2MC24, x_center=0.5,y_center=1, width=0.05, height=0.03,
                   line_width=1, color='black', zorder=1000)
add_axis_break(ax2MC24, x_center=0.5,y_center=0, width=0.05, height=0.03,
                   line_width=1, color='black', zorder=1000)

if plotMC24_1000:
    # Customize the third subplot (MC24_1000)
    axMC24_1000.set_ylim(-1, len(rmse_data_MC24_1000['Hyperparameter']))
    axMC24_1000.set_yticks(x)
    axMC24_1000.set_yticklabels([])
    axMC24_1000.set_xlabel('RMSE', x=RMSEpos)
    xminMC24_1000 = np.floor(rmse_data_MC24_1000['Best_Model_RMSE'].min()*100)/100
    xmaxMC24_1000 = np.ceil(rmse_data_MC24_1000['Baseline_RMSE'].max()*100)/100
    axMC24_1000.set_xlim([xminMC24_1000, (xmaxMC24_1000 * 2 - xminMC24_1000)*0.95])
    axMC24_1000.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    xticks = np.linspace(xminMC24_1000, xmaxMC24_1000-0.01, num=ticknums)
    axMC24_1000.set_xticks(xticks)
    # axMC24_1000.set_title('MeC-Meso-M')
    # axMC24_1000.set_title("(c)", fontweight='bold')

    ax2MC24_1000.set_xlabel('SSIM', x=SSIMpos)
    minSSIMMC24_1000 = np.min([ssim_data_MC24_1000['Baseline_SSIM'].min(), ssim_data_MC24_1000['Best_Model_SSIM'].min()])
    xminMC24_1000 = np.floor(minSSIMMC24_1000*100)/100
    xmaxMC24_1000 = np.ceil(ssim_data_MC24_1000['Best_Model_SSIM'].max()*100)/100
    ax2MC24_1000.set_xlim([(2*minSSIMMC24_1000 - xmaxMC24_1000)*0.93, xmaxMC24_1000 * 1.01])
    ax2MC24_1000.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    ax2MC24_1000.tick_params(axis='y', left=False, labelleft=False)
    xticks2 = np.linspace(xminMC24_1000, xmaxMC24_1000, num=ticknums)
    ax2MC24_1000.set_xticks(xticks2)

    add_axis_break(ax2MC24_1000, x_center=0.5, y_center=1, width=0.05, height=0.03,
                line_width=1, color='black', zorder=1000)
    add_axis_break(ax2MC24_1000, x_center=0.5, y_center=0, width=0.05, height=0.03,
                line_width=1, color='black', zorder=1000)

if plotMC24_1000:
    # axLFC18.get_legend_handles_labels()[0]

    # fig.legend(handles=[axLFC18.get_legend_handles_labels()[0][0], axMC24_1000.get_legend_handles_labels()[0][0],axMC24_1000.get_legend_handles_labels()[0][1]], labels=[axLFC18.get_legend_handles_labels()[1][0], axMC24_1000.get_legend_handles_labels()[1][0],axMC24_1000.get_legend_handles_labels()[1][1]], 
    #         loc="upper left", ncol=2, bbox_to_anchor=(0, 1))
    # create blank rectangle
    extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    
    #Create organized list containing all handles for table. Extra represent empty space
    # legend_handle = [extra, extra, extra, extra, extra, baseline_bars_LFC18, baseline_bars_MC24_1000, best_bars_LFC18, extra, baseline_stars_LFC18, baseline_stars_MC24_1000, best_stars_LFC18]
    legend_handle = [extra, extra, extra, extra, baseline_bars_LFC18, baseline_stars_LFC18, extra, baseline_bars_MC24_1000, baseline_stars_MC24_1000, extra, best_bars_LFC18, best_stars_LFC18]
    #Define the labels
    label_col_1 = ["", "RMSE", "SSIM"]
    label_j_1 = ["Baseline"]
    label_j_2 = ["Modified Baseline"]
    label_j_3 = ["One-at-a-time Optimised"]
    label_empty = [""]

    #organize labels for table construction
    legend_labels = np.concatenate([label_col_1, label_j_1, label_empty * 2, label_j_2, label_empty * 2, label_j_3, label_empty * 2])


    #Create legend
    fig.legend(legend_handle, legend_labels, 
            loc = 2, ncol = 4, shadow = False, handletextpad = -2, bbox_to_anchor=(0.26, 1.15))
    # fig.legend(handles=[baseline_bars_LFC18, baseline_bars_MC24_1000, best_bars_LFC18,
    #                     baseline_stars_LFC18, baseline_stars_MC24_1000, best_stars_LFC18],
    #           labels=['Baseline', 'Modified Baseline', 'One-at-a-time Optimised',
    #                   'Baseline', 'Modified Baseline', 'One-at-a-time Optimised'],
    #           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15),
    #           title='RMSE (bars) / SSIM (markers)')

else:
    fig.legend(handles=axLFC18.get_legend_handles_labels()[0], labels=axLFC18.get_legend_handles_labels()[1], 
            loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
# Add horizontal dotted lines between the yticks for both ax and ax2
for axis in [axLFC18, ax2LFC18]:
    for ytick in range(len(x)):
        # mid_point = (x[ytick] + x[ytick + 1]) / 2
        mid_point = ytick
        axis.axhline(y=mid_point, color='grey', linewidth=0.2, zorder=0)
for axis in [axMC24, ax2MC24]:
    for ytick in range(len(x)):
        # mid_point = (x[ytick] + x[ytick + 1]) / 2
        mid_point = ytick
        axis.axhline(y=mid_point, color='grey', linewidth=0.2, zorder=0)

if plotMC24_1000:
    for axis in [axMC24_1000, ax2MC24_1000]:
        for ytick in range(len(x)):
            # mid_point = (x[ytick] + x[ytick + 1]) / 2
            mid_point = ytick
            axis.axhline(y=mid_point, color='grey', linewidth=0.2, zorder=0)

# Adjust layout to make it tight
plt.tight_layout( w_pad=1.25)
# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
# Plot SSIM values as stars on top of the bars on a secondary x-axis
# Create a second x-axis for SSIM values
# ax2 = ax.twiny()

# # Plot SSIM values as stars on the second x-axis
# baseline_stars = ax2.scatter(ssim_data['Baseline_SSIM'], x, marker='*', color=back_color, s=20, zorder=4)
# best_stars = ax2.scatter(ssim_data['Best_Model_SSIM'], x, marker='*', color=front_color, s=20, zorder=4)

# # Customize the second x-axis
# ax2.set_xlabel('SSIM')
# ax2.set_xlim([0.0, 1.0])  # Adjust the range as needed

# Add RMSE values next to the bars
# for bar in baseline_bars:
#     width = bar.get_width()
#     ax.annotate(f'{width:.2f}', 
#                 xy=(width, bar.get_y() + bar.get_height() / 2),
#                 xytext=(3, 0),  # Offset text by 3 points
#                 textcoords="offset points",
#                 ha='left', va='center', fontsize=6, color='black')


# for bar in best_bars:
#     width = bar.get_width()
#     ax.annotate(f'{width:.2f}', 
#                 xy=(width, bar.get_y() + bar.get_height() / 2),
#                 xytext=(-3, 0),  # Offset text inside the bar
#                 textcoords="offset points",
#                 ha='right', va='center', fontsize=6, color='white')




# Customize the plot
# ax.set_yticks(x)
# # Calculate the limits for the x-axis
# best_rmse = rmse_data['Best_Model_RMSE'].min()
# worst_rmse = rmse_data['Baseline_RMSE'].max()
# ax.set_xlim([best_rmse * 0.9, worst_rmse * 2 - best_rmse * 0.9])

# # Calculate the limits for the x-axis of SSIM
# best_ssim = ssim_data['Best_Model_SSIM'].max()
# worst_ssim = ssim_data['Baseline_SSIM'].min()
# # ax2.set_xlim([worst_ssim / 2, best_ssim * 1.05])

# ax.set_yticklabels(rmse_data['Hyperparameter'])
# ax.set_xlabel('RMSE')
# # ax.set_title('Baseline vs Best Model RMSE')
# ax.legend(loc='lower left', bbox_to_anchor=(1, 0))  # outside the plot

# # Add grid
# ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

# Save the plot
# folderPath = r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_HP_Op_results'
# output_path = os.path.join(folderPath, "Baseline_vs_Best_performance_horizontal.pdf")
# plt.savefig(output_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)
# print(f"Horizontal bar chart saved to {output_path}")
# folderPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures'
folderPath = r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures'
output_path = os.path.join(folderPath, "InitialSweep_Results_v3.pdf")
plt.savefig(output_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)
print(f"Horizontal bar chart saved to {output_path}")

plt.show()


# %% Create a single table for individual hyperparameter results
LFC18_table = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_HP_Op_results\hyperparameter_table_LFC18.csv")
MC24_table = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_HP_Op_results\hyperparameter_table_MC24.csv")


# %%

gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)

axs = gs.subplots(sharex='col')

# fig, axs = plt.subplots(2, 2,sharex='col', sharey='row')
fig.set_figheight(figHeight*2)
fig.set_figwidth(figWidth*2)
plotparam_s = ['RMSE','SSIM']
plotDs_s = [dataset]
lett = np.array(['(a)', '(b)'])

# RMSE #################################

# Extract RMSE data from summary_results
RMSEdf = pd.DataFrame(summary_results)[['Hyperparameter', 'Baseline_RMSE', 'Best_Model_RMSE']]
# Calculate percentage improvement
RMSEdf['Improvement [%]'] = 100 * (RMSEdf['Baseline_RMSE'] - RMSEdf['Best_Model_RMSE']) / RMSEdf['Baseline_RMSE']
# Rename columns for clarity
RMSEdf.rename(columns={'Baseline_RMSE': 'Baseline', 'Best_Model_RMSE': 'Best'}, inplace=True)
# Melt the dataframe for easier plotting
RMSEdf = RMSEdf.melt(id_vars=['Hyperparameter', 'Improvement [%]'], value_vars=['Baseline', 'Best'], 
                     var_name='Value', value_name='RMSE')
# Plot the RMSE data
p = sns.pointplot(ax=axs[0], data=RMSEdf, x='Hyperparameter', y='RMSE', hue='Value', 
                  dodge=0.2, linestyle="none", errorbar=None, marker="_")

# # Annotate percentage improvement
for n in range(len(summary_df)):
    bsrow = RMSEdf.loc[(RMSEdf['Hyperparameter'] == summary_df.iloc[n]['Hyperparameter']) & (RMSEdf['Value'] == 'Baseline')]
    bestrow = RMSEdf.loc[(RMSEdf['Hyperparameter'] == summary_df.iloc[n]['Hyperparameter']) & (RMSEdf['Value'] == 'Best')]
    
    if not bsrow['RMSE'].to_numpy() == bestrow['RMSE'].to_numpy():
        axs[0].annotate("",
            xy=(n, bsrow['RMSE']), xycoords='data',
            xytext=(n, bestrow['RMSE']), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3", color='k', lw=0.5),
            )
        
        # Annotate with percentage improvement
        annotationPer = str(np.round(bsrow['Improvement [%]'].to_numpy()[0],1)) + ' %' 
        axs[0].annotate(annotationPer,
                xy=(n, bestrow['RMSE']), xycoords='data',rotation=90,
                xytext=(2,4), textcoords='offset points', bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8),
                )

axs[0].label_outer()
axs[0].tick_params(axis='x',labelrotation=90)

axs[0].set_xlabel('')

axs[0].grid()
h,l = axs[0].get_legend_handles_labels()
axs[0].get_legend().remove()


# SSIM #################################

# Extract SSIM data from summary_results
SSIMdf = pd.DataFrame(summary_results)[['Hyperparameter', 'Baseline_SSIM', 'Best_Model_SSIM']]
# Calculate percentage improvement
SSIMdf['Improvement [%]'] = 100 * (SSIMdf['Best_Model_SSIM']-SSIMdf['Baseline_SSIM']) / SSIMdf['Baseline_SSIM']
# Rename columns for clarity
SSIMdf.rename(columns={'Baseline_SSIM': 'Baseline', 'Best_Model_SSIM': 'Best'}, inplace=True)
# Melt the dataframe for easier plotting
SSIMdf = SSIMdf.melt(id_vars=['Hyperparameter', 'Improvement [%]'], value_vars=['Baseline', 'Best'], 
                     var_name='Value', value_name='SSIM')
# Plot the SSIM data
p = sns.pointplot(ax=axs[1], data=SSIMdf, x='Hyperparameter', y='SSIM', hue='Value', 
                  dodge=0.2, linestyle="none", errorbar=None, marker="_")

# # Annotate percentage improvement
for n in range(len(summary_df)):
    bsrow = SSIMdf.loc[(SSIMdf['Hyperparameter'] == summary_df.iloc[n]['Hyperparameter']) & (SSIMdf['Value'] == 'Baseline')]
    bestrow = SSIMdf.loc[(SSIMdf['Hyperparameter'] == summary_df.iloc[n]['Hyperparameter']) & (SSIMdf['Value'] == 'Best')]
    
    if not bsrow['SSIM'].to_numpy() == bestrow['SSIM'].to_numpy():
        axs[1].annotate("",
            xy=(n, bsrow['SSIM']), xycoords='data',
            xytext=(n, bestrow['SSIM']), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3", color='k', lw=0.5),
            )
        
        # Annotate with percentage improvement
        annotationPer = str(np.round(bsrow['Improvement [%]'].to_numpy()[0],1)) + ' %' 
        axs[1].annotate(annotationPer,
                xy=(n, max(bestrow['SSIM'].to_numpy(),bsrow['SSIM'].to_numpy())), xycoords='data',rotation=90,
                xytext=(2,-22), textcoords='offset points', bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8),
                )

axs[1].label_outer()
axs[1].tick_params(axis='x',labelrotation=90)

axs[1].set_xlabel('')

axs[1].grid()
h,l = axs[1].get_legend_handles_labels()
axs[1].get_legend().remove()



fig.legend(title='Score',handles = h,labels=l, 
        loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.15))

plt.savefig(f'MeanRMSESSIM_Improvements{dataset}.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)



plt.show()


# %%
    difVals = plotDf.loc[plotDf['variable'] == LFC18_default.iloc[0].index[n]]
    if not difVals.iloc[0][plotparam] == difVals.iloc[1][plotparam]:
        # if plotparam == 'RMSE': # RMSE is better if it's lower
        axs[pli,dsi].annotate("",
        xy=(n, difVals.iloc[0][plotparam]), xycoords='data',
        xytext=(n, difVals.iloc[1][plotparam]), textcoords='data',
        arrowprops=dict(arrowstyle="<-",
                        connectionstyle="arc3", color='k', lw=0.5),
        )

        # Annotate with percentage improvement
        annotationPer = perImpro.loc[(perImpro["variable"] == LFC18_default.iloc[0].index[n]) & (perImpro['Value'] == 'Best'),'Improvement [%]'].values[0]
        # if annotationPer > 0.5:
        if plotparam == 'RMSE':
            annotationPer = -annotationPer
        annotationPer = str(round(annotationPer,1)) + ' %'
        
        if plotparam == 'RMSE':
            axs[pli,dsi].annotate(annotationPer,
            xy=(n, difVals.iloc[1][plotparam]), xycoords='data',rotation=90,
            xytext=(2,3), textcoords='offset points', bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8),
            )
        else:
            axs[pli,dsi].annotate(annotationPer,
            xy=(n, difVals.iloc[1][plotparam]), xycoords='data',rotation=90,
            xytext=(2,-17), textcoords='offset points', bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8),
            )



# %%

for pli, plotparam in enumerate(plotparam_s):
    if plotparam == 'RMSE':
        plotDf = RMSE_Means.loc[RMSE_Means['Dataset'] == plotDs]
    else:
        plotDf = SSIM_Means.loc[RMSE_Means['Dataset'] == plotDs]

    axs[pli].annotate(lett[pli], xy=(0, 1.05), xycoords="axes fraction", fontsize = 12,bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8))

    # Percentage improvement
    perImpro = plotDf
    perImpro['Improvement'] = 0
    perImpro['Improvement [%]'] = 0
    for HP in pd.unique(plotDf['variable']):

        best = perImpro.loc[(perImpro["variable"] == HP) & (perImpro['Value'] == 'Best')][plotparam].values[0]
        baseline = perImpro.loc[(perImpro["variable"] == HP) & (perImpro['Value'] == 'Default')][plotparam].values[0]
        improvement = np.abs(best-baseline)
        relImprovement = 100*np.abs(best-baseline)/baseline # Percentage improvement
        perImpro.loc[(perImpro["variable"] == HP) & (perImpro['Value'] == 'Best'),'Improvement']= improvement
        perImpro.loc[(perImpro["variable"] == HP) & (perImpro['Value'] == 'Best'),'Improvement [%]']= relImprovement
    

    # ax0 = axs[pli,dsi].twinx()
    # p = sns.pointplot(ax = ax0,data=perImpro,x='variable', y='Improvement [%]', linestyle="none", errorbar=None,marker="_", color = 'green')





    p = sns.pointplot(ax = axs[pli,dsi],data=plotDf,x='variable', y=plotparam, hue='Value',dodge=.2, linestyle="none", errorbar=None,marker="_")

    for n in range(len(MC24_default.iloc[0])):
        difVals = plotDf.loc[plotDf['variable'] == LFC18_default.iloc[0].index[n]]
        if not difVals.iloc[0][plotparam] == difVals.iloc[1][plotparam]:
            # if plotparam == 'RMSE': # RMSE is better if it's lower
            axs[pli,dsi].annotate("",
            xy=(n, difVals.iloc[0][plotparam]), xycoords='data',
            xytext=(n, difVals.iloc[1][plotparam]), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3", color='k', lw=0.5),
            )

            # Annotate with percentage improvement
            annotationPer = perImpro.loc[(perImpro["variable"] == LFC18_default.iloc[0].index[n]) & (perImpro['Value'] == 'Best'),'Improvement [%]'].values[0]
            # if annotationPer > 0.5:
            if plotparam == 'RMSE':
                annotationPer = -annotationPer
            annotationPer = str(round(annotationPer,1)) + ' %'
            
            if plotparam == 'RMSE':
                axs[pli,dsi].annotate(annotationPer,
                xy=(n, difVals.iloc[1][plotparam]), xycoords='data',rotation=90,
                xytext=(2,3), textcoords='offset points', bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8),
                )
            else:
                axs[pli,dsi].annotate(annotationPer,
                xy=(n, difVals.iloc[1][plotparam]), xycoords='data',rotation=90,
                xytext=(2,-17), textcoords='offset points', bbox=dict(boxstyle='square,pad=0',facecolor='white', edgecolor='none', alpha = 0.8),
                )

                # ax.arrow(x = n, y = difVals.iloc[0][plotparam], dx = 0, dy = difVals.iloc[1][plotparam]-difVals.iloc[0][plotparam],width=.002)

            # else 
            # ax.annotate("",
            #     xy=(n, difVals.iloc[1][plotparam]), xycoords='data',
            #     xytext=(n, difVals.iloc[0][plotparam]), textcoords='data',
            #     arrowprops=dict(arrowstyle="<-",
            #                     connectionstyle="arc3", color='k', lw=0.5),
            #     )

    axs[pli,dsi].label_outer()
    axs[pli,dsi].tick_params(axis='x',labelrotation=90)
    
    axs[pli,dsi].set_xlabel('')
    
    axs[pli,dsi].grid()
    h,l = axs[pli,dsi].get_legend_handles_labels()
    axs[pli,dsi].get_legend().remove()


    fig.legend(title='Score',handles = h,labels=l, 
            loc="lower center", ncol=2,bbox_to_anchor=(0.55, -0.15))



plt.savefig('MeanRMSESSIM_Improvements.pdf', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0.1)
plt.show()

# # Save the summary to a CSV file
# summary_csv_path = os.path.join(folder_path, "summary_results.csv")
# summary_df.to_csv(summary_csv_path, index=False)
# print(f"Summary results saved to {summary_csv_path}")


# %%
# Load the CSV file

# Baselines
# file_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250806_LFC18_crossVal_Baseline_resDf.xlsx"
# file_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250806_crossVal\20250806_MC24_crossVal_Baseline_resDf.xlsx"
# file_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Baseline_resDf.xlsx"

# Optimised
# file_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_resDf.xlsx"
file_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_resDf.xlsx"
df = pd.read_excel(file_path)

# Define y_range for NRMSE calculation
# y_range = 4.472809791564941-0.7046534419059753 # LFC18
# y_range = 1-0.20981520414352417 # MC24
y_range = 1-0.16222809255123138 # MC24_1000

# Function to calculate mean and coefficient of variation
def calculate_mean_cv(data):
    mean = np.mean(data)
    cv = np.std(data) / mean if mean != 0 else 0
    return mean, cv

# Filter data for train and validation
train_data = df[df['data'] == 'train']
val_data = df[df['data'] == 'val']

# Calculate metrics for train data
train_rmse_mean, train_rmse_cv = calculate_mean_cv(train_data['RMSE'])
train_nrmse_mean, train_nrmse_cv = calculate_mean_cv(train_data['RMSE'] / y_range)
train_ssim_mean, train_ssim_cv = calculate_mean_cv(train_data['SSIM_metric'])

# Calculate metrics for validation data
val_rmse_mean, val_rmse_cv = calculate_mean_cv(val_data['RMSE'])
val_nrmse_mean, val_nrmse_cv = calculate_mean_cv(val_data['RMSE'] / y_range)
val_ssim_mean, val_ssim_cv = calculate_mean_cv(val_data['SSIM_metric'])

# Print results
print("Train Data:")
print(f"Mean RMSE: {train_rmse_mean:.3f}, Coefficient of Variation RMSE: {train_rmse_cv*100:.2f}%")
print(f"Mean NRMSE: {train_nrmse_mean:.3f}, Coefficient of Variation NRMSE: {train_nrmse_cv*100:.2f}%")
print(f"Mean SSIM: {train_ssim_mean:.3f}, Coefficient of Variation SSIM: {train_ssim_cv*100:.2f}%")

print("\nValidation Data:")
print(f"Mean RMSE: {val_rmse_mean:.3f}, Coefficient of Variation RMSE: {val_rmse_cv*100:.2f}%")
print(f"Mean NRMSE: {val_nrmse_mean:.3f}, Coefficient of Variation NRMSE: {val_nrmse_cv*100:.2f}%")
print(f"Mean SSIM: {val_ssim_mean:.3f}, Coefficient of Variation SSIM: {val_ssim_cv*100:.2f}%")