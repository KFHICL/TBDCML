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
from matplotlib.lines import Line2D

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

# 12052025 LFC18 optimisation run 2 (4-layer deep networks)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_trainValSplit_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_batchSize_run2" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_kernelSize_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_optimizer_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_activationFunction_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_loss_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_dropout_run2" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_initialLr_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_lrDecay_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_maxPool_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_batchNorm_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_modelDepth_run2" # NO RUN 2 FOR MODEL DEPTH
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_dataAug_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_skinConnections_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_decoderAct_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_LFC18_epsilon_run2"

# 12052025 MC24 optimisation run 2 (4-layer deep networks)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_trainValSplit_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_batchSize_run2" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_kernelSize_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_optimizer_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_activationFunction_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_loss_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_dropout_run2" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_initialLr_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_lrDecay_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_maxPool_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_batchNorm_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_modelDepth_run2" # NO RUN 2 FOR MODEL DEPTH
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_dataAug_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_skinConnections_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_decoderAct_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24_epsilon_run2" # Failed need to re-run

# 12052025 MC24x optimisation run 2 (4-layer deep networks)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_trainValSplit_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_batchSize_run2" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_kernelSize_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_optimizer_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_activationFunction_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_loss_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_dropout_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_initialLr_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_lrDecay_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_maxPool_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_batchNorm_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_modelDepth_run2" # NO RUN 2 FOR MODEL DEPTH
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_dataAug_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_skinConnections_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_decoderAct_run2"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250512_MC24x_epsilon_run2" 

# 21052025 LFC18 optimisation run 3 (factor 2 downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_trainValSplit_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_batchSize_run3" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_kernelSize_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_optimizer_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_activationFunction_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_loss_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_dropout_run3" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_initialLr_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_lrDecay_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_maxPool_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_batchNorm_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_modelDepth_run3" # NO RUN 2 FOR MODEL DEPTH
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_dataAug_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_skinConnections_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_decoderAct_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_LFC18_epsilon_run3"

# 21052025 MC24 optimisation run 3 (factor 2 downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_trainValSplit_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_batchSize_run3" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_kernelSize_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_optimizer_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_activationFunction_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_loss_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_dropout_run3" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_initialLr_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_lrDecay_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_maxPool_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_batchNorm_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_modelDepth_run3" # NO RUN 2 FOR MODEL DEPTH
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_dataAug_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_skinConnections_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_decoderAct_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_epsilon_run3"

# 21052025 MC24x optimisation run 3 (factor 2 downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_trainValSplit_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_batchSize_run3" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_kernelSize_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_optimizer_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_activationFunction_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_loss_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_dropout_run3" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_initialLr_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_lrDecay_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_maxPool_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_batchNorm_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_modelDepth_run3" # NO RUN 2 FOR MODEL DEPTH
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_dataAug_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_skinConnections_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24x_decoderAct_run3"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250521_MC24_epsilon_run3"

# 11062025 LFC18 optimisation run 4 (pseudo-optimal model from runs 1-3)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_trainValSplit_run4" # FAILED NEED TO RE-RUN
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_batchSize_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_kernelSize_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_optimizer_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_activationFunction_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_loss_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_dropout_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_initialLr_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_lrDecay_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_maxPool_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_batchNorm_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_modelDepth_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_dataAug_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_skinConnections_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_decoderAct_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_LFC18_epsilon_run4"

# 11062025 MC24 optimisation run 4 (pseudo-optimal model from runs 1-3)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_trainValSplit_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_batchSize_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_kernelSize_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_optimizer_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_activationFunction_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_loss_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_dropout_run4" # FAILED RERUN
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_initialLr_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_lrDecay_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_maxPool_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_batchNorm_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_modelDepth_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_dataAug_run4" # FAILED RETRUN
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_skinConnections_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_decoderAct_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24_epsilon_run4"

# 11062025 MC24x optimisation run 4 (pseudo-optimal model from runs 1-3)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_trainValSplit_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_batchSize_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_kernelSize_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_optimizer_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_activationFunction_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_loss_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_dropout_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_initialLr_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_lrDecay_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_maxPool_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_batchNorm_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_modelDepth_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_dataAug_run4" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_skinConnections_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_decoderAct_run4"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250611_MC24x_epsilon_run4"

# 25072025 LFC18 optimisation run 5 (pseudo-optimal model from runs 1-3, relu instead of tanh)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_trainValSplit_run5" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_batchSize_run5" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_kernelSize_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_optimizer_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_activationFunction_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_loss_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_dropout_run5" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_initialLr_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_lrDecay_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_maxPool_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_batchNorm_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_modelDepth_run5" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_dataAug_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_skinConnections_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_decoderAct_run5"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250725_LFC18_epsilon_run5"

# 20250806 LFC18 crossValidation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_LFC18_crossVal_Baseline" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_LFC18_crossVal_Opti" 

# 20250806 MC24 crossValidation
jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_MC24_crossVal_Baseline" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_MC24_crossVal_Opti" 


# 20250806 MC24x crossValidation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_MC24x_crossVal_Baseline" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_MC24x_crossVal_Opti" 

# 07082025 LFC18 optimisation run 6 (no data leakage, with downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_trainValSplit_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_batchSize_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_kernelSize_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_optimizer_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_activationFunction_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_loss_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_dropout_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_initialLr_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_lrDecay_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_maxPool_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_batchNorm_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_modelDepth_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_dataAug_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_skinConnections_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_decoderAct_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_epsilon_run6"

# 07082025 MC24 optimisation run 6 (no data leakage, with downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_trainValSplit_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_batchSize_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_kernelSize_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_optimizer_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_activationFunction_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_loss_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_dropout_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_initialLr_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_lrDecay_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_maxPool_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_batchNorm_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_modelDepth_run6" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_dataAug_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_skinConnections_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_decoderAct_run6"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_epsilon_run6"

# 07082025 LFC18 optimisation run 7 (no data leakage, without downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_trainValSplit_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_batchSize_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_kernelSize_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_optimizer_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_activationFunction_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_loss_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_dropout_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_initialLr_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_lrDecay_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_maxPool_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_batchNorm_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_modelDepth_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_dataAug_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_skinConnections_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_decoderAct_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_epsilon_run7"

# 07082025 MC24 optimisation run 7 (no data leakage, without downsampling)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_trainValSplit_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_batchSize_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_kernelSize_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_optimizer_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_activationFunction_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_loss_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_dropout_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_initialLr_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_lrDecay_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_maxPool_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_batchNorm_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_modelDepth_run7" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_dataAug_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_skinConnections_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_decoderAct_run7"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_epsilon_run7"

# 07082025 LFC18 optimisation run 8 (no data leakage, without downsampling, pseudo optimal)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_trainValSplit_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_batchSize_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_kernelSize_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_optimizer_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_activationFunction_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_loss_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_dropout_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_initialLr_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_lrDecay_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_maxPool_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_batchNorm_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_modelDepth_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_dataAug_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_skinConnections_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_decoderAct_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_LFC18_epsilon_run8"

# 07082025 MC24 optimisation run 8 (no data leakage, without downsampling, pseudo optimal)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_trainValSplit_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_batchSize_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_kernelSize_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_optimizer_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_activationFunction_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_loss_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_dropout_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_initialLr_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_lrDecay_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_maxPool_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_batchNorm_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_modelDepth_run8" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_dataAug_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_skinConnections_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_decoderAct_run8"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250807_MC24_epsilon_run8"


# 08082025 LFC18 crossValidation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250806_LFC18_crossVal_Baseline" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_LFC18_crossVal_Opti" 

# 08082025 MC24_ConstVf optimisation run 9 (no data leakage, without downsampling, pseudo optimal)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_trainValSplit_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_batchSize_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_kernelSize_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_optimizer_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_activationFunction_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_loss_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_dropout_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_initialLr_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_lrDecay_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_maxPool_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_batchNorm_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_modelDepth_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_dataAug_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_skinConnections_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_decoderAct_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_ConstVf_epsilon_run9"

# 08082025 MC24_1000 optimisation run 9 (no data leakage, without downsampling, pseudo optimal)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_trainValSplit_run9" # FAILED, RERUN
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_batchSize_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_kernelSize_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_optimizer_run9" # Failed, rerun longer walltime
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_activationFunction_run9" # Failed, rerun longer walltime
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_loss_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_dropout_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_initialLr_run9" # failed, rerun
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_lrDecay_run9" # failed, rerun
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_maxPool_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_batchNorm_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_modelDepth_run9" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_dataAug_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_skinConnections_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_decoderAct_run9"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250808_MC24_1000_epsilon_run9" # fails, walltime

# 13082025 MC24 optimisation run 10 (no data leakage, without downsampling, baseline sweep)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_trainValSplit_run10" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_batchSize_run10" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_kernelSize_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_optimizer_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_activationFunction_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_loss_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_dropout_run10" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_initialLr_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_lrDecay_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_maxPool_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_batchNorm_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_modelDepth_run10" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_dataAug_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_skinConnections_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_decoderAct_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_epsilon_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_downSample_run10"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_filterScale_run10"

# 13082025 MC24 optimisation run 11 (no data leakage, without downsampling, baseline sweep, 0.2 valSize)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_trainValSplit_run11" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_batchSize_run11" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_kernelSize_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_optimizer_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_activationFunction_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_loss_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_dropout_run11" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_initialLr_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_lrDecay_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_maxPool_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_batchNorm_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_modelDepth_run11" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_dataAug_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_skinConnections_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_decoderAct_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_epsilon_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_downSample_run11"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_filterScale_run11"

# 13082025 MC24_1000 optimisation run 12 (Somewhat optimal MC24_1000 model)
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_trainValSplit_run12" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_batchSize_run12" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_kernelSize_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_optimizer_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_activationFunction_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_loss_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_dropout_run12" # repeat 2, model 3 failed.
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_initialLr_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_lrDecay_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_maxPool_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_batchNorm_run12" # fails repeat 3, rerun
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_modelDepth_run12" # fails repeat 2, rerun
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_dataAug_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_skinConnections_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_decoderAct_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_epsilon_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_downSample_run12"
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250813_MC24_1000_filterScale_run12"

# 14082025 MC24_1000 crossValidation
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250814_MC24_1000_crossVal_Baseline" 
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250814_MC24_1000_crossVal_Opti" 

# 09092025 MC24x run 16 - somewhat optimal MC24x model and literature models
# jobPath = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\20250909_MC24x_run16"

# jobIdx = [1,2,3,4,5,6,7,8,9,10,11,14,17] # Which jobs to include (1-indexed)
jobIdx = [1,2,3,4,5,6,7,8,9,10,11,14,17] # Which jobs to include (1-indexed)

crossVal = True # Cross validation or not
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

# Mask results_files and history_files using jobIdx (convert to 0-based index)
# results_files = results_files[:, [i - 1 for i in jobIdx]]
# history_files = history_files[:, [i - 1 for i in jobIdx]]


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
        if "RMSE" in df.columns:
            df = df.drop(columns=["RMSE"])
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
    print(histDf['model'].unique())
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
    print(len(base_palette), len(sweepVals))
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
    if len(sweepVals)>4: # More models, need to increase height
        figHeight = 2*figHeight
        figWidth = 1.5*figWidth
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
    
    
    # Ensure RMSEmeans is ordered "train, val, test" for each model
    RMSEmeans = resDf.groupby(['model', 'data'])['RMSE'].mean().reset_index()
    RMSEmeans['data'] = pd.Categorical(RMSEmeans['data'], categories=['train', 'val', 'test'], ordered=True)
    RMSEmeans = RMSEmeans.sort_values(['model', 'data'])
    sns.pointplot(ax=ax, data=RMSEmeans, x='model', y='RMSE', hue='data', palette=RMSE_palette, dodge=.4, linestyle="none", errorbar=None,
    marker="_", markersize=12, markeredgewidth=3)

    SSIMmeans = resDf.groupby(['model', 'data'])['SSIM_metric'].mean().reset_index()
    SSIMmeans['data'] = pd.Categorical(SSIMmeans['data'], categories=['train', 'val', 'test'], ordered=True)
    SSIMmeans = SSIMmeans.sort_values(['model', 'data'])
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
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if len(sweepVals)>4: # Rotate x-tick labels if there are many
        ax.set_xticklabels(sweepVals, rotation=90)
        # Add vertical dotted lines halfway between xticks
        xticks = ax.get_xticks()
        for i in range(len(xticks) - 1):
            mid = (xticks[i] + xticks[i + 1]) / 2
            ax.axvline(mid, color='gray', linestyle=':', linewidth=0.5, zorder=0)
    else:
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
        for m, model in enumerate(resDf['model'].unique()):
            filtered_df = resDf[resDf['model'] == model]
            for data_type in filtered_df['data'].unique():
                data_filtered_df = filtered_df[filtered_df['data'] == data_type]
                RMSE_mean = data_filtered_df['RMSE'].mean()
                RMSE_std = data_filtered_df['RMSE'].std()
                SSIM_mean = data_filtered_df['SSIM_metric'].mean()
                SSIM_std = data_filtered_df['SSIM_metric'].std()
                summary_data.append([model, sweepVals[m], data_type, 'RMSE', RMSE_mean, RMSE_std])
                summary_data.append([model, sweepVals[m], data_type, 'SSIM', SSIM_mean, SSIM_std])

        summary_df = pd.DataFrame(summary_data, columns=['Model', sweepName, 'Data', 'Metric', 'Mean', 'Std'])
    print(summary_df)
    summary_df.to_csv(table_output_path, index=False)
    print(f"Metrics summary saved to {table_output_path}")

    plt.show()





# %%
histDf = formatHistory(histories,histCols = histCols)
resDf = formatResults(results,resCols = resCols)


# %% Make plots for each sween
# idx = [1] # Activation func

# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250806_crossVal"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run6"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run7"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run8"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run6"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run7"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run10"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run11"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_run12"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run8"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_constVf_run9"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_1000_run9"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run5_pOpti2"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run4_pOpti"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run4_pOpti"
# output_dir = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run4_pOpti"
# output_dir = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run4_pOpti"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run3_downsample"
# output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_run2_4layers"
output_dir = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250806_crossVal" # Baseline cross validation (optimised models were found with tonnes of data leakage)
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
# sweepVals = list(range(1,5)) # Values of the sweep parameter
# sweepVals = list(range(1,6)) # Values of the sweep parameter
# sweepVals = list(range(1, 7)) # Values of the sweep parameter

# sweepName = 'Data augmentation' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'Skip Connections' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'Decoder activations' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter
# sweepName = 'epsilon' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepName = 'downSample' # Name of the sweep parameter
# sweepVals = ['On'   , 'Off']
# sweepName = 'filterScale' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()




# sweepName = 'batchSize' # Name of the sweep parameter
# sweepVals = sweepDef[sweepName].to_numpy()
# sweepVals = list(range(1, 7))
# sweepVals = ['On'   , 'Off'] # Values of the sweep parameter

sweepName = 'type' # Name of the sweep parameter
sweepVals = sweepDef[sweepName].to_numpy()
# sweepVals = sweepVals[np.array(jobIdx)-1]

# jobIdx = [1,2,3,4,5,6,7,8,9,10,11,14,17] # Which jobs to include (1-indexed)
idx = list(range(1, 100))
# idx = [1,2,3,4,5,6,7,8,9,10,11,12,13]

# idx = [4,13,5,7,6,8,2,3,9,10,11,12,1]

# idx = [1,10,11,14,17]
# idxHist = [1,10,11,14,17]
# idx = [1,10,11,12,13]
# idxHist = [1,10,11,12,13]
job_name = job_name + "tmp"
# sweepVals = sweepVals[np.array(idx)-1]

plotHist(histDf, idx, sweepName, sweepVals, output_dir, job_name)
plotResults(resDf, idx, sweepName, sweepVals, output_dir, job_name)
# Save figures to the specified directory with the job name appended
# Save resDf as a spreadsheet in the output directory
resDf_output_path = os.path.join(output_dir, f"{job_name}_resDf.xlsx")
resDf.to_excel(resDf_output_path, index=False)
print(f"resDf saved to {resDf_output_path}")

print(f"Figures saved to {output_dir} with job name appended.")

plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# resToPlot = pd.read_csv(r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_metrics.csv")
resToPlot = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24x_run16\20250909_MC24x_run16_metrics.csv")
# Rename 'type' where it equals 'default' to 'This work'
resToPlot.loc[resToPlot['type'] == 'default', 'type'] = 'TBDCNet-Meso-L'
# VGG16 (Simonyan and Zisserman 2014)
# UNet (Ronneberger et al. 2015) 
# ResNet50 (He et al. 2016a)
# InceptionV3 (Szegedy et al. 2016)
# ResNet50V2 (He et al. 2016b)
# InceptionResNetV2 (Szegedy et al. 2017)
# Xception (Chollet 2017)
# MobilenetV2 (Howard et al. 2017)
# DenseNet121 (Huang et al. 2017)
# NasNetMobile (Zoph et al. 2018)
# EfficientNetV2S (Tan and Le 2021)
# ConvNextTiny (Liu et al. 2022)

# Define the desired order of types
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

# Reorder the DataFrame based on the type_order
resToPlot['type'] = pd.Categorical(resToPlot['type'], categories=type_order, ordered=True)
resToPlot = resToPlot.sort_values('type')
RMSEs = resToPlot[resToPlot['Metric'] == 'RMSE']
SSIMs = resToPlot[resToPlot['Metric'] == 'SSIM']
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


# Create a barplot with error bars
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)

ax = sns.barplot(
    # data=RMSEs[RMSEs['Data']=='test'],
    data=RMSEs,
    x='type',
    y='Mean',
    hue='Data',
    hue_order=['train', 'val', 'test'],  # Specify the order of the hue categories
    palette=RMSE_palette
)

# x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
# y_coords = [p.get_height() for p in ax.patches]

x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches[:-3]]
y_coords = [p.get_height() for p in ax.patches[:-3]]
# ax.errorbar(x=x_coords, y=y_coords, yerr=RMSEs[RMSEs['Data']=='test']['Std'], fmt="none", c="k")
ax.errorbar(x=x_coords, y=y_coords, yerr=RMSEs['Std'], fmt="none", c="k")
ax.set_ylabel('RMSE', fontsize=12)
ax.set_xlabel('', fontsize=12)
ax.legend(
    title='Data',
    loc='upper left',
    # bbox_to_anchor=(1, 1),
    fontsize=10,
    title_fontsize=10
)
ax.set_ylim(0.035, None)  # Start y-axis at 0
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

ax.grid(axis='y', linestyle='--')

plt.subplot(2, 1, 2)
ax2 = sns.barplot(
    data=SSIMs,
    x='type',
    y='Mean',
    hue='Data',
    hue_order=['train', 'val', 'test'],  # Specify the order of the hue categories
    palette=SSIM_palette
)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax2.patches[:-3]]
y_coords = [p.get_height() for p in ax2.patches[:-3]]
# ax.errorbar(x=x_coords, y=y_coords, yerr=RMSEs[RMSEs['Data']=='test']['Std'], fmt="none", c="k")
ax2.errorbar(x=x_coords, y=y_coords, yerr=SSIMs['Std'], fmt="none", c="k")
ax2.set_ylim(0.5, None)  # Start y-axis at 0
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90, fontsize=12, ha='center',va = 'top')
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=10)
ax2.set_ylabel('SSIM', fontsize=12)
ax2.set_xlabel('', fontsize=12)
ax2.legend(
    title='Data',
    loc='upper left',
    # bbox_to_anchor=(1, 1),
    fontsize=10,
    title_fontsize=10
)

# ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=10)
ax.set_xticklabels([''] * len(ax.get_xticks()), fontsize=1)  # Remove xtick labels on ax
ax2.grid(axis='y', linestyle='--')
plt.tight_layout()

# Save the figure as a high-resolution PDF
output_pdf_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures\high_resolution_figure.pdf"
plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved to {output_pdf_path}")

plt.show()

# %% 


# %%
# Plot trainTime for each model as a bar chart
plt.figure(figsize=(8, 4))
train_times = histDf.groupby(['model', 'repeat'])['trainTime'].max().groupby('model').mean()
sns.barplot(x=train_times.index, y=train_times.values, palette='colorblind')
plt.xlabel('Model')
plt.ylabel('Mean Train Time [s]')
plt.title('Mean Train Time per Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.xticks(ticks=range(len(sweepVals)), labels=sweepVals, rotation=90)
plt.show()

# %%
# Calculate overall mean, std, and coefficient of variation for SSIM
train_ssim = resDf[resDf['data'] == 'train']['SSIM_metric']
val_ssim = resDf[resDf['data'] == 'val']['SSIM_metric']

train_ssim_mean = train_ssim.mean()
train_ssim_std = train_ssim.std()
train_ssim_cv = train_ssim_std / train_ssim_mean if train_ssim_mean != 0 else np.nan

val_ssim_mean = val_ssim.mean()
val_ssim_std = val_ssim.std()
val_ssim_cv = val_ssim_std / val_ssim_mean if val_ssim_mean != 0 else np.nan

print(f"Train SSIM: mean={train_ssim_mean:.4f}, std={train_ssim_std:.4f}, CV={train_ssim_cv:.4f}")
print(f"Val SSIM: mean={val_ssim_mean:.4f}, std={val_ssim_std:.4f}, CV={val_ssim_cv:.4f}")

# Calculate overall mean, std, and coefficient of variation for RMSE
train_rmse = resDf[resDf['data'] == 'train']['RMSE']
val_rmse = resDf[resDf['data'] == 'val']['RMSE']

train_rmse_mean = train_rmse.mean()
train_rmse_std = train_rmse.std()
train_rmse_cv = train_rmse_std / train_rmse_mean if train_rmse_mean != 0 else np.nan

val_rmse_mean = val_rmse.mean()
val_rmse_std = val_rmse.std()
val_rmse_cv = val_rmse_std / val_rmse_mean if val_rmse_mean != 0 else np.nan

print(f"Train RMSE: mean={train_rmse_mean:.4f}, std={train_rmse_std:.4f}, CV={train_rmse_cv:.4f}")
print(f"Val RMSE: mean={val_rmse_mean:.4f}, std={val_rmse_std:.4f}, CV={val_rmse_cv:.4f}")
# %%
# Load baseline and optimised resDf files
# baseline_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Baseline_resDf.xlsx"
# optimised_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_resDf.xlsx"
# optimised_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_resDf.xlsx"
baseline_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250806_crossVal\20250806_MC24_crossVal_Baseline_resDf.xlsx"
optimised_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Baseline_resDf.xlsx"
baseline_df = pd.read_excel(baseline_path)
optimised_df = pd.read_excel(optimised_path)

baseline_df['ModelType'] = 'Baseline'
optimised_df['ModelType'] = 'Optimised'

combined_df = pd.concat([baseline_df, optimised_df], ignore_index=True)

# Only keep train and val data
combined_df = combined_df[combined_df['data'].isin(['train', 'val'])]

# Swarmplot for RMSE
plt.figure(figsize=(8, 4))
sns.swarmplot(data=combined_df, x='ModelType', y='RMSE', hue='data', dodge=True, palette={'train': '#66D3B3', 'val': '#029E73'}, alpha=1)
plt.ylabel('RMSE', fontsize=14)
plt.xlabel('', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend(
    title='Data',
    loc='upper left',
    bbox_to_anchor=(1, 1),
    fontsize=12,
    title_fontsize=12
)
swarm_pdf_path = os.path.join(os.path.dirname(optimised_path), "RMSE_swarm.pdf")
plt.savefig(swarm_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# Swarmplot for SSIM
plt.figure(figsize=(8, 4))
sns.swarmplot(data=combined_df, x='ModelType', y='SSIM_metric', hue='data', dodge=True, palette={'train': '#66ADD6', 'val': '#0173B2'}, alpha=1)
plt.ylabel('SSIM', fontsize=14)
plt.xlabel('', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend(
    title='Data',
    loc='upper left',
    bbox_to_anchor=(1, 1),
    fontsize=12,
    title_fontsize=12
)
swarm_pdf_path = os.path.join(os.path.dirname(optimised_path), "SSIM_swarm.pdf")
plt.savefig(swarm_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# Scatter plot: SSIM (x-axis) vs RMSE (y-axis)
plt.figure(figsize=(7, 5))

# Plot train (light color)
train_plot = sns.scatterplot(
    data=combined_df[combined_df['data'] == 'train'],
    x='SSIM_metric',
    y='RMSE',
    hue='ModelType',
    palette={'Baseline': '#66ADD6', 'Optimised': '#66D3B3'},
    alpha=0.5,
    s=60,
    marker='o',
    # label='Train'
)
# Plot val (dark color)
val_plot = sns.scatterplot(
    data=combined_df[combined_df['data'] == 'val'],
    x='SSIM_metric',
    y='RMSE',
    hue='ModelType',
    palette={'Baseline': '#0173B2', 'Optimised': '#029E73'},
    alpha=0.9,
    s=60,
    marker='X',
    # label='Val'
)

plt.xlabel('SSIM', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.grid(axis='both')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Custom legend: combine ModelType and Data
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor="#383838", markersize=8, alpha=0.5),
    Line2D([0], [0], marker='X', color='w', label='Val', markerfacecolor="#383838", markersize=8, alpha=1),
    # Line2D([0], [0], marker='o', color='w', label='Optimised', markerfacecolor='#029E73', markersize=8, alpha=1),
    # Line2D([0], [0], marker='o', color='w', label='Baseline', markerfacecolor='#0173B2', markersize=8, alpha=1)
    Line2D([0], [0], marker='o', color='w', label='MeC-Meso-S', markerfacecolor='#0173B2', markersize=8, alpha=1),
        Line2D([0], [0], marker='o', color='w', label='MeC-Meso-M', markerfacecolor='#029E73', markersize=8, alpha=1),
]
plt.legend(handles=legend_elements, title='Legend', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=12)
# Save the scatter plot as a high-quality PDF in the same folder as the optimised resDf
# scatter_pdf_path = os.path.join(os.path.dirname(optimised_path), "scatter_SSIM_vs_RMSE.pdf")


scatter_pdf_path = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\MC24_HP_Op_results\datasetSize_scatter_SSIM_vs_RMSE.pdf"
plt.savefig(scatter_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Scatter plot saved to {scatter_pdf_path}")
plt.show()



# %%

import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity as ssim
os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

samplePath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\FlorianAbaqusFiles\datain\Unnotched_TBDC_2022_28.csv"
# samplePath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MatLabModelFiles\20240703_1417_100Samples\sample_14.csv"
sampleShape = [55,20]
# sampleShape = [60,20]
samplesPerFile = 1
winKernel = 5
# winKernel = 7
xNames = ['E11','E22','E12']
# xNames = ['Ex','Ey','Gxy','Vf','c2']
yNames = ['FI']
# modelPath_Baseline = r"C:\Users\kfh23\Desktop\model_20250806_LFC18_crossVal_Baseline_1_1.keras"
modelPath_Baseline = r"C:\Users\kfh23\Desktop\model_20250808_LFC18_crossVal_Opti_1_1.keras"
modelPath_Optimised = r"C:\Users\kfh23\Desktop\model_20250808_LFC18_crossVal_Opti_1_1.keras"
# modelPath_Baseline = r"C:\Users\kfh23\Desktop\model_20250814_MC24_1000_crossVal_Baseline_1_10.keras"
# modelPath_Baseline = r"C:\Users\kfh23\Desktop\model_20250814_MC24_1000_crossVal_Opti_1_1.keras" # Do untrained example (re-initialize weights)
# modelPath_Optimised = r"C:\Users\kfh23\Desktop\model_20250814_MC24_1000_crossVal_Opti_1_1.keras"

_, file_extension = os.path.splitext(samplePath)
match file_extension:
    case '.csv':
        sample = pd.read_csv(samplePath)
        # samples = sample.to_numpy()
        samples = np.array(sample)
    case '.parquet':
        sample = pd.read_parquet(samplePath, engine='auto')
        samples = [y for x, y in sample.groupby('specimen')]
        # samples = samples[specIdx] # Take the first sample for testing
        
        sample = np.array(samples)
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

# Load both baseline and optimised models
model_baseline = keras.models.load_model(modelPath_Baseline, custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss})
if  modelPath_Baseline == modelPath_Optimised:
    # Re-initialize the weights for the baseline model
    for layer in model_baseline.layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            weight_shapes = [w.shape for w in layer.get_weights()]
            if weight_shapes:
                # Re-initialize kernel and bias
                new_weights = []
                if hasattr(layer, 'kernel_initializer'):
                    new_weights.append(layer.kernel_initializer(weight_shapes[0]))
                if len(weight_shapes) > 1 and hasattr(layer, 'bias_initializer'):
                    new_weights.append(layer.bias_initializer(weight_shapes[1]))
                layer.set_weights(new_weights)

model_optimised = keras.models.load_model(modelPath_Optimised, custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss})

# Make predictions
prediction_baseline = model_baseline.predict(ds.batch(1))
prediction_optimised = model_optimised.predict(ds.batch(1))

# Reshape ground truth and predictions
ground_truth = Y[0, :, :, 0].reshape(sampleShape)
predicted_baseline = prediction_baseline[0, :, :, 0].reshape(sampleShape)
predicted_optimised = prediction_optimised[0, :, :, 0].reshape(sampleShape)

# Calculate errors
error_baseline = np.abs(ground_truth - predicted_baseline)
error_optimised = np.abs(ground_truth - predicted_optimised)
errorSquared_baseline = error_baseline ** 2

errorSquared_optimised = error_optimised ** 2

# Get colorbar limits from the ground truth
vmin, vmax = 0, ground_truth.max()
# Plot the ground truth, prediction, and error fields side by side for both baseline and optimised models
plt.figure(figsize=(5, 6))

# Compute squared error for optimised model
errorSquared_optimised = error_optimised ** 2

# Get colorbar limits for squared error based on the baseline model
err_vmin = 0
err_vmax = np.max(errorSquared_baseline)

# Calculate RMSE and SSIM for baseline and optimised predictions

# Flatten for RMSE calculation
rmse_baseline = np.sqrt(np.mean((ground_truth - predicted_baseline) ** 2))
rmse_optimised = np.sqrt(np.mean((ground_truth - predicted_optimised) ** 2))

# SSIM calculation (expects 2D arrays)
ssim_baseline = ssim(ground_truth, predicted_baseline, data_range=ground_truth.max() - ground_truth.min())
ssim_optimised = ssim(ground_truth, predicted_optimised, data_range=ground_truth.max() - ground_truth.min())

print(f"Baseline RMSE: {rmse_baseline:.4f}, SSIM: {ssim_baseline:.4f}")
print(f"Optimised RMSE: {rmse_optimised:.4f}, SSIM: {ssim_optimised:.4f}")


# --- Baseline row ---
# plt.subplot(2, 3, 1)
# --- Easily adjustable variables for plot appearance ---
main_fontsize = 12
colorbar_fontsize = 10
fig_width = 5
fig_height = 6
cmap_name = 'viridis'
save_path = 'ground_truth_prediction_error_baseline_optimised_LFC18.pdf'

# --- Baseline row ---
plt.figure(figsize=(fig_width, fig_height))

plt.subplot(2, 3, 1)
im1 = plt.imshow(ground_truth, cmap=cmap_name, vmin=vmin, vmax=vmax)
plt.title('Ground Truth', fontsize=main_fontsize)
cbar1 = plt.colorbar(im1)
cbar1.ax.tick_params(labelsize=colorbar_fontsize)
plt.axis('off')

plt.subplot(2, 3, 2)
# im2 = plt.imshow(predicted_baseline, cmap=cmap_name, vmin=vmin, vmax=vmax)
im2 = plt.imshow(predicted_baseline, cmap=cmap_name)
plt.title('Prediction', fontsize=main_fontsize)
cbar2 = plt.colorbar(im2)
cbar2.ax.tick_params(labelsize=colorbar_fontsize)
plt.axis('off')

plt.subplot(2, 3, 3)
im3 = plt.imshow(errorSquared_baseline, cmap=cmap_name, vmin=err_vmin, vmax=err_vmax)
plt.title('Squared Error', fontsize=main_fontsize)
cbar3 = plt.colorbar(im3)
cbar3.ax.tick_params(labelsize=colorbar_fontsize)
plt.axis('off')

# --- Optimised row ---
plt.subplot(2, 3, 4)
im4 = plt.imshow(ground_truth, cmap=cmap_name, vmin=vmin, vmax=vmax)
cbar4 = plt.colorbar(im4)
cbar4.ax.tick_params(labelsize=colorbar_fontsize)
plt.axis('off')

plt.subplot(2, 3, 5)
im5 = plt.imshow(predicted_optimised, cmap=cmap_name, vmin=vmin, vmax=vmax)
cbar5 = plt.colorbar(im5)
cbar5.ax.tick_params(labelsize=colorbar_fontsize)
plt.axis('off')

plt.subplot(2, 3, 6)
# im6 = plt.imshow(errorSquared_optimised, cmap=cmap_name, vmin=err_vmin, vmax=err_vmax)
im6 = plt.imshow(errorSquared_optimised, cmap=cmap_name)
cbar6 = plt.colorbar(im6)
cbar6.ax.tick_params(labelsize=colorbar_fontsize)
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures',save_path), format='pdf', bbox_inches='tight')
plt.show()

# Plot the distribution of ground truth, predictions, and error
# plt.figure(figsize=(10, 6))
# sns.kdeplot(ground_truth.flatten(), label='Ground Truth', color='blue', fill=True, alpha=0.5)
# sns.kdeplot(predicted_field.flatten(), label='Prediction', color='orange', fill=True, alpha=0.5)
# sns.kdeplot(error_field.flatten(), label='Error', color='red', fill=True, alpha=0.5)
# plt.title('Distribution of Ground Truth, Predictions, and Error')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
# plt.grid()
# plt.show()

# # Find the 5 areas with the highest values in the ground truth
# flat_ground_truth = ground_truth.flatten()
# flat_predicted_field = predicted_field.flatten()
# top_5_indices = np.argpartition(flat_ground_truth, -5)[-5:]
# top_5_indices = top_5_indices[np.argsort(flat_ground_truth[top_5_indices])[::-1]]

# print("Top 5 highest values in the ground truth and corresponding predictions:")
# for idx in top_5_indices:
#     row, col = divmod(idx, sampleShape[1])
#     gt_value = ground_truth[row, col]
#     pred_value = predicted_field[row, col]
#     print(f"Location: ({row}, {col}), Ground Truth: {gt_value:.4f}, Prediction: {pred_value:.4f}")





# %%
# Load and summarise a CNN model
def summarise_model(model_path):
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
        custom_objects={'SSIM_metric': SSIM_metric, 'custom_loss': custom_loss}
    )
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

# Example usage:
summarise_model(r"C:\Users\kfh23\Desktop\model_20250806_LFC18_crossVal_Opti_1_1.keras")
# summarise_model(r"C:\Users\kfh23\Desktop\model_20250725_LFC18_loss_run5_1_2.keras")
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
# Plot loss curves for each history file in separate subplots
fig, axes = plt.subplots(nrows=len(history_files), ncols=1, figsize=(8, 3 * len(history_files)), sharex=True)
if len(history_files) == 1:
    axes = [axes]
for i, hist_path in enumerate(history_files):
    with open(hist_path, 'r') as f:
        hist_data = json.load(f)
    df = pd.DataFrame.from_dict(hist_data)
    axes[i].plot(df['loss'], label='Training Loss')
    if 'val_loss' in df.columns:
        axes[i].plot(df['val_loss'], label='Validation Loss')
    axes[i].set_title(f'Loss Curve: {os.path.basename(hist_path)}')
    axes[i].set_ylabel('Loss')
    axes[i].set_yscale('log')  # Make y-axis logarithmic
    axes[i].legend()
    axes[i].grid(True)
axes[-1].set_xlabel('Epoch')
plt.tight_layout()
plt.show()
