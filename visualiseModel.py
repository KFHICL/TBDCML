# %%
#update
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
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow_probability as tfp

from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["TF_USE_LEGACY_KERAS"]="1" # Needed to import models saved before keras 3.0 release
import tf_keras as keras # Legacy keras version which is equal to the one on the HPC

# %% Load input data 
# trainDat_path = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'
trainDat_path = r'C:\Users\kaspe\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'
# trainDat_path = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\datain\MatLabModel2024_224_4kSamples'

# MC24 standard
# trainDat_name = 'MatLabModel2024' 
# sampleShape = [60,20]
# trainDat_path = os.path.join(trainDat_path,'MatLabModel2024')
# xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
# samplesPerFile = 1
# winKernel = 7
# nSpecimens = 100

# MC24_1000 standard
# trainDat_name = 'MatLabModel2024_1000' 
# sampleShape = [60,20]
# trainDat_path = os.path.join(trainDat_path,trainDat_name)
# xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
# samplesPerFile = 1
# winKernel = 7
# nSpecimens = 1000

# MC24_1000 const vf
# trainDat_name = 'MatLabModel2024_1000SamplesVfConstant' 
# sampleShape = [60,20]
# trainDat_path = os.path.join(trainDat_path,trainDat_name)
# xNames = ['Ex','Ey','Gxy','Vf','c2'] # Use all available features
# samplesPerFile = 1
# winKernel = 7
# nSpecimens = 1000

# # LFC18 standard
trainDat_name = 'Gaudron2018' 
sampleShape = [55,20]
xNames = ['E11','E22','E12'] # Names of input features in input csv
trainDat_path = os.path.join(trainDat_path,'Gaudron2018') # Path for training data samples
samplesPerFile = 1
winKernel = 5
nSpecimens = 100

# MC24x standard
# sampleShape = [224,224]
# xNames = ['Ex','Ey','Gxy','Vf','c2']# Names of input features in input csv
# samplesPerFile = 40
# winKernel = 17
# nSpecimens = 4000

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
    # files = files[2:3]

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



def get_padding_shape(height, width, multiple=32): # Can do up to 4 levels of downsampling with 64x32 images
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return ((pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2))

def peak_loss(y_true,y_pred):
  # Compute the 95th percentile value of y_true
  percentile_95 = tfp.stats.percentile(y_true, 95.0, interpolation='linear')
  # Create a mask for values >= 95th percentile
  mask = tf.greater_equal(y_true, percentile_95)
  # Compute squared error
  squared_error = tf.math.square(y_true - y_pred)
  # Only keep errors where y_true >= 95th percentile
  masked_error = tf.where(mask, squared_error, tf.zeros_like(squared_error))
  # Compute mean over the selected elements (avoid dividing by all elements)
  num_selected = tf.reduce_sum(tf.cast(mask, tf.float32))
  loss = tf.reduce_sum(masked_error) / (num_selected + 1e-8)
  return loss

def custom_loss(y_true,y_pred):
  SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
  loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.nn.relu(y_true))))
  loss = tf.reduce_mean(loss)
  return loss

def custom_loss5(y_true,y_pred):
  SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
  loss = tf.math.multiply(SE_base,(tf.math.add(tf.constant(1,dtype=tf.float32),tf.math.multiply(tf.nn.relu(y_true),5))))
  loss = tf.reduce_mean(loss)
  return loss

def custom_loss_power(y_true, y_pred, alpha=10, beta=10):
    SE_base = tf.math.square(tf.math.subtract(y_true,y_pred))
    weights = tf.math.add(
      tf.constant(1.0, dtype=tf.float32),
      tf.math.multiply(
        tf.constant(alpha, dtype=tf.float32),
        tf.math.pow(y_true, tf.constant(beta, dtype=tf.float32))
      )
    )
    return tf.reduce_mean(SE_base * weights)

custom_objects = {"SSIM_metric": SSIM_metric, 'get_padding_shape': get_padding_shape, 'peak_loss': peak_loss, 'custom_loss': custom_loss, 'custom_loss5': custom_loss5, 'custom_loss_power': custom_loss_power}


# Optimised LFC18 with MSE
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_Opti_MSE\model_default_20251215184541_1.keras"

# modelPath = r"C:\Users\kfh23\Desktop\tempFiles\model_20250401_MC24_CrossValidation_PostOpti_1_1.keras"
# modelPath = r"C:\Users\kfh23\Desktop\tempFiles\model_20250311_LFC18_CrossValidation_PreOpti_1_1.keras"
# modelPath = r"C:\Users\kfh23\Desktop\tempFiles\model_20250401_LFC18_CrossValidation_PostOpti_1_1.keras"
# modelPath = r"C:\Users\kfh23\Desktop\model_20250430_MC24x_modelDepth_1_4.keras"
# modelPath = r"C:\Users\kfh23\Desktop\MC24_standard_downsample_postBestWeights.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250806_LFC18_crossVal_Baseline_1\dataout\model_20250806_LFC18_crossVal_Baseline_1_1.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\model_20250808_LFC18_crossVal_Opti_1_4.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\model_20250808_LFC18_crossVal_Opti_1_5.keras"
modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\model_20250808_LFC18_crossVal_Opti_1_4.keras"

# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Baseline_1\dataout\model_20250814_MC24_1000_crossVal_Baseline_1_1.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\model_20250814_MC24_1000_crossVal_Opti_1_1.keras"
# modelPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1\dataout\model_20250814_MC24_1000_crossVal_Opti_1_1.keras"

# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250814_MC24_1000_constVf_Opti\model_20250814_MC24_1000_constVf_Opti_1.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_custom5\model_20250819_MC24_1000_custom5_1.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_customPower\model_20250819_MC24_1000_customPower_1.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250820_MC24_1000_customPower\model_20250820_MC24_1000_customPower_1.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250820_MC24_1000_peakLoss\model_20250820_MC24_1000_peakLoss_1.keras"
# modelPath = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250814_MC24_1000_constVf_Opti\model_20250814_MC24_1000_constVf_Opti_1.keras"
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
model = tf.keras.models.load_model(modelPath, custom_objects = custom_objects)
# model = tf.keras.models.load_model(modelPath, custom_objects = custom_objects)
# model = keras.models.load_model(modelPath, custom_objects = custom_objects)
# with keras.saving.custom_object_scope(custom_objects):
#     model = keras.models.load_model(modelPath)
modelName = os.path.basename(modelPath).split('.')[0] # Extract model name from path


# model = CNNModel
# Load "samples" and "trainHist" JSON files from the same folder as the model
model_folder = os.path.dirname(modelPath)
model_base = os.path.splitext(os.path.basename(modelPath))[0].removeprefix("model_")
samples_json_path = os.path.join(model_folder, f"samples_{model_base}.json")
train_hist_json_path = os.path.join(model_folder, f"trainHist_{model_base}.json")


# with open(samples_json_path, "r") as f:
#     samples_json = json.load(f)

# with open(train_hist_json_path, "r") as f:
#     train_hist_json = json.load(f)
# # %%
# paramPath = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_1\dataout\parameters_20250808_LFC18_crossVal_Opti_1_1.json"
# with open(paramPath, "r") as p:
#     params = json.load(p)

# print("Model parameters:")

## %%
# ResPaths = [r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_1',
#             r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_2',
#             r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_3'
#             ]

# # For each ResPath, load the model and corresponding samples JSON to get train/val sample IDs
# for respath in ResPaths:
#     dataout_folder = os.path.join(respath, "dataout")
#     # Find the model file (assume only one .keras file per folder)
#     model_files = [f for f in os.listdir(dataout_folder) if f.endswith('.keras')]
    
#     for m, modelPath in enumerate(model_files):
#         model_base = os.path.splitext(os.path.basename(modelPath))[0].removeprefix("model_")
#         samples_json_path = os.path.join(model_folder, f"samples_{model_base}.json")
#         with open(samples_json_path, "r") as f:
#             samples = json.load(f)

#     model_file = model_files[0]
#     model_path = os.path.join(dataout_folder, model_file)
#     # Load the model (optional, can skip if only IDs are needed)
#     # loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#     # Find the samples JSON file
#     model_base = os.path.splitext(model_file)[0].removeprefix("model_")
#     samples_json_path = os.path.join(dataout_folder, f"samples_{model_base}.json")
#     if not os.path.exists(samples_json_path):
#         print(f"No samples JSON found in {dataout_folder}")
#         continue
#     with open(samples_json_path, "r") as f:
#         samples_json = json.load(f)
#     # Extract train and validation sample IDs
#     train_ids = samples_json.get("train", [])
#     val_ids = samples_json.get("val", [])
#     print(f"ResPath: {respath}")
#     print(f"  Train IDs: {train_ids}")
#     print(f"  Val IDs: {val_ids}")

# # %%
# model.evaluate(samples.batch(10))


## %% Compare predictions to ground truth for all samples as a heatmap
# Extract all ground truths and predictions
all_ground_truths = []
all_predictions = []
gt_nonFlat = []
pred_nonFlat = []
subset = 100

for sample in samples.take(subset).batch(1):
    ground_truth = sample[1].numpy()  # Extract ground truth
    prediction = model.predict(sample[0])  # Get model prediction
    all_ground_truths.append(ground_truth.flatten())
    all_predictions.append(prediction.flatten())
    gt_nonFlat.append(ground_truth)
    pred_nonFlat.append(prediction)

# Convert lists to numpy arrays
all_ground_truths = np.concatenate(all_ground_truths)
all_predictions = np.concatenate(all_predictions)
# Calculate mean error, mean squared error, and root mean squared error
mean_error = np.mean(abs(all_predictions - all_ground_truths))
mean_error_relData = np.mean(abs(all_predictions - all_ground_truths)/all_ground_truths)
mean_squared_error = np.mean((all_predictions - all_ground_truths) ** 2)
root_mean_squared_error = np.sqrt(mean_squared_error)

# Calculate mean error as a percentage of the mean ground truth value
mean_ground_truth = np.mean(all_ground_truths)
mean_error_percentage = (mean_error / mean_ground_truth) * 100

print(f"Mean Error: {mean_error}")
print(f"Mean Squared Error: {mean_squared_error}")
print(f"Root Mean Squared Error: {root_mean_squared_error}")
print(f"Mean Error Percentage: {mean_error_percentage:.2f}%")

SSIM_s = []
for i, preds in enumerate(pred_nonFlat):
    SSIM = tf.image.ssim(
  img1 = gt_nonFlat[i]* 3.768 /0.82417405,
  img2 = preds * 3.768 /0.82417405,
  max_val = 1,
  filter_size=winKernel,
  filter_sigma=1.5,
  k1=0.01,
  k2=0.03,
  return_index_map=False
  )
    SSIM_s.append(SSIM)

print(np.average(SSIM_s))
print(np.max(all_ground_truths)-np.min(all_ground_truths)) # = 3.768 for LFC18
print(np.max(all_ground_truths)-np.min(all_ground_truths))


errors = all_predictions - all_ground_truths

# Find the 95th percentile threshold of ground truth values
percentile_95 = np.percentile(all_ground_truths, 95)

# Mask for points where ground truth is in the top 5th percentile
top_5_mask = all_ground_truths >= percentile_95

# Errors for all points and for top 5th percentile points
errors_all = errors
errors_top5 = errors[top_5_mask]
print("Mean absolute error for all points: ", np.mean(np.abs(errors_all)))

 # = 0.82417405 for MC24
# %%
# Set text size variables for consistent plot styling
tick_label_size = 12
axis_label_size = 14
legend_size = 12

plt.rc('xtick', labelsize=tick_label_size)
plt.rc('ytick', labelsize=tick_label_size)
plt.rc('axes', labelsize=axis_label_size)
plt.rc('legend', fontsize=legend_size)


# Create a DataFrame for easier plotting
data = pd.DataFrame({
    'Ground Truth': all_ground_truths,
    'Prediction': all_predictions
})

# Take n0000 random points for the scatterplot
scatter_data = data.sample(n=20000, random_state=42)

# Draw a combo histogram, scatterplot, and density contours
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x=scatter_data['Ground Truth'], 
    y=scatter_data['Prediction'], 
    s=5, 
    color="0", 
    # label="Data Points"
)
hist = sns.histplot(
    x=data['Ground Truth'], 
    y=data['Prediction'], 
    bins=100, 
    cbar=True, 
    pthresh=0.1,
    cbar_kws={'label': 'Density'}, 
    cmap="viridis",
    label="Point concentration"
)

# Plot the perfect prediction line
plt.plot(
    [data['Ground Truth'].min(), data['Ground Truth'].max()], 
    [data['Ground Truth'].min(), data['Ground Truth'].max()], 
    color='black', 
    linestyle='--',
    linewidth=1, 
    # label='Perfect Prediction'
)

plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.axis('equal')
plt.xlim([data['Ground Truth'].min(), data['Ground Truth'].max()])
plt.ylim([data['Ground Truth'].min(), data['Ground Truth'].max()])

plt.title('Comparison of Predictions to Ground Truth')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# hist.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.tight_layout()

# Save the figure in high quality
# output_path = r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal'
# output_path = r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal'
# output_path = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures'
output_path = r'C:\Users\kaspe\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures'
output_filepath = os.path.join(output_path, 'PredGT_Scatter_{modelName}.png'.format(modelName=modelName))
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')

plt.show()

# %%
# Plot the distribution of prediction errors as a function of ground truth value

errors = all_predictions - all_ground_truths

plt.figure(figsize=(10, 6))
# Use a 2D histogram (hexbin or hist2d) to show error vs ground truth value
hb = plt.hexbin(
    all_ground_truths, errors, 
    gridsize=80, cmap='viridis', 
    mincnt=1, linewidths=0.2
)
plt.colorbar(hb, label='Count')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction Error (Prediction - Ground Truth)')
plt.title('Prediction Error Distribution vs Ground Truth')
plt.tight_layout()

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
output_path = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures'
output_filepath = os.path.join(output_path, f'Prediction_Error_vs_GT_{modelName}.pdf')
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.show()


# %%
# Calculate prediction errors
errors = all_predictions - all_ground_truths

# Find the 95th percentile threshold of ground truth values
percentile_95 = np.percentile(all_ground_truths, 95)

# Mask for points where ground truth is in the top 5th percentile
top_5_mask = all_ground_truths >= percentile_95

# Errors for all points and for top 5th percentile points
errors_all = errors
errors_top5 = errors[top_5_mask]

# Use seaborn's colorblind palette
colorblind_palette = sns.color_palette("colorblind")
sns.set_palette(colorblind_palette)

plt.figure(figsize=(8, 4))
sns.kdeplot(errors_all, label='All Predictions', color=colorblind_palette[0], fill=True)
sns.kdeplot(errors_top5, label='Top 1% FI locations', color=colorblind_palette[1], fill=True)
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.xlim([-0.7,0.7])
plt.ylim([0,7])
plt.axvline(0, color='black', linestyle=':', linewidth=1)
plt.title('Error Distribution: All vs Top 5% FI')
plt.grid(axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5, which='both')
plt.minorticks_on()
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
output_filepath = os.path.join(output_path, f'ErrorDistributions_{modelName}.pdf')
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.show()

# %%
# Visualize loss functions: x = error (y_true - y_pred), y = y_true, color = loss value

loss_functions = {
  "MSE": lambda y_true, error: error**2,
  # "MAE": lambda y_true, error: np.abs(error),
  # "Custom": lambda y_true, error: (error**2) * (1 + np.maximum(y_true, 0)),
  # Peak: MSE only for y_true >= 95th percentile of y_true
  "Peak": lambda y_true, error: np.where(y_true >= np.percentile(y_true, 95), error**2, 0),
  # "Custom5": lambda y_true, error: (error**2) * (1 + 5 * np.maximum(y_true, 0)),
  "CustomPower": lambda y_true, error: (error**2) * (1 + 10 * np.power(y_true, 10)),
}

error_range = np.linspace(-2, 2, 200)
y_true_range = np.linspace(0, 1, 200)
E, Y = np.meshgrid(error_range, y_true_range)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes = axes.flatten()

for idx, (name, func) in enumerate(loss_functions.items()):
  Z = func(Y, E)
  ax = axes[idx]
  c = ax.contourf(E, Y, Z, levels=50, cmap='viridis')
  ax.set_title(name)
  ax.set_xlabel("Error (y_true - y_pred)")
  ax.set_ylabel("y_true")
  fig.colorbar(c, ax=ax)
for ax in axes[len(loss_functions):]:
  ax.axis('off')
plt.tight_layout()
output_filepath = os.path.join(output_path, f'LossFunctions_{modelName}.png')
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.show()

# %%


# Plot training and validation MSE and SSIM curves on the same subplot with dual y-axes
custom_ticks = np.array([0.4, 0.3, 0.2, 0.1,0.05])
############# SET THIS

epochs = np.arange(1, len(train_hist_json['mean_squared_error']) + 1)
train_loss = np.array(train_hist_json['mean_squared_error'])
val_loss = np.array(train_hist_json['val_mean_squared_error'])
train_ssim = np.array(train_hist_json['SSIM_metric'])
val_ssim = np.array(train_hist_json['val_SSIM_metric'])

fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'
color4 = 'tab:red'

# Define custom color palettes
RMSE_palette = {
    'train': '#66D3B3',  # lighter green
    'val': '#029E73',    # base green
    'test': '#01523D'    # darker green
}

SSIM_palette = {
    'train': '#66ADD6',  # lighter blue
    'val': '#0173B2',    # base blue
    'test': '#01436A'    # darker blue
}

# Plot RMSE on left y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('RMSE', color=RMSE_palette['train'])
l1 = ax1.plot(epochs, np.sqrt(train_loss), label='Train RMSE', color=RMSE_palette['train'])
l2 = ax1.plot(epochs, np.sqrt(val_loss), label='Val RMSE', color=RMSE_palette['val'])
ax1.tick_params(axis='y', labelcolor=RMSE_palette['train'])
import matplotlib.ticker as mticker
ax1.set_yscale('log')
ax1.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
ax1.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(max(custom_ticks),min(custom_ticks)), numticks=100))
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax1.minorticks_on()
ax1.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Plot SSIM on right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('SSIM', color=SSIM_palette['train'])
l3 = ax2.plot(epochs, train_ssim, label='Train SSIM', color=SSIM_palette['train'])
l4 = ax2.plot(epochs, val_ssim, label='Val SSIM', color=SSIM_palette['val'])
ax2.tick_params(axis='y', labelcolor=SSIM_palette['train'])



# Set grid and minor ticks
# ax1.grid(axis='y')
# ax1.minorticks_on()
ax1.tick_params(axis='x', colors='black')
ax1.tick_params(axis='y', colors='black')
ax2.tick_params(axis='y', colors='black')
# Custom tick locations for log scale (e.g., 1.0, 0.8, 0.6, ..., 0.1)

# Define custom ticks (descending order for log scale)

ax1.set_yticks(custom_ticks)
# ax1.get_yaxis().set_major_formatter(ScalarFormatter())
# ax1.set_ylim([custom_ticks[-1], custom_ticks[0]])
# Set colored axis labels with background boxes
ax1.set_ylabel('RMSE', color='black', bbox=dict(facecolor="#029E73", edgecolor="#029E73", pad=0.2, alpha=0.5, boxstyle='Round'))
ax2.set_ylabel('SSIM', color='black', bbox=dict(facecolor="#0173B2", edgecolor="#0173B2", pad=0.2, alpha=0.5, boxstyle='Round'))

# Gather legend handles and labels from both axes
h, l = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

# Only keep the last two (train/val) for each axis
unique_data_types = 2
h, l = h[-unique_data_types:], l[-unique_data_types:]
h2, l2 = h2[-unique_data_types:], l2[-unique_data_types:]

# Remove legends from individual axes if present
if ax1.get_legend() is not None:
    ax1.get_legend().remove()
if ax2.get_legend() is not None:
    ax2.get_legend().remove()

# Add combined legends to the figure
fig.legend(title='RMSE', handles=h, labels=l, 
           loc="lower left", ncol=1, bbox_to_anchor=(0, -.2))
fig.legend(title='SSIM', handles=h2, labels=l2, 
           loc="lower right", ncol=1, bbox_to_anchor=(1, -.2))

plt.title('Training and Validation RMSE & SSIM')
output_filepath = os.path.join(output_path, f'TrainingCurves_{modelName}.png')
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.tight_layout()
plt.show()
# %%


# Plot ground truth vs prediction distributions
# Use seaborn's colorblind palette for consistent, accessible colors
colorblind_palette = sns.color_palette("colorblind")
sns.set_palette(colorblind_palette)
plt.figure(figsize=(10, 6))
sns.kdeplot(all_ground_truths, label='Ground Truth', color='blue', fill=True, alpha=0.5)
sns.kdeplot(all_predictions, label='Prediction', color='orange', fill=True, alpha=0.5)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Ground Truth vs Prediction Distribution')
plt.legend()
plt.tight_layout()

output_filepath = os.path.join(output_path, 'PredGT_Distribution_{modelName}.png'.format(modelName=modelName))
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')

plt.show()

# %% Quantify datasets (LFC18 and MC24)
trainDat_path = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'
# MC24 standard
trainDat_name = 'MatLabModel2024'
sampleShape = [60,20]
mc24_path = os.path.join(trainDat_path, 'MatLabModel2024')
xNames = ['Ex','Ey','Gxy','Vf','c2']
samplesPerFile = 1
winKernel = 7
nSpecimens = 100
numSamples = len(os.listdir(mc24_path)) * samplesPerFile
headers_mc24, samples_mc24 = load_all_samples(mc24_path, numSamples)

# MC24_1000 standard
trainDat_name = 'MatLabModel2024_1000'
sampleShape = [60,20]
mc24_1000_path = os.path.join(trainDat_path, 'MatLabModel2024_1000')
xNames = ['Ex','Ey','Gxy','Vf','c2']
samplesPerFile = 1
winKernel = 7
nSpecimens = 1000
numSamples = len(os.listdir(mc24_1000_path)) * samplesPerFile
headers_mc24_1000, samples_mc24_1000 = load_all_samples(mc24_1000_path, numSamples)

# MC24_1000 const vf
trainDat_name = 'MatLabModel2024_1000SamplesVfConstant'
sampleShape = [60,20]
mc24_1000_constvf_path = os.path.join(trainDat_path, 'MatLabModel2024_1000SamplesVfConstant')
xNames = ['Ex','Ey','Gxy','Vf','c2']
samplesPerFile = 1
winKernel = 7
nSpecimens = 1000
numSamples = len(os.listdir(mc24_1000_constvf_path)) * samplesPerFile
headers_mc24_1000_constvf, samples_mc24_1000_constvf = load_all_samples(mc24_1000_constvf_path, numSamples)

# LFC18 standard
trainDat_name = 'Gaudron2018'
sampleShape = [55,20]
lfc18_path = os.path.join(trainDat_path, 'Gaudron2018')
xNames = ['E11','E22','E12']
samplesPerFile = 1
winKernel = 5
nSpecimens = 100
numSamples = len(os.listdir(lfc18_path)) * samplesPerFile
headers_lfc18, samples_lfc18 = load_all_samples(lfc18_path, numSamples)

#%% Convert all loaded samples datasets to numpy arrays for easier manipulation
def dataset_to_numpy(samples_dataset):
    # Add a new dimension to the dataset (first dim)
    samples_dataset = samples_dataset.map(lambda X, Y: (tf.expand_dims(X, 0) if X.ndim == 3 else X, tf.expand_dims(Y, 0) if Y.ndim == 3 else Y))
    X_list, Y_list = [], []
    for X, Y in samples_dataset:
        X_list.append(X.numpy())
        Y_list.append(Y.numpy())
    X_np = np.concatenate(X_list, axis=0)
    Y_np = np.concatenate(Y_list, axis=0)
    return X_np, Y_np

X_mc24, Y_mc24 = dataset_to_numpy(samples_mc24)
X_mc24_1000, Y_mc24_1000 = dataset_to_numpy(samples_mc24_1000)
X_mc24_1000_constvf, Y_mc24_1000_constvf = dataset_to_numpy(samples_mc24_1000_constvf)
X_lfc18, Y_lfc18 = dataset_to_numpy(samples_lfc18)

# %%
from scipy.stats import pearsonr
def compute_pearson_correlations(X, Y, feature_names, label_names):
    # X: (samples, h, w, features), Y: (samples, h, w, labels)
    # Flatten spatial dims and samples
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1, Y.shape[-1])
    results = []
    for i, fname in enumerate(feature_names):
        for j, lname in enumerate(label_names):
            # Remove NaNs for correlation
            mask = ~np.isnan(X_flat[:, i]) & ~np.isnan(Y_flat[:, j])
            if np.any(mask):
                corr, pval = pearsonr(X_flat[mask, i], Y_flat[mask, j])
            else:
                corr, pval = np.nan, np.nan
            results.append((fname, lname, corr, pval))
    return results

# Prepare feature and label names for each dataset
datasets = [
    # ("MC24", X_mc24, Y_mc24, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    # ("MC24_1000", X_mc24_1000, Y_mc24_1000, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    # ("MC24_1000_constVf", X_mc24_1000_constvf, Y_mc24_1000_constvf, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    ("LFC18", X_lfc18, Y_lfc18, ['Ex','Ey','Gxy'], ['FI']),
    ("MC24", X_mc24_1000, Y_mc24_1000, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    ("MC24_constVf", X_mc24_1000_constvf, Y_mc24_1000_constvf, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    
]

# Collect results
summary = {}
for name, X, Y, xnames, ynames in datasets:
    summary[name] = compute_pearson_correlations(X, Y, xnames, ynames)

# Display as table
# for dataset, results in summary.items():
#     print(f"\nPearson Correlations for {dataset}:")
#     print(f"{'Feature':<10} {'Label':<10} {'Correlation':>12} {'p-value':>12}")
#     for feat, lab, corr, pval in results:
#         print(f"{feat:<10} {lab:<10} {corr:12.4f} {pval:12.2e}")

# # Optionally, plot as barplot for each dataset
# for dataset, results in summary.items():
#     features = [r[0] for r in results]
#     corrs = [r[2] for r in results]
#     plt.figure(figsize=(6, 3))
#     plt.bar(features, corrs, zorder=2)
#     plt.grid(axis='y', zorder=1)
#     plt.title(f'Pearson Correlation ({dataset})')
#     plt.ylabel('Correlation with label')
#     plt.ylim(-1, 1)
#     plt.tight_layout()
    
#     plt.show()
    

# Organize the summary variable (Pearson correlations) into a DataFrame for plotting
corr_rows = []
for dataset, results in summary.items():
    for feat, lab, corr, pval in results:
        corr_rows.append({
            "Dataset": dataset,
            "Feature": feat,
            "Label": lab,
            "Correlation": corr,
            "p-value": pval
        })
corrDf = pd.DataFrame(corr_rows)

# If you want to plot, e.g., correlation with label 'FI' only:
corrDf_plot = corrDf[corrDf["Label"] == "FI"]

plt.style.use("seaborn-v0_8-colorblind")
fig = plt.figure()
ax = plt.subplot(1,1,1)
bp = sns.barplot(data=corrDf_plot, x='Feature', y='Correlation', hue='Dataset', zorder=2)
plt.grid(zorder=1)
plt.xlabel('')
plt.ylabel('Correlation', labelpad=-2)
# plt.title('Pearson correlation with FI')
h,l = ax.get_legend_handles_labels()
ax.get_legend().remove()
# ax.set_yscale('log')
fig.legend(title='Dataset', handles=h, labels=l, 
           loc="upper left", ncol=1, bbox_to_anchor=(1, 1))

# Add a checkbox widget to hide/show datasets interactively (requires ipywidgets and Jupyter)
# If not in Jupyter, you can manually set which datasets to show by filtering corrDf_plot

# Example: Hide a dataset by setting its alpha to 0 (invisible) but keeping its space
# Let's say you want to hide "MC24_constVf"
# hide_dataset = "MC24_constVf"
# for patch, (_, row) in zip(bp.patches, corrDf_plot.iterrows()):
#     if row["Dataset"] == hide_dataset:
#         patch.set_alpha(0)  # Hide bar but keep space

# hide_dataset = "MC24"
# for patch, (_, row) in zip(bp.patches, corrDf_plot.iterrows()):
#     if row["Dataset"] == hide_dataset:
#         patch.set_alpha(0)  # Hide bar but keep space

# Optionally, you can add a legend entry for the hidden dataset with a note
# handles, labels = ax.get_legend_handles_labels()
# handles.append(Patch(alpha=0, label=f"{hide_dataset} (hidden)"))
# fig.legend(handles=handles, labels=labels, title='Dataset', loc="upper left", ncol=1, bbox_to_anchor=(1, 1))
output_filepath = os.path.join(output_path, f'Correlation_LFC18_MC24_MC24ConstVf.png')
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.show()






# %%
from sklearn.feature_selection import mutual_info_regression
def compute_mutual_information(X, Y, feature_names, label_names):
    # X: (samples, h, w, features), Y: (samples, h, w, labels)
    # Flatten spatial dims and samples
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1, Y.shape[-1])
    results = []
    for i, fname in enumerate(feature_names):
        for j, lname in enumerate(label_names):
            # Remove NaNs for MI calculation
            mask = ~np.isnan(X_flat[:, i]) & ~np.isnan(Y_flat[:, j])
            if np.any(mask):
                # MI expects 1d y, 2d X
                mi = mutual_info_regression(X_flat[mask, [i]].reshape(-1, 1), Y_flat[mask, j].reshape(-1, 1), random_state=0)[0]
            else:
                mi = np.nan
            results.append((fname, lname, mi))
    return results

# Compute MI for all datasets
mi_summary = {}
for name, X, Y, xnames, ynames in datasets:
    mi_summary[name] = compute_mutual_information(X, Y, xnames, ynames)

# Display as table
for dataset, results in mi_summary.items():
    print(f"\nMutual Information for {dataset}:")
    print(f"{'Feature':<10} {'Label':<10} {'Mutual Info':>12}")
    for feat, lab, mi in results:
        print(f"{feat:<10} {lab:<10} {mi:12.4f}")

# Plot as barplot for each dataset
for dataset, results in mi_summary.items():
    features = [r[0] for r in results]
    mis = [r[2] for r in results]
    plt.figure(figsize=(6, 3))
    plt.bar(features, mis)
    plt.title(f'Mutual Information ({dataset})')
    plt.ylabel('MI with label')
    plt.tight_layout()
    plt.show()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Helper function to flatten X and Y for regression
def flatten_for_regression(X, Y):
    # X: (samples, h, w, features), Y: (samples, h, w, labels)
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1, Y.shape[-1])
    return X_flat, Y_flat

# Store results for each dataset
regression_results = {}

for name, X, Y, xnames, ynames in datasets:
    X_flat, Y_flat = flatten_for_regression(X, Y)
    # Remove NaNs
    mask = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(Y_flat).any(axis=1)
    X_clean = X_flat[mask]
    Y_clean = Y_flat[mask]
    # Split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X_clean, Y_clean, test_size=0.1, random_state=42)
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    # Predict on test set
    Y_pred = reg.predict(X_test)
    # Metrics
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    regression_results[name] = {
        "model": reg,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "mse": mse,
        "r2": r2,
        "xnames": xnames,
        "ynames": ynames
    }
    print(f"{name}: Test MSE={mse:.4f}, R2={r2:.4f}")

    # Plot: Linear fit vs test data
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test, Y_pred, s=1, alpha=0.3, label="Test Data")
    min_val = min(Y_test.min(), Y_pred.min())
    max_val = max(Y_test.max(), Y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Fit")
    plt.xlabel("Ground Truth")
    plt.ylabel("Linear Regression Prediction")
    plt.title(f"{name}: Linear Regression Fit\nMSE={mse:.4f}, R2={r2:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% PCA

from sklearn.decomposition import PCA

# PCA visualization for all 4 datasets
# Prepare dataset info: (name, X, Y, feature_names, label_names)
dataset_info = [
    ("MC24", X_mc24, Y_mc24, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    ("MC24_1000", X_mc24_1000, Y_mc24_1000, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    ("MC24_1000_constVf", X_mc24_1000_constvf, Y_mc24_1000_constvf, ['Ex','Ey','Gxy','Vf','c2'], ['FI']),
    ("LFC18", X_lfc18, Y_lfc18, ['E11','E22','E12'], ['FI']),
]

# Number of points to plot (limit for performance/clarity)
n_plot = min(100000, min([X.shape[0]*X.shape[1]*X.shape[2] for _, X, _, _, _ in dataset_info]))

# Create a 1x4 subplot for each dataset
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=100)

for idx, (name, X, Y, xnames, ynames) in enumerate(dataset_info):
    # Flatten spatial and sample dimensions for PCA
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1, Y.shape[-1])
    # Remove rows with NaNs in either X or Y
    mask = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(Y_flat).any(axis=1)
    X_flat = X_flat[mask]
    Y_flat = Y_flat[mask]
    # Fit PCA to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    # Randomly sample points for plotting if too many
    if X_pca.shape[0] > n_plot:
        plot_idx = np.random.choice(X_pca.shape[0], n_plot, replace=False)
    else:
        plot_idx = np.arange(X_pca.shape[0])
    ax = axes[idx]
    # Scatter plot: color by first label (e.g., FI)
    sc = ax.scatter(
        X_pca[plot_idx, 0],
        X_pca[plot_idx, 1],
        c=Y_flat[plot_idx, 0],
        cmap='viridis',
        alpha=0.75,
        s=2
    )
    ax.set_title(f'PCA: {name}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # Add colorbar for the label
    plt.colorbar(sc, ax=ax, label=ynames[0])
    ax.grid(True)
    # Set scientific notation for both axes
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.tight_layout()
output_filepath = os.path.join(output_path, 'PCA_Datasets_{modelName}.png'.format(modelName=modelName))
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')

plt.show()


# %% PCA captured variance

# PCA captured variance for all 4 datasets

import matplotlib.ticker

# Prepare dataset info: (name, X, feature_names)
dataset_info = [
    ("MC24", X_mc24, ['Ex','Ey','Gxy','Vf','c2']),
    ("MC24_1000", X_mc24_1000, ['Ex','Ey','Gxy','Vf','c2']),
    ("MC24_1000_constVf", X_mc24_1000_constvf, ['Ex','Ey','Gxy','Vf','c2']),
    ("LFC18", X_lfc18, ['E11','E22','E12']),
]

explained_variances = {}
num_components = {}

for name, X, feature_names in dataset_info:
    n_features = X.shape[-1]
    max_components = min(10, n_features)
    nums = np.arange(1, max_components + 1)
    var_ratios = []
    X_flat = X.reshape(-1, n_features)
    for num in nums:
        pca = PCA(n_components=num)
        pca.fit(X_flat)
        var_ratios.append(np.sum(pca.explained_variance_ratio_))
    explained_variances[name] = var_ratios
    num_components[name] = nums

# Plot cumulative explained variance for all datasets
plt.figure(figsize=(7, 7))
ax = plt.gca()
markers = ['o', 's', 'D', '^']
for idx, (name, nums) in enumerate(num_components.items()):
    sns.lineplot(x=nums, y=explained_variances[name], marker=markers[idx % len(markers)], label=name)

# Print explained variance (not cumulative) for each principal component for each dataset
print("\nExplained variance ratio (per principal component):")
for name, X, feature_names in dataset_info:
    n_features = X.shape[-1]
    max_components = min(10, n_features)
    X_flat = X.reshape(-1, n_features)
    pca = PCA(n_components=max_components)
    pca.fit(X_flat)
    print(f"\n{name}:")
    print("PC\tExplained Variance Ratio")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"{i+1}\t{var:.4f}")
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance ratio')
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.legend(title='Dataset', ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.35))
plt.grid()
plt.tight_layout()
output_filepath = os.path.join(output_path, 'PCA_ExplainedVariance_{modelName}.png'.format(modelName=modelName))
plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.show()

# %% Shannon Entropy
from scipy.stats import entropy
def compute_shannon_entropy(Y, label_names, bins=100):
    # Y: (samples, h, w, labels)
    # Compute entropy for each label
    entropies = {}
    for i, lname in enumerate(label_names):
        y_flat = Y[..., i].flatten()
        y_flat = y_flat[~np.isnan(y_flat)]
        hist, bin_edges = np.histogram(y_flat, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
        shannon_entropy = entropy(hist, base=2)
        entropies[lname] = shannon_entropy
    return entropies

for name, X, Y, xnames, ynames in datasets:
    entropies = compute_shannon_entropy(Y, ynames)
    for lname, ent in entropies.items():
        print(f"Shannon entropy of {lname} in {name}: {ent:.4f} bits")

# %% Variance of field gradients
# Compute mean and variance of gradient magnitudes for all datasets, normalized by label range

def compute_gradient_stats_normalized(Y, label_names):
    # Y: (samples, h, w, labels)
    stats = {}
    for i, lname in enumerate(label_names):
        y = Y[..., i]
        # Compute range for normalization (avoid divide by zero)
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        y_range = y_max - y_min if y_max > y_min else 1.0
        # Compute gradients along spatial axes for each sample
        grad_mags = []
        for sample in y:
            gy, gx = np.gradient(sample)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_mags.append((grad_mag / y_range).flatten())
        grad_mags = np.concatenate(grad_mags)
        stats[lname] = {
            "mean_grad": np.mean(grad_mags),
            "var_grad": np.var(grad_mags),
            "range": y_range
        }
    return stats

for name, X, Y, xnames, ynames in datasets:
    grad_stats = compute_gradient_stats_normalized(Y, ynames)
    for lname, stat in grad_stats.items():
        print(f"{name} - {lname}: Mean |∇y|/range = {stat['mean_grad']:.4f}, Var |∇y|/range = {stat['var_grad']:.4f}, Range = {stat['range']:.4f}")

# Optionally, plot histogram of normalized gradient magnitudes for each dataset
for name, X, Y, xnames, ynames in datasets:
    grad_mags = []
    for i, lname in enumerate(ynames):
        y = Y[..., i]
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        y_range = y_max - y_min if y_max > y_min else 1.0
        for sample in y:
            gy, gx = np.gradient(sample)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_mags.append((grad_mag / y_range).flatten())
    grad_mags = np.concatenate(grad_mags)
    plt.figure(figsize=(6, 3))
    plt.hist(grad_mags, bins=100, alpha=0.7)
    plt.title(f'Normalized Gradient Magnitude Histogram ({name})')
    plt.xlabel('|∇y| / range')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# %% Fourier analysis
# Fourier analysis of label fields (e.g., FI) for all datasets

def compute_2d_fft_energy(Y, label_names):
    # Y: (samples, h, w, labels)
    fft_energies = {}
    for i, lname in enumerate(label_names):
        y = Y[..., i]
        # Compute 2D FFT for each sample, then average the power spectrum
        power_spectra = []
        for sample in y:
            # Remove NaNs (replace with 0 for FFT)
            sample = np.nan_to_num(sample)
            fft2 = np.fft.fft2(sample)
            fft2_shifted = np.fft.fftshift(fft2)
            power = np.abs(fft2_shifted) ** 2
            power_spectra.append(power)
        mean_power = np.mean(power_spectra, axis=0)
        fft_energies[lname] = mean_power
    return fft_energies

# Plot the mean 2D FFT energy for each dataset
for name, X, Y, xnames, ynames in datasets:
    fft_energies = compute_2d_fft_energy(Y, ynames)
    for lname, energy in fft_energies.items():
        plt.figure(figsize=(6, 5))
        plt.imshow(
            np.log1p(energy), 
            cmap='viridis', 
            aspect='auto', 
            extent=[-0.5, 0.5, -0.5, 0.5]
        )
        plt.colorbar(label='log(1 + Power)')
        plt.title(f'2D FFT Power Spectrum\n{name} - {lname}')
        plt.xlabel('Normalized Frequency (x)')
        plt.ylabel('Normalized Frequency (y)')
        plt.tight_layout()
        plt.show()

# Optionally, plot the radial average of the power spectrum (energy vs frequency)
def radial_profile(power):
    # Compute radial average of a 2D array
    y, x = np.indices(power.shape)
    center = np.array([(x.max() - x.min())/2.0, (y.max() - y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

for name, X, Y, xnames, ynames in datasets:
    fft_energies = compute_2d_fft_energy(Y, ynames)
    for lname, energy in fft_energies.items():
        rp = radial_profile(energy)
        plt.figure(figsize=(6, 4))
        plt.plot(rp / rp.max())
        plt.xlabel('Radial Frequency Bin')
        plt.ylabel('Normalized Energy')
        plt.title(f'Radial FFT Energy Profile\n{name} - {lname}')
        plt.tight_layout()
        plt.show()

# %%
# Load multiple Excel files and concatenate them into a single DataFrame

def load_excel_files_separately(file_list):
    dfs = []
    for file in file_list:
        df = pd.read_excel(file)
        dfs.append(df)
    return dfs

# Example usage:
file_list = [
    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250808_LFC18_crossVal\20250808_LFC18_crossVal_Opti_resDf.xlsx",
    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\20250814_MC24_1000_crossVal\20250814_MC24_1000_crossVal_Opti_resDf.xlsx",
]
res_dfs = load_excel_files_separately(file_list)

# Create an artificial DataFrame with the same columns as the loaded DataFrames
if res_dfs:
    artificial_df = pd.DataFrame([{
        'repeat': 1,
        'model': 1,
        'data': 'val',
        'loss': 0.0070392517,
        'mean_absolute_error': 0.0654064715,
        'mean_squared_error': 0.0070392517,
        'SSIM_metric': 0.8054416776,
        'RMSE': 0.0839002487
    }])
    artificial_df['source'] = 'MC24_100'
    res_dfs.append(artificial_df)

ranges = [3.77,0.838,0.79]

# Add a column to each dataframe where mean_absolute_error is normalized by the corresponding range
for df, rng in zip(res_dfs, ranges):
    df['mean_absolute_error_norm'] = df['mean_absolute_error'] / rng

# %%
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "8"
# Load the two CSV files into pandas DataFrames
csv1 = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run7\20250807_LFC18_modelDepth_run7_metrics.csv"
csv2 = r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run8\20250807_LFC18_modelDepth_run8_metrics.csv"

# csv1 = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run7\20250807_LFC18_modelDepth_run7_metrics.csv"
# csv2 = r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\HP_Optimisation_Results\LFC18_run8\20250807_LFC18_modelDepth_run8_metrics.csv"
palette = sns.color_palette("colorblind")

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

df_run7 = pd.read_csv(csv1)
df_run8 = pd.read_csv(csv2)

# Plot comparison of run 7 and run 8: Model depth vs RMSE (left y), with error bars, train/val distinguished


def plot_depth_vs_rmse(df1, df2, label1="Initial HP sweep", label2="Combination HP sweep\n(increased regularisation)"):
    fig, ax = plt.subplots(figsize=(6, 3))



    def plot_df(df, run_label, color, alpha=1.0):
        for data_type in ["train", "val"]:
            rmse = df[(df["Metric"] == "RMSE") & (df["Data"] == data_type)]
            linestyle = "-" if data_type == "train" else "--"
            marker = "o" if data_type == "train" else "s"
            linewidth = 1 if data_type == "train" else 0.75
            ax.errorbar(
                rmse["Model depth"], rmse["Mean"],
                #   yerr=rmse["Std"],
                label=f"{run_label} {data_type}",
                color=color, marker=marker, linestyle=linestyle, alpha=alpha, linewidth=linewidth, markersize=5
            )

    plot_df(df1[df1['Data']=='train'], label1, color=NRMSE_palette["val"], alpha=1.0)  # blue
    plot_df(df1[df1['Data']=='val'], label1, color=NRMSE_palette["val"], alpha=1.0)  # blue
    plot_df(df2[df2['Data']=='train'], label2, color=RMSE_palette["val"], alpha=1.0)  # orange
    plot_df(df2[df2['Data']=='val'], label2, color=RMSE_palette["val"], alpha=1.0)  # orange

     # Customize plot
    ax.set_xlabel("Model depth")
    ax.set_ylabel("RMSE")
    ax.set_xticks(sorted(df1["Model depth"].unique()))
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Only show legend for run colors (not train/val)
    legend_handles = [
        Patch(color=NRMSE_palette["val"], label=label1),
        Patch(color=RMSE_palette["val"], label=label2)
    ]
    style_lines = [
    plt.Line2D([0], [0], color='black', linestyle='-', label='Training', marker='o', linewidth=1),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Validation', marker='s', linewidth=0.75)
    ]
    legend2 = plt.legend(handles=style_lines, title="Data", loc="upper left", bbox_to_anchor=(1.03, 0.7), borderaxespad=0.)
    # legend1 = plt.legend(lines, model_labels[:len(lines)], title="MC24 dataset", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.015, 1.0), title="Run")
    ax.add_artist(legend2)  # Add the model legend back

    # ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Run")
    # plt.title("Model Depth vs RMSE (Run 7 vs Run 8)")
    plt.tight_layout()
    ax.set_ylim([0.07, 0.24])
    # output_filepath = os.path.join(r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures', 'ModelDepth_vs_RMSE_Run7vs8.png')
    # output_filepath = os.path.join(r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures', 'ModelDepth_vs_RMSE_Run7vs8_v2.pdf')
    output_filepath = os.path.join(r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures', 'ModelDepth_vs_RMSE_Run7vs8_v3.pdf')
    plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
    plt.show()

plot_depth_vs_rmse(df_run7, df_run8)



# %%
# Given a list of folders, load the model, trainHist, and results from each folder
import ast
import matplotlib
def load_model_and_results_from_folders(folders, custom_objects):
    models = []
    train_hists = []
    results = []
    samples = []
    for folder in folders:
        # only one .keras file per folder)
        model_files = [f for f in os.listdir(folder) if f.endswith('.keras')]
        if not model_files:
            print(f"No model found in folder: {folder}")
            continue
        model_path = os.path.join(folder, model_files[0])
        # Extract model base name for JSON files
        model_base = os.path.splitext(model_files[0])[0].removeprefix("model_")
        train_hist_json_path = os.path.join(folder, f"trainHist_{model_base}.json")
        samples_json_path = os.path.join(folder, f"samples_{model_base}.json")
        results_json_path = os.path.join(folder, f"results_{model_base}.json")
        # Load model
        # with keras.saving.custom_object_scope(custom_objects):
        #     try:
        #         model = keras.models.load_model(model_path)
        #     except Exception:
        #         model = tf.keras.models.load_model(model_path)
        # models.append(model)
        # Load trainHist
        if os.path.exists(train_hist_json_path):
            with open(train_hist_json_path, "r") as f:
                train_hist = json.load(f)
        else:
            train_hist = None
        train_hists.append(train_hist)
        # Load results (samples)
        if os.path.exists(samples_json_path):
            with open(samples_json_path, "r") as f:
                sample = json.load(f)
        else:
            sample = None
        samples.append(sample)
        
        if os.path.exists(results_json_path):
            with open(results_json_path, "r") as f:
                result = json.load(f)
        else:
            result = None
        results.append(result)
    return models, train_hists, results, samples

# Example usage:
folders = [
    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_Opti_trainedOn_MC24",
    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_Opti_trainedOn_MC24_200",
    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_Opti_trainedOn_MC24_500",
    r"C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250814_MC24_1000_crossVal_Opti",
]
# folders = [
#     r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_Opti_trainedOn_MC24",
#     r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_Opti_trainedOn_MC24_200",
#     r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250819_MC24_1000_Opti_trainedOn_MC24_500",
#     r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\20250814_MC24_1000_crossVal_Opti",
# ]
models, train_hists, results, samples = load_model_and_results_from_folders(folders, custom_objects)


# Plot loss curves from all train_hists on the same plot
# Allow manual legend labels for each model
model_labels = [
    "100",
    "200",
    "500",
    "1000"
]
# If you want to set custom labels, edit the above list to match your folders/models

model_colors = sns.color_palette("colorblind", n_colors=len(train_hists))
#  Training curves
plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
px = 1/plt.rcParams['figure.dpi']  # Inches per pixelmatplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams['axes.linewidth'] = 0.25

plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = "8"
# plt.rcParams['grid.linewidth'] = 0.2
latexWidth = 315
figWidth = latexWidth*px
figHeight = 2*figWidth/1.618 # Golden ratio
tick_locator = matplotlib.ticker.MaxNLocator(nbins=3) # Number of ticks on colorbars
cBarBins = 3
resolution_scaling = 1 # Manually scale DPI and text accordingly
fig = plt.figure(layout="constrained", dpi = resolution_scaling*100) # 100 is default size
fig.set_figheight(figHeight)

fig.set_figwidth(figWidth*2/2)
lines = []
for idx, hist in enumerate(train_hists):
    if hist is not None and 'mean_squared_error' in hist and 'val_mean_squared_error' in hist:
        epochs = np.arange(1, len(hist['mean_squared_error']) + 1)
        train_mse = np.array(hist['mean_squared_error'])
        val_mse = np.array(hist['val_mean_squared_error'])
        color = model_colors[idx]
        # Plot train and val, but only keep one handle for legend
        l_train, = plt.plot(epochs, train_mse, linestyle='-', color=color, linewidth=0.7, alpha=1)
        plt.plot(epochs, val_mse, linestyle='--', color=color, linewidth=0.7, alpha=0.7)

        lines.append(l_train)

plt.yscale('log')  # Set y-axis to log scale
# plt.ylim([min([train_mse.min(), val_mse.min()]), max([train_mse.max(), val_mse.max()])])
plt.ylim([0.001, 0.10171668231487274])
# plt.ylim([0.001, 0.015])
plt.xlim([0, 1050])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True, axis='y', linestyle='--', alpha=0.75, which='major')
plt.minorticks_on()
plt.grid(True, axis='y', linestyle=':', alpha=0.5, which='minor')
# plt.title('Training and Validation MSE Curves')

# Add custom legend for models (outside, right)


# Add a legend entry for line styles (train/val), also outside
style_lines = [
    plt.Line2D([0], [0], color='black', linestyle='-', label='Train', linewidth=0.7, alpha=1),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Val', linewidth=0.7, alpha=0.7)
]
# legend2 = plt.legend(handles=style_lines, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.12), title="Data")
legend2 = fig.legend(handles=style_lines, title="Data", loc="lower center",ncol=2, bbox_to_anchor=(0.55, -0.12))
legend1 = plt.legend(lines, model_labels[:len(lines)], title="MeC-Meso \nspecimens", loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
plt.gca().add_artist(legend2)  # Add the model legend back
plt.title('')









modelName = 'MC24_1000_optimised'
# plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
output_filepath_pdf = os.path.join(r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures', f'TrainingCurves_datasetSize_{modelName}.pdf')
# plt.savefig(output_filepath_pdf, dpi=400, bbox_inches='tight', format='pdf')


# output_filepath = os.path.join(r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures', 'TrainingCurves_datasetSize4_{modelName}.png'.format(modelName=modelName))
# plt.savefig(output_filepath, dpi=400, bbox_inches='tight')

plt.show()

# %%
# Plot RMSE and SSIM for training and validation as a function of dataset size (from results)



final_train_rmse = []
final_val_rmse = []
final_train_mae = []
final_val_mae = []
final_train_ssim = []
final_val_ssim = []

for res in results:
    if res is not None:
        # If result is a string (from JSON), parse it
        if isinstance(res, str):
            res_dict = ast.literal_eval(res)
        elif isinstance(res, dict):
            res_dict = res
        else:
            res_dict = json.loads(res)
        # Handle possible key capitalization
        ssim_key = None
        for k in res_dict.keys():
            if k.lower() == "ssim_metric":
                ssim_key = k
                break
        # Extract values
        final_train_rmse.append(res_dict["RMSE"]["train"])
        final_val_rmse.append(res_dict["RMSE"]["val"])
        final_train_mae.append(res_dict["mean_absolute_error"]["train"])
        final_val_mae.append(res_dict["mean_absolute_error"]["val"])
        if ssim_key:
            final_train_ssim.append(res_dict[ssim_key]["train"])
            final_val_ssim.append(res_dict[ssim_key]["val"])
        else:
            final_train_ssim.append(np.nan)
            final_val_ssim.append(np.nan)
    else:
        final_train_rmse.append(np.nan)
        final_val_rmse.append(np.nan)
        final_train_ssim.append(np.nan)
        final_val_ssim.append(np.nan)

x_labels = model_labels[:len(final_train_rmse)]

model_colors = sns.color_palette("colorblind", n_colors=len(train_hists))
train_colors = model_colors
val_colors = [sns.utils.set_hls_values(c) for c in train_colors]  # Lighten the colors for validation


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
ax = fig.add_subplot(1, 2, 2)
ax2 = ax.twiny()

fig.set_figheight(figHeight)

fig.set_figwidth((figWidth *2))

# Bar width
bar_width = 0.8
x = np.arange(len(x_labels))
# Create the colours from the seaborn colourblind palette
# palette = sns.color_palette("colorblind")
# train_color = palette[:4]  # First 4 colors from the palette
# val_color = [sns.utils.set_hls_values(c, l=0.7) for c in train_color]  # Lighten the colors for validation

bar_width = 0.35  # Adjust the bar width for side-by-side bars
x = np.arange(len(x_labels))  # Define the x positions for the groups

RMSEBarsTrain = ax.barh(x - bar_width / 2, np.flip(final_train_rmse), height=bar_width, color=np.flip(train_colors, axis=0), alpha=1, label='Train', zorder=3)
# Option 1: Use hatch pattern '///' to signify equivalence to a dotted line
# RMSEBarsVal = ax.barh(x + bar_width / 2, np.flip(final_val_rmse), height=bar_width, color=np.flip(val_color, axis=0), alpha=1, label='Val', zorder=0, hatch='///')

# Option 2: Use a different hatch pattern like '...' for a dotted effect
# RMSEBarsVal = ax.barh(x + bar_width / 2, np.flip(final_val_rmse), height=bar_width, color=np.flip(val_color, axis=0), alpha=1, label='Val', zorder=0, hatch='...')

# Option 3: Use a combination of hatch and reduced alpha for a more distinct visual
# RMSEBarsVal = ax.barh(x + bar_width / 2, np.flip(final_val_rmse), height=bar_width, color=np.flip(val_color, axis=0), alpha=0.7, label='Val', zorder=0, hatch='///')

# Option 4: Use a completely transparent fill with only the hatch pattern visible
RMSEBarsVal = ax.barh(x + bar_width / 2, np.flip(final_val_rmse), height=bar_width, color='none', edgecolor=np.flip(val_colors, axis=0), alpha=1, label='Val', zorder=0, hatch='/////')

for i, xi in enumerate(x):
    SSIM_val= ax2.scatter(np.flip(final_val_ssim)[i], xi + bar_width / 2,marker = 's',color='none',  edgecolors=np.flip(val_colors, axis=0)[i], s=50, zorder=4,alpha = 1, label='Val',hatch='/////')
SSIM_Train = ax2.scatter(np.flip(final_train_ssim), x - bar_width / 2, marker='s', color=np.flip(train_colors, axis=0), s=50, zorder=4, alpha=1, label='Train', linewidths=1.5,)


RMSEpos = 0.21
SSIMpos = 0.78



from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle
import matplotlib.lines as mlines
from matplotlib.patches import Patch


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


# Customize the first subplot (LFC18))
ax.set_yticks(np.flip(x))
# ax.set_yticklabels([])
ax.set_yticklabels([f'{label}' for label in x_labels],rotation=90, va='center', ha='right')
ax.set_ylabel('MeC-Meso-M specimens', rotation=90)
# ax.yaxis.set_label_coords(0,1.02)
ax.set_xlabel('RMSE', x=RMSEpos)
xminLFC18 = np.floor(np.min(final_train_rmse)*100)/100
xmaxLFC18 = np.ceil(np.max(final_val_rmse)*100)/100
ax.set_xlim([xminLFC18, xmaxLFC18 * 2 - xminLFC18])
# ax.legend(loc='lower left', bbox_to_anchor=(1, 0))  # outside the plot
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
# Manually set xticks for axLFC18 at reasonable intervals with least possible digits
xticks = np.linspace(xminLFC18, xmaxLFC18-0.01, num=3)
# xticks = np.linspace(np.round(rmse_data_LFC18['Best_Model_RMSE'].min(),2), np.round(rmse_data_LFC18['Baseline_RMSE'].max(),2), num=3)
# xticks = np.round(np.linspace(rmse_data_LFC18['Best_Model_RMSE'].min(), rmse_data_LFC18['Baseline_RMSE'].max(), num=3), 2)
ax.set_xticks(xticks)



ax2.set_xlabel('SSIM', x=SSIMpos)
minSSIM = np.min([np.min(final_train_ssim), np.min(final_val_ssim)])
xminLFC18 =  np.floor(minSSIM*100)/100
xmaxLFC18 = np.ceil(np.max(final_val_ssim)*100)/100
ax2.set_xlim([(2*xminLFC18 - xmaxLFC18)*0.9 , xmaxLFC18 * 1.01])
ax2.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
ax2.tick_params(axis='y', left=False, labelleft=False)
xticks2 = np.linspace(xminLFC18+0.01, xmaxLFC18, num=3)
# xticks2 = np.linspace(np.round(minSSIM,2), np.round(ssim_data_LFC18['Best_Model_SSIM'].max(),2), num=3)
ax2.set_xticks(xticks2)
add_axis_break(ax2, x_center=0.5,y_center=1, width=0.03, height=0.03,
                   line_width=1, color='black', zorder=1000)
add_axis_break(ax2, x_center=0.5,y_center=0, width=0.03, height=0.03,
                   line_width=1, color='black', zorder=1000)

# Create a monochrome legend for the hatching
monochrome_legend = [
    Patch(facecolor='black', edgecolor='black', label='Train'),
    Patch(facecolor='none', edgecolor='black', hatch='/////', label='Val')
]

# Add the monochrome legend
# ax.set_ylabel('MeC-Meso specimens')
fig.legend(handles=monochrome_legend, loc="lower center", ncol=2, bbox_to_anchor=(0.8, -0.1), title="Data")




ax0 = fig.add_subplot(1, 2, 1)
nSpecimens = [100, 200, 500, 1000]

lines = []
for idx, hist in enumerate(train_hists):
    if hist is not None and 'mean_squared_error' in hist and 'val_mean_squared_error' in hist:
        # All have batch size 32
        epochs = np.arange(1, len(hist['mean_squared_error']) + 1)
        train_mse = np.array(hist['mean_squared_error'])
        val_mse = np.array(hist['val_mean_squared_error'])
        
        # Convert epochs to steps based on specimens and batch size
        steps_per_epoch = nSpecimens[idx] // 32  # 32 is batch size
        steps = epochs * steps_per_epoch

        
        
        # epochs = np.arange(1, len(hist['mean_squared_error']) + 1)
        # train_mse = np.array(hist['mean_squared_error'])
        # val_mse = np.array(hist['val_mean_squared_error'])
        # color = model_colors[idx]
        t_color = train_colors[idx]
        v_color = val_colors[idx]
        # Plot train and val, but only keep one handle for legend
        # l_train, = ax0.plot(steps, train_mse, linestyle='-', color=color, linewidth=0.7, alpha=1)
        # ax0.plot(steps, val_mse, linestyle='--', color=color, linewidth=0.7, alpha=0.7)
        l_train, = ax0.plot(epochs, train_mse, linestyle='-', color=t_color, linewidth=0.7, alpha=0.8)
        ax0.plot(epochs, val_mse, linestyle='--', color=v_color, linewidth=0.5, alpha=0.9)

        lines.append(l_train)

ax0.set_yscale('log')  # Set y-axis to log scale
# ax0.set_ylim([min([train_mse.min(), val_mse.min()]), max([train_mse.max(), val_mse.max()])])
ax0.set_ylim([0.001, 0.10171668231487274])
# ax0.set_ylim([0.001, 0.01])
# ax0.set_ylim([0.001, 0.015])
# ax0.set_xlim([0, 1050])
ax0.set_xlabel('Epoch')
ax0.set_ylabel('MSE')
ax0.grid(True, axis='y', linestyle='--', alpha=0.75, which='major')
ax0.minorticks_on()
ax0.grid(True, axis='y', linestyle=':', alpha=0.5, which='minor')
# ax0.set_title('Training and Validation MSE Curves')
# Add custom legend for models (outside, right)


# Add a legend entry for line styles (train/val), also outside
style_lines = [
    plt.Line2D([0], [0], color='black', linestyle='-', label='Train', linewidth=0.7, alpha=0.8),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Val', linewidth=0.5, alpha=0.9)
]
# legend2 = plt.legend(handles=style_lines, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.12), title="Data")
legend2 = fig.legend(handles=style_lines, title="Data", loc="lower center",ncol=2, bbox_to_anchor=(0.29, -0.1))
# fig.legend(handles=monochrome_legend, loc="lower center", ncol=2, bbox_to_anchor=(0.8, -0.1), title="Data")
legend1 = ax0.legend(lines, model_labels[:len(lines)], title="MeC-Meso-M \nspecimens", loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
ax0.add_artist(legend2)  # Add the model legend back
ax0.set_title('')









modelName = 'MC24_1000_optimised'
# plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# output_filepath_pdf = os.path.join(r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures', f'RMSE_SSIM_datasetSize_{modelName}.pdf')
# output_filepath_pdf = os.path.join(r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures', f'RMSE_SSIM_datasetSize_{modelName}_v2.pdf')
output_filepath_pdf = os.path.join(r'C:\Users\kfh23\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Paper\Figures', f'RMSE_SSIM_datasetSize_{modelName}_v3.pdf')
plt.savefig(output_filepath_pdf, dpi=400, bbox_inches='tight', format='pdf')


# Add a legend for the second axis (ax2) for SSIM
# ssim_legend = [
#     plt.Line2D([0], [0], marker='s', color='black', label='Train', markersize=6, linestyle='None'),
#     plt.Line2D([0], [0], marker='s', color='black', label='Val', markersize=6, linestyle='None')
# ]
# # fig.legend(handles=monochrome_legend, loc="center", ncol=2, bbox_to_anchor=(0.7, 1), title="SSIM")
# fig.legend(handles=ssim_legend, loc="center", ncol=2, bbox_to_anchor=(0.78, 1.01), title="SSIM")
plt.show()
# %%




fig, ax1 = plt.subplots(figsize=(6, 4))

color_rmse = '#029E73'
color_ssim = '#0173B2'

# RMSE (left y-axis)
ax1.set_xlabel('Dataset Size')
ax1.set_ylabel('RMSE', color=color_rmse)
# l1 = ax1.plot(x_labels, final_train_rmse, marker='o', linestyle='-', color=color_rmse, label='Train RMSE')
# l2 = ax1.plot(x_labels, final_val_rmse, marker='s', linestyle='--', color=color_rmse, label='Val RMSE')
l1 = ax1.plot(x_labels, final_train_mae, marker='o', linestyle='-', color=color_rmse, label='Train MAE')
l2 = ax1.plot(x_labels, final_val_mae, marker='s', linestyle='--', color=color_rmse, label='Val MAE')
ax1.tick_params(axis='y', labelcolor=color_rmse)
# ax1.set_yscale('log')
ax1.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# # SSIM (right y-axis)
# ax2 = ax1.twinx()
# ax2.set_ylabel('SSIM', color=color_ssim)
# l3 = ax2.plot(x_labels, final_train_ssim, marker='o', linestyle='-', color=color_ssim, label='Train SSIM')
# l4 = ax2.plot(x_labels, final_val_ssim, marker='s', linestyle='--', color=color_ssim, label='Val SSIM')
# ax2.tick_params(axis='y', labelcolor=color_ssim)

ax1.tick_params(axis='x', colors='black')
ax1.tick_params(axis='y', colors='black')
# ax2.tick_params(axis='y', colors='black')
# Custom tick locations for log scale (e.g., 1.0, 0.8, 0.6, ..., 0.1)

# Define custom ticks (descending order for log scale)

# ax1.set_yticks(custom_ticks)
# ax1.get_yaxis().set_major_formatter(ScalarFormatter())
# ax1.set_ylim([custom_ticks[-1], custom_ticks[0]])
# Set colored axis labels with background boxes
# ax1.set_ylabel('RMSE', color='black', bbox=dict(facecolor="#029E73", edgecolor="#029E73", pad=0.2, alpha=0.5, boxstyle='Round'))
ax1.set_ylabel('MAE', color='black')
# ax2.set_ylabel('SSIM', color='black', bbox=dict(facecolor="#0173B2", edgecolor="#0173B2", pad=0.2, alpha=0.5, boxstyle='Round'))

# Legends
lines = l1 + l2 + l3 + l4
# labels = ['Train RMSE', 'Val RMSE', 'Train SSIM', 'Val SSIM']
labels = ['Training data', 'Validation data']
fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.12))

# plt.title('Final RMSE and SSIM vs Dataset Size')
plt.tight_layout()

# Save the figure as a PDF
output_filepath_pdf = os.path.join(r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures', f'MAE_Dataset_Size_{modelName}.pdf')
plt.savefig(output_filepath_pdf, dpi=400, bbox_inches='tight', format='pdf')

# output_filepath = os.path.join(r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures', f'MAE_Dataset_Size_{modelName}.png')
# plt.savefig(output_filepath, dpi=400, bbox_inches='tight')
plt.show()

# %%
# Plot RMSE vs SSIM as a function of epoch for train and val data
# Extract metrics from train_hist_json
epochs = np.arange(1, len(train_hist_json['mean_squared_error']) + 1)
train_mse = np.array(train_hist_json['mean_squared_error'])
val_mse = np.array(train_hist_json['val_mean_squared_error'])
train_ssim = np.array(train_hist_json['SSIM_metric'])
val_ssim = np.array(train_hist_json['val_SSIM_metric'])

# Compute RMSE
train_rmse = np.sqrt(train_mse)
val_rmse = np.sqrt(val_mse)

# Create alpha values that increase with epoch (start transparent, end solid)
n_epochs = len(epochs)
alphas = np.linspace(0.2, 1.0, n_epochs)  # Adjust start alpha as needed

plt.figure(figsize=(8, 6))

# Plot train points with varying alpha
for i in range(n_epochs):
    plt.scatter(train_ssim[i], train_rmse[i], color='blue', alpha=alphas[i], label='Train' if i == 0 else "")

# Plot val points with varying alpha
for i in range(n_epochs):
    plt.scatter(val_ssim[i], val_rmse[i], color='orange', alpha=alphas[i], label='Val' if i == 0 else "")

plt.grid()
plt.xlabel('SSIM')
plt.ylabel('RMSE')
plt.title('RMSE vs SSIM per Epoch')
plt.legend()
plt.tight_layout()
plt.show()


# %%  Calculate model outputs
if samplesPerFile > 1:
   samples = samples.take(1) # Only 1 specimen
inputSpecimen = samples.batch(1) # Only 1 specimen

# intermediate representations for all layers except the first layer.
layer_outputs = [layer.output for layer in model.layers]
# visual_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)
visual_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

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

# %% Make specific plots for poster

for layer_idx, feature_map in enumerate(feature_maps):
    # Calculate variance for each feature map in the layer
    variances = [np.var(feature_map[0, :, :, i]) for i in range(feature_map.shape[-1])]
    output_folder = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures\Poster research showcase'  # Replace with your desired folder path
    os.makedirs(output_folder, exist_ok=True)

    if feature_map.shape[-1] >= 3:
        # Find the indices of the 3 feature maps with the largest variances
        top_3_indices = np.argsort(variances)[-3:]
        # Extract the feature maps with the largest variances
        top_3_feature_maps = [feature_map[0, :, :, idx] for idx in top_3_indices]
        
        # Save the 3 feature maps side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(top_3_feature_maps[i], cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(output_folder, f'Layer_{layer_idx + 1}_Top3Features.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        # If fewer than 3 feature maps, just save the one with the largest variance
        top_index = np.argmax(variances)
        top_feature_map = feature_map[0, :, :, top_index]
        
        plt.figure(figsize=(5, 5))
        plt.imshow(top_feature_map, cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(output_folder, f'Layer_{layer_idx + 1}_Feature_{top_index}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()



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
# %% Visualize Ground Truth vs Prediction for a specific sample

# Choose the sample index (i)
i = 7  # Replace with the desired sample index

# Extract the specific sample from the dataset
selected_sample = samples.skip(i).take(1).batch(1)  # Skip to the i-th sample and take it

# Extract ground truth and predictions for the chosen sample
ground_truth = next(iter(selected_sample))[1].numpy()[0]  # Extract the ground truth for sample i
prediction = model.predict(selected_sample)[0]  # Get the model's prediction for the single sample

# Calculate the error (absolute difference)
error = np.abs(ground_truth - prediction)

# Assuming the ground truth, prediction, and error have multiple channels, visualize each channel separately
num_channels = ground_truth.shape[-1]
fig, axes = plt.subplots(3, num_channels, figsize=(15, 15))

for j in range(num_channels):
    # Determine the color scale limits based on the ground truth
    vmin, vmax = ground_truth[:, :, j].min(), ground_truth[:, :, j].max()

    # Plot ground truth
    ax_gt = axes[0, j] if num_channels > 1 else axes[0]
    im_gt = ax_gt.imshow(ground_truth[:, :, j], cmap='viridis', vmin=vmin, vmax=vmax)
    ax_gt.set_title(f'Ground Truth - Channel {j + 1}')
    ax_gt.axis('off')
    fig.colorbar(im_gt, ax=ax_gt, orientation='vertical', fraction=0.046, pad=0.04)

    # Plot prediction
    ax_pred = axes[1, j] if num_channels > 1 else axes[1]
    im_pred = ax_pred.imshow(prediction[:, :, j], cmap='viridis', vmin=vmin, vmax=vmax)
    ax_pred.set_title(f'Prediction - Channel {j + 1}')
    ax_pred.axis('off')
    fig.colorbar(im_pred, ax=ax_pred, orientation='vertical', fraction=0.046, pad=0.04)

    # Plot error
    ax_err = axes[2, j] if num_channels > 1 else axes[2]
    # im_err = ax_err.imshow(error[:, :, j], cmap='viridis', vmin=vmin, vmax=vmax)
    im_err = ax_err.imshow(error[:, :, j], cmap='viridis')
    ax_err.set_title(f'Error - Channel {j + 1}')
    ax_err.axis('off')
    fig.colorbar(im_err, ax=ax_err, orientation='vertical', fraction=0.046, pad=0.04)

plt.tight_layout()
output_path = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Figures\Poster research showcase\ExamplePrediction.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# %% Compare prediction distributions with ground truth distributions
import seaborn as sns
from matplotlib.ticker import FixedLocator, ScalarFormatter
import ast
from matplotlib.patches import Patch


# Extract all ground truths and predictions
all_ground_truths = []
all_predictions = []
nSpecimens = 100

for sample in samples.take(nSpecimens).batch(1):
    ground_truth = sample[1].numpy()  # Extract ground truth
    prediction = model.predict(sample[0])  # Get model prediction
    all_ground_truths.append(ground_truth.flatten())
    all_predictions.append(prediction.flatten())

# Convert lists to numpy arrays
all_ground_truths = np.concatenate(all_ground_truths)
all_predictions = np.concatenate(all_predictions)

# Plot distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(all_ground_truths, label='Ground Truth', color='#0072B2', fill=True, alpha=0.5)  # Blue from colorblind palette
sns.kdeplot(all_predictions, label='Predictions', color='#D55E00', fill=True, alpha=0.5)  # Orange from colorblind palette
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution of Predictions vs Ground Truth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
batch_sizes = [1, 8, 16, 32, 64]  # Define different batch sizes to test
for batch_size in batch_sizes:
    start_time = time.time()
    model.predict(samples.batch(batch_size))
    end_time = time.time()
    print(f"Batch size: {batch_size}, Time taken for prediction: {end_time - start_time:.3f} seconds")