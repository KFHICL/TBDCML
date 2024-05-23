#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# sweepIdx = int('1')

# # Sweep parameters
# # Batch size
# # Train/val ratio

# # sweepPath = os.path.join('IndividualProject','CNNTraining','sweep_definition.csv')
# sweepPath = 'sweep_definition.csv'
# sweep_params = pd.read_csv(sweepPath)

# sweep_params = sweep_params.set_index("Index")

# # sweep_params.head()
# params = sweep_params.loc[sweepIdx]

# print(params['convActivation']=='tanh')
#%%
sweepIdxPath = r'C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Code\TBDCML_Clone\TBDCML\compareIndex_fineTune.csv'

sweepIdx = pd.read_csv(sweepIdxPath)
for i in sweepIdx.index:
    plotParams = sweepIdx.apply(lambda row: row[row == 1].index.tolist(), axis=1)[i]
    sweepnums = np.fromstring(sweepIdx['sweepIdx'][i],dtype=int, sep=',')
    sweepnums = np.hstack((baselineIdx,sweepnums))


    sweepPlot(sweep=sweepnums, paramVariables = plotParams, figname = sweepIdx['sweepName'][i])
# %%
x1 = np.linspace(-2.5,2.5)
ytanh = tf.keras.activations.tanh(x1).numpy()
yrelu = tf.keras.activations.relu(x1).numpy()
ysoftplus = tf.keras.activations.softplus(x1).numpy()
yelu = tf.keras.activations.elu(x1).numpy()
ylrelu = tf.keras.activations.leaky_relu(x1).numpy()
ygelu = tf.keras.activations.gelu(x1).numpy()
ysilu = tf.keras.activations.silu(x1).numpy()

plt.style.use("seaborn-v0_8-colorblind")
plt.figure(figsize=(5,3), layout = "constrained", dpi = 300)
plt.subplot(1,2,1)
plt.plot(x1, ytanh)
plt.plot(x1, yrelu)
plt.plot(x1, ysoftplus)
plt.plot(x1, ygelu)
plt.legend(['Tanh', 'Relu', 'SoftPlus','Gelu'])
plt.grid()
plt.xlabel('x')
plt.ylabel('Activation function (x)')
plt.subplot(1,2,2)
plt.plot(x1, yelu)
plt.plot(x1, ylrelu)

plt.plot(x1, ysilu)
plt.legend(['Elu', 'Lrelu',  'Silu'])
plt.grid()
plt.xlabel('x')
plt.ylabel('Activation function (x)')
# plt.legend(['Tanh', 'Relu', 'SoftPlus', 'Elu', 'Lrelu', 'Gelu', 'Silu'])

plt.show()



# %%
numSamples = 100
kFold = 10
k = 10
count = np.linspace(1,100,100)
trainValRatio = (k-1)/k # Training and validation data split ratio
train_length = round(numSamples * trainValRatio)
X = count  # Input data is always stiffness components
yFI = np.linspace(1001,1100,100) # Labels for failure index

# Create tensor datasets
# ds_e22 = tf.data.Dataset.from_tensor_slices((X, ye22))
# ds_FI = tf.data.Dataset.from_tensor_slices((X, yFI))
ds_FI = tf.data.Dataset.from_tensor_slices((X,yFI))
# for x in ds_FI:
#     print(x)
# Split into training and validation datasets using k-fold
valIdx = int((kFold-1)*(numSamples/k)) # The k fold variable is 1-indexed, valIdx marks the start of the validation set in this fold

# val_ds_e22 = ds_e22.skip(valIdx) # Skip until the point where the validation set starts
# val_ds_e22 = val_ds_e22.take(int(numSamples/k)) # Take the validation set
val_ds_FI = ds_FI.skip(valIdx) # Skip until the point where the validation set starts

val_ds_FI = val_ds_FI.take(int(numSamples/k)) # Take the validation set
for x in val_ds_FI:
    print(x)
# train_ds_e22_first = ds_e22.take(valIdx) # Take samples up to the start of the validation set
# train_ds_e22_second = ds_e22.skip(int(valIdx+(numSamples/k))) # Take samples after the end of the validation set
# train_ds_e22 = train_ds_e22_first.concatenate(train_ds_e22_second) # Concatenate the two parts of the training dataset

train_ds_FI_first = ds_FI.take(valIdx) # Take samples up to the start of the validation set
train_ds_FI_second = ds_FI.skip(int(valIdx+(numSamples/k))) # Take samples after the end of the validation set
train_ds_FI = train_ds_FI_first.concatenate(train_ds_FI_second) # Concatenate the two parts of the training dataset
for x in train_ds_FI:
    print(x)
# %%
