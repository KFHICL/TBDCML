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
