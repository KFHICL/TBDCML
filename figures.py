#####################################################################
# Description
#####################################################################
# This script is used to generate some plots for use in reports and 
# thesis.




#####################################################################
# Imports
#####################################################################
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d.proj3d import proj_transform
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# from matplotlib.patches import FancyArrowPatch


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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

r1 = pd.Series([0.13932267,	0.1312578,	0.13298206], name='Baseline')  # Baseline model
r2 = pd.Series([0.12348962,	0.1269816,	0.12398586], name='Smaller batch size') # Batch size of 4
r3 = pd.Series([0.12679584,	0.13527173,	0.12746832], name='Larger convolution window') # Kernel of size 4
r4 = pd.Series([0.10533698,	0.11521004,	0.11750075], name='Longer training') # 4-layer model, longer training with early stopping
r5 = pd.Series([0.112298615,0.115193091,0.112744972], name='Silu activation function') # Silu activation
r6 = pd.Series([0.09844742,	0.10513075,	0.11276722], name = 'Final model')

# Final model with 10-fold cross validation:
r7 = pd.Series([0.10891952,	0.10808814,	0.10763301,	0.09643763,	0.09844742,	0.11448937,	0.10834616,	0.10191214,	0.11064257,	0.10917973,	0.10927559,	0.10266547,	0.11111066,	0.10206658,	0.10513075,	0.09653068,	0.10023911,	0.0969057,	0.11271869,	0.11323597,	0.11356502,	0.09728512,	0.09374001,	0.10241187,	0.11276722,	0.10735505,	0.11108778,	0.09843335,	0.11136408,	0.09723821], name='Final Model Crossvalidation')




RMSEPoints = pd.concat([r1,r2,r3,r4,r5,r6,r7], axis=1)
# RMSEPoints = pd.concat([r1,r2,r3,r4,r5,r6], axis=1)
# print(RMSEPoints)
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
# plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis
axis = plt.subplot(1,1,1)
sns.boxplot(ax=axis, data=RMSEPoints,palette = sns.color_palette('colorblind', 1))
highlightBox = axis.patches[0]
highlightBox.set_facecolor('#DE8F05')

highlightBox = axis.patches[-2]
highlightBox.set_facecolor('#CC78BC')
highlightBox = axis.patches[-1]
highlightBox.set_facecolor('#029E73')
# highlightBox.set_edgecolor('black')
# highlightBox.set_linewidth(3)

plt.grid()
plt.ylabel('Root mean squared error')
plt.title('Performance of selected models during hyperparameter optimisation')

plt.show()
# sns.pointplot(data=RMSEPoints, x="name", y="body_mass_g")



# %% Plot of longitudinal stiffness and FI of specimen from new matlab model
Qxx = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\Qxx.csv",header=None)
FI = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\FI.csv",header=None)
gridX = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridx.csv",header=None)
gridY = pd.read_csv(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridy.csv",header=None)


# Qxx = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\Qxx.csv", delimiter=',')
# FI = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\FI.csv", delimiter=',')
# gridX = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridx.csv", delimiter=',')
# gridY = np.genfromtxt(r"C:\Users\kaspe\OneDrive\UNIVERSITY\YEAR 4\Individual Project\Data\MiscData_ForPlots\Matlab_Specimen_PreliminarySeminar\gridy.csv", delimiter=',')





# print(FI)
# FI = np.pad(FI, pad_width = ((0, 1),(0,1)), mode='constant', constant_values = 1)
# Qxx = np.pad(Qxx, pad_width = ((0, 1),(0,1)), mode='constant', constant_values = 1)
# print(FI)
FI = np.pad(FI, pad_width = ((0, 1),(0,1)), mode='edge')
Qxx = np.pad(Qxx, pad_width = ((0, 1),(0,1)), mode='edge')


px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(300*px, 300*px), layout="constrained")
plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis

ax = plt.subplot(1,2,1) # E_xx
CS = ax.contourf(gridX,gridY,Qxx/1000)
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Stiffness [GPa]')
plt.title('E_xx')
ax.set_xticks([])
ax.set_yticks([])


ax = plt.subplot(1,2,2) # E_xx
CS2 = ax.contourf(gridX,gridY,FI)
cbar = fig.colorbar(CS2)
cbar.ax.set_ylabel('Failure Index')
plt.title('Failure Index')
ax.set_xticks([])
ax.set_yticks([])

plt.show()


# %% Activation functions
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



# %% Loss functions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

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

ax = plt.subplot(2,2,1)
CS = plt.contourf(Y,P, MAE_Mesh,levels = 100)
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Absolute Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')

ax = plt.subplot(2,2,2)
CS2 = plt.contourf(Y,P, MSE_Mesh,levels = 100)
cbar = fig.colorbar(CS2)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Squared Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')

ax = plt.subplot(2,2,3)
CS3 = plt.contourf(Y,P, MSLE_Mesh,levels = 100)
cbar = fig.colorbar(CS3)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Squared Logarithmic Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')

ax = plt.subplot(2,2,4)
CS4 = plt.contourf(Y,P, Custom_Mesh,levels = 100)
cbar = fig.colorbar(CS4)
cbar.ax.set_ylabel('Loss')
plt.title('Custom loss function')
plt.xlabel('True value')
plt.ylabel('Predicted value')

# Also do 3D plots:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS1 = ax.plot_surface(Y, P, MAE_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS1)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Absolute Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS2 = ax.plot_surface(Y, P, MSE_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS2)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Squared Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS3 = ax.plot_surface(Y, P, MSLE_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS3)
cbar.ax.set_ylabel('Loss')
plt.title('Mean Squared Logarithmic Error Loss')
plt.xlabel('True value')
plt.ylabel('Predicted value')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
CS4 = ax.plot_surface(Y, P, Custom_Mesh, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
cbar = fig.colorbar(CS4)
cbar.ax.set_ylabel('Loss')
plt.title('Custom loss function')
plt.xlabel('True value')
plt.ylabel('Predicted value')

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

