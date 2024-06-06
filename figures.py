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

r1 = pd.Series([0.13932267,	0.1312578,	0.13298206], name='Baseline')  # Baseline model
r2 = pd.Series([0.12348962,	0.1269816,	0.12398586], name='Smaller batch size') # Batch size of 4
r3 = pd.Series([0.12679584,	0.13527173,	0.12746832], name='Larger convolution window') # Kernel of size 4
r4 = pd.Series([0.10533698,	0.11521004,	0.11750075], name='Longer training') # 4-layer model, longer training with early stopping
r5 = pd.Series([0.112298615,0.115193091,0.112744972], name='Silu activation function') # Silu activation
r6 = pd.Series([0.09849966,	0.10574671,	0.10506879], name = 'Final model')

# Final model with 10-fold cross validation:
r7 = pd.Series([0.09849966,	0.08307547,	0.09041375,	0.09272594,	0.10196505,	0.10486203,	0.10931589,	0.10525837,	0.11376371,	0.10530864,	0.10574671,	0.10420603,	0.08745328,	0.10823236,	0.10726699,	0.10785414,	0.10911972,	0.1154059,	0.10777717,	0.11049049,	0.10506879,	0.11161773,	0.09534075,	0.10674551,	0.09966194,	0.10608425,	0.10241564,	0.11438299,	0.10887758,	0.09320892], name='Final Model Crossvalidation')




# RMSEPoints = pd.concat([r1,r2,r3,r4,r5,r6,r7], axis=1)
RMSEPoints = pd.concat([r1,r2,r3,r4,r5,r6], axis=1)
# print(RMSEPoints)
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig = plt.figure(figsize=(1200*px, 800*px), layout="constrained")
# plt.style.use("seaborn-v0_8-colorblind") # For consitency use this colour scheme and viridis
axis = plt.subplot(1,1,1)
sns.boxplot(ax=axis, data=RMSEPoints,palette = sns.color_palette('colorblind', 1))
highlightBox = axis.patches[0]
highlightBox.set_facecolor('#DE8F05')

highlightBox = axis.patches[-1]
highlightBox.set_facecolor('#029E73')
# highlightBox.set_edgecolor('black')
# highlightBox.set_linewidth(3)

plt.grid()
plt.ylabel('Root mean squared error')
plt.title('Performance of selected models during hyperparameter optimisation')

plt.show()
# sns.pointplot(data=RMSEPoints, x="name", y="body_mass_g")



# %%
