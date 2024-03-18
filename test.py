#%%
import pandas as pd
import numpy as np
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
