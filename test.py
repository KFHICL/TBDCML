#%%
import pandas as pd
sweepIdx = int('1')

# Sweep parameters
# Batch size
# Train/val ratio

# sweepPath = os.path.join('IndividualProject','CNNTraining','sweep_definition.csv')
sweepPath = 'sweep_definition.csv'
sweep_params = pd.read_csv(sweepPath)

sweep_params = sweep_params.set_index("Index")

# sweep_params.head()
params = sweep_params.loc[sweepIdx]

print(params['convActivation']=='tanh')

