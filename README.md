# TBDCML
#### MSc research project concerning machine learning applications to TBDC materials
These scripts are used to train and evaluate simple encoder-decoder CNNs. 
A manual hyperparameter optimisation can be used to highlight and investigate
the impact of modelling choices and data characteristics.

### Scripts

1. MultiDimTBDCNet HPC tuning.py: The main model defnition and training script
to be used in hyperparameter sweeps on the HPC. The script is launched on the HPC and
reads the training data and sweep definitions placed using the folder structure defined
below. After training of models, the results will be available in the hyperparameter
sweep folder that the user created.

2. HPC submission job script: shell file (or other tex-file readable by the HPC PBSPro
workload manager) which submits the hyperparameter optimisation sweep of a single
repetition. This file contains the paths to input data and the scripts needed. If multiple
repetitions are to be run, you need to copy this file and rename/alter it to reflect this
such that you have 1 job submission for each repetition. See below for an example of
this script.

3. CrossValidation.py: Used to define and train model folds using the k-fold crossvalidation
routine. By default, k=10. The script reads the sweep definition csv file in
the same way as the MultiDimTBDCNet HPC tuning script, and is also written to run
on the HPC.

4. Comparemodels.py: Used to create visualisation figures for model hyperparameter
sweeps. Early in the script the path to the local hyperparameter sweep results folder
can be defined. The script is called with by passing the jobname as the argument ”-j”
and the number of repetitions done on the sweep as the argument ”-rp”, such that a
summary of the sweep with jobname ”Epsilon2908 ” containing 3 repetitions is called
using py CompareModels.py -j Epsilon2908 -rp 3. The comparemodels script requires
a file named ”compareIndex JOBNAME.csv to be placed within the results folder of the
first repetition.

5. SummariseModel.py: Script to summarise a single model from a hyperparameter
sweep. Early in the script the path to the local hyperparameter sweep results folder
can be defined. The script is called with by passing the jobname as the argument ”-j”,
the repetition you would like to summarise (1-indexed) as the argument ”-rp”, and the index of the model as defined by the sweep definition (1-indexed) as the argument ”-i”.
An example call could be py SummariseModel.py -j CrossValidation2808 -rp 1 -i 1.

6. test.py: Used to test model training locally. Is effectively the same code as the MultiDimTBDCNet
HPC tuning script, but runs locally using a sweep definition csv file
placed in the same folder as the script.

7. FullyConnectedNN.py: Script written to locally train and evaluate a fully/densely
connected NN model (FFNN in the thesis) using a sweep definition placed in the same
folder as the script.

8. TransferLearning.py: Script written to locally train and evaluate transfer learning
models using a sweep definition placed in the same folder as the script. This is very
preliminary work.

9. figures.py: Allows creation of various figures used in this thesis.

10. HPSweep Figures.py: Allows creation of various figures related to the hyperparameter
sweeps used in this thesis.

11. GenerateResultSummary.py: Allows extraction of RMSEs from all model sweeps in
the results folder and compilation of these in a single spreadsheet.


### Folders
TEMPLATES_AND_EXAMPLES contains examples of HPC submission folders and the format of results folders



