#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -J 1-10

# Final confirmation run: k-fold cross-validation (k=10, one PBS array task
# per fold -- see CrossValidation.py) of ONE fixed hyperparameter
# configuration, namely the winner of the Optuna search
# (Submit_optuna.sh + BenchMarks_Optuna.py). This is the paper's "TBDCNet
# evaluation" step (Figure 6) -- unlike the search phase, this is meant to
# be run once, on the config you have already decided on.

jobName=TEMPLATE
curName=$jobName_confirm

mkdir -p $HOME/IndividualProject/CNNTraining/$jobName/dataout

module add tools/prod
eval "$(~/miniforge3/bin/conda shell.bash hook)"
source activate TBDCNet_GPU

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

#PBS -N $curName

# best_params.json comes from `python export_best_params.py ...` after the
# Optuna search study has run for a while -- see hpc_templates/README.md.
cp $HOME/IndividualProject/CNNTraining/$jobName/dataout/best_params.json $TMPDIR
cp -r $HOME/IndividualProject/CNNTraining/datain $TMPDIR
cp -r $HOME/IndividualProject/CNNTraining/dataout $TMPDIR
cd $TMPDIR

python3 $HOME/IndividualProject/CNNTraining/$jobName/CrossValidation.py \
  -p $PBS_ARRAY_INDEX -j ${jobName}_1 \
  --params-file best_params.json

cp -r $TMPDIR/dataout/. $HOME/IndividualProject/CNNTraining/$jobName/dataout/
