#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -J 1-TEMPLATE_NUM_WORKERS

jobName=TEMPLATE
curName=$jobName

# Persistent, NFS-backed location every array task/worker can see. This is
# where the shared Optuna study (study_$jobName.journal) and every trial's
# result files live -- do NOT point this at $TMPDIR, which is node-local
# scratch and invisible to the other array tasks running at the same time.
sharedDir=$HOME/IndividualProject/CNNTraining/$jobName/dataout
mkdir -p $sharedDir

module add tools/prod
eval "$(~/miniforge3/bin/conda shell.bash hook)"
source activate TBDCNet_GPU

## Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

#PBS -N $curName

# Only datain + this worker's own study config need to be staged into
# $TMPDIR for fast local disk I/O; results go straight to $sharedDir.
cp $HOME/IndividualProject/CNNTraining/$jobName/study_config_$jobName.json $TMPDIR
cp -r $HOME/IndividualProject/CNNTraining/datain $TMPDIR
cd $TMPDIR

# The --timeout-seconds value below should be a little less than the
# walltime declared above (in seconds) so this worker always exits cleanly,
# instead of being killed mid-trial by the scheduler.
python3 $HOME/IndividualProject/CNNTraining/$jobName/BenchMarks_Optuna.py \
  -p $PBS_ARRAY_INDEX -j $jobName \
  --timeout-seconds TEMPLATE_TIMEOUT_SECONDS \
  --shared-dir $sharedDir

# No copy-back step needed: BenchMarks_Optuna.py already wrote every result
# directly to $sharedDir as it went, so nothing is lost even if this task
# is killed by the scheduler before reaching this line.
