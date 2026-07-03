#####################################################################
# Description
#####################################################################
'''
HPC driver for an Optuna hyperparameter study. Replaces BenchMarks.py's
"one PBS array task = one row of sweep_definition_<jobname>.csv" model with
"one PBS array task = one worker pulling trials from a shared Optuna study".

How the parallelism works
--------------------------
Every worker (PBS array task) opens the *same* Optuna study, stored as a
JournalStorage file on the shared filesystem (NFS-safe: no database server
needed, unlike RDBStorage, and safer than SQLite over NFS). Optuna
coordinates which hyperparameters each worker tries next -- workers do not
need to know about each other beyond that shared file. This means:

  * The number of PBS array tasks (`#PBS -J 1-N`) is now simply "how many
    workers to run in parallel", not "how many hyperparameter values to
    test" -- you can use any N you have GPU/CPU budget for.
  * The old "3 repeats via Submit1.sh/Submit2.sh/Submit3.sh" pattern is
    gone. Repeats (if you want them) happen *inside* one trial's objective
    (see N_REPEATS below) instead of via separate submit scripts.

Inputs
------
-j / --jobname     : job name (folder name under CNNTraining/), no more
                      trailing "_1"/"_2"/"_3" repeat suffix needed.
-p / --parallel    : PBS_ARRAY_INDEX; used only to disambiguate this
                      worker's own output filenames/logs, never to select
                      hyperparameters (Optuna does that).
--n-trials         : stop this worker after completing this many trials.
--timeout-seconds  : (alternative to --n-trials, and the recommended choice
                      on the HPC) stop this worker after this many seconds,
                      so it always exits cleanly before PBS walltime kills
                      it. Give at least one of --n-trials/--timeout-seconds.
--shared-dir        : a persistent, NFS-backed directory every worker can
                      see (e.g. $HOME/IndividualProject/CNNTraining/$jobName
                      /dataout). This is NOT $TMPDIR: $TMPDIR is node-local
                      scratch, invisible to the *other* array tasks running
                      on other nodes at the same time, so the Optuna study
                      (and its cross-worker coordination) would silently
                      fall apart if the journal lived there. `datain` and
                      training checkpoints still live in the current working
                      directory ($TMPDIR) for fast local disk I/O -- only
                      the journal and the small per-trial result files use
                      --shared-dir.

study_config_<jobname>.json (placed next to this script, one per job
folder -- the direct analogue of the old sweep_definition_<jobname>.csv,
but describing *search ranges* instead of enumerated literal values):
{
  "dataset": "MC24x",
  "mc24_features": "All",
  "active_params": null,           // null = search everything
  "fixed_params": {"Epochs": 1000, "testSize": 0.0, "type": "default"},
  "batch_size_choices": [32, 64, 128, 256],
  "max_model_depth": 4,
  "n_repeats": 1,
  "save_models": false             // keep false during the search phase;
}                                   // see note near SAVE_MODELS below.

Outputs (all written to --shared-dir)
----------------------------------------------------------------
study_<jobname>.journal      : the shared Optuna study (do not delete
                                between array tasks -- every worker appends
                                to it).
trainHist_<jn>_t<trial>.json : training history for that trial's repeats
parameters_<jn>_t<trial>.json: full hyperparameter dict for that trial
results_<jn>_t<trial>.json   : RMSE/SSIM for that trial
model_<jn>_t<trial>.keras    : only written if "save_models": true
'''

#####################################################################
# Imports
#####################################################################
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from optuna_integration import TFKerasPruningCallback

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from tbdcml_workflow import (
    load_shuffled_pool,
    resolve_dataset_spec,
    seed_everything,
    split_and_prepare,
    suggest_baseline_cnn_params,
    train_once,
)

#####################################################################
# Settings
#####################################################################

yNames = ['FI']
normalizerLength = 20
seed = 0
seed_everything(seed)

#####################################################################
# Args + study config
#####################################################################

argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--parallel", required=True, help="PBS array index; used only for this worker's own output naming")
argParser.add_argument("-j", "--jobname", required=True, help="Job name")
argParser.add_argument("--n-trials", type=int, default=None, help="Stop this worker after N completed trials")
argParser.add_argument("--timeout-seconds", type=int, default=None, help="Stop this worker after this many seconds (recommended on HPC)")
argParser.add_argument("--shared-dir", required=True, help=(
    "Persistent, NFS-backed directory (NOT $TMPDIR) that every worker/array "
    "task can see. The Optuna study journal and every trial's result JSON "
    "are written here -- this is what makes the parallel workers coordinate "
    "with each other at all. `datain` and training checkpoints stay in the "
    "current working directory ($TMPDIR) for fast local I/O."
))
args = argParser.parse_args()

if args.n_trials is None and args.timeout_seconds is None:
    raise SystemExit("Provide --n-trials and/or --timeout-seconds so this worker knows when to stop.")

os.makedirs(args.shared_dir, exist_ok=True)

with open(f"study_config_{args.jobname}.json") as f:
    study_config = json.load(f)

DATASET = study_config["dataset"]
MC24_FEATURES = study_config.get("mc24_features", "All")
ACTIVE_PARAMS = study_config.get("active_params")  # null -> None -> search everything
ACTIVE_PARAMS = set(ACTIVE_PARAMS) if ACTIVE_PARAMS else None
FIXED_PARAMS = study_config.get("fixed_params", {})
BATCH_SIZE_CHOICES = tuple(study_config.get("batch_size_choices", (32, 64, 128, 256)))
MAX_MODEL_DEPTH = study_config.get("max_model_depth", 4)
N_REPEATS = study_config.get("n_repeats", 1)
SAVE_MODELS = study_config.get("save_models", False)
# Saving every trial's full .keras model quickly exhausts HPC storage quotas
# across a search of dozens/hundreds of trials. Leave this false during the
# search phase; re-train (and save) only the winning configuration via
# CrossValidation.py once the study has converged.

#####################################################################
# Dataset: load once per worker process, re-used by every trial it runs
#####################################################################

dataset_spec = resolve_dataset_spec(DATASET, MC24_FEATURES)
trainDat_path = os.path.join('datain', dataset_spec.train_dat_name)
numSamples = len(os.listdir(trainDat_path)) * dataset_spec.samples_per_file

pool = load_shuffled_pool(
    trainDat_path,
    numSamples,
    dataset_spec.x_names,
    yNames,
    tuple(dataset_spec.sample_shape),
    dataset_spec.samples_per_file,
    seed=seed,
)

#####################################################################
# Shared study: JournalStorage on the persistent (non-$TMPDIR) filesystem
#####################################################################

journal_path = os.path.join(args.shared_dir, f"study_{args.jobname}.journal")
storage = JournalStorage(JournalFileBackend(journal_path, lock_obj=JournalFileOpenLock(journal_path)))

study = optuna.create_study(
    study_name=args.jobname,
    storage=storage,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    load_if_exists=True,
)

#####################################################################
# Objective
#####################################################################


def objective(trial: optuna.Trial) -> float:
    params = suggest_baseline_cnn_params(
        trial,
        dataset=DATASET,
        mc24_features=MC24_FEATURES,
        active_params=ACTIVE_PARAMS,
        fixed=FIXED_PARAMS,
        batch_size_choices=BATCH_SIZE_CHOICES,
        max_model_depth=MAX_MODEL_DEPTH,
    )

    data = split_and_prepare(
        pool,
        numSamples,
        batch_size=params["batchSize"],
        val_fraction=params["valSize"],
        test_fraction=params["testSize"],
        augment=params["dsAugmentation"] == 1,
        seed=seed,
        normalizer_length=normalizerLength,
    )

    repeat_rmses, repeat_ssims, histories = [], [], []
    for repeat in range(N_REPEATS):
        extra_callbacks = []
        if N_REPEATS == 1:
            extra_callbacks.append(TFKerasPruningCallback(trial, "val_loss"))

        checkpoint_dir = os.path.join(
            f"training_checkpoints_{args.jobname}", f"trial_{trial.number:05d}_repeat_{repeat}"
        )
        result = train_once(
            params,
            data,
            seed=seed + repeat,
            win_kernel=dataset_spec.win_kernel,
            checkpoint_dir=checkpoint_dir,
            loss_variant="hpc",
            patience=60,
            extra_callbacks=extra_callbacks,
            verbose=2,
            return_model=SAVE_MODELS and repeat == 0,
        )
        repeat_rmses.append(result["val_rmse"])
        repeat_ssims.append(result["val_ssim"])
        histories.append(result["history"])

        if SAVE_MODELS and repeat == 0:
            result["model"].save(os.path.join(args.shared_dir, f"model_{args.jobname}_t{trial.number:05d}.keras"))

    trial.set_user_attr("val_rmse_repeats", repeat_rmses)
    trial.set_user_attr("val_ssim_repeats", repeat_ssims)
    trial.set_user_attr("params_full", params)
    trial.set_user_attr("worker", args.parallel)

    # Lightweight per-trial artefacts (always written; cheap compared to a
    # full .keras model). Naming by `trial.number` is safe across workers
    # because Optuna guarantees trial numbers are unique within a study.
    with open(os.path.join(args.shared_dir, f"trainHist_{args.jobname}_t{trial.number:05d}.json"), "w") as f:
        json.dump(histories, f, indent=2)
    with open(os.path.join(args.shared_dir, f"parameters_{args.jobname}_t{trial.number:05d}.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(args.shared_dir, f"results_{args.jobname}_t{trial.number:05d}.json"), "w") as f:
        json.dump({"val_rmse_repeats": repeat_rmses, "val_ssim_repeats": repeat_ssims}, f, indent=2)

    return float(np.mean(repeat_rmses))


#####################################################################
# Run this worker's share of the study
#####################################################################

study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_seconds)

print(f"Worker {args.parallel} finished. Study now has {len(study.trials)} total trials.")
if study.best_trial is not None:
    print(f"Best so far: trial #{study.best_trial.number}, val RMSE = {study.best_value:.5f}")
