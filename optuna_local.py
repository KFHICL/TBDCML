# %%
#####################################################################
# Description
#####################################################################
'''
Local Optuna hyperparameter search, meant to be run cell-by-cell (# %%) in
VS Code exactly like test.py, but driving an Optuna study instead of
reading one row of a sweep_definition_*.csv at a time.

What this replaces
-------------------
Old workflow: hand-edit a sweep_definition_<name>.csv (one column per
hyperparameter, one row per run), run test.py / BenchMarks.py once per row.

New workflow: describe *ranges* for each hyperparameter once, in
tbdcml_workflow/search_space.py, and let Optuna decide which combination to
try next based on everything it has learned from previous trials (a
Tree-structured Parzen Estimator by default) -- optionally stopping a bad
trial early (pruning) instead of always training to completion.

Everything data/model related is unchanged: this script still calls
`tbdcml_workflow.build_model_from_params`, `resolve_dataset_spec`, etc. The
only thing that changes is how the `params` dict fed into those functions
is produced (Optuna's `trial.suggest_*` instead of a CSV row) and that the
raw dataset is loaded from disk ONCE up front and re-used by every trial
(see `tbdcml_workflow.pipeline` for why).

How to read the results
------------------------
This script writes, under `localResults/optuna/<STUDY_NAME>/`:
  study.db                 -- the Optuna study (sqlite). Re-running this
                               script with the same STUDY_NAME resumes it.
  best_params.json         -- the full, ready-to-train params dict for the
                               best trial found so far (feed this to
                               CrossValidation.py for a final MCCV/k-fold
                               confirmation run).
  trials_dataframe.csv     -- one row per trial: every hyperparameter value
                               tried plus the resulting RMSE. This is the
                               direct analogue of the old sweep CSV outputs.
  param_importances.html   -- which HPs mattered most (replaces the manual
                               "one-at-a-time" bar charts).
  parallel_coordinate.html -- how HP combinations relate to performance
                               (replaces the manual "combination sweep"
                               plots, e.g. Figure 14 in the paper).
  slice.html, contour.html -- per-HP and pairwise-HP views.
Open the .html files directly in a browser; no server needed.

Optuna dashboard (optional, very beginner friendly): once you have
optuna-dashboard installed (`pip install optuna-dashboard`), run
    optuna-dashboard sqlite:///localResults/optuna/<STUDY_NAME>/study.db
and open the printed local URL for the same plots interactively, live,
while the study is still running.
'''
#####################################################################
# Imports
#####################################################################
import os
import json
import math
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna_integration import TFKerasPruningCallback

from tbdcml_workflow import (
    load_shuffled_pool,
    resolve_dataset_spec,
    seed_everything,
    split_and_prepare,
    suggest_baseline_cnn_params,
    train_once,
)

# %%
#####################################################################
# Settings -- the only cell you should need to edit day to day
#####################################################################

DATASET = "LFC18"          # Smallest dataset -- good for fast local iteration.
MC24_FEATURES = "All"       # Ignored unless DATASET starts with "MC24".
SEED = 0
TESTING = 0                 # 1 = only load 10 sample files, for a fast dry run.

# Which hyperparameters to search. Set to None to search everything defined
# in tbdcml_workflow/search_space.py (a full joint study). Restrict this set
# to reproduce a narrower "combination sweep" of just a few HPs, e.g.:
#   ACTIVE_PARAMS = {"model_depth", "dropout", "data_augmentation"}
ACTIVE_PARAMS = None

# Hyperparameters that should never be searched -- always use these values.
# `Epochs`/`testSize` here match the paper's protocol (Section 3.2, 2.2).
FIXED_PARAMS = {
    "Epochs": 1000,
    "testSize": 0.0,
    "type": "default",  # "default"/"dense" = TBDCNet-style CNN, "UNet" = custom U-Net
}

N_TRIALS = 30                # Number of Optuna trials to run *in this call*.
N_REPEATS = 1                # Training repeats per trial (see note below).
PATIENCE = 60                # Early-stopping patience, matches the paper.

# --- On N_REPEATS and pruning ---------------------------------------------
# The paper's MCCV repeats an entire train/val split+train cycle several
# times to get a *reliable* performance estimate for one fixed HP config.
# Doing that inside every Optuna trial is the most faithful translation, but
# multiplies compute by N_REPEATS and disables pruning (a trial can't be
# "obviously bad" from one repeat if the next repeat might disagree).
#
# Recommended two-phase workflow instead:
#   1) SEARCH:  N_REPEATS = 1, pruning ON.  Cheaply explore the space to
#      find a promising region (this script).
#   2) CONFIRM: take study.best_trial's params (best_params.json) and run
#      them through CrossValidation.py, which already does proper k-fold /
#      MCCV -- exactly the paper's "TBDCNet evaluation" step (Figure 6).
# ----------------------------------------------------------------------------

STUDY_NAME = f"{DATASET}_optuna_{datetime.datetime.now().strftime('%Y%m%d')}"
RESULT_DIR = os.path.join(os.getcwd(), "localResults", "optuna", STUDY_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)
STUDY_DB_PATH = os.path.join(RESULT_DIR, "study.db")

seed_everything(SEED)

# %%
#####################################################################
# Load the raw dataset ONCE (expensive, disk-bound -- do not repeat per trial)
#####################################################################

localDatain_root = r'C:\Users\kfh23\OneDrive - Imperial College London\KFH23_GENERAL\PROJECTS\20241029_MSc_Paper\Data\datain'

dataset_spec = resolve_dataset_spec(DATASET, MC24_FEATURES)
train_dat_path = os.path.join(localDatain_root, dataset_spec.train_dat_name)

num_samples = len(os.listdir(train_dat_path)) * dataset_spec.samples_per_file
if TESTING:
    num_samples = min(num_samples, 10 * dataset_spec.samples_per_file)

pool = load_shuffled_pool(
    train_dat_path,
    num_samples,
    dataset_spec.x_names,
    ["FI"],
    tuple(dataset_spec.sample_shape),
    dataset_spec.samples_per_file,
    seed=SEED,
    testing=bool(TESTING),
)

print(f"Loaded {num_samples} specimens from {DATASET} once; reusing for every trial.")

# %%
#####################################################################
# Objective function -- one Optuna trial = one (or N_REPEATS) trained model(s)
#####################################################################


def objective(trial: optuna.Trial) -> float:
    params = suggest_baseline_cnn_params(
        trial,
        dataset=DATASET,
        mc24_features=MC24_FEATURES,
        active_params=ACTIVE_PARAMS,
        fixed=FIXED_PARAMS,
        # MeC-Macro / MeC-Meso-S-sized datasets: cap batch size (paper footnote, Table 2).
        batch_size_choices=(4, 8, 16, 32, 64),
        max_model_depth=4,
    )

    data = split_and_prepare(
        pool,
        num_samples,
        batch_size=params["batchSize"],
        val_fraction=params["valSize"],
        test_fraction=params["testSize"],
        augment=params["dsAugmentation"] == 1,
        seed=SEED,
        normalizer_length=40,
    )

    repeat_rmses, repeat_ssims = [], []
    for repeat in range(N_REPEATS):
        extra_callbacks = []
        if N_REPEATS == 1:
            # Pruning only makes sense when there is a single, directly
            # comparable run per trial -- see the note in the Settings cell.
            extra_callbacks.append(TFKerasPruningCallback(trial, "val_loss"))

        checkpoint_dir = os.path.join(
            RESULT_DIR, "checkpoints", f"trial_{trial.number:04d}_repeat_{repeat}"
        )
        result = train_once(
            params,
            data,
            seed=SEED + repeat,
            win_kernel=dataset_spec.win_kernel,
            checkpoint_dir=checkpoint_dir,
            loss_variant="local",
            patience=PATIENCE,
            extra_callbacks=extra_callbacks,
            verbose=0,
        )
        repeat_rmses.append(result["val_rmse"])
        repeat_ssims.append(result["val_ssim"])

    trial.set_user_attr("val_rmse_repeats", repeat_rmses)
    trial.set_user_attr("val_ssim_mean", float(np.mean(repeat_ssims)))
    trial.set_user_attr("params_full", params)  # full dict, incl. non-searched HPs

    return float(np.mean(repeat_rmses))


# %%
#####################################################################
# Run the study
#####################################################################

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=f"sqlite:///{STUDY_DB_PATH}",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    load_if_exists=True,  # re-running this script resumes the same study
)

study.optimize(objective, n_trials=N_TRIALS)

# %%
#####################################################################
# Save results (transparent, beginner-readable outputs)
#####################################################################

print(f"\nBest trial: #{study.best_trial.number}  val RMSE = {study.best_value:.5f}")
print("Best (searched) hyperparameters:")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

best_params_full = study.best_trial.user_attrs["params_full"]
with open(os.path.join(RESULT_DIR, "best_params.json"), "w") as f:
    json.dump(best_params_full, f, indent=2)

trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(RESULT_DIR, "trials_dataframe.csv"), index=False)

# %%
#####################################################################
# Visualisations -- the direct replacement for the old manual sweep plots
#####################################################################
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if len(completed_trials) >= 2:
    plot_param_importances(study).write_html(os.path.join(RESULT_DIR, "param_importances.html"))
    plot_parallel_coordinate(study).write_html(os.path.join(RESULT_DIR, "parallel_coordinate.html"))
    plot_slice(study).write_html(os.path.join(RESULT_DIR, "slice.html"))
    plot_optimization_history(study).write_html(os.path.join(RESULT_DIR, "optimization_history.html"))
    searched_params = list(study.best_trial.params.keys())
    if len(searched_params) >= 2:
        plot_contour(study, params=searched_params[:2]).write_html(
            os.path.join(RESULT_DIR, "contour.html")
        )
    print(f"\nSaved plots + best_params.json + trials_dataframe.csv to:\n  {RESULT_DIR}")
else:
    print("\nFewer than 2 completed trials -- skipping plots (increase N_TRIALS).")
