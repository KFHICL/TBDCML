# HPC Optuna templates

Used by `copyOptunaSweepFolder.py` to provision one self-contained job
folder per Optuna study. You normally don't need to touch these files
directly -- edit the call to `create_optuna_study_folder(...)` at the
bottom of `copyOptunaSweepFolder.py` instead.

- `study_config_TEMPLATE.json` -- default search-space policy (which HPs to
  search, which to hold fixed). One real copy of this, named
  `study_config_<jobName>.json`, is placed in every job folder.
- `Submit_optuna.sh` -- PBS array job template for the search phase. Array
  size = number of parallel Optuna workers (not number of HP values, unlike
  the old sweep CSVs). Every worker shares one Optuna study via a
  `JournalStorage` file on the persistent filesystem.
- `Submit_confirm.sh` -- PBS array job template (k=10, one task per fold)
  for the final confirmation phase: k-fold cross-validation of one fixed,
  already-chosen configuration, using `CrossValidation.py --params-file`.

## Three-step workflow for one dataset

1. **Search**: `qsub <jobName>_Submit_optuna.sh`. Each worker pulls the
   next hyperparameter configuration to try from the shared study and
   trains it (with early-stopping/pruning). Re-submit the same script
   later to keep extending the same study -- it resumes automatically
   (`load_if_exists=True`).
2. **Export**: once satisfied,
   `python export_best_params.py --journal dataout/study_<jobName>.journal --study-name <jobName>`
   writes `best_params.json` plus the importance/interaction plots into
   `dataout/`.
3. **Confirm**: `qsub <jobName>_Submit_confirm.sh` runs the paper's
   k-fold/MCCV evaluation (`CrossValidation.py`) on exactly the winning
   configuration from step 2 -- this is the number you report, not any
   single trial's validation RMSE from the search phase.
