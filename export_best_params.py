#####################################################################
# Description
#####################################################################
'''
Close out an Optuna study (local sqlite or HPC journal file) and write out
its results in beginner-readable form: the winning hyperparameter dict
(ready to feed straight into CrossValidation.py for a final k-fold/MCCV
confirmation run) plus a CSV of every trial and the standard Optuna
importance/interaction plots.

This is the one place that looks across *every* trial in a study, which
matters for the HPC workflow: each BenchMarks_Optuna.py worker only ever
prints the best trial *it happened to see*, not the study-wide best, since
workers run in separate processes/nodes.

Usage
-----
Local study (written by optuna_local.py):
    python export_best_params.py --sqlite localResults/optuna/<STUDY_NAME>/study.db --study-name <STUDY_NAME>

HPC study (written by BenchMarks_Optuna.py, --shared-dir points at the
same directory that was passed to every worker):
    python export_best_params.py --journal /path/to/shared_dir/study_<jobname>.journal --study-name <jobname>

Both forms write best_params.json / trials_dataframe.csv / the *.html plots
next to whichever storage file you pointed at (override with --out-dir).
'''

import argparse
import json
import os

import optuna

argParser = argparse.ArgumentParser()
storage_group = argParser.add_mutually_exclusive_group(required=True)
storage_group.add_argument("--sqlite", help="Path to a study.db written by optuna_local.py")
storage_group.add_argument("--journal", help="Path to a study_<jobname>.journal written by BenchMarks_Optuna.py")
argParser.add_argument("--study-name", required=True, help="Name the study was created with (-j jobname on the HPC, or STUDY_NAME locally)")
argParser.add_argument("--out-dir", default=None, help="Where to write outputs (default: alongside the storage file)")
args = argParser.parse_args()

if args.sqlite:
    storage = f"sqlite:///{args.sqlite}"
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.sqlite))
else:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    storage = JournalStorage(JournalFileBackend(args.journal, lock_obj=JournalFileOpenLock(args.journal)))
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.journal))

os.makedirs(out_dir, exist_ok=True)

study = optuna.load_study(study_name=args.study_name, storage=storage)
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Study '{args.study_name}': {len(study.trials)} trials total, {len(completed)} completed.")

if not completed:
    raise SystemExit("No completed trials yet -- nothing to export.")

print(f"\nBest trial: #{study.best_trial.number}  value = {study.best_value:.5f}")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")

best_params_full = study.best_trial.user_attrs.get("params_full")
if best_params_full is None:
    print(
        "\nWarning: best trial has no 'params_full' user attribute (only the "
        "searched subset is known). Falling back to study.best_trial.params."
    )
    best_params_full = study.best_trial.params

with open(os.path.join(out_dir, "best_params.json"), "w") as f:
    json.dump(best_params_full, f, indent=2)

study.trials_dataframe().to_csv(os.path.join(out_dir, "trials_dataframe.csv"), index=False)

if len(completed) >= 2:
    from optuna.visualization import (
        plot_contour,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )

    plot_param_importances(study).write_html(os.path.join(out_dir, "param_importances.html"))
    plot_parallel_coordinate(study).write_html(os.path.join(out_dir, "parallel_coordinate.html"))
    plot_slice(study).write_html(os.path.join(out_dir, "slice.html"))
    plot_optimization_history(study).write_html(os.path.join(out_dir, "optimization_history.html"))
    searched_params = list(study.best_trial.params.keys())
    if len(searched_params) >= 2:
        plot_contour(study, params=searched_params[:2]).write_html(os.path.join(out_dir, "contour.html"))

print(f"\nWrote best_params.json, trials_dataframe.csv, and plots to:\n  {out_dir}")
