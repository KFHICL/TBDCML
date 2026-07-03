# %%
'''
Provision one HPC job folder per Optuna study, replacing copySweepFolder.py.

What's different from copySweepFolder.py
------------------------------------------
The old script multiplied one CSV row into N rows (one literal value per
row) and copied 3 submit scripts (one per repeat) for every single
hyperparameter you wanted to sweep -- e.g. 18 job folders for 18
one-at-a-time HP sweeps (see the `jobNames`/`jobParameters` lists in
copySweepFolder.py).

Now, a single Optuna study can search *all* (or any subset of) the
hyperparameters at once, so you typically only need ONE job folder per
dataset (or per deliberately-narrow "combination sweep", see
`active_params` in tbdcml_workflow/search_space.py) instead of one per HP.
Each job folder gets:
  * BenchMarks_Optuna.py, CrossValidation.py, export_best_params.py, and a
    copy of the tbdcml_workflow/ package (self-contained, like the old
    per-job BenchMarks.py/CrossValidation.py copies).
  * study_config_<jobName>.json  (search-space policy; see hpc_templates/study_config_TEMPLATE.json)
  * Submit_optuna.sh             (PBS array = number of parallel workers)
  * Submit_confirm.sh            (k-fold confirmation of the eventual winner)

Before running this for the first time on a new dataset, deploy your conda
environment with `pip install optuna optuna-integration` -- this repo's
`tbdcml_workflow` package now depends on both (see requirements note in
README).
'''
# %%
from __future__ import annotations

import os
import shutil
import json
from datetime import datetime

def _read_lines(path):
    with open(path, "r") as f:
        return f.readlines()


def _write_lines(path, lines):
    with open(path, "w", newline="\n") as f:
        f.writelines(lines)


def _substitute_submit_script(lines, *, job_name, num_workers, timeout_seconds, use_gpu, use_extra_memory):
    out = []
    for line in lines:
        if "jobName=TEMPLATE" in line:
            line = line.replace("jobName=TEMPLATE", f"jobName={job_name}")
        if "#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1" in line and not use_gpu:
            mem = "128gb" if use_extra_memory else "64gb"
            line = line.replace("#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1", f"#PBS -l select=1:ncpus=4:mem={mem}")
        if "#PBS -J 1-TEMPLATE_NUM_WORKERS" in line:
            line = line.replace("TEMPLATE_NUM_WORKERS", str(num_workers))
        if "TEMPLATE_TIMEOUT_SECONDS" in line:
            line = line.replace("TEMPLATE_TIMEOUT_SECONDS", str(timeout_seconds))
        out.append(line)
    return out


def create_optuna_study_folder(
    *,
    template_dir: str,
    source_code_dir: str,
    dest_dir: str,
    job_name: str,
    dataset: str,
    mc24_features: str = "All",
    active_params=None,
    fixed_params=None,
    batch_size_choices=(32, 64, 128, 256),
    max_model_depth: int = 4,
    n_repeats: int = 1,
    save_models: bool = False,
    num_workers: int = 8,
    walltime_hours: int = 24,
    confirm_walltime_hours: int | None = None,
    use_gpu: bool = False,
    use_extra_memory: bool = True,
):
    """Create one self-contained HPC job folder for one Optuna study.

    `template_dir` is `hpc_templates/` (Submit_optuna.sh, Submit_confirm.sh,
    study_config_TEMPLATE.json). `source_code_dir` is this repo's root
    (BenchMarks_Optuna.py, CrossValidation.py, export_best_params.py,
    tbdcml_workflow/).
    """
    dest = os.path.join(dest_dir, job_name)
    if os.path.exists(dest):
        raise FileExistsError(f"{dest} already exists -- pick a different job_name or remove it first.")
    os.makedirs(dest)

    # --- Python code: self-contained copy, same spirit as the old script ---
    for fname in ("BenchMarks_Optuna.py", "CrossValidation.py", "export_best_params.py"):
        shutil.copy(os.path.join(source_code_dir, fname), dest)
    shutil.copytree(os.path.join(source_code_dir, "tbdcml_workflow"), os.path.join(dest, "tbdcml_workflow"))

    # --- Study config: the search-space policy for this job ---
    study_config = {
        "dataset": dataset,
        "mc24_features": mc24_features,
        "active_params": sorted(active_params) if active_params else None,
        "fixed_params": fixed_params or {"Epochs": 1000, "testSize": 0.0, "type": "default"},
        "batch_size_choices": list(batch_size_choices),
        "max_model_depth": max_model_depth,
        "n_repeats": n_repeats,
        "save_models": save_models,
    }
    with open(os.path.join(dest, f"study_config_{job_name}.json"), "w") as f:
        json.dump(study_config, f, indent=2)

    # --- Submit scripts ---
    def _set_walltime(lines, hours):
        return [
            line.replace("#PBS -l walltime=24:00:00", f"#PBS -l walltime={hours:02d}:00:00")
            for line in lines
        ]

    timeout_seconds = int(walltime_hours * 3600 * 0.95)  # leave headroom to exit cleanly
    optuna_lines = _substitute_submit_script(
        _read_lines(os.path.join(template_dir, "Submit_optuna.sh")),
        job_name=job_name, num_workers=num_workers, timeout_seconds=timeout_seconds,
        use_gpu=use_gpu, use_extra_memory=use_extra_memory,
    )
    optuna_lines = _set_walltime(optuna_lines, walltime_hours)
    _write_lines(os.path.join(dest, f"{job_name}_Submit_optuna.sh"), optuna_lines)

    confirm_lines = _substitute_submit_script(
        _read_lines(os.path.join(template_dir, "Submit_confirm.sh")),
        job_name=job_name, num_workers=num_workers, timeout_seconds=timeout_seconds,
        use_gpu=use_gpu, use_extra_memory=use_extra_memory,
    )
    confirm_lines = _set_walltime(confirm_lines, confirm_walltime_hours or walltime_hours)
    _write_lines(os.path.join(dest, f"{job_name}_Submit_confirm.sh"), confirm_lines)

    print(f"Created {dest}")
    print(f"  1) qsub {job_name}_Submit_optuna.sh          # runs the Optuna search")
    print(f"  2) python export_best_params.py --journal dataout/study_{job_name}.journal --study-name {job_name}")
    print(f"  3) qsub {job_name}_Submit_confirm.sh          # k-fold-confirms the winner")
    return dest


# %%
#####################################################################
# Example usage -- edit and run this cell
#####################################################################
if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))  # this repo's root
    template_dir = os.path.join(code_dir, "hpc_templates")

    # Where job folders get created, e.g. the RDS-mounted CNNTraining dir.
    dest_dir = os.path.join(code_dir, "hpc_jobs")  # change to your CNNTraining path

    today = datetime.now().strftime("%Y%m%d")

    create_optuna_study_folder(
        template_dir=template_dir,
        source_code_dir=code_dir,
        dest_dir=dest_dir,
        job_name=f"{today}_MC24x_fullSearch",
        dataset="MC24x",
        active_params=None,  # search every hyperparameter jointly
        fixed_params={"Epochs": 1000, "testSize": 0.0, "type": "default"},
        num_workers=16,       # PBS array size = number of parallel Optuna workers
        walltime_hours=24,
        use_gpu=True,
    )
