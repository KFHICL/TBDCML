"""Optuna search space for the TBDCNet-style baseline CNN.

This module is the *single* place where "which hyperparameters can be
tuned, and over which values" is defined. Both the local development
script (``optuna_local.py``) and the HPC driver (``BenchMarks_Optuna.py``)
import :func:`suggest_baseline_cnn_params` from here, so the search space
never has to be kept in sync across several sweep-definition CSVs again.

Each ``if searching(...)`` block below corresponds to exactly one row of
Table 2 in the paper. The dict keys written into ``params`` match the
column names already used by ``tbdcml_workflow.custom_models`` and
``tbdcml_workflow.modeling`` (e.g. ``layer1Kernel``, ``batchNorm``), so the
rest of the training pipeline does not need to change at all -- only how
that params dict gets constructed changes.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import optuna

# ---------------------------------------------------------------------------
# Baseline ("row 1" / Table 2 "baseline CNN" column) hyperparameter values.
#
# Whenever a hyperparameter is *not* being searched in a given study (see
# `active_params` below), its value is taken from here instead, so a narrow
# study (e.g. only searching "dropout" and "model_depth") still produces a
# complete, runnable params dict. Adjust these to match your exact baseline
# CNN definition (Section 3.1 / Table 2 bold column) if you need an exact
# reproduction.
# ---------------------------------------------------------------------------
BASELINE_DEFAULTS: dict[str, Any] = {
    "layer1Kernel": 3, "layer2Kernel": 3, "layer3Kernel": 3,
    "layer4Kernel": 3, "layer5Kernel": 3, "layer6Kernel": 3,
    "conv1Activation": "relu", "conv2Activation": "relu", "conv3Activation": "relu",
    "conv4Activation": "relu", "conv5Activation": "relu", "conv6Activation": "relu",
    "pooling": 1,
    "downSample": 0,
    "dropout": 0.10,
    "batchNorm": 1,
    "layer2": 1, "layer3": 1, "layer4": 0, "layer5": 0, "layer6": 0,  # model depth = 3
    "L1kernel_regularizer": 0.0,
    "L2kernel_regularizer": 0.0,
    "optimizer": "Adam",
    "epsilon": 1e-7,
    "initial_lr": 0.001,
    "lr_decay_rate": 1.0,
    "skipConnections": 0,
    "dsAugmentation": 0,
    "standardisation": "Standard",
    "loss": "MSE",
    "ActivationUp": 1,
    "filterScale": 1.0,
    "valSize": 0.1,
    "testSize": 0.0,
    "batchSize": 8,
    "Epochs": 1000,
    "type": "default",
}

# Ordered so that `_apply_model_depth` can flip flags 2..N on and N+1..6 off.
_MODEL_DEPTH_FLAGS = ("layer2", "layer3", "layer4", "layer5", "layer6")


def _apply_model_depth(params: dict[str, Any], depth: int) -> None:
    """`depth` = number of Conv blocks in the encoder (Figure 5). depth=1
    means only the first (always-present) block is used."""
    for i, flag in enumerate(_MODEL_DEPTH_FLAGS, start=2):
        params[flag] = 1 if depth >= i else 0


def suggest_baseline_cnn_params(
    trial: optuna.Trial,
    *,
    dataset: str,
    mc24_features: str = "All",
    active_params: Iterable[str] | None = None,
    fixed: Mapping[str, Any] | None = None,
    batch_size_choices: Iterable[int] = (4, 8, 16, 32, 64, 128, 256),
    max_model_depth: int = 4,
) -> dict[str, Any]:
    """Sample one concrete hyperparameter configuration for `trial`.

    Parameters
    ----------
    trial:
        The Optuna trial requesting values. Every ``trial.suggest_*`` call
        below registers that hyperparameter with the trial/study, which is
        what powers the importance/interaction plots later.
    dataset, mc24_features:
        Passed straight through to ``resolve_dataset_spec`` elsewhere; not
        searched.
    active_params:
        If ``None`` (default), every hyperparameter below is searched --
        use this for a full joint study. Pass an explicit set/list of the
        short names used in the ``searching(...)`` calls (e.g.
        ``{"dropout", "model_depth", "data_augmentation"}``) to reproduce a
        narrower, Section-5.2-style "combination sweep" of just a few HPs
        while holding everything else at its baseline value.
    fixed:
        Structural overrides that should never be searched (e.g. per-dataset
        ``valSize``/``testSize``/``batchSize`` caps, or ``Epochs``). Applied
        after the baseline defaults and before any `trial.suggest_*` calls,
        so `fixed` always wins for keys it sets -- unless that same key is
        also in `active_params`, in which case the search overrides it.
    batch_size_choices:
        Candidate mini-batch sizes. The paper notes that only batch sizes up
        to 64 were trialled on the small datasets -- pass a smaller tuple
        here for MeC-Macro/MeC-Meso-S-sized studies.
    max_model_depth:
        Largest number of encoder Conv blocks to consider (paper searched
        1-4; the encoder/decoder supports up to 6).
    """
    fixed = dict(fixed or {})
    params: dict[str, Any] = dict(BASELINE_DEFAULTS)
    params["Dataset"] = dataset
    params["MC24_Features"] = mc24_features
    params.update(fixed)

    active = set(active_params) if active_params is not None else None

    def searching(name: str) -> bool:
        return active is None or name in active

    if searching("activation"):
        activation = trial.suggest_categorical(
            "activation",
            ["relu", "tanh", "softplus", "elu", "leaky_relu", "silu", "gelu"],
        )
        for k in ("conv1Activation", "conv2Activation", "conv3Activation",
                  "conv4Activation", "conv5Activation", "conv6Activation"):
            params[k] = activation

    if searching("kernel_size"):
        kernel_size = trial.suggest_int("kernel_size", 1, 7)
        for k in ("layer1Kernel", "layer2Kernel", "layer3Kernel",
                  "layer4Kernel", "layer5Kernel", "layer6Kernel"):
            params[k] = kernel_size

    if searching("batch_norm"):
        params["batchNorm"] = trial.suggest_categorical("batch_norm", [0, 1])

    if searching("max_pooling"):
        params["pooling"] = trial.suggest_categorical("max_pooling", [0, 1])

    if searching("downsampling"):
        params["downSample"] = trial.suggest_categorical("downsampling", [0, 1])

    if searching("dropout"):
        params["dropout"] = trial.suggest_categorical(
            "dropout", [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
        )

    if searching("model_depth"):
        depth = trial.suggest_int("model_depth", 1, max_model_depth)
        _apply_model_depth(params, depth)

    if searching("skip_connections"):
        params["skipConnections"] = trial.suggest_categorical("skip_connections", [0, 1])

    if searching("decoder_activations"):
        params["ActivationUp"] = trial.suggest_categorical("decoder_activations", [0, 1])

    if searching("filter_scaling"):
        params["filterScale"] = trial.suggest_categorical(
            "filter_scaling", [1.0, 0.75, 0.5, 0.25]
        )

    if searching("optimizer"):
        params["optimizer"] = trial.suggest_categorical(
            "optimizer", ["Adam", "Nadam", "Adadelta"]
        )

    if searching("epsilon"):
        params["epsilon"] = trial.suggest_categorical(
            "epsilon", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        )

    if searching("initial_lr"):
        params["initial_lr"] = trial.suggest_categorical(
            "initial_lr", [0.0001, 0.0005, 0.001, 0.005, 0.01]
        )

    if searching("lr_decay_rate"):
        params["lr_decay_rate"] = trial.suggest_categorical(
            "lr_decay_rate", [1.0, 0.5, 0.1, 0.01]
        )

    if searching("loss"):
        params["loss"] = trial.suggest_categorical("loss", ["MSE", "MAE", "Custom"])

    if searching("data_augmentation"):
        params["dsAugmentation"] = trial.suggest_categorical("data_augmentation", [0, 1])

    if searching("batch_size"):
        params["batchSize"] = trial.suggest_categorical(
            "batch_size", list(batch_size_choices)
        )

    if searching("val_fraction"):
        params["valSize"] = trial.suggest_categorical("val_fraction", [0.1, 0.2, 0.3])

    return params
