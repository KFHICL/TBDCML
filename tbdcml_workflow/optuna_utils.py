"""Shared "train one concrete configuration and score it" helper.

This is the piece that used to live inline in `test.py` / `BenchMarks.py`
(model instantiation, checkpointing, early stopping, evaluation). It is
factored out here so both the local Optuna script and the HPC Optuna driver
call the *same* training routine -- one bug fix, one place.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from . import architectures as shared_architectures
from . import custom_models as shared_custom_models
from .losses import make_ssim_metric, resolve_loss
from .modeling import build_model_from_params


def get_model_map(normalizer, seed: int) -> dict:
    """Same architecture registry used by test.py / BenchMarks.py, centralised."""
    return {
        "Xception": shared_architectures.Xception_Model,
        "MobileNetV2": shared_architectures.mobileNetV2_Model,
        "VGG16": shared_architectures.VGG16_Model,
        "ResNet50": shared_architectures.ResNet50_Model,
        "ResNet50V2": shared_architectures.ResNet50V2_Model,
        "InceptionV3": shared_architectures.InceptionV3_Model,
        "InceptionResNetV2": shared_architectures.InceptionResNetV2_Model,
        "DenseNet121": shared_architectures.DenseNet121_Model,
        "NASNetMobile": shared_architectures.NASNetMobile_Model,
        "EfficientNetV2S": shared_architectures.EfficientNetV2S_Model,
        "EfficientNetV2M": shared_architectures.EfficientNetV2M_Model,
        "EfficientNetV2L": shared_architectures.EfficientNetV2L_Model,
        "ConvNeXtTiny": shared_architectures.ConvNeXtTiny_Model,
        "ConvNeXtSmall": shared_architectures.ConvNeXtSmall_Model,
        "ConvNeXtLarge": shared_architectures.ConvNeXtLarge_Model,
        "UNet": lambda inputShape, outputShape, params: shared_custom_models.build_tbdcnet_unet(
            inputShape, outputShape, params, normalizer, seed
        ),
        "default": lambda inputShape, outputShape, params: shared_custom_models.build_tbdcnet_model_cnn(
            inputShape, outputShape, params, normalizer
        ),
        "dense": lambda inputShape, outputShape, params: shared_custom_models.build_tbdcnet_model_cnn(
            inputShape, outputShape, params, normalizer
        ),
        "applyDecoder": shared_architectures.applyDecoder,
    }


def train_once(
    params: dict[str, Any],
    data: dict[str, Any],
    *,
    seed: int,
    win_kernel: int,
    checkpoint_dir: str,
    loss_variant: str = "hpc",
    patience: int = 60,
    extra_callbacks: Optional[list] = None,
    verbose: int = 0,
    return_model: bool = False,
) -> dict[str, Any]:
    """Train a single model for one concrete `params` dict and one prepared
    dataset bundle (`data`, from `pipeline.split_and_prepare`).

    Returns a dict of scalar metrics (+ the Keras history) for this one
    repeat. Call this multiple times with different `seed`/`checkpoint_dir`
    values to reproduce the paper's Monte-Carlo-cross-validation repeats.
    """
    tf.keras.backend.clear_session()

    ssim_metric = make_ssim_metric(win_kernel)
    lossfunc = resolve_loss(params["loss"], variant=loss_variant)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params["initial_lr"],
        decay_steps=data["steps_per_epoch"] * params["Epochs"],
        decay_rate=params["lr_decay_rate"],
    )

    model_map = get_model_map(data["normalizer"], seed)
    model = build_model_from_params(
        model_map=model_map,
        params=params,
        input_shape=data["input_shape"],
        output_shape=data["output_shape"],
        normalizer=data["normalizer"],
        lr_schedule=lr_schedule,
        lossfunc=lossfunc,
        ssim_metric=ssim_metric,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    cp_path = os.path.join(checkpoint_dir, "best.weights.h5")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_path, save_weights_only=True, save_best_only=True, monitor="val_loss", verbose=0,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=False, verbose=0,
    )

    callbacks = [early_stop, cp_callback] + list(extra_callbacks or [])

    history = model.fit(
        data["train_ds"],
        epochs=params["Epochs"],
        steps_per_epoch=int(data["steps_per_epoch"]),
        validation_data=data["val_ds"],
        callbacks=callbacks,
        verbose=verbose,
    )

    model.load_weights(cp_path)

    val_results = model.evaluate(data["val_ds_eval"], verbose=0, return_dict=True)
    train_results = model.evaluate(data["train_ds_eval"], verbose=0, return_dict=True)

    result = {
        "val_rmse": float(np.sqrt(val_results["mean_squared_error"])),
        "val_ssim": float(_find_metric(val_results, "ssim")),
        "train_rmse": float(np.sqrt(train_results["mean_squared_error"])),
        "train_ssim": float(_find_metric(train_results, "ssim")),
        "n_epochs_trained": len(history.history.get("loss", [])),
        "history": history.history,
    }
    if data.get("test_ds_eval") is not None:
        test_results = model.evaluate(data["test_ds_eval"], verbose=0, return_dict=True)
        result["test_rmse"] = float(np.sqrt(test_results["mean_squared_error"]))
        result["test_ssim"] = float(_find_metric(test_results, "ssim"))
    if return_model:
        result["model"] = model
    return result


def _find_metric(results: dict[str, float], name_contains: str) -> float:
    """Keras normalises metric names (e.g. the `SSIM_metric` function becomes
    the `ssim_metric` key in `model.evaluate(..., return_dict=True)`), and
    that normalisation has changed between Keras versions. Look the metric
    up case-insensitively instead of hardcoding one exact key spelling."""
    for key, value in results.items():
        if name_contains.lower() in key.lower():
            return value
    return float("nan")
