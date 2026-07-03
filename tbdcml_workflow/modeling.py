from __future__ import annotations

import tensorflow as tf


def compile_model(CNNModel, params, lr_schedule, lossfunc, ssim_metric, include_ssim=True):
    metrics = ["mean_absolute_error", "mean_squared_error"]
    if include_ssim:
        metrics.append(ssim_metric)

    if params["optimizer"] == "Adadelta":
        CNNModel.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=lr_schedule, epsilon=params["epsilon"]),
            loss=lossfunc,
            metrics=metrics,
        )
    elif params["optimizer"] == "Nadam":
        CNNModel.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=lr_schedule, epsilon=params["epsilon"]),
            loss=lossfunc,
            metrics=metrics,
        )
    else:
        CNNModel.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=params["epsilon"]),
            loss=lossfunc,
            metrics=metrics,
        )
    return CNNModel


def build_model_from_params(model_map, params, input_shape, output_shape, normalizer, lr_schedule, lossfunc, ssim_metric):
    model_type = params["type"]

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_builder = model_map[model_type]
    if model_type in {"default", "dense", "UNet"}:
        CNNModel = model_builder(inputShape=input_shape, outputShape=output_shape, params=params)
    else:
        input_layer, output_layer = model_builder(inputShape=input_shape)
        CNNModel = model_map["applyDecoder"](input_layer, output_layer, outputShape=output_shape, params=params)

    return compile_model(
        CNNModel,
        params=params,
        lr_schedule=lr_schedule,
        lossfunc=lossfunc,
        ssim_metric=ssim_metric,
        include_ssim=(model_type != "dense"),
    )
