from __future__ import annotations

import tensorflow as tf


def make_ssim_metric(win_kernel: int):
    def SSIM_metric(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(
            tf.image.ssim(
                img1=y_true,
                img2=y_pred,
                max_val=1,
                filter_size=win_kernel,
                filter_sigma=1.5,
                k1=0.01,
                k2=0.03,
                return_index_map=False,
            )
        )

    return SSIM_metric


def custom_loss(y_true, y_pred):
    se_base = tf.math.square(tf.math.subtract(y_true, y_pred))
    loss = tf.math.multiply(se_base, tf.math.add(tf.constant(1, dtype=tf.float32), tf.nn.relu(y_true)))
    return tf.reduce_mean(loss)


def custom_loss5(y_true, y_pred):
    se_base = tf.math.square(tf.math.subtract(y_true, y_pred))
    loss = tf.math.multiply(
        se_base,
        tf.math.add(tf.constant(1, dtype=tf.float32), tf.math.multiply(tf.nn.relu(y_true), 5)),
    )
    return tf.reduce_mean(loss)


def custom_loss_power(y_true, y_pred, alpha=10, beta=10):
    se_base = tf.math.square(tf.math.subtract(y_true, y_pred))
    weights = tf.math.add(
        tf.constant(1.0, dtype=tf.float32),
        tf.math.multiply(tf.constant(alpha, dtype=tf.float32), tf.math.pow(y_true, tf.constant(beta, dtype=tf.float32))),
    )
    return tf.reduce_mean(se_base * weights)


def peak_loss_hpc(y_true, y_pred):
    peak_val = tf.reduce_max(y_true, keepdims=True)
    cond = tf.equal(y_true, peak_val)
    error_grid = tf.math.subtract(y_true, y_pred)
    zero_grid = tf.math.subtract(y_true, y_true)
    loss = tf.where(cond, error_grid, zero_grid)
    return tf.reduce_mean(loss)


def peak_loss_local(y_true, y_pred):
    import tensorflow_probability as tfp

    percentile_95 = tfp.stats.percentile(y_true, 95.0, interpolation="linear")
    mask = tf.greater_equal(y_true, percentile_95)
    squared_error = tf.math.square(y_true - y_pred)
    masked_error = tf.where(mask, squared_error, tf.zeros_like(squared_error))
    num_selected = tf.reduce_sum(tf.cast(mask, tf.float32))
    loss = tf.reduce_sum(masked_error) / (num_selected + 1e-8)
    return loss


def peak_loss2(y_true, y_pred):
    flat_idx = tf.argmax(tf.reshape(y_pred, [-1]))
    shape = tf.cast(tf.shape(y_pred), tf.int64)
    coords = tf.unravel_index(flat_idx, shape)
    pred_peak = tf.gather_nd(y_pred, [coords])
    true_peak = tf.gather_nd(y_true, [coords])
    return tf.abs(pred_peak - true_peak)


def denseLoss(y_true, failCoord):
    if isinstance(failCoord, (tuple, list)):
        row, col = int(failCoord[0]), int(failCoord[1])
        vals = y_true[:, row, col, 0] if y_true.ndim == 4 else y_true[:, row, col]
    else:
        batch_indices = tf.range(tf.shape(y_true)[0])
        row = tf.cast(failCoord[:, 0], tf.int32)
        col = tf.cast(failCoord[:, 1], tf.int32)
        indices = tf.stack([batch_indices, row, col], axis=1)
        vals = tf.gather_nd(y_true, indices)
    return tf.reduce_mean(tf.abs(vals - 1.0))


def resolve_loss(loss_name: str, variant: str = "hpc"):
    if loss_name == "MSE":
        return tf.keras.losses.MeanSquaredError()
    if loss_name == "MAE":
        return tf.keras.losses.MeanAbsoluteError()
    if loss_name == "Custom":
        return custom_loss
    if loss_name == "Peak":
        return peak_loss_hpc if variant == "hpc" else peak_loss_local
    if loss_name == "Peak2":
        return peak_loss2
    if loss_name == "Custom5":
        return custom_loss5
    if loss_name == "CustomPower":
        return custom_loss_power
    if loss_name == "dense":
        return denseLoss
    raise ValueError(f"Unsupported loss type: {loss_name}")
