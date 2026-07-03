"""Dataset pipeline for Optuna-driven training.

The old per-run scripts (`test.py`, `BenchMarks.py`) re-read every sample
file from disk, re-shuffled, re-split, and re-built the whole `tf.data`
pipeline on every single run. That was fine when each HPC array task was
one independent process anyway, but it is wasteful once many Optuna trials
run one after another *inside the same process* (which is exactly what
`study.optimize()` does).

This module splits that work in two:

* :func:`load_shuffled_pool` -- the expensive, disk-bound step (reading
  every specimen file, shuffling once with a fixed seed). This only depends
  on the dataset itself, never on hyperparameters, so call it *once* per
  process and reuse the result for every trial.
* :func:`split_and_prepare` -- the cheap, per-trial step (train/val/test
  split, batching, augmentation, normalisation). This *does* depend on
  hyperparameters that are searched (batch size, validation fraction, data
  augmentation), so it is called once per trial, but it never touches disk.

Because the pool is shuffled once with a fixed seed and
`reshuffle_each_iteration=False`, splitting it differently per trial (e.g.
a 10% vs 20% validation fraction) is equivalent to re-shuffling from
scratch each time -- so trial-to-trial results stay comparable to the
original one-CSV-row-per-run behaviour.
"""

from __future__ import annotations

import math
from typing import Any

import tensorflow as tf

from .data import drop_sample_ids, get_sample_ids, load_all_samples


class Augment(tf.keras.layers.Layer):
    """Applies the same random horizontal/vertical flip to a feature/label pair."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)

    def call(self, inputs, labels):
        return self.augment_inputs(inputs), self.augment_labels(labels)


def load_shuffled_pool(
    train_dat_path: str,
    num_samples: int,
    x_names: list[str],
    y_names: list[str],
    sample_shape: tuple[int, int],
    samples_per_file: int,
    seed: int,
    testing: bool = False,
) -> tf.data.Dataset:
    """Load every specimen once and shuffle it deterministically.

    Call this exactly once per process (before the Optuna study starts) and
    pass the result into every call of :func:`split_and_prepare`.
    """
    _, samples = load_all_samples(
        train_dat_path, num_samples, x_names, y_names, sample_shape, samples_per_file, testing=testing
    )
    samples = samples.cache()
    samples = samples.shuffle(buffer_size=num_samples, seed=seed, reshuffle_each_iteration=False)
    return samples


def split_and_prepare(
    pool: tf.data.Dataset,
    num_samples: int,
    *,
    batch_size: int,
    val_fraction: float,
    test_fraction: float = 0.0,
    augment: bool = False,
    seed: int = 0,
    normalizer_length: int = 40,
) -> dict[str, Any]:
    """Split the pre-loaded `pool` and build train/val/(test) `tf.data`
    pipelines for one concrete hyperparameter configuration.

    Returns a dict bundle consumed by
    :func:`tbdcml_workflow.optuna_utils.train_once`.
    """
    val_size = math.floor(val_fraction * num_samples)
    test_size = math.floor(test_fraction * num_samples)
    train_length = num_samples - val_size - test_size
    if train_length <= 0:
        raise ValueError(
            f"val_fraction={val_fraction} + test_fraction={test_fraction} leaves no training samples "
            f"out of {num_samples} total."
        )

    train_ds = pool.take(train_length)
    remaining = pool.skip(train_length)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size) if test_size > 0 else None

    sample_ids = {
        "train": get_sample_ids(train_ds),
        "val": get_sample_ids(val_ds),
        "test": get_sample_ids(test_ds) if test_ds is not None else [],
    }

    train_ds = drop_sample_ids(train_ds)
    val_ds = drop_sample_ids(val_ds)
    test_ds = drop_sample_ids(test_ds) if test_ds is not None else None

    # Un-repeated, un-augmented copies used for final RMSE/SSIM evaluation.
    train_ds_eval = train_ds.batch(batch_size).cache()
    val_ds_eval = val_ds.batch(batch_size).cache()
    test_ds_eval = test_ds.batch(batch_size).cache() if test_ds is not None else None

    # Normalisation statistics are derived only from a slice of the training
    # set, to avoid leaking validation/test data into the scaling.
    feature_ds = train_ds.take(normalizer_length).map(lambda x, y: x)
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(feature_ds)

    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=train_length, seed=seed)
    if augment:
        train_ds = train_ds.map(Augment(seed=seed))
    train_ds = train_ds.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = train_length // batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"batch_size={batch_size} is larger than the training set ({train_length} samples).")

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_ds_eval": train_ds_eval,
        "val_ds_eval": val_ds_eval,
        "test_ds_eval": test_ds_eval,
        "normalizer": normalizer,
        "steps_per_epoch": steps_per_epoch,
        "input_shape": tuple(train_ds_eval.element_spec[0].shape[1:]),
        "output_shape": tuple(train_ds_eval.element_spec[1].shape[1:]),
        "sample_ids": sample_ids,
    }
