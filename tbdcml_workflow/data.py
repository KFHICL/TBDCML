from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tensorflow as tf


def load_sample_new(
    path: str,
    x_names: list[str],
    y_names: list[str],
    sample_shape: tuple[int, int],
    samples_per_file: int,
):
    _, file_extension = os.path.splitext(path)
    match file_extension:
        case ".csv":
            sample = pd.read_csv(path)
            samples = np.array(sample)
        case ".parquet":
            sample = pd.read_parquet(path, engine="auto")
            samples = [y for _, y in sample.groupby("specimen")]
            samples = np.array(list(map(lambda x: x.to_numpy(), samples)))
        case _:
            raise ValueError(f"Unsupported file type: {file_extension}")

    headers = np.array(sample.columns.values.tolist())
    samples = samples.reshape(samples_per_file, sample_shape[0], sample_shape[1], -1)

    feature_idx = []
    for name in x_names:
        feature_idx += [np.where(headers == name)[0][0]]

    gt_idx = []
    for name in y_names:
        gt_idx += [np.where(headers == name)[0][0]]

    x_values = np.asarray(samples[:, :, :, feature_idx]).astype("float32")
    y_values = np.asarray(samples[:, :, :, gt_idx]).astype("float32")

    base_id = os.path.basename(path).replace(".", "_")
    sample_ids = [f"{base_id}_{i}" for i in range(x_values.shape[0])]

    dataset = tf.data.Dataset.from_tensor_slices((x_values, y_values, sample_ids))

    if "coordinates" in headers:
        headers = np.concatenate(([[headers[0], "x_coord", "y_coord"], headers[2:]]))

    return headers, dataset


def load_all_samples(
    train_dat_path: str,
    num_samples: int,
    x_names: list[str],
    y_names: list[str],
    sample_shape: tuple[int, int],
    samples_per_file: int,
    testing: bool = False,
):
    files = [os.path.join(train_dat_path, file) for file in os.listdir(train_dat_path)]
    if testing:
        files = files[0:10]

    def process_file(filepath: str):
        return load_sample_new(filepath, x_names, y_names, sample_shape, samples_per_file)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for i, future in enumerate(as_completed(futures)):
            if i == 0:
                headers, samples = future.result()
            else:
                samples = samples.concatenate(future.result()[1])

            print(
                "Now loading file number {num} out of {total}".format(
                    num=i + 1,
                    total=num_samples / samples_per_file,
                )
            )

    return headers, samples


def get_sample_ids(dataset):
    ids = []
    for _, _, sample_id in dataset.as_numpy_iterator():
        ids.append(sample_id.decode() if isinstance(sample_id, bytes) else sample_id)
    return ids


def drop_sample_ids(dataset):
    return dataset.map(lambda x, y, _: (x, y))


def check_overlapping_samples(train_ids, val_ids):
    train_set = set(train_ids)
    val_set = set(val_ids)
    return train_set.intersection(val_set)
