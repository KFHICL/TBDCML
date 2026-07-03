from __future__ import annotations

from dataclasses import dataclass
import os
import random

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class DatasetSpec:
    train_dat_name: str
    sample_shape: tuple[int, int]
    x_names: list[str]
    samples_per_file: int
    win_kernel: int


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_dataset_spec(dataset: str, mc24_features: str | None) -> DatasetSpec:
    if dataset == "LFC18":
        return DatasetSpec("Gaudron2018", (55, 20), ["E11", "E22", "E12"], 1, 5)
    if dataset == "MC24":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024", (60, 20), x_names, 1, 7)
    if dataset == "MC24_200":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_200", (60, 20), x_names, 1, 7)
    if dataset == "MC24_500":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_500", (60, 20), x_names, 1, 7)
    if dataset == "MC24_1000":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_1000", (60, 20), x_names, 1, 7)
    if dataset == "MC24x":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_224_4kSamples", (224, 224), x_names, 40, 17)
    if dataset == "MC24_VarVf":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_100SamplesVfVariable", (60, 20), x_names, 1, 7)
    if dataset == "MC24_ConstVf":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_100SamplesVfConstant", (60, 20), x_names, 1, 7)
    if dataset == "MC24_1000_ConstVf":
        x_names = ["Ex", "Ey", "Gxy"]
        if mc24_features == "Vf_c2":
            x_names = ["Vf", "c2"]
        elif mc24_features == "All":
            x_names = ["Ex", "Ey", "Gxy", "Vf", "c2"]
        return DatasetSpec("MatLabModel2024_1000SamplesVfConstant", (60, 20), x_names, 1, 7)
    raise ValueError(f"Unsupported dataset: {dataset}")
