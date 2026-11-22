"""Compute and save per-channel mean and std from the train set.

The stats come from the training images only (never val/test, which would leak information
about them) and are in [0, 255] pixel units, the scale transforms.normalise expects.
They are saved to paths.stats_json so prediction and retraining normalise the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def compute_channel_stats(dataset: tf.data.Dataset) -> tuple[list[float], list[float], int]:
    """Mean and std per RGB channel over every pixel of an (images, labels) dataset.

    Pass the un-augmented, un-normalised train set. Returns (mean, std, number of images).
    """
    total = np.zeros(3)
    total_sq = np.zeros(3)
    pixels = 0
    images = 0
    for x, _ in dataset:
        x = x.numpy().astype(np.float64).reshape(-1, 3)
        total += x.sum(axis=0)
        total_sq += (x ** 2).sum(axis=0)
        pixels += x.shape[0]
        images += int(_.shape[0])
    if pixels == 0:
        raise ValueError("dataset is empty")
    mean = total / pixels
    std = np.sqrt(np.maximum(total_sq / pixels - mean ** 2, 1e-12))
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std], images


def save_stats(path: Path, mean: list[float], std: list[float], image_size: int, n_images: int) -> None:
    """Write the stats, with what they were computed on, to a small JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean, "std": std, "image_size": image_size,
                                "n_train_images": n_images, "units": "pixels 0-255, RGB"},
                               indent=2) + "\n", encoding="utf-8", newline="\n")


def load_stats(path: Path) -> tuple[list[float], list[float]]:
    """(mean, std) as saved by save_stats; the form loaders and transforms.normalise take."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; compute the train stats first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["mean"], data["std"]
