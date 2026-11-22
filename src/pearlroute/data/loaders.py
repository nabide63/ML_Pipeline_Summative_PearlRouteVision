"""tf.data input pipelines.

Reads the folders written by scripts/split_dataset.py (data/processed/<split>/<class>/*.jpg),
or any folder with the same <class>/ layout such as an unzipped retrain sample.

Pipeline per split:
    paths -> read + resize (uint8) -> cache -> [shuffle] -> batch -> float32
          -> [augment, train only] -> [normalise] -> prefetch

Labels are integer indices into the sorted class names (cfg.class_names), the same order the
model's output units use. Images come out as float32 in [0, 255] unless normalisation stats are
given; the augmentation and stats modules plug in through `augment` and `stats`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import tensorflow as tf

from pearlroute.config import DataConfig, ModelConfig, TrainConfig
from pearlroute.logging_utils import get_logger
from pearlroute.preprocessing.transforms import normalise, read_and_resize

log = get_logger(__name__)

AUTOTUNE = tf.data.AUTOTUNE
SPLITS = ("train", "val", "test")
# tf.io.decode_image reads these; .webp would need PIL (the downloader saves everything as JPEG).
READABLE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

Augment = Callable[[tf.Tensor], tf.Tensor]
Stats = tuple[Sequence[float], Sequence[float]]


def list_images(root: Path, class_names: Sequence[str]) -> tuple[list[str], list[int]]:
    """Image paths under root/<class>/ and their label indices, in a stable (sorted) order."""
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/split_dataset.py first")
    extra = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in class_names)
    if extra:
        log.warning("%s has folders that are not classes and will be ignored: %s", root, extra)
    paths: list[str] = []
    labels: list[int] = []
    for index, name in enumerate(class_names):
        files = sorted(p for p in (root / name).glob("*") if p.suffix.lower() in READABLE_EXTENSIONS)
        paths += [str(p) for p in files]
        labels += [index] * len(files)
    if not paths:
        raise ValueError(f"no images found under {root}")
    return paths, labels


def make_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    image_size: int,
    batch_size: int,
    *,
    training: bool,
    seed: int,
    cache: bool = True,
    augment: Augment | None = None,
    stats: Stats | None = None,
) -> tf.data.Dataset:
    """Batched (images, labels) dataset. Shuffles, and augments if given, only when training."""
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(lambda p, y: (read_and_resize(p, image_size), y), num_parallel_calls=AUTOTUNE,
                deterministic=True)
    if cache:
        ds = ds.cache()  # decoded uint8, so ~77 KB per image at 160 px
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if training and augment is not None:
        ds = ds.map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE)
    if stats is not None:
        mean, std = stats
        ds = ds.map(lambda x, y: (normalise(x, mean, std), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def dataset_from_directory(
    root: Path,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    training: bool,
    augment: Augment | None = None,
    stats: Stats | None = None,
) -> tf.data.Dataset:
    """Dataset for any folder laid out as root/<class>/<image>."""
    paths, labels = list_images(root, data_cfg.class_names)
    return make_dataset(paths, labels, model_cfg.image_size, train_cfg.batch_size, training=training,
                        seed=data_cfg.seed, cache=train_cfg.cache, augment=augment, stats=stats)


def load_split(
    split: str,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    augment: Augment | None = None,
    stats: Stats | None = None,
) -> tf.data.Dataset:
    """One of train / val / test from data/processed. Only train is shuffled and augmented."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    return dataset_from_directory(data_cfg.paths.processed_dir / split, data_cfg, model_cfg, train_cfg,
                                  training=split == "train", augment=augment, stats=stats)


def load_datasets(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    augment: Augment | None = None,
    stats: Stats | None = None,
) -> dict[str, tf.data.Dataset]:
    """{'train': ..., 'val': ..., 'test': ...}"""
    return {s: load_split(s, data_cfg, model_cfg, train_cfg, augment=augment, stats=stats) for s in SPLITS}


def load_class_weights(data_cfg: DataConfig) -> dict[int, float] | None:
    """Class weights from counts.json keyed by label index (for model.fit), or None if not needed."""
    path = data_cfg.paths.counts_json
    if not path.exists():
        return None
    weights = json.loads(path.read_text(encoding="utf-8")).get("split", {}).get("class_weights")
    if not weights:
        return None
    return {i: float(weights[name]) for i, name in enumerate(data_cfg.class_names)}
