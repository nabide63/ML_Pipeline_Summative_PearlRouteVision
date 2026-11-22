"""Offline tests for the tf.data loaders and the shared resize/normalise transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from pearlroute.config import (DataConfig, ModelConfig, TrainConfig, load_data_config,
                               load_model_config, load_train_config)
from pearlroute.data.loaders import (dataset_from_directory, list_images, load_class_weights,
                                     load_datasets, make_dataset)
from pearlroute.preprocessing.transforms import normalise, read_and_resize, resize

SIZE = 32


def _cfgs(tmp: Path) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data = load_data_config()
    data = data.model_copy(update={"paths": data.paths.model_copy(
        update={"processed_dir": tmp / "processed", "counts_json": tmp / "counts.json"})})
    return data, ModelConfig(image_size=SIZE), TrainConfig(batch_size=4, cache=True)


def _fill(root: Path, class_names: list[str], per_class: int) -> None:
    """Class i gets solid images whose red channel is 10 * (i + 1), so labels can be checked from pixels."""
    for i, name in enumerate(class_names):
        (root / name).mkdir(parents=True)
        for j in range(per_class):
            w, h = (60 + 7 * j, 45)  # mixed sizes and aspect ratios
            Image.new("RGB", (w, h), (10 * (i + 1), 100, 200)).save(root / name / f"{j}.png")


def test_configs_load() -> None:
    assert load_model_config().image_size > 0
    assert load_train_config().batch_size > 0


def test_resize_and_read(tmp_path: Path) -> None:
    Image.new("RGB", (90, 30), (250, 0, 5)).save(tmp_path / "a.jpg", quality=100)
    img = read_and_resize(tf.constant(str(tmp_path / "a.jpg")), SIZE)
    assert img.dtype == tf.uint8 and img.shape == (SIZE, SIZE, 3)
    assert abs(int(img[16, 16, 0]) - 250) <= 3
    out = resize(np.full((50, 80, 3), 255, np.uint8), SIZE)
    assert out.dtype == tf.float32 and float(tf.reduce_max(out)) <= 255.0


def test_normalise() -> None:
    x = tf.fill((2, 2, 3), 110.0)
    np.testing.assert_allclose(normalise(x, [100, 110, 120], [10, 10, 10]).numpy()[0, 0], [1, 0, -1])


def test_list_images_orders_by_class_and_ignores_others(tmp_path: Path) -> None:
    names = ["alpha", "beta"]
    _fill(tmp_path, names, 3)
    (tmp_path / "stray").mkdir()
    (tmp_path / "alpha" / "notes.txt").write_text("x")
    paths, labels = list_images(tmp_path, names)
    assert labels == [0, 0, 0, 1, 1, 1]
    assert all(Path(p).parent.name == names[l] for p, l in zip(paths, labels))
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing", names)


def test_make_dataset_shapes_labels_and_shuffling(tmp_path: Path) -> None:
    names = ["alpha", "beta", "gamma"]
    _fill(tmp_path, names, 5)
    paths, labels = list_images(tmp_path, names)

    val = make_dataset(paths, labels, SIZE, 4, training=False, seed=0)
    x, y = next(iter(val))
    assert x.shape == (4, SIZE, SIZE, 3) and x.dtype == tf.float32 and y.dtype == tf.int32
    order = np.concatenate([y.numpy() for _, y in val])
    assert order.tolist() == labels                                     # not shuffled
    for xb, yb in val:                                                  # pixels match labels
        np.testing.assert_allclose(xb.numpy()[:, 0, 0, 0], 10 * (yb.numpy() + 1), atol=1)

    train = make_dataset(paths, labels, SIZE, 4, training=True, seed=0)
    epoch1 = np.concatenate([y.numpy() for _, y in train])
    epoch2 = np.concatenate([y.numpy() for _, y in train])
    assert sorted(epoch1) == sorted(labels) and epoch1.tolist() != labels   # shuffled
    assert epoch1.tolist() != epoch2.tolist()                               # reshuffled per epoch
    again = make_dataset(paths, labels, SIZE, 4, training=True, seed=0)
    assert np.concatenate([y.numpy() for _, y in again]).tolist() == epoch1.tolist()  # seeded


def test_augment_only_on_train_and_stats_everywhere(tmp_path: Path) -> None:
    _fill(tmp_path, ["alpha"], 4)
    paths, labels = list_images(tmp_path, ["alpha"])
    zero = lambda x: x * 0.0
    stats = ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    train_x, _ = next(iter(make_dataset(paths, labels, SIZE, 4, training=True, seed=0, augment=zero)))
    val_x, _ = next(iter(make_dataset(paths, labels, SIZE, 4, training=False, seed=0, augment=zero,
                                      stats=stats)))
    assert float(tf.reduce_max(train_x)) == 0.0
    assert float(tf.reduce_max(val_x)) == pytest.approx(100.0, abs=1)   # 200 / 2, not augmented


def test_load_datasets_and_class_weights(tmp_path: Path) -> None:
    data, model, train = _cfgs(tmp_path)
    for split in ("train", "val", "test"):
        _fill(data.paths.processed_dir / split, data.class_names, 2)
    sets = load_datasets(data, model, train)
    assert set(sets) == {"train", "val", "test"}
    assert sum(int(y.shape[0]) for _, y in sets["val"]) == 2 * len(data.class_names)
    assert dataset_from_directory(data.paths.processed_dir / "test", data, model, train, training=False)

    assert load_class_weights(data) is None
    weights = {c: 1.0 + i for i, c in enumerate(data.class_names)}
    data.paths.counts_json.write_text(json.dumps({"split": {"class_weights": weights}}))
    assert load_class_weights(data) == {i: 1.0 + i for i in range(len(data.class_names))}


def test_channel_stats_round_trip(tmp_path: Path) -> None:
    from pearlroute.preprocessing.stats import compute_channel_stats, load_stats, save_stats
    images = tf.constant(np.stack([np.full((4, 4, 3), v, np.float32) for v in (0.0, 100.0)]))
    ds = tf.data.Dataset.from_tensor_slices((images, [0, 1])).batch(1)
    mean, std, n = compute_channel_stats(ds)
    assert n == 2 and mean == [50.0] * 3 and std == [50.0] * 3
    save_stats(tmp_path / "stats.json", mean, std, image_size=4, n_images=n)
    assert load_stats(tmp_path / "stats.json") == (mean, std)
