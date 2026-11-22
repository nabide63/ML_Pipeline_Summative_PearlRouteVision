"""Tests for pearlroute.models, training and evaluation (tiny synthetic data, runs in seconds)."""

from __future__ import annotations

from pathlib import Path

import keras
import matplotlib
import numpy as np
import pytest
import tensorflow as tf

matplotlib.use("Agg")

from pearlroute.config import BaselineConfig, ModelConfig, TrainConfig  # noqa: E402
from pearlroute.evaluation.metrics import classification_metrics, evaluate_model  # noqa: E402
from pearlroute.evaluation.plots import plot_confusion_matrix, plot_history  # noqa: E402
from pearlroute.models.io import load_model, save_model  # noqa: E402
from pearlroute.models.registry import available_models, build_model  # noqa: E402
from pearlroute.training.trainer import train_model  # noqa: E402

SIZE, CLASSES = 32, ["a", "b", "c"]


def _model_cfg() -> ModelConfig:
    return ModelConfig(image_size=SIZE, baseline=BaselineConfig(filters=[8, 16], dense_units=16, dropout=0.1))


def _toy_data(n: int = 48) -> tf.data.Dataset:
    """Class k images are bright in channel k, so a tiny CNN can learn them quickly."""
    rng = np.random.default_rng(0)
    y = np.arange(n) % len(CLASSES)
    x = rng.normal(0, 0.3, (n, SIZE, SIZE, 3)).astype("float32")
    x[np.arange(n), :, :, y] += 2.0
    return tf.data.Dataset.from_tensor_slices((x, y.astype("int32"))).batch(8)


def test_registry_and_baseline_shape() -> None:
    assert "baseline" in available_models()
    model = build_model("baseline", _model_cfg(), len(CLASSES))
    out = model(tf.zeros((2, SIZE, SIZE, 3)))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.numpy().sum(axis=1), 1.0, atol=1e-5)      # softmax
    with pytest.raises(ValueError):
        build_model("nope", _model_cfg(), 3)


def test_baseline_default_config_size() -> None:
    model = build_model("baseline", ModelConfig(image_size=160), 6)
    assert 1_000_000 < model.count_params() < 3_000_000                       # a small model


def test_train_evaluate_save_load(tmp_path: Path) -> None:
    keras.utils.set_random_seed(0)
    data = _toy_data()
    model = build_model("baseline", _model_cfg(), len(CLASSES))
    cfg = TrainConfig(batch_size=8, epochs=8, learning_rate=3e-3, early_stopping_patience=8)
    history = train_model(model, data, data, cfg, tmp_path)
    assert list(history.index[:2]) == [1, 2] and {"loss", "val_loss", "val_accuracy"} <= set(history.columns)
    assert history["loss"].iloc[-1] < history["loss"].iloc[0]                 # it learns
    assert (tmp_path / "checkpoints" / "best.keras").exists()

    metrics = evaluate_model(model, data, CLASSES)
    assert metrics["accuracy"] > 0.8 and metrics["n"] == 48

    save_model(model, tmp_path / "saved", CLASSES, SIZE, [1, 2, 3], [4, 5, 6], test_accuracy=0.5)
    loaded, meta = load_model(tmp_path / "saved")
    assert meta["class_names"] == CLASSES and meta["normalisation"]["mean"] == [1, 2, 3]
    x, _ = next(iter(data))
    np.testing.assert_allclose(loaded(x).numpy(), model(x).numpy(), atol=1e-5)


def test_classification_metrics_known_values() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.eye(3)[[0, 0, 1, 2, 2, 2]] * 0.8 + 0.2 / 3
    m = classification_metrics(y_true, y_prob, CLASSES)
    assert m["accuracy"] == pytest.approx(5 / 6, abs=1e-3)
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert m["per_class"]["b"]["recall"] == 0.5 and m["per_class"]["c"]["support"] == 2
    assert 0 < m["roc_auc_macro"] <= 1 and m["loss"] > 0


def test_plots_return_figures() -> None:
    import pandas as pd
    history = pd.DataFrame({"loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "accuracy": [0.4, 0.6],
                            "val_accuracy": [0.35, 0.5]}, index=pd.Index([1, 2], name="epoch"))
    assert plot_history(history).axes
    assert plot_confusion_matrix([[2, 1], [0, 3]], ["x", "y"]).axes
