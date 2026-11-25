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


def test_pearlnet_shape_size_and_blocks() -> None:
    from pearlroute.config import PearlNetConfig
    model = build_model("pearlnet", ModelConfig(image_size=160), 6)
    assert model.output_shape == (None, 6)
    assert model.get_layer("global_pool").output.shape[-1] == 256
    names = [layer.name for layer in model.layers]
    assert any(n.endswith("_se_scale") for n in names) and any(n.endswith("_add") for n in names)
    assert "flatten" not in names                                            # GAP head, no big dense layer
    assert model.get_layer("probabilities").count_params() == 256 * 6 + 6
    tiny = ModelConfig(image_size=SIZE, pearlnet=PearlNetConfig(stem_filters=8, filters=[8, 16], blocks=[1, 2]))
    out = build_model("pearlnet", tiny, 3)(tf.zeros((2, SIZE, SIZE, 3)), training=False)
    np.testing.assert_allclose(out.numpy().sum(axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        build_model("pearlnet", ModelConfig(image_size=SIZE, pearlnet=PearlNetConfig(filters=[8], blocks=[1, 1])), 3)


def test_label_smoothing_loss_matches_manual_value() -> None:
    from pearlroute.training.trainer import smoothed_sparse_crossentropy
    loss = smoothed_sparse_crossentropy(num_classes=4, label_smoothing=0.2)
    y_pred = tf.constant([[0.7, 0.1, 0.1, 0.1]])
    target = np.array([0.8 + 0.05, 0.05, 0.05, 0.05])                        # 0.2 spread over 4 classes
    expected = -(target * np.log([0.7, 0.1, 0.1, 0.1])).sum()
    assert float(loss(tf.constant([0]), y_pred)[0]) == pytest.approx(expected, rel=1e-4)


def test_loss_and_accuracy_accept_integer_and_soft_labels() -> None:
    from pearlroute.training.trainer import label_accuracy, smoothed_sparse_crossentropy
    probs = tf.constant([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    loss = smoothed_sparse_crossentropy(3, 0.0)
    as_int = loss(tf.constant([[0], [2]]), probs)
    as_soft = loss(tf.one_hot([0, 2], 3), probs)
    np.testing.assert_allclose(as_int.numpy(), as_soft.numpy(), rtol=1e-6)
    blended = loss(tf.constant([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]), probs).numpy()
    np.testing.assert_allclose(blended[0], -(np.log(0.7) + np.log(0.2)) / 2, rtol=1e-5)
    acc = label_accuracy(3)
    acc.update_state(tf.constant([[0.6, 0.4, 0.0], [1.0, 0.0, 0.0]]), probs)  # 2nd is wrong
    assert float(acc.result()) == pytest.approx(0.5)


def test_adamw_cosine_schedule_and_decay_exclusions() -> None:
    from pearlroute.training.trainer import build_optimizer
    cfg = TrainConfig(batch_size=8, epochs=10, learning_rate=1e-3, optimizer="adamw", weight_decay=0.05,
                      lr_schedule="cosine", warmup_epochs=2)
    opt = build_optimizer(cfg, steps_per_epoch=10)
    schedule = opt._learning_rate
    assert float(schedule(0)) == pytest.approx(1e-5)
    assert float(schedule(20)) == pytest.approx(1e-3)          # end of warmup = peak
    assert float(schedule(100)) == pytest.approx(1e-5, rel=1e-3)  # last step ~ 1% of peak
    model = build_model("pearlnet", _model_cfg(), len(CLASSES))
    decayed = {v.name for v in model.trainable_variables if opt._use_weight_decay(v)}
    assert decayed == {"kernel"}
    with pytest.raises(ValueError):
        build_optimizer(cfg)  # cosine without steps per epoch


def test_training_with_cosine_schedule_logs_learning_rate(tmp_path: Path) -> None:
    keras.utils.set_random_seed(0)
    cfg = TrainConfig(batch_size=8, epochs=3, learning_rate=1e-2, optimizer="adamw", weight_decay=0.01,
                      lr_schedule="cosine", warmup_epochs=1, models_dir=tmp_path, figures_dir=tmp_path)
    model = build_model("baseline", _model_cfg(), len(CLASSES))
    history = train_model(model, _toy_data(), _toy_data(), cfg, tmp_path)
    assert len(history) == 3
    lr = history["learning_rate"].to_numpy()
    assert lr[0] == pytest.approx(1e-2) and lr[2] < lr[1] < lr[0]
