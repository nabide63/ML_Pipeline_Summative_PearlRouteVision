"""fit() wrapper that returns the training history."""

from __future__ import annotations

from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from pearlroute.config import TrainConfig
from pearlroute.training.callbacks import HISTORY_FILE, build_callbacks


def smoothed_sparse_crossentropy(num_classes: int, label_smoothing: float):
    """Cross-entropy on integer labels with label smoothing.

    The target for a photo labelled k becomes (1 - s) on class k plus s spread evenly over all classes,
    so the model is never pushed to be 100% sure. That limits how hard it fits wrongly labelled photos.
    (Keras' SparseCategoricalCrossentropy has no label_smoothing option, hence this wrapper.)
    """
    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_classes)
        return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    loss.__name__ = "smoothed_sparse_crossentropy"
    return loss


def compile_model(model: keras.Model, cfg: TrainConfig) -> keras.Model:
    """Adam + cross-entropy on class indices (with label smoothing if configured), tracking accuracy."""
    num_classes = model.output_shape[-1]
    loss = (smoothed_sparse_crossentropy(num_classes, cfg.label_smoothing) if cfg.label_smoothing
            else keras.losses.SparseCategoricalCrossentropy())
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
                  loss=loss,
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: TrainConfig,
    out_dir: Path,
    class_weight: dict[int, float] | None = None,
    epochs: int | None = None,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> pd.DataFrame:
    """Compile and fit with the configured callbacks (always starts from a fresh optimiser).

    Returns the per-epoch history (loss, accuracy, val_loss, val_accuracy, learning_rate), which is
    also written to out_dir/history.csv. The model ends up with the best epoch's weights.
    """
    compile_model(model, cfg)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs or cfg.epochs,
              callbacks=build_callbacks(cfg, out_dir) + list(extra_callbacks or []),
              class_weight=class_weight, verbose=2)
    return load_history(out_dir)


def load_history(out_dir: Path) -> pd.DataFrame:
    """history.csv written during training, indexed by epoch (starting at 1)."""
    history = pd.read_csv(out_dir / HISTORY_FILE)
    history["epoch"] += 1
    return history.set_index("epoch")
