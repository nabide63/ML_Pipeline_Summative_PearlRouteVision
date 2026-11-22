"""fit() wrapper that returns the training history."""

from __future__ import annotations

from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from pearlroute.config import TrainConfig
from pearlroute.training.callbacks import HISTORY_FILE, build_callbacks


def compile_model(model: keras.Model, cfg: TrainConfig) -> keras.Model:
    """Adam + sparse categorical cross-entropy (labels are class indices), tracking accuracy."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
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
) -> pd.DataFrame:
    """Compile and fit with the configured callbacks (always starts from a fresh optimiser).

    Returns the per-epoch history (loss, accuracy, val_loss, val_accuracy, learning_rate), which is
    also written to out_dir/history.csv. The model ends up with the best epoch's weights.
    """
    compile_model(model, cfg)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs or cfg.epochs,
              callbacks=build_callbacks(cfg, out_dir), class_weight=class_weight, verbose=2)
    return load_history(out_dir)


def load_history(out_dir: Path) -> pd.DataFrame:
    """history.csv written during training, indexed by epoch (starting at 1)."""
    history = pd.read_csv(out_dir / HISTORY_FILE)
    history["epoch"] += 1
    return history.set_index("epoch")
