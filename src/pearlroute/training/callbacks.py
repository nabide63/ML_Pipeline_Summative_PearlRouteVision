"""Keras callbacks built from configs/train.yaml.

- EarlyStopping: stop when validation loss has not improved for a while, and roll back to the
  best epoch's weights (so the saved model is the best one, not the last one).
- ReduceLROnPlateau: halve the learning rate when validation loss stalls, for finer steps.
- ModelCheckpoint: keep the best model on disk, so an interrupted run is not lost.
- CSVLogger: write loss/accuracy per epoch to history.csv for the learning-curve plots.
"""

from __future__ import annotations

from pathlib import Path

import keras

from pearlroute.config import TrainConfig

HISTORY_FILE = "history.csv"
CHECKPOINT_FILE = "checkpoints/best.keras"


def build_callbacks(cfg: TrainConfig, out_dir: Path) -> list[keras.callbacks.Callback]:
    """Callbacks for model.fit, writing their files into out_dir."""
    (out_dir / CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.reduce_lr_factor,
                                          patience=cfg.reduce_lr_patience, min_lr=cfg.min_learning_rate,
                                          verbose=1),
        keras.callbacks.ModelCheckpoint(out_dir / CHECKPOINT_FILE, monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(out_dir / HISTORY_FILE),
    ]
