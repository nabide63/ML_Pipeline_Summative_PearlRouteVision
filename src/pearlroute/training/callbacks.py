"""Keras callbacks built from configs/train.yaml.

- EarlyStopping: stop when validation loss has not improved for a while, and roll back to the
  best epoch's weights (so the saved model is the best one, not the last one).
- ReduceLROnPlateau: halve the learning rate when validation loss stalls, for finer steps
  (only with lr_schedule: plateau; the cosine schedule sets the rate itself).
- ModelCheckpoint: keep the best model on disk, so an interrupted run is not lost.
- CSVLogger: write loss/accuracy per epoch to history.csv for the learning-curve plots.
"""

from __future__ import annotations

from pathlib import Path

import keras

from pearlroute.config import TrainConfig

HISTORY_FILE = "history.csv"
CHECKPOINT_FILE = "checkpoints/best.keras"


class LogLearningRate(keras.callbacks.Callback):
    """Put the epoch's final learning rate into the logs (and so history.csv) when a schedule sets it."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is not None:
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))


def build_callbacks(cfg: TrainConfig, out_dir: Path) -> list[keras.callbacks.Callback]:
    """Callbacks for model.fit, writing their files into out_dir."""
    (out_dir / CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    if cfg.lr_schedule == "cosine":
        lr_callback: keras.callbacks.Callback = LogLearningRate()
    else:
        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.reduce_lr_factor,
                                                        patience=cfg.reduce_lr_patience,
                                                        min_lr=cfg.min_learning_rate, verbose=1)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience,
                                      restore_best_weights=True, verbose=1),
        lr_callback,
        keras.callbacks.ModelCheckpoint(out_dir / CHECKPOINT_FILE, monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(out_dir / HISTORY_FILE),
    ]
