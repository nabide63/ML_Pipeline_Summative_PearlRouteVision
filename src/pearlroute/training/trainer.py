"""fit() wrapper that returns the training history."""

from __future__ import annotations

from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from pearlroute.config import TrainConfig
from pearlroute.training.callbacks import HISTORY_FILE, build_callbacks


def _dense_labels(y_true, num_classes: int):
    """Labels as (batch, n_classes) floats: integer labels become one-hot, soft labels pass through.

    Validation and test batches carry integer class indices; training batches carry soft labels
    when CutMix/MixUp is on (see data.loaders). The static shape tells them apart.
    """
    if y_true.shape.rank == 2 and y_true.shape[-1] == num_classes and num_classes > 1:
        return tf.cast(y_true, tf.float32)
    return tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_classes)


def smoothed_sparse_crossentropy(num_classes: int, label_smoothing: float):
    """Cross-entropy on integer or soft labels, with optional label smoothing.

    With smoothing s the target becomes (1 - s) times the label plus s spread evenly over all
    classes, so the model is never pushed to be 100% sure. That limits how hard it fits wrongly
    labelled photos. (Keras' SparseCategoricalCrossentropy has no label_smoothing option.)
    """
    def loss(y_true, y_pred):
        return keras.losses.categorical_crossentropy(_dense_labels(y_true, num_classes), y_pred,
                                                     label_smoothing=label_smoothing)
    loss.__name__ = "smoothed_sparse_crossentropy"
    return loss


def label_accuracy(num_classes: int) -> keras.metrics.Metric:
    """Accuracy for integer or soft labels (a mixed photo counts as its larger share's class)."""
    def accuracy(y_true, y_pred):
        y_true = tf.argmax(_dense_labels(y_true, num_classes), axis=-1)
        return tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1)), tf.float32)
    return keras.metrics.MeanMetricWrapper(accuracy, name="accuracy")


def build_optimizer(cfg: TrainConfig, steps_per_epoch: int | None = None) -> keras.optimizers.Optimizer:
    """Adam or AdamW at cfg.learning_rate, or a warmup + cosine schedule peaking there.

    AdamW's weight decay shrinks weights directly each step instead of going through the loss
    as L2 does; with Adam's per-weight step sizes that regularises more evenly. Batch-norm scales
    and shifts and all biases are left out of it.

    The cosine schedule ramps the learning rate up from ~0 over cfg.warmup_epochs, then lowers it
    along half a cosine to ~0 at the last epoch: large steps while the network is far from good,
    small ones at the end, without reacting to every noisy validation epoch.
    """
    lr: float | keras.optimizers.schedules.LearningRateSchedule = cfg.learning_rate
    if cfg.lr_schedule == "cosine":
        if not steps_per_epoch:
            raise ValueError("the cosine schedule needs the number of training steps per epoch")
        warmup = cfg.warmup_epochs * steps_per_epoch
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate * 0.01 if warmup else cfg.learning_rate,
            decay_steps=max(cfg.epochs * steps_per_epoch - warmup, 1), alpha=0.01,
            warmup_target=cfg.learning_rate if warmup else None, warmup_steps=warmup)
    if cfg.optimizer == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay)
        optimizer.exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])
        return optimizer
    return keras.optimizers.Adam(learning_rate=lr)


def compile_model(model: keras.Model, cfg: TrainConfig, steps_per_epoch: int | None = None) -> keras.Model:
    """Optimiser from cfg + cross-entropy (with label smoothing if configured), tracking accuracy."""
    num_classes = model.output_shape[-1]
    model.compile(optimizer=build_optimizer(cfg, steps_per_epoch),
                  loss=smoothed_sparse_crossentropy(num_classes, cfg.label_smoothing),
                  metrics=[label_accuracy(num_classes)])
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
    steps = int(train_ds.cardinality())
    compile_model(model, cfg, steps_per_epoch=steps if steps > 0 else None)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs or cfg.epochs,
              callbacks=build_callbacks(cfg, out_dir) + list(extra_callbacks or []),
              class_weight=class_weight, verbose=2)
    return load_history(out_dir)


def load_history(out_dir: Path) -> pd.DataFrame:
    """history.csv written during training, indexed by epoch (starting at 1)."""
    history = pd.read_csv(out_dir / HISTORY_FILE)
    history["epoch"] += 1
    return history.set_index("epoch")
