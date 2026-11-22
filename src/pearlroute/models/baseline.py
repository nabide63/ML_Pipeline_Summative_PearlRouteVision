"""Model A: vanilla CNN baseline.

The simplest reasonable image classifier: a stack of convolution + max-pooling blocks that learn
edges, then textures, then parts, followed by one hidden dense layer and a softmax over the classes.
No batch normalisation, residual connections or attention; those are what PearlNet adds, so this
model is the score PearlNet has to beat.

Input: normalised images (see preprocessing.transforms.normalise). Output: class probabilities.
"""

from __future__ import annotations

import keras

from pearlroute.config import BaselineConfig


def build_baseline(image_size: int, num_classes: int, cfg: BaselineConfig) -> keras.Model:
    """Conv(3x3, ReLU) + MaxPool per entry in cfg.filters, then Flatten -> Dense -> Dropout -> softmax."""
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    for i, filters in enumerate(cfg.filters, start=1):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = keras.layers.MaxPooling2D(name=f"pool{i}")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="baseline")
