"""Augmentation block, applied to training data only.

Each training batch gets small random changes a real tourist photo could plausibly have:
mirrored left-right, slightly tilted, zoomed or shifted, brighter or darker, more or less
contrast. The model then sees a slightly different version of every photo each epoch, which
makes it rely on the subject (a raft, a gorilla) rather than memorising exact pixels.

Deliberately left out: vertical flips (skies are up, waterfalls fall down) and hue shifts
(colour is a real cue here: green forest, white water, golden savanna).

Works on float images in [0, 255], before normalisation; loaders.make_dataset applies it there.
"""

from __future__ import annotations

from typing import Callable

import keras
import tensorflow as tf

from pearlroute.config import AugmentationConfig


def build_augmentation(cfg: AugmentationConfig, seed: int | None = None) -> keras.Sequential:
    """The augmentation layers as a Keras model. Layers with a strength of 0 are left out."""
    layers: list[keras.layers.Layer] = []
    if cfg.flip_horizontal:
        layers.append(keras.layers.RandomFlip("horizontal", seed=seed))
    if cfg.rotation:
        layers.append(keras.layers.RandomRotation(cfg.rotation, fill_mode="reflect", seed=seed))
    if cfg.zoom:
        layers.append(keras.layers.RandomZoom(cfg.zoom, fill_mode="reflect", seed=seed))
    if cfg.translation:
        layers.append(keras.layers.RandomTranslation(cfg.translation, cfg.translation,
                                                     fill_mode="reflect", seed=seed))
    if cfg.brightness:
        layers.append(keras.layers.RandomBrightness(cfg.brightness, value_range=(0, 255), seed=seed))
    if cfg.contrast:
        layers.append(keras.layers.RandomContrast(cfg.contrast, value_range=(0, 255), seed=seed))
    return keras.Sequential(layers, name="augmentation")


def make_augment(cfg: AugmentationConfig, seed: int | None = None) -> Callable[[tf.Tensor], tf.Tensor]:
    """A function for loaders' `augment` argument: always in training mode, output kept in [0, 255].

    Keras random layers do nothing unless called with training=True, so the model is wrapped.
    """
    model = build_augmentation(cfg, seed)
    if not model.layers:  # everything turned off; Keras cannot run an empty Sequential
        return lambda images: images

    def augment(images: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(model(images, training=True), 0.0, 255.0)

    return augment
