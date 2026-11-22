"""Resize and normalise: the single source of truth for preprocessing.

Training (tf.data), prediction and retraining all go through these functions, so an image is
resized the same way everywhere. Images are resized to a square without cropping (the aspect
ratio is squashed, nothing is cut off), with antialiasing so downscaled photos stay smooth.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


def resize(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Resize an HxWx3 (or NxHxWx3) image to image_size x image_size; returns float32 in [0, 255]."""
    image = tf.image.resize(tf.cast(image, tf.float32), (image_size, image_size),
                            method="bilinear", antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def resize_to_uint8(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """resize() then round back to whole pixel values (uint8), exactly as training images are stored."""
    image = tf.cast(tf.round(resize(image, image_size)), tf.uint8)
    image.set_shape((*image.shape[:-3], image_size, image_size, 3))
    return image


def read_and_resize(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Read an image file (JPEG/PNG/GIF/BMP) and resize it; returns uint8, compact enough to cache."""
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return resize_to_uint8(image, image_size)


def normalise(image: tf.Tensor, mean: Sequence[float], std: Sequence[float]) -> tf.Tensor:
    """Per-channel standardisation of a [0, 255] image: (x - mean) / std, with stats in the same units."""
    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t = tf.constant(std, dtype=tf.float32)
    return (tf.cast(image, tf.float32) - mean_t) / std_t
