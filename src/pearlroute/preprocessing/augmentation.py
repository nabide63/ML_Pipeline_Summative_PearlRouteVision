"""Augmentation block, applied to training data only.

Each training batch gets small random changes a real tourist photo could plausibly have:
mirrored left-right, slightly tilted, zoomed or shifted, brighter or darker, more or less
contrast. The model then sees a slightly different version of every photo each epoch, which
makes it rely on the subject (a raft, a gorilla) rather than memorising exact pixels.

Deliberately left out: vertical flips (skies are up, waterfalls fall down) and hue shifts
(colour is a real cue here: green forest, white water, golden savanna).

Works on float images in [0, 255], before normalisation; loaders.make_dataset applies it there.

On top of that, make_mix blends pairs of photos within a batch (MixUp: a see-through overlay of
the whole image; CutMix: a rectangle of one photo pasted onto another) and blends their labels in
the same proportion, e.g. 70% waterfalls / 30% rafting. The model can no longer memorise single
training photos and has to recognise a class from part of a picture, which narrows the gap
between training and validation accuracy when training from scratch.
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


Mix = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


def _sample_beta(alpha: float) -> tf.Tensor:
    """One draw from Beta(alpha, alpha): the share of the batch's own photo that is kept."""
    a = tf.random.gamma([], alpha)
    b = tf.random.gamma([], alpha)
    return a / (a + b)


def _mixup(images: tf.Tensor, labels: tf.Tensor, perm: tf.Tensor, alpha: float):
    lam = _sample_beta(alpha)
    return (lam * images + (1 - lam) * tf.gather(images, perm),
            lam * labels + (1 - lam) * tf.gather(labels, perm))


def _cutmix(images: tf.Tensor, labels: tf.Tensor, perm: tf.Tensor, alpha: float):
    """Paste one random rectangle (same place for the whole batch) from the partner photos.

    The rectangle covers about 1 - lam of the image; it is clipped at the borders, so the label
    weight is recomputed from the area actually pasted.
    """
    height, width = tf.shape(images)[1], tf.shape(images)[2]
    side = tf.sqrt(1.0 - _sample_beta(alpha))
    cut_h = tf.cast(side * tf.cast(height, tf.float32), tf.int32)
    cut_w = tf.cast(side * tf.cast(width, tf.float32), tf.int32)
    cy = tf.random.uniform([], 0, height, dtype=tf.int32)
    cx = tf.random.uniform([], 0, width, dtype=tf.int32)
    y1, y2 = tf.clip_by_value(cy - cut_h // 2, 0, height), tf.clip_by_value(cy + cut_h // 2, 0, height)
    x1, x2 = tf.clip_by_value(cx - cut_w // 2, 0, width), tf.clip_by_value(cx + cut_w // 2, 0, width)
    rows, cols = tf.range(height), tf.range(width)
    inside = ((rows >= y1) & (rows < y2))[:, None] & ((cols >= x1) & (cols < x2))[None, :]
    mask = tf.cast(inside, images.dtype)[None, :, :, None]
    pasted = tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(height * width, tf.float32)
    return (images * (1 - mask) + tf.gather(images, perm) * mask,
            (1 - pasted) * labels + pasted * tf.gather(labels, perm))


def make_mix(cfg: AugmentationConfig, num_classes: int) -> Mix | None:
    """A function (images, integer labels) -> (images, soft one-hot labels) for training batches.

    A share cfg.mix_prob of batches is mixed, by CutMix or MixUp (half each when both are on);
    the rest keep their photos and get plain one-hot labels. None if mixing is turned off.
    Random draws follow TensorFlow's global seed (keras.utils.set_random_seed); no per-op seeds,
    because ops sharing one seed would draw the same numbers.
    """
    modes = [m for m, alpha in (("mixup", cfg.mixup_alpha), ("cutmix", cfg.cutmix_alpha)) if alpha > 0]
    if cfg.mix_prob <= 0 or not modes:
        return None

    def mix(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        soft = tf.one_hot(tf.cast(labels, tf.int32), num_classes, dtype=images.dtype)
        perm = tf.random.shuffle(tf.range(tf.shape(images)[0]))
        mixed = {"mixup": lambda: _mixup(images, soft, perm, cfg.mixup_alpha),
                 "cutmix": lambda: _cutmix(images, soft, perm, cfg.cutmix_alpha)}
        if len(modes) == 2:
            pick = lambda: tf.cond(tf.random.uniform([]) < 0.5, mixed["mixup"], mixed["cutmix"])
        else:
            pick = mixed[modes[0]]
        return tf.cond(tf.random.uniform([]) < cfg.mix_prob, pick, lambda: (images, soft))

    return mix
