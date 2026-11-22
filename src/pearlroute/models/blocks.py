"""Reusable layers: conv_bn_relu, residual block, SE block.

Functional-API helpers: each takes a tensor and returns a tensor, naming its layers with `name`
so model.summary() reads like the architecture.
"""

from __future__ import annotations

import keras
from keras import layers


# Batch norm keeps running averages of each channel's mean/variance for use at prediction time.
# Keras' default momentum (0.99) updates them so slowly that with ~90 steps per epoch they lag far
# behind the weights, and validation scores look much worse than they are; 0.9 keeps them current.
BN_MOMENTUM = 0.9


def _l2(weight_decay: float) -> keras.regularizers.Regularizer | None:
    return keras.regularizers.L2(weight_decay) if weight_decay else None


def conv_bn_relu(x, filters: int, kernel_size: int = 3, strides: int = 1, *, weight_decay: float = 0.0,
                 name: str):
    """Convolution -> batch normalisation -> ReLU.

    Batch norm rescales each channel's outputs to a stable range per batch, which lets the network
    train faster and with less sensitivity to the learning rate. The conv has no bias because
    batch norm's own shift replaces it.
    """
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False,
                      kernel_regularizer=_l2(weight_decay), name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def se_block(x, ratio: int = 8, *, name: str):
    """Squeeze-and-excitation: learn a 0-1 weight per channel from the whole image and rescale by it.

    'Squeeze' averages each channel over the image; 'excite' passes that through a small two-layer
    network ending in a sigmoid. Channels useful for this image get turned up, others down.
    """
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_squeeze")(x)
    s = layers.Dense(max(channels // ratio, 8), activation="relu", name=f"{name}_reduce")(s)
    s = layers.Dense(channels, activation="sigmoid", name=f"{name}_expand")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])


def residual_block(x, filters: int, strides: int = 1, *, se_ratio: int | None = 8, weight_decay: float = 0.0,
                   name: str):
    """Two 3x3 conv-BN layers (+ optional SE) whose output is ADDED to the block's input.

    The shortcut means the block only has to learn a correction to its input, so deep stacks
    still train well. When the block changes size (strides > 1 or new channel count), the
    shortcut is a 1x1 conv + BN so the shapes match for the addition.
    """
    shortcut = x
    y = conv_bn_relu(x, filters, 3, strides, weight_decay=weight_decay, name=f"{name}_a")
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=_l2(weight_decay),
                      name=f"{name}_b_conv")(y)
    y = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_b_bn")(y)
    if se_ratio:
        y = se_block(y, se_ratio, name=f"{name}_se")
    if strides != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, use_bias=False,
                                 kernel_regularizer=_l2(weight_decay), name=f"{name}_proj_conv")(x)
        shortcut = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_proj_bn")(shortcut)
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.ReLU(name=f"{name}_out")(y)
