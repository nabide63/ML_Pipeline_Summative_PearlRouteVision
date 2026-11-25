"""Model B: PearlNet, a custom CNN trained from scratch.

    image (160x160x3)
    stem:    conv-BN-ReLU 3x3 /2, conv-BN-ReLU 3x3          -> 80x80
    stage 1: residual block(s) with SE, first one /2         -> 40x40
    stage 2: ...                                             -> 20x20
    stage 3: ...                                             -> 10x10
    stage 4: ...                                             ->  5x5
    head:    global average pooling -> dropout -> softmax

What it changes compared with the baseline, and why:
- Batch normalisation after every convolution: steadier, faster training.
- Residual (skip) connections: deeper network without the training getting stuck.
- Squeeze-and-excitation in every block: channels are reweighted per image (focus on what matters).
- Global average pooling instead of Flatten + a big Dense layer: removes the ~1.6 M weights that
  sit in a single layer of the baseline.
- Weight decay (optional L2 here, or AdamW's decoupled decay set in configs/train.yaml), plus
  dropout before the classifier.

Result on this dataset (see models/*/metrics.json): higher test accuracy and macro F1 than the
baseline, but a wide train-validation gap (84% vs 63% for the Adam + L2 run), i.e. it overfits.
Training therefore adds CutMix/MixUp, AdamW and a cosine learning-rate schedule (see
preprocessing.augmentation and training.trainer).
"""

from __future__ import annotations

import keras
from keras import layers

from pearlroute.config import PearlNetConfig
from pearlroute.models.blocks import conv_bn_relu, residual_block


def build_pearlnet(image_size: int, num_classes: int, cfg: PearlNetConfig) -> keras.Model:
    """PearlNet with stage widths cfg.filters and cfg.blocks residual blocks per stage."""
    if len(cfg.filters) != len(cfg.blocks):
        raise ValueError("pearlnet.filters and pearlnet.blocks must have the same length")
    wd = cfg.weight_decay
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = conv_bn_relu(inputs, cfg.stem_filters, 3, strides=2, weight_decay=wd, name="stem1")
    x = conv_bn_relu(x, cfg.stem_filters, 3, weight_decay=wd, name="stem2")
    for stage, (filters, blocks) in enumerate(zip(cfg.filters, cfg.blocks), start=1):
        for block in range(1, blocks + 1):
            x = residual_block(x, filters, strides=2 if block == 1 else 1, se_ratio=cfg.se_ratio,
                               weight_decay=wd, name=f"stage{stage}_block{block}")
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=keras.regularizers.L2(wd) if wd else None,
                           name="probabilities")(x)
    return keras.Model(inputs, outputs, name="pearlnet")
