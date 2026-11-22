"""build_model(name, cfg) lookup."""

from __future__ import annotations

from typing import Callable

import keras

from pearlroute.config import ModelConfig
from pearlroute.models.baseline import build_baseline

_BUILDERS: dict[str, Callable[[ModelConfig, int], keras.Model]] = {
    "baseline": lambda cfg, n: build_baseline(cfg.image_size, n, cfg.baseline),
}


def available_models() -> list[str]:
    """Names accepted by build_model."""
    return sorted(_BUILDERS)


def build_model(name: str, cfg: ModelConfig, num_classes: int) -> keras.Model:
    """Build an uncompiled model by name, with its settings from configs/model.yaml."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; choose from {available_models()}")
    return _BUILDERS[name](cfg, num_classes)
