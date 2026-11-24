"""Class names and the class-to-destination map.

Both come from configs/data.yaml (the single place they are defined); these helpers just give the
rest of the code (predictor, API, UI) one import for them.
"""

from __future__ import annotations

from pearlroute.config import load_data_config


def class_names() -> list[str]:
    """Class names in model output order (sorted)."""
    return load_data_config().class_names


def class_destinations() -> dict[str, list[str]]:
    """East African destinations where each experience can be had, e.g. waterfalls -> Sipi Falls (Uganda), ..."""
    cfg = load_data_config()
    return {name: list(cfg.classes[name].destinations) for name in cfg.class_names}


def display_name(class_name: str) -> str:
    """'whitewater_rafting' -> 'Whitewater rafting'."""
    return class_name.replace("_", " ").capitalize()
