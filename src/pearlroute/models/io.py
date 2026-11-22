"""Save/load a model with its class names and normalisation stats.

A saved model is a folder:
    model.keras     the trained Keras model
    metadata.json   class names (in output order), image size, normalisation stats, and any extras
                    such as test metrics, so prediction needs nothing else to use the model correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import keras

MODEL_FILE = "model.keras"
METADATA_FILE = "metadata.json"


def save_model(model: keras.Model, out_dir: Path, class_names: list[str], image_size: int,
               mean: list[float], std: list[float], **extra: Any) -> Path:
    """Write model.keras and metadata.json into out_dir. Returns out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / MODEL_FILE)
    metadata = {"name": model.name, "class_names": class_names, "image_size": image_size,
                "normalisation": {"mean": mean, "std": std}, **extra}
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
                                         newline="\n")
    return out_dir


def load_metadata(model_dir: Path) -> dict[str, Any]:
    """metadata.json of a saved model folder."""
    return json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))


def load_model(model_dir: Path) -> tuple[keras.Model, dict[str, Any]]:
    """(model, metadata) from a folder written by save_model."""
    if not (model_dir / MODEL_FILE).exists():
        raise FileNotFoundError(f"no {MODEL_FILE} in {model_dir}; train the model first")
    # compile=False: loading is for prediction; the training loss (e.g. label smoothing) isn't needed.
    return keras.models.load_model(model_dir / MODEL_FILE, compile=False), load_metadata(model_dir)
