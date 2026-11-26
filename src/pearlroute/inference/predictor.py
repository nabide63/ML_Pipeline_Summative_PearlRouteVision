"""Predictor: load the active model once, predict(image).

    predictor = get_predictor()                  # loads models/<active model> once, then reuses it
    result = predictor.predict("photo.jpg")      # or bytes from an upload, a PIL image, a numpy array
    result.label, result.confidence, result.top_k, result.destinations

Preprocessing matches training exactly: same resize (preprocessing.transforms.resize_to_uint8) and the
normalisation stats stored in the model's own metadata.json, so a model always sees photos prepared
the way it was trained, even after the training set's stats change.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from pearlroute.config import ModelConfig, TrainConfig, load_model_config, load_train_config
from pearlroute.constants import class_destinations, display_name
from pearlroute.models.io import load_model
from pearlroute.preprocessing.transforms import normalise, resize_to_uint8

ImageInput = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]
ACTIVE_FILE = "active.json"   # written by retraining to switch models: {"model": "<folder under models/>"}


class InvalidImageError(ValueError):
    """The input could not be read as an image."""


@dataclass
class Prediction:
    """One photo's result. probabilities covers every class; top_k is the most likely few."""

    label: str
    display_name: str
    confidence: float
    uncertain: bool
    top_k: list[dict[str, Any]]
    probabilities: dict[str, float]
    destinations: list[str] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXIF_ORIENTATION = 0x0112


def _apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Turn a decoded array upright according to its EXIF orientation (same result as PIL.exif_transpose)."""
    ops = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1], 4: lambda a: a[::-1],
           5: lambda a: a.transpose(1, 0, 2), 6: lambda a: np.rot90(a, -1),
           7: lambda a: a.transpose(1, 0, 2)[::-1, ::-1], 8: lambda a: np.rot90(a, 1)}
    return np.ascontiguousarray(ops[orientation](arr)) if orientation in ops else arr


def load_rgb(image: ImageInput) -> np.ndarray:
    """Any supported input -> HxWx3 uint8 RGB array.

    JPEGs (files or bytes) are decoded with TensorFlow, the same decoder the training pipeline uses:
    PIL's JPEG decoder rounds slightly differently (pixels off by 1-3), which would be a small
    train/serve mismatch. Everything else goes through PIL. Phone-camera EXIF rotation is applied,
    and transparency (PNG/WebP with alpha) is flattened onto white.
    """
    if isinstance(image, (str, Path, bytes, bytearray)):
        data = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
        try:
            img = Image.open(io.BytesIO(data))
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"could not read image: {type(exc).__name__}") from exc
        if img.format == "JPEG" and img.mode in ("RGB", "L"):
            try:
                arr = tf.io.decode_jpeg(data, channels=3).numpy()
            except tf.errors.InvalidArgumentError as exc:
                raise InvalidImageError("could not decode JPEG") from exc
            return _apply_orientation(arr, img.getexif().get(EXIF_ORIENTATION, 1))
        image = img
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise InvalidImageError(f"expected an HxWx3 image array, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr[..., :3]
    if not isinstance(image, Image.Image):
        raise InvalidImageError(f"unsupported input type {type(image).__name__}")
    img = image
    try:
        img.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"could not read image: {type(exc).__name__}") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    return np.asarray(img.convert("RGB"))


def resolve_model_dir(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Path:
    """models/active.json's choice if present, else serving.active_model from configs/model.yaml."""
    models_dir = train_cfg.models_dir
    name = model_cfg.serving.active_model
    pointer = models_dir / ACTIVE_FILE
    if pointer.exists():
        name = json.loads(pointer.read_text(encoding="utf-8"))["model"]
    return models_dir / name


class Predictor:
    """A loaded model plus everything needed to prepare photos for it and explain its answers."""

    def __init__(self, model_dir: Path | None = None, model_cfg: ModelConfig | None = None,
                 train_cfg: TrainConfig | None = None) -> None:
        model_cfg = model_cfg or load_model_config()
        self.model_dir = Path(model_dir) if model_dir else resolve_model_dir(model_cfg, train_cfg or load_train_config())
        self.model, self.metadata = load_model(self.model_dir)
        self.class_names: list[str] = self.metadata["class_names"]
        self.image_size: int = self.metadata["image_size"]
        self.mean = self.metadata["normalisation"]["mean"]
        self.std = self.metadata["normalisation"]["std"]
        self.top_k = min(model_cfg.serving.top_k, len(self.class_names))
        self.min_confidence = model_cfg.serving.min_confidence
        self.destinations = class_destinations()
        # One compiled graph for any batch size: ~6x faster per call on CPU than calling the
        # Keras model eagerly (26 vs 156 ms per photo in the container), with identical outputs.
        model = self.model
        self._infer = tf.function(
            tf.autograph.experimental.do_not_convert(lambda batch: model(batch, training=False)),
            input_signature=[tf.TensorSpec([None, self.image_size, self.image_size, 3], tf.float32)])

    @property
    def info(self) -> dict[str, Any]:
        """What is being served, for the API's health/model endpoints."""
        keys = ("name", "test_accuracy", "test_f1_macro", "best_epoch", "parameters")
        return {"model_dir": self.model_dir.name, "class_names": self.class_names,
                "image_size": self.image_size, **{k: self.metadata.get(k) for k in keys}}

    def preprocess(self, images: Sequence[ImageInput]) -> tf.Tensor:
        """Photos -> one normalised float batch (N, size, size, 3), as the model was trained on."""
        batch = [resize_to_uint8(tf.constant(load_rgb(img)), self.image_size) for img in images]
        return normalise(tf.stack(batch), self.mean, self.std)

    def predict_proba(self, images: Sequence[ImageInput]) -> np.ndarray:
        """Class probabilities, shape (N, n_classes), columns in self.class_names order."""
        if not images:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)
        return self._infer(self.preprocess(images)).numpy()

    def predict(self, image: ImageInput) -> Prediction:
        """The prediction for one photo."""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Sequence[ImageInput]) -> list[Prediction]:
        """Predictions for several photos in one model call."""
        return [self._explain(p) for p in self.predict_proba(images)]

    def _explain(self, probs: np.ndarray) -> Prediction:
        order = np.argsort(probs)[::-1]
        best = self.class_names[order[0]]
        confidence = float(probs[order[0]])
        return Prediction(
            label=best,
            display_name=display_name(best),
            confidence=round(confidence, 4),
            uncertain=confidence < self.min_confidence,
            top_k=[{"label": self.class_names[i], "display_name": display_name(self.class_names[i]),
                    "confidence": round(float(probs[i]), 4)} for i in order[: self.top_k]],
            probabilities={c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
            destinations=self.destinations.get(best, []),
            model=self.model_dir.name,
        )


_predictor: Predictor | None = None
_lock = threading.Lock()


def get_predictor() -> Predictor:
    """The shared Predictor, created on first use (loading a model takes a few seconds)."""
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                _predictor = Predictor()
    return _predictor


def loaded_predictor() -> Predictor | None:
    """The shared Predictor if it has been loaded already, else None (never triggers a load)."""
    return _predictor


def reload_predictor(model_dir: Path | None = None) -> Predictor:
    """Load a (new) model and make it the shared one, e.g. after retraining."""
    global _predictor
    new = Predictor(model_dir)
    with _lock:
        _predictor = new
    return new
