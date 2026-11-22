"""Tests for pearlroute.inference."""

from __future__ import annotations

import io
import json
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from pearlroute.config import BaselineConfig, ModelConfig, ServingConfig, TrainConfig
from pearlroute.data.loaders import make_dataset
from pearlroute.inference.predictor import (InvalidImageError, Predictor, load_rgb, resolve_model_dir)
from pearlroute.models.io import save_model
from pearlroute.models.registry import build_model

SIZE = 32
CLASSES = ["cultural_heritage", "waterfalls", "wildlife_safari"]
MEAN, STD = [120.0, 110.0, 100.0], [60.0, 55.0, 50.0]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    keras.utils.set_random_seed(0)
    cfg = ModelConfig(image_size=SIZE, baseline=BaselineConfig(filters=[4], dense_units=8, dropout=0.0))
    out = tmp_path_factory.mktemp("models") / "tiny"
    save_model(build_model("baseline", cfg, len(CLASSES)), out, CLASSES, SIZE, MEAN, STD, test_accuracy=0.5)
    return out


@pytest.fixture(scope="module")
def predictor(model_dir: Path) -> Predictor:
    cfg = ModelConfig(image_size=SIZE, serving=ServingConfig(top_k=2, min_confidence=0.9))
    return Predictor(model_dir, model_cfg=cfg)


def _photo(path: Path, size=(70, 45), colour=(200, 120, 40)) -> Path:
    rng = np.random.default_rng(1)
    arr = (np.array(colour) + rng.integers(-30, 30, (size[1], size[0], 3))).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path, quality=95)
    return path


def test_prediction_fields(predictor: Predictor, tmp_path: Path) -> None:
    result = predictor.predict(_photo(tmp_path / "a.jpg"))
    assert result.label in CLASSES and result.model == "tiny"
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert [t["label"] for t in result.top_k][0] == result.label and len(result.top_k) == 2
    assert result.top_k[0]["confidence"] >= result.top_k[1]["confidence"]
    assert result.uncertain == (result.confidence < 0.9)
    assert result.display_name == result.label.replace("_", " ").capitalize()
    assert result.destinations                                    # real class names -> real destinations
    assert json.dumps(result.to_dict())                            # API-ready


def test_preprocessing_matches_training_pipeline(predictor: Predictor, tmp_path: Path) -> None:
    """No train/serve skew: a file prepared by the predictor equals the same file from the tf.data loader."""
    path = _photo(tmp_path / "b.jpg", size=(123, 77))
    from_loader, _ = next(iter(make_dataset([str(path)], [0], SIZE, 1, training=False, seed=0, stats=(MEAN, STD))))
    np.testing.assert_allclose(predictor.preprocess([path]).numpy(), from_loader.numpy(), atol=1e-4)


def test_all_input_types_agree(predictor: Predictor, tmp_path: Path) -> None:
    path = _photo(tmp_path / "c.png", colour=(30, 160, 90))     # PNG: lossless, so all forms are identical
    arr = np.asarray(Image.open(path).convert("RGB"))
    inputs = [path, str(path), path.read_bytes(), Image.open(path), arr]
    probs = predictor.predict_proba(inputs)
    for row in probs[1:]:
        np.testing.assert_allclose(row, probs[0], atol=1e-5)
    assert [p.label for p in predictor.predict_batch(inputs)] == [predictor.predict(path).label] * 5


def test_load_rgb_handles_alpha_greyscale_webp_and_rotation(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (10, 6), (255, 0, 0, 0))              # fully transparent -> white
    assert load_rgb(rgba)[0, 0].tolist() == [255, 255, 255]
    assert load_rgb(Image.new("L", (10, 6), 77)).shape == (6, 10, 3)
    buf = io.BytesIO(); Image.new("RGB", (10, 6), (0, 0, 255)).save(buf, "WEBP"); webp = buf.getvalue()
    assert load_rgb(webp).shape == (6, 10, 3)
    rotated = Image.new("RGB", (10, 6), (0, 255, 0))
    exif = rotated.getexif(); exif[0x0112] = 6                      # "rotate 90" as phones store it
    buf = io.BytesIO(); rotated.save(buf, "JPEG", exif=exif)
    assert load_rgb(buf.getvalue()).shape == (10, 6, 3)            # upright: height and width swapped


def test_invalid_input_raises(predictor: Predictor) -> None:
    with pytest.raises(InvalidImageError):
        predictor.predict(b"not an image")
    with pytest.raises(InvalidImageError):
        predictor.predict(np.zeros((4, 4, 7)))
    assert predictor.predict_proba([]).shape == (0, 3)


def test_active_model_pointer(tmp_path: Path) -> None:
    train_cfg = TrainConfig(batch_size=1, models_dir=tmp_path)
    cfg = ModelConfig(image_size=SIZE)
    assert resolve_model_dir(cfg, train_cfg) == tmp_path / "pearlnet"     # config default
    (tmp_path / "active.json").write_text(json.dumps({"model": "pearlnet_retrained"}))
    assert resolve_model_dir(cfg, train_cfg) == tmp_path / "pearlnet_retrained"


def test_info(predictor: Predictor) -> None:
    info = predictor.info
    assert info["class_names"] == CLASSES and info["image_size"] == SIZE and info["test_accuracy"] == 0.5


@pytest.mark.skipif(not Path("models/pearlnet/model.keras").exists(), reason="trained model not present")
def test_real_model_loads_and_predicts() -> None:
    from pearlroute.inference.predictor import get_predictor
    p = get_predictor()
    assert p is get_predictor()                                         # loaded once
    assert p.image_size == 160 and len(p.class_names) == 6
    result = p.predict(np.full((200, 300, 3), 128, np.uint8))
    assert 0 < result.confidence <= 1


@pytest.mark.parametrize("orientation", range(1, 9))
def test_jpeg_orientation_matches_pil(orientation: int) -> None:
    from PIL import ImageOps
    arr = np.zeros((6, 10, 3), np.uint8); arr[0, 0] = 255; arr[0, 9] = (0, 255, 0); arr[5, 0] = (0, 0, 255)
    img = Image.fromarray(arr); exif = img.getexif(); exif[0x0112] = orientation
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=100, exif=exif); data = buf.getvalue()
    expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    got = load_rgb(data)
    assert got.shape == expected.shape
    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 12    # same layout; decoders differ slightly
