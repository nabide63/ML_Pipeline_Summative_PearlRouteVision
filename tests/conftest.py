"""Shared fixtures: a tiny saved model and an API client that serves it (fast, no real model needed)."""

from __future__ import annotations

import io
from pathlib import Path

import keras
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pearlroute.config import BaselineConfig, ModelConfig, ServingConfig
from pearlroute.models.io import save_model
from pearlroute.models.registry import build_model

TINY_CLASSES = ["cultural_heritage", "waterfalls", "wildlife_safari"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    keras.utils.set_random_seed(0)
    cfg = ModelConfig(image_size=32, baseline=BaselineConfig(filters=[4], dense_units=8, dropout=0.0))
    out = tmp_path_factory.mktemp("models") / "tiny"
    save_model(build_model("baseline", cfg, len(TINY_CLASSES)), out, TINY_CLASSES, 32,
               [120.0, 110.0, 100.0], [60.0, 55.0, 50.0], test_accuracy=0.5, test_f1_macro=0.5)
    (out / "metrics.json").write_text('{"test": {"accuracy": 0.5}}', encoding="utf-8")
    return out


@pytest.fixture
def api_client(tiny_model_dir: Path) -> TestClient:
    """TestClient whose /predict uses the tiny model, with small upload limits."""
    from pearlroute.api.dependencies import predictor_dependency, serving_config
    from pearlroute.api.main import create_app
    from pearlroute.inference.predictor import Predictor

    serving = ServingConfig(top_k=2, min_confidence=0.4, max_upload_mb=0.05, max_batch=3)
    predictor = Predictor(tiny_model_dir, model_cfg=ModelConfig(image_size=32, serving=serving))
    app = create_app(preload_model=False)
    app.dependency_overrides[predictor_dependency] = lambda: predictor
    app.dependency_overrides[serving_config] = lambda: serving
    return TestClient(app)


def jpeg_bytes(size=(64, 48), colour=(30, 140, 200)) -> bytes:
    rng = np.random.default_rng(0)
    arr = (np.array(colour) + rng.integers(-20, 20, (size[1], size[0], 3))).clip(0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()
