"""Shared Predictor and DB session dependencies.

Routes ask for these with FastAPI's Depends(), so tests can swap in a tiny model
(app.dependency_overrides[predictor_dependency] = ...) without touching the real one.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from pearlroute.config import ServingConfig, load_model_config
from pearlroute.inference.predictor import Predictor, get_predictor


def predictor_dependency() -> Predictor:
    """The shared, already-loaded Predictor; 503 if no trained model is available."""
    try:
        return get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"No trained model available: {exc}") from exc


def serving_config() -> ServingConfig:
    """Upload limits and other serving settings from configs/model.yaml."""
    return load_model_config().serving
