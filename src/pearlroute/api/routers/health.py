"""GET /health."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends

from pearlroute import __version__
from pearlroute.api.dependencies import predictor_dependency
from pearlroute.api.schemas import HealthResponse, ModelInfo
from pearlroute.inference.predictor import Predictor, loaded_predictor

router = APIRouter(tags=["status"])
_started = time.time()


@router.get("/health", response_model=HealthResponse, summary="Is the API up, and is a model loaded?")
def health() -> HealthResponse:
    """Cheap liveness check: never loads the model itself (the app loads it at startup)."""
    predictor = loaded_predictor()
    return HealthResponse(status="ok", version=__version__, model_loaded=predictor is not None,
                          model=predictor.model_dir.name if predictor else None,
                          uptime_seconds=round(time.time() - _started, 1))


@router.get("/model", response_model=ModelInfo, summary="The model being served and its scores")
def model_info(predictor: Predictor = Depends(predictor_dependency)) -> ModelInfo:
    metrics_file = predictor.model_dir / "metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else None
    return ModelInfo(**predictor.info, metrics=metrics)
