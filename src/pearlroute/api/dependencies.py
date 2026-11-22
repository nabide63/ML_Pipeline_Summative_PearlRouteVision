"""Shared Predictor and DB session dependencies.

Routes ask for these with FastAPI's Depends(), so tests can swap in a tiny model
(app.dependency_overrides[predictor_dependency] = ...) without touching the real one.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from pearlroute.config import DataConfig, ServingConfig, load_data_config, load_model_config
from pearlroute.db.session import DatabaseNotConfigured, get_sessionmaker
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


def data_config() -> DataConfig:
    """configs/data.yaml: class names, cleaning thresholds, upload settings."""
    return load_data_config()


def db_session() -> Iterator[Session]:
    """A database session for one request. Routes commit explicitly; anything uncommitted is rolled back.
    503 if DATABASE_URL isn't set."""
    try:
        maker = get_sessionmaker()
    except DatabaseNotConfigured as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    session = maker()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
