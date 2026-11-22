"""FastAPI app factory.

Run locally:
    uvicorn pearlroute.api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive documentation.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pearlroute import __version__
from pearlroute.api.routers import health, predict
from pearlroute.inference.predictor import get_predictor
from pearlroute.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

DESCRIPTION = """
Classifies the kind of Uganda tourism experience in a photo (gorilla trekking, wildlife safari,
whitewater rafting, waterfalls, mountain hiking, cultural heritage) with PearlNet, a CNN trained
from scratch, and suggests destinations for it.
"""


def create_app(preload_model: bool | None = None) -> FastAPI:
    """Build the app. preload_model loads the model at startup (default: on, unless
    PEARLROUTE_PRELOAD_MODEL=0), so the first prediction isn't slow."""
    if preload_model is None:
        preload_model = os.environ.get("PEARLROUTE_PRELOAD_MODEL", "1") != "0"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging("INFO")
        if preload_model:
            try:
                predictor = get_predictor()
                log.info("serving model %s", predictor.model_dir.name)
            except FileNotFoundError as exc:
                log.warning("no model loaded at startup: %s", exc)
        yield

    app = FastAPI(title="PearlRoute Vision API", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(predict.router)
    return app


app = create_app()
