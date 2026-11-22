"""POST /retrain and GET /retrain/{job_id}."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from pearlroute.api.dependencies import db_session
from pearlroute.api.schemas import ModelVersionOut, RetrainJobOut
from pearlroute.config import load_train_config
from pearlroute.db import repository as repo
from pearlroute.inference.predictor import reload_predictor
from pearlroute.retraining import pipeline

router = APIRouter(prefix="/retrain", tags=["retrain"])


@router.post("", response_model=RetrainJobOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Fine-tune the active model on the uploaded photos (runs in the background)")
def start(session: Session = Depends(db_session)) -> RetrainJobOut:
    # Fail fast with a clear message instead of starting a job that stops at step 1.
    waiting = sum(repo.count_unused(session).values())
    need = load_train_config().retrain.min_new_images
    if waiting < need:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"only {waiting} new photos waiting; upload at least {need} to retrain")
    try:
        job = pipeline.start_retraining(on_promote=reload_predictor)
    except pipeline.RetrainAlreadyRunning as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return RetrainJobOut(**job.to_dict())


@router.get("/latest", response_model=RetrainJobOut | None, summary="The most recent job this server ran")
def latest() -> RetrainJobOut | None:
    job = pipeline.latest_job()
    return RetrainJobOut(**job.to_dict()) if job else None


@router.get("/versions", response_model=list[ModelVersionOut], summary="Every model version, newest first")
def versions(session: Session = Depends(db_session)) -> list[ModelVersionOut]:
    return [ModelVersionOut.model_validate(v, from_attributes=True) for v in repo.list_model_versions(session)]


@router.get("/{job_id}", response_model=RetrainJobOut, summary="Progress and result of a retraining job")
def job_status(job_id: str) -> RetrainJobOut:
    job = pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no retraining job {job_id} on this server")
    return RetrainJobOut(**job.to_dict())
