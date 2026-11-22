"""POST /predict."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from pearlroute.api.dependencies import predictor_dependency, serving_config
from pearlroute.api.schemas import BatchPredictionResponse, FileError, PredictionResponse
from pearlroute.config import ServingConfig
from pearlroute.inference.predictor import InvalidImageError, Predictor

router = APIRouter(prefix="/predict", tags=["predict"])


def read_upload(upload: UploadFile, max_mb: float) -> bytes:
    """The uploaded file's bytes, refusing empty files and files over the size limit."""
    limit = int(max_mb * 1024 * 1024)
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"{upload.filename}: larger than the {max_mb:g} MB limit")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{upload.filename}: empty file")
    return data


# Plain `def` (not async): FastAPI runs these in a worker thread, so a prediction doesn't block the server.
@router.post("", response_model=PredictionResponse, summary="Classify one photo")
def predict(file: UploadFile = File(..., description="A photo (JPEG, PNG, WebP, ...)"),
            predictor: Predictor = Depends(predictor_dependency),
            cfg: ServingConfig = Depends(serving_config)) -> PredictionResponse:
    data = read_upload(file, cfg.max_upload_mb)
    try:
        result = predictor.predict(data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{file.filename}: {exc}") from exc
    return PredictionResponse(filename=file.filename, **result.to_dict())


@router.post("/batch", response_model=BatchPredictionResponse, summary="Classify several photos at once")
def predict_batch(files: list[UploadFile] = File(..., description="Up to serving.max_batch photos"),
                  predictor: Predictor = Depends(predictor_dependency),
                  cfg: ServingConfig = Depends(serving_config)) -> BatchPredictionResponse:
    if len(files) > cfg.max_batch:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"at most {cfg.max_batch} files per request")
    readable: list[tuple[str | None, bytes]] = []
    errors: list[FileError] = []
    for upload in files:
        try:
            data = read_upload(upload, cfg.max_upload_mb)
            predictor.preprocess([data])          # validate now so one bad file doesn't sink the batch
            readable.append((upload.filename, data))
        except HTTPException as exc:
            errors.append(FileError(filename=upload.filename, error=str(exc.detail)))
        except InvalidImageError as exc:
            errors.append(FileError(filename=upload.filename, error=str(exc)))
    results = predictor.predict_batch([data for _, data in readable])
    return BatchPredictionResponse(
        predictions=[PredictionResponse(filename=name, **r.to_dict()) for (name, _), r in zip(readable, results)],
        errors=errors)
