"""POST /upload.

Add labelled photos for the next retraining, either way:
  - photos + one `label` for all of them (e.g. five raft photos, label=whitewater_rafting)
  - a zip with one folder per class (waterfalls/x.jpg, wildlife_safari/y.jpg), like data/retrain_sample.zip;
    a single top folder around them (my_photos/waterfalls/x.jpg) is fine too
Photos are checked and stored in the database (see db.repository.save_batch); every rejected
photo comes back with a reason.
"""

from __future__ import annotations

import io
import zipfile
from collections import Counter
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from pearlroute.api.dependencies import data_config, db_session, serving_config
from pearlroute.api.routers.predict import read_upload
from pearlroute.api.schemas import UploadResponse, UploadStats
from pearlroute.config import DataConfig, ServingConfig
from pearlroute.db import repository as repo

router = APIRouter(prefix="/upload", tags=["upload"])


def is_zip(upload: UploadFile) -> bool:
    return (upload.filename or "").lower().endswith(".zip") or upload.content_type in (
        "application/zip", "application/x-zip-compressed")


def items_from_zip(data: bytes, cfg: DataConfig, default_label: str | None = None
                   ) -> tuple[list[repo.UploadItem], list[dict]]:
    """Photos inside a zip, labelled by the class folder they sit in. Returns (items, rejections).

    Raises HTTPException 400/413 for a broken zip or one that would expand too much.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "not a valid zip file") from exc
    entries = [i for i in archive.infolist() if not i.is_dir()]
    unzipped_mb = sum(i.file_size for i in entries) / 1024 / 1024
    if unzipped_mb > cfg.uploads.max_unzipped_mb:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"zip expands to {unzipped_mb:.0f} MB; the limit is {cfg.uploads.max_unzipped_mb:g} MB")
    classes = set(cfg.class_names)
    items: list[repo.UploadItem] = []
    rejections: list[dict] = []
    for info in entries:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if any(part.startswith((".", "__MACOSX")) for part in path.parts):
            continue                                            # macOS/hidden system files
        if path.suffix.lower() not in cfg.cleaning.allowed_extensions:
            rejections.append({"filename": str(path), "reason": f"not an image file ({path.suffix or 'no extension'})"})
            continue
        label = next((part for part in path.parts[:-1] if part in classes), None) or default_label
        if label is None:
            rejections.append({"filename": str(path), "reason": "not inside a class folder "
                               f"(expected one of {sorted(classes)})"})
            continue
        items.append(repo.UploadItem(label=label, data=archive.read(info), filename=str(path)))
    return items, rejections


# Plain `def`: decoding and resizing photos is CPU work, so FastAPI runs it in a worker thread.
@router.post("", response_model=UploadResponse, summary="Add labelled photos for retraining")
def upload(files: list[UploadFile] = File(..., description="Photos, and/or zips with one folder per class"),
           label: str | None = Form(None, description="Class for loose photos, e.g. waterfalls"),
           session: Session = Depends(db_session),
           cfg: DataConfig = Depends(data_config),
           serving: ServingConfig = Depends(serving_config)) -> UploadResponse:
    items: list[repo.UploadItem] = []
    rejections: list[dict] = []
    for f in files:
        try:
            if is_zip(f):
                zip_items, zip_rejections = items_from_zip(read_upload(f, cfg.uploads.max_zip_mb), cfg, label)
                items += zip_items
                rejections += zip_rejections
            elif not label:
                rejections.append({"filename": f.filename, "reason": "no label: send label=<class>, "
                                   "or upload a zip with one folder per class"})
            else:
                items.append(repo.UploadItem(label=label, data=read_upload(f, serving.max_upload_mb),
                                             filename=f.filename))
        except HTTPException as exc:
            if is_zip(f):
                raise                                           # a broken/oversized zip fails the request
            rejections.append({"filename": f.filename, "reason": str(exc.detail)})
    if len(items) > cfg.uploads.max_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"{len(items)} photos; at most {cfg.uploads.max_files} per upload")
    if not items and not rejections:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no photos found in the upload")

    batch = repo.save_batch(session, items, cfg, source="api", rejected=rejections)
    session.commit()
    return UploadResponse(batch_id=batch.id, accepted=batch.n_accepted, rejected=batch.n_rejected,
                          rejections=batch.rejections or [],
                          accepted_per_label=dict(Counter(i.label for i in batch.images)),
                          waiting_for_retraining=repo.count_unused(session))


@router.get("/stats", response_model=UploadStats, summary="Uploaded photos: used vs waiting, per class")
def upload_stats(session: Session = Depends(db_session)) -> UploadStats:
    return UploadStats(**repo.upload_stats(session))
