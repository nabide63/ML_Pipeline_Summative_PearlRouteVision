"""Queries: save_batch, fetch_unused, mark_used, model versions.

Every function takes an open Session (see db.session.session_scope) so callers control the transaction.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from pearlroute.config import DataConfig
from pearlroute.data.downloader import perceptual_hash
from pearlroute.db.models import ModelVersion, TrainingImage, UploadBatch
from pearlroute.inference.predictor import InvalidImageError, load_rgb


@dataclass
class UploadItem:
    """One photo to store: its class label, raw file bytes, and original file name."""

    label: str
    data: bytes
    filename: str | None = None


@dataclass
class PreparedImage:
    data: bytes
    width: int
    height: int
    sha256: str
    phash: int


def prepare_image(raw: bytes, cfg: DataConfig) -> PreparedImage:
    """Decode (upright, RGB), check it is big enough, shrink to store_max_side and re-encode as JPEG.

    Raises InvalidImageError with a human-readable reason if the photo can't be used.
    """
    img = Image.fromarray(load_rgb(raw))
    if min(img.size) < cfg.cleaning.min_short_side:
        raise InvalidImageError(f"too small ({img.size[0]}x{img.size[1]}); "
                                f"the short side must be at least {cfg.cleaning.min_short_side}px")
    img.thumbnail((cfg.uploads.store_max_side, cfg.uploads.store_max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=cfg.uploads.jpeg_quality)
    data = buf.getvalue()
    return PreparedImage(data, img.width, img.height, hashlib.sha256(data).hexdigest(), perceptual_hash(img))


# Dataset images an upload may NOT duplicate: everything the model was trained, tuned or scored on,
# plus the demo photos. The retrain sample is deliberately allowed: it was held out of every split
# so it can be uploaded to demonstrate retraining.
UPLOADABLE_SPLITS = {"retrain"}


@lru_cache
def dataset_hashes(splits_csv: Path) -> tuple[int, ...]:
    """pHashes of the dataset images that uploads must not duplicate (all splits except the retrain sample)."""
    if not splits_csv.exists():
        return ()
    df = pd.read_csv(splits_csv, usecols=["phash", "split"], dtype=str)
    return tuple(int(h, 16) for h in df.loc[~df["split"].isin(UPLOADABLE_SPLITS), "phash"].dropna())


def _near(phash: int, others: Iterable[int], max_distance: int) -> bool:
    return any((phash ^ o).bit_count() <= max_distance for o in others)


def save_batch(session: Session, items: list[UploadItem], cfg: DataConfig, source: str = "api",
               rejected: list[dict] | None = None) -> UploadBatch:
    """Store the usable photos of one upload; the rest are listed in batch.rejections with a reason.

    Rejected: unknown label, unreadable or too-small image, an exact or near duplicate of a photo
    already uploaded, or a near duplicate of a train/val/test/demo photo (so test photos can't leak
    into training). Photos from data/retrain_sample.zip are accepted: that sample exists to be uploaded.
    `rejected` adds problems the caller already found (e.g. a file with no label) to the same record.
    """
    classes = set(cfg.class_names)
    max_distance = cfg.cleaning.phash_max_distance
    known = list(dataset_hashes(cfg.paths.splits_csv))
    known += [int(h, 16) for h in session.scalars(select(TrainingImage.phash))]
    batch = UploadBatch(source=source)
    rejections: list[dict] = list(rejected or [])
    for item in items:
        reason = None
        if item.label not in classes:
            reason = f"unknown label {item.label!r}; expected one of {sorted(classes)}"
        else:
            try:
                prepared = prepare_image(item.data, cfg)
            except InvalidImageError as exc:
                reason = str(exc)
            else:
                if _near(prepared.phash, known, max_distance):
                    reason = "duplicate of a photo already in the dataset or uploads"
        if reason:
            rejections.append({"filename": item.filename, "reason": reason})
            continue
        known.append(prepared.phash)
        batch.images.append(TrainingImage(label=item.label, filename=item.filename, sha256=prepared.sha256,
                                          phash=f"{prepared.phash:016x}", width=prepared.width,
                                          height=prepared.height, data=prepared.data))
    batch.n_accepted, batch.n_rejected, batch.rejections = len(batch.images), len(rejections), rejections or None
    session.add(batch)
    session.flush()  # assigns ids
    return batch


def fetch_unused(session: Session, label: str | None = None, limit: int | None = None) -> list[TrainingImage]:
    """Uploaded photos no retrained model has used yet, oldest first."""
    query = select(TrainingImage).where(TrainingImage.used_by_id.is_(None)).order_by(TrainingImage.id)
    if label:
        query = query.where(TrainingImage.label == label)
    if limit:
        query = query.limit(limit)
    return list(session.scalars(query))


def count_unused(session: Session) -> dict[str, int]:
    """Unused uploaded photos per label."""
    rows = session.execute(select(TrainingImage.label, func.count()).where(TrainingImage.used_by_id.is_(None))
                           .group_by(TrainingImage.label))
    return {label: n for label, n in rows}


def mark_used(session: Session, image_ids: Iterable[int], model_version: ModelVersion) -> int:
    """Record that these photos went into model_version. Returns how many rows changed."""
    ids = list(image_ids)
    if not ids:
        return 0
    result = session.execute(update(TrainingImage).where(TrainingImage.id.in_(ids))
                             .values(used_by_id=model_version.id))
    return result.rowcount


def add_model_version(session: Session, name: str, *, parent: str | None = None, n_new_images: int = 0,
                      metrics: dict | None = None, notes: str | None = None, active: bool = False) -> ModelVersion:
    """Register a model (val/test scores taken from its metrics.json content if given)."""
    metrics = metrics or {}
    version = ModelVersion(name=name, parent=parent, n_new_images=n_new_images, notes=notes, metrics=metrics or None,
                           val_f1_macro=metrics.get("val", {}).get("f1_macro"),
                           test_accuracy=metrics.get("test", {}).get("accuracy"),
                           test_f1_macro=metrics.get("test", {}).get("f1_macro"))
    session.add(version)
    session.flush()
    if active:
        set_active(session, name)
    return version


def set_active(session: Session, name: str) -> ModelVersion:
    """Make `name` the only active model version."""
    version = session.scalar(select(ModelVersion).where(ModelVersion.name == name))
    if version is None:
        raise KeyError(f"no model version named {name!r}")
    session.execute(update(ModelVersion).values(is_active=False))
    version.is_active = True
    session.flush()
    return version


def get_active(session: Session) -> ModelVersion | None:
    return session.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))


def get_model_version(session: Session, name: str) -> ModelVersion | None:
    return session.scalar(select(ModelVersion).where(ModelVersion.name == name))


def list_model_versions(session: Session) -> list[ModelVersion]:
    """All model versions, newest first."""
    return list(session.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())))


def upload_stats(session: Session) -> dict:
    """Totals for the UI: batches, photos stored, used vs waiting, per label."""
    total = session.scalar(select(func.count()).select_from(TrainingImage)) or 0
    unused = count_unused(session)
    return {"batches": session.scalar(select(func.count()).select_from(UploadBatch)) or 0,
            "images": total, "unused": sum(unused.values()), "used": total - sum(unused.values()),
            "unused_per_label": unused}
