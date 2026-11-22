"""SQLAlchemy tables: training_images, upload_batches, model_versions.

    upload_batches   1 ──< training_images >── 0..1  model_versions
    (one upload)          (one labelled photo)       (the retrained model that first used it)

Photos are stored in the database itself (resized JPEG bytes), not on local disk: a hosted
container's disk is wiped on restart, the database is not.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UploadBatch(Base):
    """One upload request: a set of labelled photos added for retraining."""

    __tablename__ = "upload_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source: Mapped[str] = mapped_column(String(50), default="api")       # api, ui, zip, seed ...
    n_accepted: Mapped[int] = mapped_column(Integer, default=0)
    n_rejected: Mapped[int] = mapped_column(Integer, default=0)
    rejections: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [{"filename", "reason"}]

    images: Mapped[list[TrainingImage]] = relationship(back_populates="batch")


class ModelVersion(Base):
    """A model that has been served or produced by retraining, with its scores."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)          # folder under models/
    parent: Mapped[str | None] = mapped_column(String(100), nullable=True)  # model it was retrained from
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    n_new_images: Mapped[int] = mapped_column(Integer, default=0)
    val_f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)


class TrainingImage(Base):
    """One uploaded, labelled photo, waiting to be used (or already used) for retraining."""

    __tablename__ = "training_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("upload_batches.id"), index=True)
    label: Mapped[str] = mapped_column(String(50), index=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)   # as uploaded
    sha256: Mapped[str] = mapped_column(String(64), unique=True)               # of the stored bytes
    phash: Mapped[str] = mapped_column(String(16), index=True)                 # near-duplicate check
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    data: Mapped[bytes] = mapped_column(LargeBinary)                           # resized JPEG
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    used_by_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True, index=True)

    batch: Mapped[UploadBatch] = relationship(back_populates="images")
    used_by: Mapped[ModelVersion | None] = relationship()
