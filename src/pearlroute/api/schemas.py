"""Pydantic request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str = Field(examples=["whitewater_rafting"])
    display_name: str = Field(examples=["Whitewater rafting"])
    confidence: float = Field(ge=0, le=1, examples=[0.91])


class PredictionResponse(BaseModel):
    """The model's answer for one photo."""

    filename: str | None = Field(default=None, examples=["raft.jpg"])
    label: str = Field(examples=["whitewater_rafting"])
    display_name: str = Field(examples=["Whitewater rafting"])
    confidence: float = Field(ge=0, le=1, examples=[0.91])
    uncertain: bool = Field(description="True when confidence is below the serving.min_confidence threshold")
    top_k: list[ClassScore]
    probabilities: dict[str, float] = Field(description="Probability for every class (sums to 1)")
    destinations: list[str] = Field(description="Uganda destinations for the predicted experience",
                                    examples=[["Jinja (the Nile)"]])
    model: str = Field(examples=["pearlnet"])


class FileError(BaseModel):
    filename: str | None
    error: str


class BatchPredictionResponse(BaseModel):
    """One entry per readable photo; unreadable ones are listed in errors instead of failing the batch."""

    predictions: list[PredictionResponse]
    errors: list[FileError] = []


class ModelInfo(BaseModel):
    """What the API is serving and how well it scored."""

    model_dir: str
    name: str | None = None
    class_names: list[str]
    image_size: int
    parameters: int | None = None
    best_epoch: int | None = None
    test_accuracy: float | None = None
    test_f1_macro: float | None = None
    metrics: dict[str, Any] | None = Field(default=None, description="metrics.json: val/test scores, per class")


class Rejection(BaseModel):
    filename: str | None
    reason: str


class UploadResponse(BaseModel):
    """What happened to an upload: stored photos are waiting to be used by the next retraining."""

    batch_id: int
    accepted: int
    rejected: int
    rejections: list[Rejection] = []
    accepted_per_label: dict[str, int] = Field(examples=[{"waterfalls": 12}])
    waiting_for_retraining: dict[str, int] = Field(description="Unused uploaded photos per class, all batches")


class UploadStats(BaseModel):
    batches: int
    images: int
    used: int
    unused: int
    unused_per_label: dict[str, int]


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    version: str
    model_loaded: bool
    model: str | None = Field(default=None, examples=["pearlnet"])
    uptime_seconds: float
