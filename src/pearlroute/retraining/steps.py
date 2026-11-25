"""One function per retraining step (7 steps).

Each step takes the run's RetrainContext, does one job, and stores its result on the context:

    1 check_new_data       uploaded photos not used yet; stop if fewer than retrain.min_new_images
    2 export_new_data      write them as JPEGs into <work_dir>/<run>/new/<class>/
    3 load_parent          the active model, its metrics and its own normalisation stats
    4 build_datasets       train = original train split + new photos; val/test unchanged
    5 fine_tune            continue training the parent at a low learning rate
    6 evaluate_and_compare same val/test as the parent; promote if val macro F1 is not worse
    7 save_and_register    save models/<name>/, record it in the database; if promoted, make it
                           active, mark the photos used and switch the live predictor
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import keras
import pandas as pd
from sqlalchemy.orm import Session

from pearlroute.config import DataConfig, ModelConfig, TrainConfig
from pearlroute.data.loaders import list_images, make_dataset
from pearlroute.db import repository as repo
from pearlroute.evaluation.metrics import evaluate_model
from pearlroute.inference.predictor import ACTIVE_FILE, resolve_model_dir
from pearlroute.models.io import load_model, save_model
from pearlroute.preprocessing.augmentation import make_augment
from pearlroute.training.trainer import train_model


class RetrainError(RuntimeError):
    """A step can't continue (e.g. not enough new photos); the message is shown to the user."""


@dataclass
class RetrainContext:
    """Everything a retraining run needs and produces; steps fill it in as they go."""

    data_cfg: DataConfig
    model_cfg: ModelConfig
    train_cfg: TrainConfig
    session: Session
    run_id: str
    on_promote: Callable[[Path], Any] | None = None          # e.g. reload the API's predictor
    on_epoch: Callable[[int, dict], Any] | None = None       # progress reporting

    images: list = field(default_factory=list)               # TrainingImage rows
    new_dir: Path | None = None
    parent_dir: Path | None = None
    parent_name: str = ""
    parent_metrics: dict = field(default_factory=dict)
    stats: tuple | None = None
    model: keras.Model | None = None
    datasets: dict = field(default_factory=dict)
    history: pd.DataFrame | None = None
    out_dir: Path | None = None
    name: str = ""
    val: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    promoted: bool = False
    decision: str = ""

    @property
    def run_dir(self) -> Path:
        return self.train_cfg.retrain.work_dir / self.run_id


def check_new_data(ctx: RetrainContext) -> str:
    ctx.images = repo.fetch_unused(ctx.session)
    need = ctx.train_cfg.retrain.min_new_images
    if len(ctx.images) < need:
        raise RetrainError(f"only {len(ctx.images)} new photos waiting; upload at least {need} to retrain")
    per_label = pd.Series([i.label for i in ctx.images]).value_counts().to_dict()
    return f"{len(ctx.images)} new photos: {per_label}"


def export_new_data(ctx: RetrainContext) -> str:
    ctx.new_dir = ctx.run_dir / "new"
    for image in ctx.images:
        path = ctx.new_dir / image.label / f"upload_{image.id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image.data)
    return f"exported to {ctx.new_dir}"


def load_parent(ctx: RetrainContext) -> str:
    ctx.parent_dir = resolve_model_dir(ctx.model_cfg, ctx.train_cfg)
    ctx.model, metadata = load_model(ctx.parent_dir)
    ctx.parent_name = ctx.parent_dir.name
    if metadata["class_names"] != ctx.data_cfg.class_names:
        raise RetrainError(f"{ctx.parent_name} was trained on different classes: {metadata['class_names']}")
    ctx.stats = (metadata["normalisation"]["mean"], metadata["normalisation"]["std"])
    metrics_file = ctx.parent_dir / "metrics.json"
    ctx.parent_metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    return f"parent {ctx.parent_name} (val macro F1 {ctx.parent_metrics.get('val', {}).get('f1_macro')})"


def build_datasets(ctx: RetrainContext) -> str:
    names = ctx.data_cfg.class_names
    size = ctx.model.input_shape[1]
    batch = ctx.train_cfg.batch_size
    old_paths, old_labels = list_images(ctx.data_cfg.paths.processed_dir / "train", names)
    new_paths, new_labels = list_images(ctx.new_dir, names)
    augment = make_augment(ctx.train_cfg.augmentation, seed=ctx.data_cfg.seed)
    ctx.datasets["train"] = make_dataset(old_paths + new_paths, old_labels + new_labels, size, batch, training=True,
                                         seed=ctx.data_cfg.seed, cache=ctx.train_cfg.cache, cache_dir=ctx.train_cfg.cache_dir, augment=augment,
                                         stats=ctx.stats)
    for split in ("val", "test"):
        paths, labels = list_images(ctx.data_cfg.paths.processed_dir / split, names)
        ctx.datasets[split] = make_dataset(paths, labels, size, batch, training=False, seed=ctx.data_cfg.seed,
                                           cache=ctx.train_cfg.cache, cache_dir=ctx.train_cfg.cache_dir, stats=ctx.stats)
    return f"train {len(old_paths)} original + {len(new_paths)} new photos; val/test unchanged"


class _EpochReporter(keras.callbacks.Callback):
    def __init__(self, report: Callable[[int, dict], Any]) -> None:
        super().__init__()
        self.report = report

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        self.report(epoch + 1, {k: round(float(v), 4) for k, v in (logs or {}).items()})


def next_version_name(session: Session, parent: str) -> str:
    """pearlnet -> pearlnet_r1; pearlnet_r1 -> pearlnet_r2 (numbered from the base name)."""
    base = re.sub(r"_r\d+$", "", parent)
    taken = [v.name for v in repo.list_model_versions(session)]
    numbers = [int(m.group(1)) for n in taken if (m := re.fullmatch(rf"{re.escape(base)}_r(\d+)", n))]
    return f"{base}_r{max(numbers, default=0) + 1}"


def fine_tune(ctx: RetrainContext) -> str:
    rc = ctx.train_cfg.retrain
    cfg = ctx.train_cfg.model_copy(update={"learning_rate": rc.learning_rate, "epochs": rc.epochs,
                                           "early_stopping_patience": rc.early_stopping_patience})
    ctx.name = next_version_name(ctx.session, ctx.parent_name)
    ctx.out_dir = ctx.train_cfg.models_dir / ctx.name
    keras.utils.set_random_seed(ctx.data_cfg.seed)
    reporters = [_EpochReporter(ctx.on_epoch)] if ctx.on_epoch else []
    ctx.history = train_model(ctx.model, ctx.datasets["train"], ctx.datasets["val"], cfg, ctx.out_dir,
                              extra_callbacks=reporters)
    best = int(ctx.history["val_loss"].idxmin())
    return f"{ctx.name}: {len(ctx.history)} epochs, best epoch {best}"


def evaluate_and_compare(ctx: RetrainContext) -> str:
    names = ctx.data_cfg.class_names
    ctx.val = evaluate_model(ctx.model, ctx.datasets["val"], names)
    ctx.test = evaluate_model(ctx.model, ctx.datasets["test"], names)
    parent_f1 = ctx.parent_metrics.get("val", {}).get("f1_macro")
    new_f1 = ctx.val["f1_macro"]
    tolerance = ctx.train_cfg.retrain.promote_tolerance
    if parent_f1 is None:
        ctx.promoted, ctx.decision = True, "parent has no validation score; promoting"
    else:
        ctx.promoted = new_f1 >= parent_f1 - tolerance
        verdict = "promote" if ctx.promoted else "keep the parent"
        ctx.decision = f"val macro F1 {new_f1:.3f} vs parent {parent_f1:.3f} (tolerance {tolerance}): {verdict}"
    return ctx.decision


def save_and_register(ctx: RetrainContext) -> str:
    mean, std = ctx.stats
    best = int(ctx.history["val_loss"].idxmin())
    summary = {"best_epoch": best, "epochs_run": int(len(ctx.history)), "parameters": int(ctx.model.count_params()),
               "architecture": ctx.parent_metrics.get("architecture"), "parent": ctx.parent_name,
               "new_images": len(ctx.images), "promoted": ctx.promoted, "decision": ctx.decision,
               "run_id": ctx.run_id}
    metrics = {**summary, "val": ctx.val, "test": ctx.test}
    (ctx.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    save_model(ctx.model, ctx.out_dir, ctx.data_cfg.class_names, int(ctx.model.input_shape[1]), list(mean), list(std),
               test_accuracy=ctx.test["accuracy"], test_f1_macro=ctx.test["f1_macro"], **summary)

    version = repo.add_model_version(ctx.session, ctx.name, parent=ctx.parent_name, n_new_images=len(ctx.images),
                                     metrics=metrics, notes=ctx.decision)
    if ctx.promoted:
        repo.set_active(ctx.session, ctx.name)
        repo.mark_used(ctx.session, [i.id for i in ctx.images], version)
        (ctx.train_cfg.models_dir / ACTIVE_FILE).write_text(json.dumps({"model": ctx.name}) + "\n", encoding="utf-8")
    ctx.session.commit()
    if ctx.promoted and ctx.on_promote:
        ctx.on_promote(ctx.out_dir)
    live = "now live" if ctx.promoted else f"saved but not live; {ctx.parent_name} stays active"
    return f"{ctx.name} {live}; test accuracy {ctx.test['accuracy']:.3f}"


STEPS: list[tuple[str, Callable[[RetrainContext], str]]] = [
    ("check new data", check_new_data),
    ("export new data", export_new_data),
    ("load parent model", load_parent),
    ("build datasets", build_datasets),
    ("fine-tune", fine_tune),
    ("evaluate and compare", evaluate_and_compare),
    ("save and register", save_and_register),
]
