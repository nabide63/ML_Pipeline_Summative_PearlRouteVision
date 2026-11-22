"""Run the retraining steps in order and report status.

    job = start_retraining()          # background thread; returns at once
    get_job(job.id).to_dict()         # step, epoch progress, messages, result

Only one retraining runs at a time (training is heavy and they'd both write models/ and the database).
Job status is kept in memory: it shows live progress; the lasting record of every retrained model
is its row in the model_versions table and its models/<name>/ folder.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pearlroute.config import (DataConfig, ModelConfig, TrainConfig, load_data_config, load_model_config,
                               load_train_config)
from pearlroute.db.session import get_sessionmaker
from pearlroute.logging_utils import get_logger
from pearlroute.retraining.steps import STEPS, RetrainContext, RetrainError

log = get_logger(__name__)


@dataclass
class RetrainJob:
    id: str
    status: str = "queued"                 # queued, running, succeeded, failed
    step: int = 0                          # 1..7 while running
    step_name: str = ""
    total_steps: int = len(STEPS)
    epoch: int | None = None
    epochs: int | None = None
    last_epoch_metrics: dict | None = None
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_retraining(job: RetrainJob, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig,
                   session_factory: Callable[[], Any], on_promote: Callable[[Path], Any] | None = None) -> RetrainJob:
    """Run all steps for `job`, updating it as it goes. Never raises: failures end up in job.error."""
    job.status, job.started_at, job.epochs = "running", _now(), train_cfg.retrain.epochs

    def on_epoch(epoch: int, logs: dict) -> None:
        job.epoch, job.last_epoch_metrics = epoch, logs

    session = session_factory()
    ctx = RetrainContext(data_cfg, model_cfg, train_cfg, session, run_id=job.id, on_promote=on_promote,
                         on_epoch=on_epoch)
    try:
        for number, (name, step) in enumerate(STEPS, start=1):
            job.step, job.step_name = number, name
            start = time.time()
            message = step(ctx)
            job.log.append(f"{number}. {name}: {message} ({time.time() - start:.0f}s)")
            log.info("retrain %s step %d/%d %s: %s", job.id, number, len(STEPS), name, message)
        job.status = "succeeded"
        job.result = {"model": ctx.name, "parent": ctx.parent_name, "promoted": ctx.promoted,
                      "decision": ctx.decision, "new_images": len(ctx.images),
                      "val_f1_macro": ctx.val.get("f1_macro"), "test_accuracy": ctx.test.get("accuracy"),
                      "test_f1_macro": ctx.test.get("f1_macro"),
                      "parent_val_f1_macro": ctx.parent_metrics.get("val", {}).get("f1_macro"),
                      "parent_test_accuracy": ctx.parent_metrics.get("test", {}).get("accuracy")}
    except RetrainError as exc:                      # expected stop, e.g. not enough new photos
        session.rollback()
        job.status, job.error = "failed", str(exc)
        job.log.append(f"{job.step}. {job.step_name}: stopped: {exc}")
    except Exception as exc:  # noqa: BLE001 - report any crash on the job instead of losing it in a thread
        session.rollback()
        job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        job.log.append(f"{job.step}. {job.step_name}: crashed: {job.error}")
        log.error("retrain %s failed:\n%s", job.id, traceback.format_exc())
    finally:
        session.close()
        job.finished_at = _now()
    return job


_jobs: dict[str, RetrainJob] = {}
_lock = threading.Lock()


class RetrainAlreadyRunning(RuntimeError):
    def __init__(self, job: RetrainJob) -> None:
        super().__init__(f"retraining job {job.id} is already running")
        self.job = job


def running_job() -> RetrainJob | None:
    return next((j for j in _jobs.values() if j.status in ("queued", "running")), None)


def start_retraining(on_promote: Callable[[Path], Any] | None = None,
                     session_factory: Callable[[], Any] | None = None,
                     configs: tuple[DataConfig, ModelConfig, TrainConfig] | None = None) -> RetrainJob:
    """Start a retraining job in a background thread and return it immediately."""
    with _lock:
        if (current := running_job()) is not None:
            raise RetrainAlreadyRunning(current)
        job = RetrainJob(id=uuid.uuid4().hex[:12])
        _jobs[job.id] = job
    data_cfg, model_cfg, train_cfg = configs or (load_data_config(), load_model_config(), load_train_config())
    factory = session_factory or get_sessionmaker()
    threading.Thread(target=run_retraining, name=f"retrain-{job.id}", daemon=True,
                     args=(job, data_cfg, model_cfg, train_cfg, factory, on_promote)).start()
    return job


def get_job(job_id: str) -> RetrainJob | None:
    return _jobs.get(job_id)


def latest_job() -> RetrainJob | None:
    return next(reversed(_jobs.values()), None)
