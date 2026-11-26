"""Tests for pearlroute.retraining: the full 7-step pipeline on a tiny model, tiny dataset and SQLite."""

from __future__ import annotations

import io
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sqlalchemy.orm import sessionmaker

from pearlroute.config import BaselineConfig, ModelConfig, ServingConfig, TrainConfig, load_data_config
from pearlroute.db import repository as repo
from pearlroute.db.models import TrainingImage
from pearlroute.db.session import init_db, make_engine
from pearlroute.models.io import save_model
from pearlroute.models.registry import build_model
from pearlroute.retraining.pipeline import RetrainJob, run_retraining
from pearlroute.retraining.steps import STEPS, next_version_name

SIZE = 32


def _jpeg(seed: int, size: int = 200) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def setup(tmp_path: Path):
    """Configs pointing at a tiny dataset, a tiny saved 'pearlnet' parent and a SQLite database."""
    base = load_data_config()
    classes = base.class_names
    processed = tmp_path / "processed"
    for split, n in [("train", 3), ("val", 2), ("test", 2)]:
        for c, name in enumerate(classes):
            (processed / split / name).mkdir(parents=True)
            for i in range(n):
                (processed / split / name / f"{i}.jpg").write_bytes(_jpeg(1000 * c + 10 * i + len(split)))
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"phash": [], "split": []}).to_csv(splits, index=False)
    repo.dataset_hashes.cache_clear()
    data_cfg = base.model_copy(update={"paths": base.paths.model_copy(
        update={"processed_dir": processed, "splits_csv": splits})})

    models_dir = tmp_path / "models"
    keras.utils.set_random_seed(0)
    model_cfg = ModelConfig(image_size=SIZE, baseline=BaselineConfig(filters=[4], dense_units=8, dropout=0.0),
                            serving=ServingConfig(active_model="pearlnet"))
    parent = models_dir / "pearlnet"
    save_model(build_model("baseline", model_cfg, len(classes)), parent, classes, SIZE, [127.0] * 3, [64.0] * 3)
    train_cfg = TrainConfig(batch_size=8, models_dir=models_dir, figures_dir=tmp_path / "fig",
                            retrain={"min_new_images": 3, "epochs": 2, "work_dir": tmp_path / "runs"})

    engine = make_engine(f"sqlite:///{tmp_path / 'r.db'}")
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        repo.add_model_version(s, "pearlnet", active=True)
        repo.save_batch(s, [repo.UploadItem(classes[i % 6], _jpeg(500 + i), f"u{i}.jpg") for i in range(4)], data_cfg)
        s.commit()
    return data_cfg, model_cfg, train_cfg, factory, parent


def _parent_scores(parent: Path, val_f1: float) -> None:
    (parent / "metrics.json").write_text(json.dumps({"val": {"f1_macro": val_f1}, "test": {"accuracy": 0.5}}))


def test_promoted_run_goes_live(setup) -> None:
    data_cfg, model_cfg, train_cfg, factory, parent = setup
    _parent_scores(parent, val_f1=0.0)                                   # anything beats it
    promoted_dirs = []
    job = run_retraining(RetrainJob(id="t1"), data_cfg, model_cfg, train_cfg, factory, on_promote=promoted_dirs.append)
    assert job.status == "succeeded", job.log
    assert len(job.log) == len(STEPS) == 7 and job.epoch == len(pd.read_csv(train_cfg.models_dir / "pearlnet_r1" / "history.csv"))
    assert job.result["model"] == "pearlnet_r1" and job.result["promoted"] and job.result["new_images"] == 4
    out = train_cfg.models_dir / "pearlnet_r1"
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["parent"] == "pearlnet" and meta["normalisation"]["mean"] == [127.0] * 3     # parent's stats kept
    assert json.loads((train_cfg.models_dir / "active.json").read_text()) == {"model": "pearlnet_r1"}
    assert promoted_dirs == [out]
    with factory() as s:
        assert repo.get_active(s).name == "pearlnet_r1" and repo.count_unused(s) == {}
        assert all(i.used_by.name == "pearlnet_r1" for i in s.query(TrainingImage))
    assert len(list((train_cfg.retrain.work_dir / "t1" / "new").rglob("*.jpg"))) == 4


def test_not_promoted_keeps_parent_and_photos(setup) -> None:
    data_cfg, model_cfg, train_cfg, factory, parent = setup
    _parent_scores(parent, val_f1=1.0)                                   # can't be beaten
    job = run_retraining(RetrainJob(id="t2"), data_cfg, model_cfg, train_cfg, factory, on_promote=pytest.fail)
    assert job.status == "succeeded" and not job.result["promoted"] and "keep the parent" in job.result["decision"]
    assert (train_cfg.models_dir / "pearlnet_r1" / "model.keras").exists()          # saved for the record
    assert not (train_cfg.models_dir / "active.json").exists()
    with factory() as s:
        assert repo.get_active(s).name == "pearlnet"
        assert sum(repo.count_unused(s).values()) == 4                               # kept for next time
        assert not repo.get_model_version(s, "pearlnet_r1").is_active


def test_too_few_photos_stops_at_step_1(setup) -> None:
    data_cfg, model_cfg, train_cfg, factory, _ = setup
    strict = train_cfg.model_copy(update={"retrain": train_cfg.retrain.model_copy(update={"min_new_images": 50})})
    job = run_retraining(RetrainJob(id="t3"), data_cfg, model_cfg, strict, factory)
    assert job.status == "failed" and job.step == 1 and "upload at least 50" in job.error


def test_version_names(setup) -> None:
    *_, factory, _ = setup
    with factory() as s:
        assert next_version_name(s, "pearlnet") == "pearlnet_r1"
        repo.add_model_version(s, "pearlnet_r1"); repo.add_model_version(s, "pearlnet_r2")
        assert next_version_name(s, "pearlnet_r2") == "pearlnet_r3"


def test_retrain_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from pearlroute.api.dependencies import db_session
    from pearlroute.api.main import create_app
    from pearlroute.retraining import pipeline

    engine = make_engine(f"sqlite:///{tmp_path / 'api.db'}"); init_db(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)
    with maker() as s:
        repo.add_model_version(s, "pearlnet", active=True); s.commit()

    def session():
        s = maker()
        try:
            yield s
        finally:
            s.close()

    app = create_app(preload_model=False)
    app.dependency_overrides[db_session] = session
    client = TestClient(app)
    r = client.post("/retrain")
    assert r.status_code == 400 and "upload at least" in r.json()["detail"]        # nothing uploaded
    assert client.get("/retrain/nope").status_code == 404
    assert [v["name"] for v in client.get("/retrain/versions").json()] == ["pearlnet"]

    with maker() as s:                                                               # pretend photos exist
        monkeypatch.setattr(repo, "count_unused", lambda _s: {"waterfalls": 99})
    fake = pipeline.RetrainJob(id="abc", status="running", step=5, step_name="fine-tune", epoch=2, epochs=8)
    monkeypatch.setattr(pipeline, "start_retraining", lambda on_promote=None: pipeline._jobs.setdefault("abc", fake))
    r = client.post("/retrain")
    assert r.status_code == 202 and r.json()["id"] == "abc"
    assert client.get("/retrain/abc").json()["step_name"] == "fine-tune"
    assert client.get("/retrain/latest").json()["id"] == "abc"
    monkeypatch.setattr(pipeline, "start_retraining",
                        lambda on_promote=None: (_ for _ in ()).throw(pipeline.RetrainAlreadyRunning(fake)))
    assert client.post("/retrain").status_code == 409
    pipeline._jobs.pop("abc", None)


def test_sample_per_class_caps_each_class_and_is_seeded() -> None:
    from pearlroute.retraining.steps import sample_per_class
    paths = [f"p{i}" for i in range(30)]
    labels = [0] * 20 + [1] * 10
    p1, l1 = sample_per_class(paths, labels, 5, seed=1)
    assert l1.count(0) == 5 and l1.count(1) == 5
    assert sample_per_class(paths, labels, 5, seed=1) == (p1, l1)            # seeded
    assert sample_per_class(paths, labels, 50, seed=1) == (paths, labels)    # fewer than the cap: all
    assert sample_per_class(paths, labels, None, seed=1) == (paths, labels)  # None: all
