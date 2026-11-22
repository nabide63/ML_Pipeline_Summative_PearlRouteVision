"""Tests for pearlroute.db (on a temporary SQLite database; Supabase is never touched)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sqlalchemy.orm import sessionmaker

from pearlroute.config import load_data_config
from pearlroute.data.downloader import perceptual_hash
from pearlroute.db import repository as repo
from pearlroute.db.session import init_db, make_engine, ping, safe_url


def _jpeg(seed: int, size=(400, 300)) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def db(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    created = init_db(engine)
    assert created == ["model_versions", "training_images", "upload_batches"]
    assert init_db(engine) == []                                     # idempotent
    assert ping(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()


@pytest.fixture
def cfg(tmp_path: Path):
    """Real data config, but the 'dataset' is one known photo (seed 99)."""
    base = load_data_config()
    cleaned = tmp_path / "cleaned.csv"
    pd.DataFrame({"phash": [f"{perceptual_hash(Image.open(io.BytesIO(_jpeg(99)))):016x}"]}).to_csv(cleaned, index=False)
    repo.dataset_hashes.cache_clear()
    return base.model_copy(update={"paths": base.paths.model_copy(update={"cleaned_csv": cleaned})})


def test_save_batch_accepts_and_rejects_with_reasons(db, cfg) -> None:
    items = [repo.UploadItem("waterfalls", _jpeg(1), "a.jpg"),
             repo.UploadItem("waterfalls", _jpeg(1), "a_again.jpg"),                 # same photo twice
             repo.UploadItem("volcanoes", _jpeg(2), "b.jpg"),                        # unknown label
             repo.UploadItem("waterfalls", b"not an image", "c.txt"),
             repo.UploadItem("waterfalls", _jpeg(3, size=(100, 80)), "tiny.jpg"),    # too small
             repo.UploadItem("cultural_heritage", _jpeg(99), "from_test_set.jpg"),   # already in dataset
             repo.UploadItem("wildlife_safari", _jpeg(4, size=(2000, 1500)), "big.jpg")]
    batch = repo.save_batch(db, items, cfg, source="test"); db.commit()
    assert (batch.n_accepted, batch.n_rejected) == (2, 5)
    reasons = {r["filename"]: r["reason"] for r in batch.rejections}
    assert "duplicate" in reasons["a_again.jpg"] and "unknown label" in reasons["b.jpg"]
    assert "could not read" in reasons["c.txt"] and "too small" in reasons["tiny.jpg"]
    assert "dataset" in reasons["from_test_set.jpg"]
    big = next(i for i in batch.images if i.filename == "big.jpg")
    assert max(big.width, big.height) == 512 and Image.open(io.BytesIO(big.data)).size == (512, 384)

    # a later batch can't re-add the same photo either
    again = repo.save_batch(db, [repo.UploadItem("waterfalls", _jpeg(1), "a.jpg")], cfg); db.commit()
    assert again.n_accepted == 0 and "duplicate" in again.rejections[0]["reason"]


def test_unused_mark_used_and_stats(db, cfg) -> None:
    repo.save_batch(db, [repo.UploadItem("waterfalls", _jpeg(10)), repo.UploadItem("waterfalls", _jpeg(11)),
                         repo.UploadItem("wildlife_safari", _jpeg(12))], cfg)
    assert repo.count_unused(db) == {"waterfalls": 2, "wildlife_safari": 1}
    assert len(repo.fetch_unused(db, label="waterfalls")) == 2 and len(repo.fetch_unused(db, limit=1)) == 1
    version = repo.add_model_version(db, "pearlnet_r1", parent="pearlnet", n_new_images=2)
    assert repo.mark_used(db, [i.id for i in repo.fetch_unused(db, label="waterfalls")], version) == 2
    assert repo.count_unused(db) == {"wildlife_safari": 1}
    stats = repo.upload_stats(db)
    assert stats == {"batches": 1, "images": 3, "unused": 1, "used": 2, "unused_per_label": {"wildlife_safari": 1}}


def test_model_versions_and_single_active(db) -> None:
    metrics = {"val": {"f1_macro": 0.64}, "test": {"accuracy": 0.59, "f1_macro": 0.588}}
    repo.add_model_version(db, "pearlnet", metrics=metrics, active=True)
    repo.add_model_version(db, "pearlnet_r1", parent="pearlnet")
    assert repo.get_active(db).name == "pearlnet" and repo.get_active(db).test_accuracy == 0.59
    repo.set_active(db, "pearlnet_r1")
    assert repo.get_active(db).name == "pearlnet_r1"
    assert [v.name for v in repo.list_model_versions(db) if v.is_active] == ["pearlnet_r1"]
    assert [v.name for v in repo.list_model_versions(db)][0] == "pearlnet_r1"
    with pytest.raises(KeyError):
        repo.set_active(db, "missing")


def test_safe_url_hides_password() -> None:
    shown = safe_url("postgresql+psycopg2://user:s3cret@host:5432/db")
    assert "s3cret" not in shown and "host" in shown
