"""Tests for POST /upload (on a temporary SQLite database; Supabase is never touched)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import sessionmaker

from pearlroute.config import ServingConfig, load_data_config
from pearlroute.data.downloader import perceptual_hash
from pearlroute.db import repository as repo
from pearlroute.db.session import init_db, make_engine


def photo(seed: int, size=(400, 300)) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    from pearlroute.api.dependencies import data_config, db_session, serving_config
    from pearlroute.api.main import create_app

    engine = make_engine(f"sqlite:///{tmp_path / 'up.db'}")
    init_db(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)

    def session():
        s = maker()
        try:
            yield s
        finally:
            s.rollback(); s.close()

    base = load_data_config()
    splits = tmp_path / "splits.csv"                       # the "dataset": photo 99 is a test photo
    pd.DataFrame({"phash": [f"{perceptual_hash(Image.open(io.BytesIO(photo(99)))):016x}"],
                  "split": ["test"]}).to_csv(splits, index=False)
    repo.dataset_hashes.cache_clear()
    uploads = base.uploads.model_copy(update={"max_files": 6, "max_zip_mb": 1, "max_unzipped_mb": 3})
    cfg = base.model_copy(update={"paths": base.paths.model_copy(update={"splits_csv": splits}), "uploads": uploads})

    app = create_app(preload_model=False)
    app.dependency_overrides[db_session] = session
    app.dependency_overrides[data_config] = lambda: cfg
    app.dependency_overrides[serving_config] = lambda: ServingConfig(max_upload_mb=0.5)
    return TestClient(app)


def test_upload_photos_with_label(client: TestClient) -> None:
    r = client.post("/upload", data={"label": "waterfalls"},
                    files=[("files", ("a.jpg", photo(1), "image/jpeg")), ("files", ("b.jpg", photo(2), "image/jpeg"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["accepted"], body["rejected"]) == (2, 0)
    assert body["accepted_per_label"] == {"waterfalls": 2} and body["waiting_for_retraining"] == {"waterfalls": 2}


def test_upload_zip_with_class_folders(client: TestClient) -> None:
    z = make_zip({"my_photos/waterfalls/a.jpg": photo(3), "my_photos/wildlife_safari/b.jpg": photo(4),
                  "__MACOSX/my_photos/._a.jpg": b"junk", "my_photos/.DS_Store": b"junk",
                  "my_photos/readme.txt": b"hello", "loose.jpg": photo(5)})
    r = client.post("/upload", files=[("files", ("batch.zip", z, "application/zip"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted_per_label"] == {"waterfalls": 1, "wildlife_safari": 1}
    reasons = {x["filename"]: x["reason"] for x in body["rejections"]}
    assert set(reasons) == {"my_photos/readme.txt", "loose.jpg"}             # system files skipped silently
    assert "class folder" in reasons["loose.jpg"]


def test_rejections_are_reported_not_fatal(client: TestClient) -> None:
    files = [("files", ("ok.jpg", photo(6), "image/jpeg")),
             ("files", ("test_photo.jpg", photo(99), "image/jpeg")),            # in the test set
             ("files", ("notes.txt", b"hello", "text/plain")),
             ("files", ("huge.jpg", photo(7, size=(1600, 1200)), "image/jpeg"))]  # > 0.5 MB test limit
    body = client.post("/upload", data={"label": "cultural_heritage"}, files=files).json()
    assert body["accepted"] == 1 and body["rejected"] == 3
    reasons = {x["filename"]: x["reason"] for x in body["rejections"]}
    assert "dataset" in reasons["test_photo.jpg"] and "could not read" in reasons["notes.txt"]
    assert "limit" in reasons["huge.jpg"]


def test_missing_or_unknown_label(client: TestClient) -> None:
    body = client.post("/upload", files=[("files", ("a.jpg", photo(8), "image/jpeg"))]).json()
    assert body["accepted"] == 0 and "no label" in body["rejections"][0]["reason"]
    body = client.post("/upload", data={"label": "volcanoes"}, files=[("files", ("a.jpg", photo(8), "image/jpeg"))]).json()
    assert "unknown label" in body["rejections"][0]["reason"]


def test_bad_zip_and_limits(client: TestClient) -> None:
    assert client.post("/upload", files=[("files", ("x.zip", b"not a zip", "application/zip"))]).status_code == 400
    bomb = make_zip({"waterfalls/a.jpg": b"\0" * (4 * 1024 * 1024)})                    # expands past 3 MB
    assert client.post("/upload", files=[("files", ("bomb.zip", bomb, "application/zip"))]).status_code == 413
    many = make_zip({f"waterfalls/{i}.jpg": photo(100 + i, size=(200, 200)) for i in range(7)})   # limit 6
    assert client.post("/upload", files=[("files", ("many.zip", many, "application/zip"))]).status_code == 400


def test_stats_and_duplicates_across_batches(client: TestClient) -> None:
    client.post("/upload", data={"label": "waterfalls"}, files=[("files", ("a.jpg", photo(9), "image/jpeg"))])
    again = client.post("/upload", data={"label": "waterfalls"}, files=[("files", ("a.jpg", photo(9), "image/jpeg"))])
    assert again.json()["accepted"] == 0 and "duplicate" in again.json()["rejections"][0]["reason"]
    stats = client.get("/upload/stats").json()
    assert stats == {"batches": 2, "images": 1, "used": 0, "unused": 1, "unused_per_label": {"waterfalls": 1}}


def test_database_not_configured_gives_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pearlroute.api.main import create_app
    from pearlroute.db import session as db_session_module
    monkeypatch.setattr(db_session_module, "get_settings", lambda: type("S", (), {"database_url": None})())
    db_session_module.get_engine.cache_clear(); db_session_module.get_sessionmaker.cache_clear()
    try:
        r = TestClient(create_app(preload_model=False)).get("/upload/stats")
        assert r.status_code == 503 and "DATABASE_URL" in r.json()["detail"]
    finally:
        db_session_module.get_engine.cache_clear(); db_session_module.get_sessionmaker.cache_clear()
