"""Tests for POST /predict."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import TINY_CLASSES, jpeg_bytes


def test_predict_one_photo(api_client: TestClient) -> None:
    r = api_client.post("/predict", files={"file": ("raft.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "raft.jpg" and body["label"] in TINY_CLASSES and body["model"] == "tiny"
    assert len(body["top_k"]) == 2 and body["top_k"][0]["label"] == body["label"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert isinstance(body["uncertain"], bool) and isinstance(body["destinations"], list)


def test_predict_accepts_png_with_any_content_type(api_client: TestClient) -> None:
    buf = io.BytesIO(); Image.new("RGBA", (40, 30), (0, 200, 0, 128)).save(buf, "PNG")
    r = api_client.post("/predict", files={"file": ("x.png", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200, r.text


def test_predict_rejects_non_images_and_empty_and_too_large(api_client: TestClient) -> None:
    r = api_client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "notes.txt" in r.json()["detail"]
    assert api_client.post("/predict", files={"file": ("e.jpg", b"", "image/jpeg")}).status_code == 400
    big = jpeg_bytes(size=(900, 900))                                     # > 0.05 MB test limit
    r = api_client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
    assert api_client.post("/predict").status_code == 422                 # no file at all


def test_predict_batch_keeps_good_files_and_reports_bad_ones(api_client: TestClient) -> None:
    files = [("files", ("a.jpg", jpeg_bytes(), "image/jpeg")),
             ("files", ("bad.txt", b"nope", "text/plain")),
             ("files", ("b.jpg", jpeg_bytes(colour=(200, 50, 50)), "image/jpeg"))]
    r = api_client.post("/predict/batch", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [p["filename"] for p in body["predictions"]] == ["a.jpg", "b.jpg"]
    assert body["errors"][0]["filename"] == "bad.txt"


def test_predict_batch_limit(api_client: TestClient) -> None:
    files = [("files", (f"{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(4)]   # limit is 3
    assert api_client.post("/predict/batch", files=files).status_code == 400


def test_batch_matches_single(api_client: TestClient) -> None:
    one = api_client.post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}).json()
    batch = api_client.post("/predict/batch", files=[("files", ("a.jpg", jpeg_bytes(), "image/jpeg"))]).json()
    np.testing.assert_allclose(list(batch["predictions"][0]["probabilities"].values()),
                               list(one["probabilities"].values()), atol=1e-4)


def test_health_and_model_info(api_client: TestClient) -> None:
    health = api_client.get("/health").json()
    assert health["status"] == "ok" and "model_loaded" in health and health["version"]
    info = api_client.get("/model").json()
    assert info["class_names"] == TINY_CLASSES and info["image_size"] == 32
    assert info["metrics"]["test"]["accuracy"] == 0.5


def test_openapi_docs(api_client: TestClient) -> None:
    paths = api_client.get("/openapi.json").json()["paths"]
    assert {"/health", "/model", "/predict", "/predict/batch"} <= set(paths)
