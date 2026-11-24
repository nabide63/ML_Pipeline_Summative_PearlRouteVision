"""Every HTTP call from the UI to the API lives here.

The API address comes from PEARLROUTE_API_URL (default http://127.0.0.1:8000), so the same UI works
locally and in Docker. Errors come back as ApiError with the API's own message (its `detail`).
"""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.environ.get("PEARLROUTE_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30            # seconds; most calls are quick
PREDICT_TIMEOUT = 120   # a batch of photos on a CPU, or the first call while the model loads

FileTuple = tuple[str, bytes, str]   # (filename, content, mime type), as requests expects


class ApiError(Exception):
    """The API could not be reached or answered with an error."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _request(method: str, path: str, timeout: float = TIMEOUT, **kwargs) -> Any:
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiError(f"cannot reach the API at {API_URL}; is it running?") from exc
    except requests.Timeout as exc:
        raise ApiError(f"the API did not answer within {timeout:g} s") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):         # FastAPI validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise ApiError(f"{detail} (HTTP {response.status_code})", response.status_code)
    return response.json()


# Status
def health() -> dict:
    return _request("GET", "/health", timeout=5)


def model_info() -> dict:
    return _request("GET", "/model")


# Prediction
def predict(file: FileTuple) -> dict:
    return _request("POST", "/predict", timeout=PREDICT_TIMEOUT, files={"file": file})


def predict_batch(files: list[FileTuple]) -> dict:
    return _request("POST", "/predict/batch", timeout=PREDICT_TIMEOUT,
                    files=[("files", f) for f in files])


# Uploads for retraining
def upload(files: list[FileTuple], label: str | None = None) -> dict:
    data = {"label": label} if label else {}
    return _request("POST", "/upload", timeout=PREDICT_TIMEOUT, files=[("files", f) for f in files], data=data)


def upload_stats() -> dict:
    return _request("GET", "/upload/stats")


# Retraining
def start_retrain() -> dict:
    return _request("POST", "/retrain")


def retrain_job(job_id: str) -> dict:
    return _request("GET", f"/retrain/{job_id}")


def latest_retrain_job() -> dict | None:
    return _request("GET", "/retrain/latest")


def model_versions() -> list[dict]:
    return _request("GET", "/retrain/versions")
