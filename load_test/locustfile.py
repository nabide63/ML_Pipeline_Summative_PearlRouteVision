"""Load test: a flood of prediction requests against the PearlRoute API.

Each simulated user picks a random demo photo and asks the API to classify it, with a short
pause between requests. Most traffic is single predictions; some is batches and health checks.

Interactive (web UI on http://localhost:8089):
    locust -f load_test/locustfile.py --host http://localhost:8000

Headless, saving CSV stats and an HTML report (see load_test/run.sh):
    locust -f load_test/locustfile.py --host http://localhost:8000 --headless \
        -u 50 -r 5 -t 2m --csv reports/locust/u50 --html reports/locust/u50.html
"""

from __future__ import annotations

import random
from pathlib import Path

from locust import HttpUser, between, events, task

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"
PHOTOS: list[tuple[str, bytes]] = []


@events.init.add_listener
def load_photos(**_kwargs) -> None:
    """Read the demo photos once, so the test measures the API, not the disk."""
    files = sorted(p for p in DEMO_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"no demo photos in {DEMO_DIR}")
    PHOTOS.extend((p.name, p.read_bytes()) for p in files)


class TravellerUser(HttpUser):
    """Someone using the app: uploads a photo and waits a moment before the next one."""

    wait_time = between(1, 3)

    @task(10)
    def predict(self) -> None:
        name, data = random.choice(PHOTOS)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "label" not in resp.json():
                resp.failure("no label in response")

    @task(2)
    def predict_batch(self) -> None:
        picks = random.sample(PHOTOS, k=min(5, len(PHOTOS)))
        files = [("files", (name, data, "image/jpeg")) for name, data in picks]
        with self.client.post("/predict/batch", files=files, name="/predict/batch (5)",
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
