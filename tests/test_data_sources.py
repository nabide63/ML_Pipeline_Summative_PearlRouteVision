"""Offline tests for the image sources and the downloader's resume bookkeeping."""

from pathlib import Path

import pytest

from pearlroute.config import load_data_config
from pearlroute.data.downloader import FailedLog, MetadataStore
from pearlroute.data.sources.base import ImageRecord
from pearlroute.data.sources.wikimedia import normalise_license


@pytest.mark.parametrize("raw, expected", [
    ("cc-by-sa-4.0", "by-sa"),
    ("CC BY-SA 3.0", "by-sa"),
    ("cc-by-2.0", "by"),
    ("CC BY 4.0", "by"),
    ("cc0", "cc0"),
    ("pd", "pdm"),
    ("Public domain", "pdm"),
    ("cc-by-nc-2.0", None),
    ("CC BY-ND 4.0", None),
    ("cc-by-nc-sa-4.0", None),
    ("GFDL", None),
    ("", None),
])
def test_normalise_license(raw: str, expected: str | None) -> None:
    assert normalise_license(raw) == expected


def _record(url: str, landing: str = "") -> ImageRecord:
    return ImageRecord(url=url, source="test", source_id="1", landing_url=landing, title="t",
                       creator="c", license="by", license_url="", width=100, height=100, query="q")


def _row(url: str, label: str, landing: str = "") -> dict:
    return {"filename": f"{label}/x.jpg", "label": label, "source": "test", "source_id": "1",
            "url": url, "landing_url": landing, "title": "t", "creator": "c", "license": "by",
            "license_url": "", "query": "q", "orig_width": 1, "orig_height": 1, "width": 1,
            "height": 1, "downloaded_at": "now"}


def test_metadata_store_resumes(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    store = MetadataStore(path)
    store.add(_row("http://a.org/1.jpg", "waterfalls", "https://a.org/page1"))
    store.add(_row("https://a.org/2.jpg", "waterfalls"))

    reloaded = MetadataStore(path)
    assert reloaded.counts["waterfalls"] == 2
    assert reloaded.seen(_record("https://a.org/1.jpg"))               # http/https treated alike
    assert reloaded.seen(_record("https://other/x.jpg", "https://a.org/page1"))  # same landing page
    assert not reloaded.seen(_record("https://a.org/3.jpg"))


def test_failed_log_resumes(tmp_path: Path) -> None:
    path = tmp_path / "failed.tsv"
    FailedLog(path).add("http://a.org/bad.jpg", "http 404")
    assert "https://a.org/bad.jpg" in FailedLog(path)


def test_data_config_loads() -> None:
    cfg = load_data_config()
    assert len(cfg.class_names) == 6
    assert cfg.class_names == sorted(cfg.class_names)
    assert abs(cfg.split.train + cfg.split.val + cfg.split.test - 1.0) < 1e-9
    assert cfg.paths.raw_dir.is_absolute()
