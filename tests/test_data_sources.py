"""Offline tests for the image sources and the downloader's resume bookkeeping."""

from pathlib import Path

import pytest

from pearlroute.config import load_data_config
from pearlroute.data.downloader import FailedLog, MetadataStore
from pearlroute.data.sources.base import ImageRecord
from pearlroute.data.sources.wikimedia import commons_thumb_url, normalise_license


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


def test_commons_thumb_url() -> None:
    original = "https://upload.wikimedia.org/wikipedia/commons/4/45/Bwindi_gorilla.jpg"
    assert commons_thumb_url(original, 960, 4000) == (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Bwindi_gorilla.jpg/960px-Bwindi_gorilla.jpg")
    assert commons_thumb_url(original, 960, 800) == original      # Commons will not upscale
    assert commons_thumb_url(original, 960, None) == original
    tracked = original + "?utm_source=commons.wikimedia.org&utm_content=original"
    assert commons_thumb_url(tracked, 960, 4000) == commons_thumb_url(original, 960, 4000)
    flickr = "https://live.staticflickr.com/1/2_abc.jpg"
    assert commons_thumb_url(flickr, 960, 4000) == flickr


def test_interleave_spreads_candidates_across_terms() -> None:
    from pearlroute.data.downloader import interleave
    streams = [iter(["a1", "a2", "a3"]), iter(["b1"]), iter(["c1", "c2"])]
    assert list(interleave(streams)) == ["a1", "b1", "c1", "a2", "c2", "a3"]


def test_rejection_reason_filters_creators_and_title_words() -> None:
    from dataclasses import replace

    from pearlroute.data.downloader import rejection_reason, title_word_pattern
    pattern = title_word_pattern(["map", "poster"])
    base = _record("https://a.org/1.jpg")
    assert rejection_reason(replace(base, title="Rwenzori locator Map"), [], pattern)
    assert rejection_reason(replace(base, creator="Luigi Amedeo di Savoia"), ["luigi amedeo"], pattern)
    assert rejection_reason(replace(base, title="Mapping the Stanley Plateau"), [], pattern) is None
    assert rejection_reason(base, [], None) is None


def test_colourfulness_flags_black_and_white() -> None:
    from PIL import Image

    from pearlroute.data.downloader import colourfulness
    assert colourfulness(Image.new("RGB", (80, 80), (120, 120, 120))) < 1
    assert colourfulness(Image.new("RGB", (80, 80), (30, 140, 60))) > 6


def test_metadata_store_finds_near_duplicates(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    from pearlroute.data.downloader import perceptual_hash
    raw = tmp_path / "raw"
    (raw / "waterfalls").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    img.save(raw / "waterfalls" / "x.jpg")
    MetadataStore(tmp_path / "m.csv").add(_row("https://a.org/1.jpg", "waterfalls"))

    store = MetadataStore(tmp_path / "m.csv", raw)              # re-hashes saved files on load
    resized = perceptual_hash(img.resize((100, 100)))
    assert store.near_duplicate(resized, 6) == "waterfalls/x.jpg"
    other = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    assert store.near_duplicate(perceptual_hash(other), 6) is None
