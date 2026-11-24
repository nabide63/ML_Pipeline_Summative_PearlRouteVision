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


def _rules():
    from pearlroute.data.downloader import compile_rules
    return compile_rules(load_data_config().label_rules)


@pytest.mark.parametrize("text, expected", [
    ("Kayaking below Itanda Falls", "whitewater_rafting"),         # a kayak beats the waterfall
    ("Rafting the Nile at Jinja", "whitewater_rafting"),
    ("Boat on the river below the falls", "whitewater_rafting"),
    ("Dhow at sunset, Lamu", None),                                 # sea boats fit no class
    ("Elephants at Murchison Falls", "wildlife_safari"),            # animals beat 'falls'
    ("Sipi Falls, Uganda", "waterfalls"),
    ("Chimpanzee in Kibale", "gorilla_primate_trekking"),
    ("IMG_2041 ; Pan troglodytes schweinfurthii", "gorilla_primate_trekking"),   # Commons category
    ("Murchison Falls National Park gate", "no rule"),              # the park, not the waterfall
    ("Ugnada Murchison-Falls-Buffalos", "wildlife_safari"),          # plural of buffalo
    ("White-Water Rapids at Itanda Falls", "waterfalls"),           # rapids without a boat are not rafting
    ("Rafters in a wave, Jinja", "whitewater_rafting"),
])
def test_label_rules(text: str, expected: str | None) -> None:
    from pearlroute.data.downloader import apply_label_rules
    matched, label, _ = apply_label_rules(text, _rules())
    assert (label if matched else "no rule") == expected


def test_label_rules_use_whole_words() -> None:
    from pearlroute.data.downloader import apply_label_rules
    assert not apply_label_rules("Craftsmen at the Kasubi Tombs", _rules())[0]     # 'raft' inside 'Craftsmen'


def test_openverse_keeps_only_confident_machine_tags() -> None:
    from pearlroute.data.sources.openverse import trusted_tags
    tags = [{"name": "rafting"}, {"name": "waterfall", "accuracy": 0.55}, {"name": "river", "accuracy": 0.97}]
    assert trusted_tags(tags) == ["rafting", "river"]


def test_metadata_store_adds_missing_columns(tmp_path: Path) -> None:
    import csv
    from pearlroute.data.downloader import METADATA_COLUMNS
    path = tmp_path / "metadata.csv"
    old = [c for c in METADATA_COLUMNS if c != "keywords"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({c: "x" for c in old} | {"label": "waterfalls", "url": "https://a/1.jpg"})
    MetadataStore(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == METADATA_COLUMNS and rows[0]["keywords"] == "" and rows[0]["label"] == "waterfalls"


def test_apply_changes_moves_files_and_records_drops(tmp_path: Path) -> None:
    from pearlroute.config import DataConfig
    from pearlroute.data.downloader import METADATA_COLUMNS
    from pearlroute.data.relabel import apply_changes, plan_changes, read_metadata
    import csv

    raw = tmp_path / "raw"
    (raw / "waterfalls").mkdir(parents=True)
    (raw / "waterfalls" / "a.jpg").write_bytes(b"x")
    (raw / "waterfalls" / "b.jpg").write_bytes(b"x")
    data = load_data_config().model_dump()
    data["paths"] = {k: tmp_path / Path(v).name for k, v in data["paths"].items()} | {"raw_dir": raw}
    cfg = DataConfig.model_validate(data)
    with cfg.paths.metadata_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        w.writeheader()
        w.writerow({"filename": "waterfalls/a.jpg", "label": "waterfalls", "title": "Kayak at Sipi Falls"})
        w.writerow({"filename": "waterfalls/b.jpg", "label": "waterfalls", "title": "Ferry boat on Lake Victoria"})
    cfg.paths.train_exclude.write_text("waterfalls/a.jpg  # old note\n", encoding="utf-8")

    rows = read_metadata(cfg.paths.metadata_csv)
    changes = plan_changes(rows, cfg)
    assert {(c.filename, c.new_label) for c in changes} == {("waterfalls/a.jpg", "whitewater_rafting"),
                                                           ("waterfalls/b.jpg", None)}
    apply_changes(rows, changes, cfg)
    assert (raw / "whitewater_rafting" / "a.jpg").exists() and not (raw / "waterfalls" / "a.jpg").exists()
    assert (raw / "waterfalls" / "b.jpg").exists()                          # dropped, not deleted
    assert read_metadata(cfg.paths.metadata_csv)[0]["label"] == "whitewater_rafting"
    assert "waterfalls/b.jpg" in cfg.paths.manual_exclude.read_text(encoding="utf-8")
    assert cfg.paths.train_exclude.read_text(encoding="utf-8").startswith("whitewater_rafting/a.jpg  # old note")


class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error", response=self)  # type: ignore[arg-type]


class _FakeOpenverseClient:
    """Token endpoint is rate limited; image search works only without a token."""

    def __init__(self) -> None:
        self.auth_headers_seen: list[bool] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(429)

    def get(self, url: str, params: dict, headers: dict) -> _FakeResponse:
        self.auth_headers_seen.append("Authorization" in headers)
        item = {"id": 1, "url": "https://x/a.jpg", "license": "by", "filetype": "jpg"}
        return _FakeResponse(200, {"results": [item], "page_count": 1})


def test_openverse_falls_back_to_anonymous_when_token_fails() -> None:
    from pearlroute.data.sources.openverse import ANONYMOUS_MAX_PAGE_SIZE, OpenverseSource

    client = _FakeOpenverseClient()
    source = OpenverseSource(client, "https://api", ["by"], 100, [".jpg"],  # type: ignore[arg-type]
                             client_id="id", client_secret="secret")
    records = source.search("gorilla", 5)
    assert len(records) == 1
    assert client.auth_headers_seen == [False]
    assert source.page_size == ANONYMOUS_MAX_PAGE_SIZE


def test_failed_search_is_skipped_not_fatal() -> None:
    import requests

    from pearlroute.data.downloader import _lazy, interleave

    def broken() -> list:
        raise requests.ReadTimeout("timed out")

    good = [object(), object()]
    assert list(interleave([_lazy(broken), _lazy(lambda: good)])) == good
