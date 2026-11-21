"""Download records from sources and write data/metadata.csv."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, UnidentifiedImageError

from pearlroute.config import ClassConfig, DataConfig, Settings
from pearlroute.data.http import HttpClient
from pearlroute.data.sources.base import ImageRecord, ImageSource
from pearlroute.data.sources.openverse import OpenverseSource
from pearlroute.data.sources.wikimedia import WikimediaSource
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)

METADATA_COLUMNS = [
    "filename", "label", "source", "source_id", "url", "landing_url", "title", "creator",
    "license", "license_url", "query", "orig_width", "orig_height", "width", "height", "downloaded_at",
]


def _normalise_url(url: str) -> str:
    return url.strip().replace("http://", "https://", 1)


class MetadataStore:
    """Append-only metadata.csv; also the record of what has been downloaded, for resuming."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.urls: set[str] = set()
        self.landing_urls: set[str] = set()
        self.counts: Counter[str] = Counter()
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._remember(row)

    def _remember(self, row: dict[str, str]) -> None:
        self.urls.add(_normalise_url(row["url"]))
        if row.get("landing_url"):
            self.landing_urls.add(_normalise_url(row["landing_url"]))
        self.counts[row["label"]] += 1

    def seen(self, record: ImageRecord) -> bool:
        """True if this image (by file URL or landing page) was already downloaded under any class."""
        return (_normalise_url(record.url) in self.urls
                or (bool(record.landing_url) and _normalise_url(record.landing_url) in self.landing_urls))

    def add(self, row: dict[str, str | int | None]) -> None:
        """Append one row and flush it, so an interrupted run loses nothing."""
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self._remember({k: "" if v is None else str(v) for k, v in row.items()})


class FailedLog:
    """URLs that failed or were rejected, so a resumed run does not try them again."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.urls: set[str] = set()
        if path.exists():
            self.urls = {line.split("\t", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()}

    def add(self, url: str, reason: str) -> None:
        """Record a failed URL with a short reason."""
        url = _normalise_url(url)
        self.urls.add(url)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"{url}\t{reason}\n")

    def __contains__(self, url: str) -> bool:
        return _normalise_url(url) in self.urls


def fetch_image(client: HttpClient, url: str, min_short_side: int, max_side: int) -> Image.Image | str:
    """Download and decode one image as RGB, downscaled to max_side. Returns a reason string on failure."""
    try:
        resp = client.get(url)
    except Exception as exc:  # network failure after all retries
        return f"network: {type(exc).__name__}"
    if resp.status_code != 200:
        return f"http {resp.status_code}"
    if not resp.headers.get("Content-Type", "").startswith("image/"):
        return f"not an image ({resp.headers.get('Content-Type', '?')})"
    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return f"corrupt: {type(exc).__name__}"
    if min(img.size) < min_short_side:
        return f"too small {img.size[0]}x{img.size[1]}"
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))  # only ever shrinks
    return img


def iter_candidates(
    class_cfg: ClassConfig,
    sources: list[ImageSource],
    per_query_limit: int,
    per_category_limit: int,
    category_depth: int,
    include_fallback: bool,
) -> Iterator[ImageRecord]:
    """Yield candidates lazily: each search term from every source, then the Commons category trees."""
    terms = class_cfg.search_terms + (class_cfg.fallback_terms if include_fallback else [])
    for term in terms:
        for source in sources:
            yield from source.search(term, per_query_limit)
    for source in sources:
        if isinstance(source, WikimediaSource):
            for category in class_cfg.wikimedia_categories:
                yield from source.from_category(category, per_category_limit, category_depth)


def collect_class(
    label: str,
    candidates: Iterable[ImageRecord],
    target: int,
    cfg: DataConfig,
    download_client: HttpClient,
    store: MetadataStore,
    failed: FailedLog,
) -> int:
    """Download candidates for one class until it has `target` images. Returns how many were added."""
    out_dir = cfg.paths.raw_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for record in candidates:
        if store.counts[label] >= target:
            break
        if store.seen(record) or record.url in failed:
            continue
        result = fetch_image(download_client, record.url,
                             cfg.cleaning.min_short_side, cfg.collection.max_side_on_save)
        if isinstance(result, str):
            failed.add(record.url, result)
            log.debug("skip %s: %s", record.url[:80], result)
            continue
        filename = f"{record.source}_{hashlib.sha1(record.url.encode()).hexdigest()[:16]}.jpg"
        result.save(out_dir / filename, "JPEG", quality=cfg.collection.jpeg_quality)
        fields = asdict(record)
        store.add({
            **{k: fields[k] for k in ("source", "source_id", "url", "landing_url", "title",
                                      "creator", "license", "license_url", "query")},
            "filename": f"{label}/{filename}",
            "label": label,
            "orig_width": record.width,
            "orig_height": record.height,
            "width": result.width,
            "height": result.height,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        added += 1
        if store.counts[label] % 25 == 0:
            log.info("%s: %d / %d", label, store.counts[label], target)
    return added


def build_sources(cfg: DataConfig, settings: Settings, names: list[str]) -> list[ImageSource]:
    """Create the requested image sources, sharing one throttled API client."""
    c = cfg.collection
    api_client = HttpClient(cfg.sources.user_agent, c.timeout_seconds, c.request_sleep_seconds,
                            c.max_retries, c.backoff_seconds, c.max_retry_after_seconds)
    available: dict[str, ImageSource] = {
        "openverse": OpenverseSource(
            api_client, cfg.sources.openverse_api, c.openverse_licenses, cfg.sources.openverse_page_size,
            cfg.cleaning.allowed_extensions, settings.openverse_client_id, settings.openverse_client_secret),
        "wikimedia": WikimediaSource(
            api_client, cfg.sources.wikimedia_api, cfg.sources.wikimedia_page_size,
            cfg.sources.wikimedia_thumb_width, c.openverse_licenses),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}; choose from {sorted(available)}")
    return [available[n] for n in names]


def run_collection(
    cfg: DataConfig,
    settings: Settings,
    classes: list[str] | None = None,
    source_names: list[str] | None = None,
    target: int | None = None,
    include_fallback: bool = False,
) -> dict[str, int]:
    """Collect images for each class until it reaches the target. Safe to stop and re-run.

    Returns the number of images per class on disk after the run.
    """
    classes = classes or list(cfg.classes)
    unknown = set(classes) - set(cfg.classes)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")
    target = target or cfg.collection.target_per_class
    c = cfg.collection
    sources = build_sources(cfg, settings, source_names or ["openverse", "wikimedia"])
    download_client = HttpClient(cfg.sources.user_agent, c.timeout_seconds, c.download_sleep_seconds,
                                 c.max_retries, c.backoff_seconds, c.max_retry_after_seconds)
    store = MetadataStore(cfg.paths.metadata_csv)
    failed = FailedLog(cfg.paths.raw_dir / "failed_urls.tsv")

    for label in classes:
        have = store.counts[label]
        if have >= target:
            log.info("%s: already has %d / %d, skipping", label, have, target)
            continue
        log.info("=== %s: have %d, collecting up to %d ===", label, have, target)
        candidates = iter_candidates(cfg.classes[label], sources, c.per_query_limit,
                                     c.per_category_limit, c.wikimedia_subcategory_depth, include_fallback)
        added = collect_class(label, candidates, target, cfg, download_client, store, failed)
        log.info("=== %s: +%d, now %d / %d ===", label, added, store.counts[label], target)
        ran_out = store.counts[label] < target
        if ran_out and store.counts[label] < c.fallback_threshold and not include_fallback:
            log.warning("%s is below %d; re-run with --include-fallback to add East Africa terms",
                        label, c.fallback_threshold)
    return {label: store.counts[label] for label in cfg.classes}
