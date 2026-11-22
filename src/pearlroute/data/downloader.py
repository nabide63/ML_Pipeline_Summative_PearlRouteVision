"""Download records from sources and write data/metadata.csv."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from pearlroute.config import ClassConfig, DataConfig, Settings
from pearlroute.data.http import HttpClient
from pearlroute.data.sources.base import ImageRecord, ImageSource
from pearlroute.data.sources.openverse import OpenverseSource
from pearlroute.data.sources.wikimedia import WikimediaSource, commons_thumb_url
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)

# Failures worth trying again on a later run (rate limits, server errors, network drops).
_TRANSIENT_PREFIXES = ("http 429", "http 5", "network:")

METADATA_COLUMNS = [
    "filename", "label", "source", "source_id", "url", "landing_url", "title", "creator",
    "license", "license_url", "query", "orig_width", "orig_height", "width", "height", "downloaded_at",
]


def _normalise_url(url: str) -> str:
    return url.strip().replace("http://", "https://", 1)


def perceptual_hash(img: Image.Image) -> int:
    """64-bit pHash as an int, so Hamming distance is a cheap XOR + popcount."""
    return int(str(imagehash.phash(img)), 16)


def colourfulness(img: Image.Image) -> float:
    """Mean difference between colour channels on a 64x64 copy; near 0 for black-and-white photos."""
    a = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())


def title_word_pattern(words: list[str]) -> re.Pattern[str] | None:
    """Case-insensitive whole-word regex for the excluded title words (so 'map' does not hit 'Mapping')."""
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def rejection_reason(record: ImageRecord, exclude_creators: list[str],
                     title_pattern: re.Pattern[str] | None) -> str | None:
    """Why a candidate should be skipped before downloading (maps, posters, archive scans...), or None."""
    creator = record.creator.casefold()
    if any(name.casefold() in creator for name in exclude_creators):
        return f"excluded creator {record.creator}"
    if title_pattern and (match := title_pattern.search(record.title)):
        return f"excluded title word '{match.group(0)}'"
    return None


class MetadataStore:
    """Append-only metadata.csv; also the record of what has been downloaded, for resuming."""

    def __init__(self, path: Path, raw_dir: Path | None = None) -> None:
        self.path = path
        self.urls: set[str] = set()
        self.landing_urls: set[str] = set()
        self.counts: Counter[str] = Counter()
        self.hashes: list[tuple[int, str]] = []  # (pHash, filename) of every saved image
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                self._remember(row)
            if raw_dir is not None and rows:
                log.info("hashing %d saved images for duplicate checks...", len(rows))
                for row in rows:
                    file = raw_dir / row["filename"]
                    if file.exists():
                        with Image.open(file) as img:
                            self.hashes.append((perceptual_hash(img), row["filename"]))

    def _remember(self, row: dict[str, str]) -> None:
        self.urls.add(_normalise_url(row["url"]))
        if row.get("landing_url"):
            self.landing_urls.add(_normalise_url(row["landing_url"]))
        self.counts[row["label"]] += 1

    def seen(self, record: ImageRecord) -> bool:
        """True if this image (by file URL or landing page) was already downloaded under any class."""
        return (_normalise_url(record.url) in self.urls
                or (bool(record.landing_url) and _normalise_url(record.landing_url) in self.landing_urls))

    def near_duplicate(self, phash: int, max_distance: int) -> str | None:
        """Filename of a saved image within max_distance bits of phash (any class, any source), or None."""
        for other, filename in self.hashes:
            if (phash ^ other).bit_count() <= max_distance:
                return filename
        return None

    def add(self, row: dict[str, str | int | None], phash: int | None = None) -> None:
        """Append one row and flush it, so an interrupted run loses nothing."""
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self._remember({k: "" if v is None else str(v) for k, v in row.items()})
        if phash is not None:
            self.hashes.append((phash, str(row["filename"])))


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


def _lazy(fetch: Callable[[], list[ImageRecord]]) -> Iterator[ImageRecord]:
    """Defer an API call until the first candidate is requested."""
    yield from fetch()


def interleave(streams: list[Iterator[ImageRecord]]) -> Iterator[ImageRecord]:
    """Round-robin over the streams, one item at a time, until all are exhausted."""
    queue = deque(streams)
    while queue:
        stream = queue.popleft()
        try:
            yield next(stream)
        except StopIteration:
            continue
        queue.append(stream)


def iter_candidates(
    class_cfg: ClassConfig,
    sources: list[ImageSource],
    per_query_limit: int,
    per_category_limit: int,
    category_depth: int,
    include_fallback: bool,
) -> Iterator[ImageRecord]:
    """Yield candidates round-robin across every (search term, source) and Commons category tree.

    Taking one candidate from each stream in turn spreads the class's quota over all terms,
    instead of letting the first one or two terms fill the class on their own.
    """
    terms = class_cfg.search_terms + (class_cfg.fallback_terms if include_fallback else [])
    streams: list[Iterator[ImageRecord]] = [
        _lazy(lambda s=source, t=term: s.search(t, per_query_limit))
        for term in terms for source in sources
    ]
    for source in sources:
        if isinstance(source, WikimediaSource):
            streams += [
                _lazy(lambda s=source, c=category: s.from_category(c, per_category_limit, category_depth))
                for category in class_cfg.wikimedia_categories
            ]
    yield from interleave(streams)


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
    c = cfg.collection
    title_pattern = title_word_pattern(c.exclude_title_words)
    added = 0
    for record in candidates:
        if store.counts[label] >= target:
            break
        if store.seen(record) or record.url in failed:
            continue
        if reason := rejection_reason(record, c.exclude_creators, title_pattern):
            failed.add(record.url, reason)
            log.debug("skip %s: %s", record.url[:80], reason)
            continue
        # Openverse links Wikimedia originals directly; fetch a standard thumbnail instead.
        download_url = commons_thumb_url(record.url, cfg.sources.wikimedia_thumb_width, record.width)
        result = fetch_image(download_client, download_url,
                             cfg.cleaning.min_short_side, cfg.collection.max_side_on_save)
        if isinstance(result, str):
            if result.startswith(_TRANSIENT_PREFIXES):
                log.info("skip %s for now: %s", download_url[:80], result)
            else:
                failed.add(record.url, result)
                log.debug("skip %s: %s", download_url[:80], result)
            continue
        if c.min_colourfulness and colourfulness(result) < c.min_colourfulness:
            failed.add(record.url, "grayscale")
            log.debug("skip %s: grayscale", download_url[:80])
            continue
        phash = perceptual_hash(result)
        if twin := store.near_duplicate(phash, cfg.cleaning.phash_max_distance):
            failed.add(record.url, f"near-duplicate of {twin}")
            log.debug("skip %s: near-duplicate of %s", download_url[:80], twin)
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
        }, phash)
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
    if not any(marker in cfg.sources.user_agent for marker in ("http://", "https://", "@")):
        log.warning("sources.user_agent has no contact URL or email; upload.wikimedia.org "
                    "answers such clients with HTTP 429. Add one in configs/data.yaml.")
    sources = build_sources(cfg, settings, source_names or ["openverse", "wikimedia"])
    download_client = HttpClient(cfg.sources.user_agent, c.timeout_seconds, c.download_sleep_seconds,
                                 c.max_retries, c.backoff_seconds, c.max_retry_after_seconds)
    store = MetadataStore(cfg.paths.metadata_csv, cfg.paths.raw_dir)
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
