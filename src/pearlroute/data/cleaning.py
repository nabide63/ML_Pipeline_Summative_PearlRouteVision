"""Size filter, corrupt-file check and perceptual-hash dedupe.

Cleaning never touches data/raw. It reads data/metadata.csv, checks every image, and writes:
  - cleaned_csv: the rows that passed, with their pHash
  - dropped_csv: the rows that failed, with the reason
  - counts_json: raw / clean / dropped counts per class (the "cleaning" section)

The downloader already applies most of these filters, so on fresh data this mostly re-checks.
It matters for images collected before a filter existed, and for the manual exclude list you
fill in while looking through the images by eye.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from pearlroute.config import DataConfig
from pearlroute.data.downloader import colourfulness, perceptual_hash, rejection_reason, title_word_pattern
from pearlroute.data.sources.base import ImageRecord
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)


def read_manual_excludes(path: Path) -> set[str]:
    """Filenames (e.g. 'waterfalls/openverse_ab12.jpg') listed one per line; '#' starts a comment."""
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().replace("\\", "/")
        if line:
            names.add(line)
    return names


def _record_from_row(row: pd.Series) -> ImageRecord:
    return ImageRecord(url=row["url"], source=row["source"], source_id=str(row["source_id"]),
                       landing_url=row["landing_url"], title=row["title"], creator=row["creator"],
                       license=row["license"], license_url=row["license_url"], width=None, height=None,
                       query=row["query"])


def inspect_image(path: Path, cfg: DataConfig) -> tuple[str | None, int | None]:
    """Return (drop reason or None, pHash or None) for one image file."""
    c = cfg.cleaning
    if path.suffix.lower() not in c.allowed_extensions:
        return f"extension {path.suffix}", None
    if not path.exists():
        return "missing file", None
    try:
        with Image.open(path) as img:
            img.load()
            if min(img.size) < c.min_short_side:
                return f"too small {img.size[0]}x{img.size[1]}", None
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return f"corrupt: {type(exc).__name__}", None
    if cfg.collection.min_colourfulness and colourfulness(rgb) < cfg.collection.min_colourfulness:
        return "grayscale", None
    return None, perceptual_hash(rgb)


def clean_metadata(metadata: pd.DataFrame, cfg: DataConfig,
                   manual_excludes: set[str] = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split metadata rows into (kept, dropped). Dropped rows get a 'drop_reason' column.

    Near-duplicates are found across all classes and sources. Of each near-duplicate pair the
    larger image is kept; if a pair spans two classes, both are dropped, since the label is unclear.
    """
    df = metadata.fillna("").reset_index(drop=True)
    title_pattern = title_word_pattern(cfg.collection.exclude_title_words)
    reasons: list[str | None] = []
    hashes: list[int | None] = []
    for _, row in df.iterrows():
        if row["filename"] in manual_excludes:
            reasons.append("manual exclude")
            hashes.append(None)
            continue
        reason = rejection_reason(_record_from_row(row), cfg.collection.exclude_creators, title_pattern)
        phash = None
        if reason is None:
            reason, phash = inspect_image(cfg.paths.raw_dir / row["filename"], cfg)
        reasons.append(reason)
        hashes.append(phash)
    df["drop_reason"] = reasons
    df["phash"] = [None if h is None else f"{h:016x}" for h in hashes]

    # Dedupe: visit candidates largest first, so the kept copy of each group is the sharpest.
    max_distance = cfg.cleaning.phash_max_distance
    area = pd.to_numeric(df["width"], errors="coerce").fillna(0) * pd.to_numeric(df["height"], errors="coerce").fillna(0)
    order = area[df["drop_reason"].isna()].sort_values(ascending=False, kind="stable").index
    kept: list[int] = []
    for i in order:
        h = hashes[i]
        twin = next((k for k in kept if (h ^ hashes[k]).bit_count() <= max_distance), None)
        if twin is None:
            kept.append(i)
            continue
        df.at[i, "drop_reason"] = f"near-duplicate of {df.at[twin, 'filename']}"
        if df.at[i, "label"] != df.at[twin, "label"]:
            df.at[twin, "drop_reason"] = f"near-duplicate in another class: {df.at[i, 'filename']}"

    dropped = df[df["drop_reason"].notna()].reset_index(drop=True)
    kept_df = df[df["drop_reason"].isna()].drop(columns="drop_reason").reset_index(drop=True)
    return kept_df, dropped


def _reason_group(reason: str) -> str:
    """Collapse detailed reasons ('too small 120x90') into a category for the summary."""
    for prefix in ("near-duplicate in another class", "near-duplicate", "too small", "corrupt",
                   "excluded creator", "excluded title word", "extension"):
        if reason.startswith(prefix):
            return prefix
    return reason


def update_counts_json(path: Path, section: str, payload: dict) -> None:
    """Replace one section of counts.json, keeping the others."""
    counts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    counts[section] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")


def run_cleaning(cfg: DataConfig) -> pd.DataFrame:
    """Clean data/metadata.csv and write the cleaned and dropped manifests. Returns the kept rows."""
    p = cfg.paths
    metadata = pd.read_csv(p.metadata_csv, dtype=str, keep_default_na=False)
    excludes = read_manual_excludes(p.manual_exclude)
    unknown = excludes - set(metadata["filename"])
    if unknown:
        log.warning("%d names in %s are not in metadata.csv, e.g. %s", len(unknown), p.manual_exclude.name,
                    sorted(unknown)[0])
    log.info("cleaning %d images (%d on the manual exclude list)", len(metadata), len(excludes))
    kept, dropped = clean_metadata(metadata, cfg, excludes)

    p.cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(p.cleaned_csv, index=False)
    dropped.to_csv(p.dropped_csv, index=False)

    classes = cfg.class_names
    raw = metadata["label"].value_counts()
    clean = kept["label"].value_counts()
    by_reason = dropped["drop_reason"].map(_reason_group).value_counts()
    update_counts_json(p.counts_json, "cleaning", {
        "raw": {c: int(raw.get(c, 0)) for c in classes},
        "clean": {c: int(clean.get(c, 0)) for c in classes},
        "dropped_by_reason": {r: int(n) for r, n in by_reason.items()},
    })

    for c in classes:
        log.info("%-26s raw %4d  clean %4d", c, raw.get(c, 0), clean.get(c, 0))
        if clean.get(c, 0) < cfg.collection.fallback_threshold:
            log.warning("%s has only %d clean images (< %d); collect more with "
                        "scripts/collect_images.py --classes %s --include-fallback",
                        c, clean.get(c, 0), cfg.collection.fallback_threshold, c)
    for reason, n in by_reason.items():
        log.info("dropped %4d  %s", n, reason)
    return kept
