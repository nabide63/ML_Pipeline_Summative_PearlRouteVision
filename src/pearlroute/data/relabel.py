"""Apply configs/data.yaml label_rules to photos that are already downloaded.

New downloads are checked by the collector itself (downloader.collect_class). This handles the rest:
  - fetch_keywords: fill metadata.csv's `keywords` (Openverse tags, Commons categories) for older rows
  - plan_changes: which photos the rules move to another class, and which they drop
  - apply_changes: move the files, update metadata.csv and the exclude lists, drop via data/exclude.txt
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pearlroute.config import DataConfig, Settings
from pearlroute.data.cleaning import read_manual_excludes
from pearlroute.data.downloader import (METADATA_COLUMNS, apply_label_rules, compile_rules, rule_text,
                                       title_word_pattern, write_metadata)
from pearlroute.data.http import HttpClient
from pearlroute.data.sources.openverse import OpenverseSource, trusted_tags
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Change:
    filename: str           # as in metadata.csv, e.g. waterfalls/openverse_ab12.jpg
    old_label: str
    new_label: str | None   # None: drop the photo
    reason: str
    title: str


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: row.get(k) or "" for k in METADATA_COLUMNS} for row in csv.DictReader(f)]


NO_TAGS = "-"   # keywords value for a photo whose source has no tags, so a re-run doesn't ask again


def fetch_keywords(rows: list[dict[str, str]], cfg: DataConfig, settings: Settings, openverse: bool = True,
                   checkpoint: Callable[[], None] | None = None) -> int:
    """Fill in missing keywords: Commons categories (50 files per request), Openverse tags (one request
    per photo, so slower). `checkpoint` is called every 100 photos so progress can be saved; a re-run
    skips photos already done. Returns how many rows got keywords."""
    c = cfg.collection
    client = HttpClient(cfg.sources.user_agent, c.timeout_seconds, c.request_sleep_seconds,
                        c.max_retries, c.backoff_seconds, c.max_retry_after_seconds)
    filled = 0
    wiki = [r for r in rows if r["source"] == "wikimedia" and not r["keywords"] and r["source_id"]]
    for start in range(0, len(wiki), 50):
        batch = {r["source_id"]: r for r in wiki[start:start + 50]}
        params = {"action": "query", "format": "json", "formatversion": "2", "prop": "categories",
                  "clshow": "!hidden", "cllimit": "max", "pageids": "|".join(batch)}
        cont: dict = {}
        while True:
            resp = client.get(cfg.sources.wikimedia_api, params={**params, **cont})
            if resp.status_code != 200:
                log.warning("Commons categories: HTTP %d", resp.status_code)
                break
            data = resp.json()
            for page in data.get("query", {}).get("pages", []):
                row = batch.get(str(page.get("pageid")))
                cats = [x["title"].removeprefix("Category:") for x in page.get("categories") or []]
                if row is not None and cats:
                    filled += not row["keywords"]
                    row["keywords"] = "; ".join(filter(None, [row["keywords"], *cats]))
            if "continue" not in data:
                break
            cont = data["continue"]
        log.info("Commons categories: %d / %d files", min(start + 50, len(wiki)), len(wiki))
    if openverse:
        # One lookup per photo. Some photos' pages never answer on Openverse's side, so: a short
        # timeout, no retries, and a photo that doesn't answer is left with no tags (title only).
        lookup = HttpClient(cfg.sources.user_agent, 15, c.request_sleep_seconds, 0, 0, 0)
        source = OpenverseSource(lookup, cfg.sources.openverse_api, c.openverse_licenses,
                                 cfg.sources.openverse_page_size, cfg.cleaning.allowed_extensions,
                                 settings.openverse_client_id, settings.openverse_client_secret)
        ov = [r for r in rows if r["source"] == "openverse" and not r["keywords"] and r["source_id"]]
        for i, row in enumerate(ov, 1):
            try:
                resp = lookup.get(f"{cfg.sources.openverse_api.rstrip('/')}/images/{row['source_id']}/",
                                  headers=source._headers())
            except Exception as exc:                # timeout or dropped connection
                log.debug("Openverse %s: %s", row["source_id"], type(exc).__name__)
                row["keywords"] = NO_TAGS
                continue
            if resp.status_code == 401 and source._client_id:
                source._headers(refresh=True)       # token expired during the run; next photo uses a new one
            if resp.status_code == 200:
                tags = trusted_tags(resp.json().get("tags"))
                row["keywords"] = "; ".join(tags) or NO_TAGS
                filled += bool(tags)
            elif resp.status_code in (404, 500, 502, 503, 504):
                row["keywords"] = NO_TAGS            # removed from Openverse, or its page is broken
            if i % 100 == 0:
                log.info("Openverse tags: %d / %d photos", i, len(ov))
                if checkpoint:
                    checkpoint()
    return filled


def plan_changes(rows: list[dict[str, str]], cfg: DataConfig) -> list[Change]:
    """Photos whose label the rules would change (moved to another class, or dropped).

    Also drops photos the collector would now skip: excluded creators and title words (e.g. artwork).
    """
    rules = compile_rules(cfg.label_rules)
    skip_words = title_word_pattern(cfg.collection.exclude_title_words)
    skip_creators = [n.casefold() for n in cfg.collection.exclude_creators]
    already_dropped = read_manual_excludes(cfg.paths.manual_exclude)
    changes = []
    for row in rows:
        if row["filename"] in already_dropped:
            continue
        if skip_words and (hit := skip_words.search(rule_text(row["title"], row["keywords"]))):
            changes.append(Change(row["filename"], row["label"], None, f"excluded word '{hit.group(0)}'", row["title"]))
            continue
        if any(n in row["creator"].casefold() for n in skip_creators):
            changes.append(Change(row["filename"], row["label"], None, "excluded creator", row["title"]))
            continue
        matched, label, reason = apply_label_rules(rule_text(row["title"], row["keywords"]), rules)
        if matched and label != row["label"]:
            changes.append(Change(row["filename"], row["label"], label, reason, row["title"]))
    return changes


def _rewrite_list(path: Path, moved: dict[str, str]) -> None:
    """Point exclude-list entries at the files' new paths."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        name, sep, comment = line.partition("#")
        key = name.strip()
        if key in moved:
            line = moved[key] + ("  #" + comment if sep else "")
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def apply_changes(rows: list[dict[str, str]], changes: list[Change], cfg: DataConfig) -> None:
    """Move relabelled files between class folders, record drops in data/exclude.txt, save metadata.csv.

    Nothing is deleted: dropped photos stay in data/raw and are skipped by cleaning via exclude.txt.
    """
    raw = cfg.paths.raw_dir
    by_name = {r["filename"]: r for r in rows}
    moved: dict[str, str] = {}
    drops: list[Change] = []
    for ch in changes:
        if ch.new_label is None:
            drops.append(ch)
            continue
        new_name = f"{ch.new_label}/{Path(ch.filename).name}"
        src, dst = raw / ch.filename, raw / new_name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
        row = by_name[ch.filename]
        row["filename"], row["label"] = new_name, ch.new_label
        moved[ch.filename] = new_name
    write_metadata(cfg.paths.metadata_csv, rows)
    for path in (cfg.paths.manual_exclude, cfg.paths.train_exclude):
        _rewrite_list(path, moved)
    if drops:
        excluded = cfg.paths.manual_exclude
        have = excluded.read_text(encoding="utf-8") if excluded.exists() else ""
        new = [f"{moved.get(d.filename, d.filename)}  # {d.reason}: {d.title[:60]}"
               for d in drops if d.filename not in have]
        if new:
            with excluded.open("a", encoding="utf-8") as f:
                f.write(("" if have.endswith("\n") or not have else "\n") + "\n".join(new) + "\n")
    log.info("moved %d photos to another class, dropped %d", len(moved), len(drops))
