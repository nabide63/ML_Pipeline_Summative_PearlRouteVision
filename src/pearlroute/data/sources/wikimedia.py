"""Wikimedia Commons API image source."""

from __future__ import annotations

import html
import re
from collections import deque
from typing import Any, Iterator

from pearlroute.data.http import HttpClient
from pearlroute.data.sources.base import ImageRecord, ImageSource
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)

ALLOWED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def normalise_license(raw: str) -> str | None:
    """Map a Commons licence string to by / by-sa / cc0 / pdm, or None if it bars reuse or modification.

    Examples: "cc-by-sa-4.0" -> "by-sa", "CC BY 2.0" -> "by", "Public domain" -> "pdm", "CC BY-NC 2.0" -> None.
    """
    text = raw.lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    if "-nc" in text or "-nd" in text:
        return None
    if text.startswith("cc0") or "cc-zero" in text:
        return "cc0"
    if text.startswith("cc-by-sa"):
        return "by-sa"
    if text.startswith("cc-by"):
        return "by"
    if text.startswith("pd") or "public-domain" in text:
        return "pdm"
    return None


def _clean_html(value: str) -> str:
    return _SPACE.sub(" ", html.unescape(_TAG.sub("", value))).strip()


class WikimediaSource(ImageSource):
    """Search Wikimedia Commons by keyword or by category tree."""

    name = "wikimedia"

    def __init__(
        self,
        client: HttpClient,
        api_url: str,
        page_size: int,
        thumb_width: int,
        licenses: list[str],
    ) -> None:
        self.client = client
        self.api_url = api_url
        self.page_size = page_size
        self.thumb_width = thumb_width
        self.licenses = set(licenses)

    # --- API plumbing -------------------------------------------------------------

    def _query(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Run a query and follow 'continue' tokens, yielding each response's 'query' block."""
        base = {"action": "query", "format": "json", "formatversion": "2", **params}
        cont: dict[str, Any] = {}
        while True:
            resp = self.client.get(self.api_url, params={**base, **cont})
            if resp.status_code != 200:
                log.warning("Commons query failed with HTTP %d", resp.status_code)
                return
            data = resp.json()
            if "error" in data:
                log.warning("Commons API error: %s", data["error"].get("info", data["error"]))
                return
            if "query" in data:
                yield data["query"]
            if "continue" not in data:
                return
            cont = data["continue"]

    def _imageinfo_params(self) -> dict[str, Any]:
        return {
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": self.thumb_width,
            "iiextmetadatafilter": "Artist|LicenseShortName|License|LicenseUrl|ObjectName",
        }

    def _to_record(self, page: dict[str, Any], query: str) -> ImageRecord | None:
        info = (page.get("imageinfo") or [None])[0]
        if not info or info.get("mime") not in ALLOWED_MIME:
            return None
        meta = info.get("extmetadata") or {}

        def field(key: str) -> str:
            return _clean_html(str((meta.get(key) or {}).get("value", "")))

        license_code = normalise_license(field("License")) or normalise_license(field("LicenseShortName"))
        if license_code not in self.licenses:
            return None
        title = page.get("title", "")
        return ImageRecord(
            url=info.get("thumburl") or info["url"],
            source=self.name,
            source_id=str(page.get("pageid", "")),
            landing_url=info.get("descriptionurl", ""),
            title=field("ObjectName") or title.removeprefix("File:"),
            creator=field("Artist") or "unknown",
            license=license_code,
            license_url=field("LicenseUrl"),
            width=info.get("width"),
            height=info.get("height"),
            query=query,
        )

    def _collect(self, params: dict[str, Any], query: str, n: int, out: list[ImageRecord]) -> None:
        for block in self._query({**params, **self._imageinfo_params()}):
            for page in block.get("pages", []):
                record = self._to_record(page, query)
                if record:
                    out.append(record)
                    if len(out) >= n:
                        return

    # --- public API ---------------------------------------------------------------

    def search(self, query: str, n: int) -> list[ImageRecord]:
        """Full-text search over Commons files (photos only)."""
        records: list[ImageRecord] = []
        params = {"generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                  "gsrnamespace": 6, "gsrlimit": self.page_size}
        self._collect(params, query, n, records)
        log.info("Commons search '%s': %d candidates", query, len(records))
        return records

    def _subcategories(self, category: str) -> list[str]:
        subcats: list[str] = []
        for block in self._query({"list": "categorymembers", "cmtitle": category,
                                  "cmtype": "subcat", "cmlimit": "max"}):
            subcats.extend(m["title"] for m in block.get("categorymembers", []))
        return subcats

    def from_category(self, category: str, n: int, depth: int) -> list[ImageRecord]:
        """Files in a category and its subcategories (breadth-first, up to `depth` levels down)."""
        root = category if category.startswith("Category:") else f"Category:{category}"
        records: list[ImageRecord] = []
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        seen = {root}
        while queue and len(records) < n:
            cat, level = queue.popleft()
            params = {"generator": "categorymembers", "gcmtitle": cat,
                      "gcmtype": "file", "gcmlimit": self.page_size}
            self._collect(params, root, n, records)
            if level < depth:
                for sub in self._subcategories(cat):
                    if sub not in seen:
                        seen.add(sub)
                        queue.append((sub, level + 1))
        if not records:
            log.warning("Commons category '%s': no usable files (check the name)", root)
        else:
            log.info("Commons category '%s': %d candidates (%d categories visited)",
                     root, len(records), len(seen))
        return records
