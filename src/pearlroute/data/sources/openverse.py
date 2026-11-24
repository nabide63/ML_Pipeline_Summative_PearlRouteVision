"""Openverse API image source."""

from __future__ import annotations

from typing import Any

import requests

from pearlroute.data.http import HttpClient
from pearlroute.data.sources.base import ImageRecord, ImageSource
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)

ANONYMOUS_MAX_PAGE_SIZE = 20  # Openverse API limit for requests without a token
MIN_MACHINE_TAG_ACCURACY = 0.9  # machine-generated tags carry an accuracy; keep only confident ones


def trusted_tags(tags: list[dict[str, Any]] | None) -> list[str]:
    """Tag names added by people, plus machine tags the tagger was confident about."""
    keep = []
    for tag in tags or []:
        accuracy = tag.get("accuracy")
        if tag.get("name") and (accuracy is None or float(accuracy) >= MIN_MACHINE_TAG_ACCURACY):
            keep.append(str(tag["name"]))
    return keep


class OpenverseSource(ImageSource):
    """Search Openverse (Creative Commons images from Flickr, Wikimedia and others).

    Uses OAuth client credentials when given; otherwise makes anonymous requests,
    which have smaller pages and tighter rate limits.
    """

    name = "openverse"

    def __init__(
        self,
        client: HttpClient,
        api_url: str,
        licenses: list[str],
        page_size: int,
        allowed_filetypes: list[str],
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self.client = client
        self.api_url = api_url.rstrip("/")
        self.licenses = licenses
        self.allowed_filetypes = {ext.lstrip(".").lower() for ext in allowed_filetypes}
        self._client_id = client_id or None
        self._client_secret = client_secret or None
        self._token: str | None = None
        authenticated = bool(self._client_id and self._client_secret)
        self.page_size = page_size if authenticated else min(page_size, ANONYMOUS_MAX_PAGE_SIZE)
        log.info("Openverse: %s requests, page size %d",
                 "authenticated" if authenticated else "anonymous", self.page_size)

    def _drop_auth(self, reason: str) -> None:
        """Stop using credentials for the rest of the run and fall back to anonymous requests."""
        log.warning("Openverse: %s; continuing with anonymous requests", reason)
        self._client_id = self._client_secret = self._token = None
        self.page_size = min(self.page_size, ANONYMOUS_MAX_PAGE_SIZE)

    def _headers(self, refresh: bool = False) -> dict[str, str]:
        if not (self._client_id and self._client_secret):
            return {}
        if self._token is None or refresh:
            try:
                resp = self.client.post(
                    f"{self.api_url}/auth_tokens/token/",
                    data={"grant_type": "client_credentials",
                          "client_id": self._client_id, "client_secret": self._client_secret},
                )
                resp.raise_for_status()
                self._token = resp.json()["access_token"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                self._drop_auth(f"could not get an access token ({exc})")
                return {}
        return {"Authorization": f"Bearer {self._token}"}

    def _to_record(self, item: dict[str, Any], query: str) -> ImageRecord | None:
        filetype = (item.get("filetype") or "").lower()
        if filetype and filetype not in self.allowed_filetypes:
            return None
        license_code = (item.get("license") or "").lower()
        if license_code not in self.licenses or not item.get("url"):
            return None
        return ImageRecord(
            url=item["url"],
            source=self.name,
            source_id=str(item.get("id", "")),
            landing_url=item.get("foreign_landing_url") or "",
            title=item.get("title") or "",
            creator=item.get("creator") or "unknown",
            license=license_code,
            license_url=item.get("license_url") or "",
            width=item.get("width"),
            height=item.get("height"),
            query=query,
            keywords="; ".join(trusted_tags(item.get("tags"))),
        )

    def search(self, query: str, n: int) -> list[ImageRecord]:
        """Return up to n images for the query, filtered to the allowed licences."""
        records: list[ImageRecord] = []
        page = 1
        retried_auth = False
        while len(records) < n:
            params = {"q": query, "license": ",".join(self.licenses),
                      "page_size": self.page_size, "page": page, "mature": "false"}
            resp = self.client.get(f"{self.api_url}/images/", params=params, headers=self._headers())
            if resp.status_code == 401 and self._client_id:
                if retried_auth:
                    # A freshly issued token was rejected too, so the credentials themselves are bad.
                    self._drop_auth("access token rejected (HTTP 401)")
                else:
                    self._headers(refresh=True)  # token expired during a long run
                    retried_auth = True
                continue
            if resp.status_code != 200:
                # Anonymous users hit a page cap, which shows up as a 4xx; stop paging this query.
                log.info("Openverse '%s': stopped at page %d (HTTP %d)", query, page, resp.status_code)
                break
            data = resp.json()
            results = data.get("results") or []
            records.extend(r for r in (self._to_record(item, query) for item in results) if r)
            if not results or page >= int(data.get("page_count") or page):
                break
            page += 1
        log.info("Openverse '%s': %d candidates", query, min(len(records), n))
        return records[:n]
