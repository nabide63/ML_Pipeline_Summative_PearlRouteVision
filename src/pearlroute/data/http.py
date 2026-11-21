"""HTTP client shared by the image sources and the downloader: throttling and retries."""

from __future__ import annotations

import time
from typing import Any

import requests

from pearlroute.logging_utils import get_logger

log = get_logger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class HttpClient:
    """A requests.Session that waits between calls and retries rate limits and server errors."""

    def __init__(
        self,
        user_agent: str,
        timeout: float,
        sleep_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        max_retry_after: float,
    ) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.max_retry_after = max_retry_after
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self.sleep_seconds - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _retry_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                return min(float(retry_after), self.max_retry_after)
        return min(self.backoff_seconds * 2**attempt, self.max_retry_after)

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Send a request, retrying network errors, 429 and 5xx with exponential backoff.

        Returns the final response (which may still be an error status); raises only if
        every attempt failed at the network level.
        """
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise
                delay = self._retry_delay(attempt)
                log.warning("%s on %s; retry %d in %.0fs", type(exc).__name__, url[:80], attempt + 1, delay)
                time.sleep(delay)
                continue
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                delay = self._retry_delay(attempt, response)
                log.warning("HTTP %d from %s; retry %d in %.0fs",
                            response.status_code, url[:80], attempt + 1, delay)
                time.sleep(delay)
                continue
            return response
        raise RuntimeError("unreachable")  # loop always returns or raises

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """GET with throttling and retries."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """POST with throttling and retries."""
        return self.request("POST", url, **kwargs)
