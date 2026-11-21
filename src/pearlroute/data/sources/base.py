"""ImageSource interface: search(query, n) -> records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRecord:
    """One candidate image found by a source, with the attribution needed to reuse it."""

    url: str                 # direct link to the image file that will be downloaded
    source: str              # which ImageSource found it, e.g. "openverse"
    source_id: str           # the source's own id for the image
    landing_url: str         # human-readable page for the image (used for attribution and dedupe)
    title: str
    creator: str
    license: str             # normalised: by, by-sa, cc0 or pdm
    license_url: str
    width: int | None        # original size reported by the source
    height: int | None
    query: str               # search term or category that found it


class ImageSource(ABC):
    """A searchable collection of openly licensed images.

    Adding a new provider (e.g. Flickr) means one new subclass in this package.
    """

    name: str

    @abstractmethod
    def search(self, query: str, n: int) -> list[ImageRecord]:
        """Return up to n reusable images matching the query."""
