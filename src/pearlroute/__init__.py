"""PearlRoute Vision: East African tourism experience classifier."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pearlroute")   # from pyproject.toml
except PackageNotFoundError:               # running from source without installing
    __version__ = "0.0.0"
