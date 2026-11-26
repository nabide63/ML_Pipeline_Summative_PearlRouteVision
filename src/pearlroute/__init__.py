"""PearlRoute Vision: East African tourism experience classifier."""

from importlib.metadata import PackageNotFoundError, version

# psycopg2 must be loaded before TensorFlow. psycopg2-binary bundles its own OpenSSL, and when
# TensorFlow's libraries load first, an SSL connection (e.g. to Supabase) crashes with a
# segmentation fault on Linux. Every entry point imports this package first, so this sets the order.
try:
    import psycopg2  # noqa: F401
except ImportError:  # the database is optional for prediction
    pass

try:
    __version__ = version("pearlroute")   # from pyproject.toml
except PackageNotFoundError:               # running from source without installing
    __version__ = "0.0.0"
