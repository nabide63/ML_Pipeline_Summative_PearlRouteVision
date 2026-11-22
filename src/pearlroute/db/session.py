"""Engine and session factory.

The database URL comes from DATABASE_URL in .env (see .env.example), e.g. Supabase's session pooler:
    postgresql+psycopg2://postgres.<project-ref>:<password>@aws-1-eu-west-1.pooler.supabase.com:5432/postgres
(`postgresql+psycopg2://`, not `postgresql://`: SQLAlchemy 2.1 would otherwise look for psycopg v3.)
Tests pass a SQLite URL instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from pearlroute.config import get_settings
from pearlroute.db.models import Base


class DatabaseNotConfigured(RuntimeError):
    """DATABASE_URL is not set."""


def safe_url(url: str) -> str:
    """The URL with the password hidden, for logs and error messages."""
    return make_url(url).render_as_string(hide_password=True)


def make_engine(url: str) -> Engine:
    """An engine that checks connections before use (poolers drop idle ones) and keeps the pool small."""
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=2, pool_recycle=1800)


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    """The app's engine, built once from DATABASE_URL (or the given URL)."""
    url = url or get_settings().database_url
    if not url:
        raise DatabaseNotConfigured("DATABASE_URL is not set; copy .env.example to .env and fill it in")
    return make_engine(url)


@lru_cache
def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """A session that commits on success and rolls back on error."""
    session = get_sessionmaker(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> list[str]:
    """Create any missing tables (existing ones are left untouched). Returns the tables that were created."""
    engine = engine or get_engine()
    before = set(inspect(engine).get_table_names())
    Base.metadata.create_all(engine)
    return sorted(set(inspect(engine).get_table_names()) - before)


def ping(engine: Engine | None = None) -> bool:
    """True if the database answers a trivial query."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return conn.execute(text("select 1")).scalar() == 1
