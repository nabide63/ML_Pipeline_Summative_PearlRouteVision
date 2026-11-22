"""CLI: create the database tables and register the trained models.

Safe to re-run: existing tables and already-registered models are left as they are.
Uses DATABASE_URL from .env (see .env.example).

Example:
    python scripts/init_db.py
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pearlroute.config import load_model_config, load_train_config  # noqa: E402
from pearlroute.db import repository as repo  # noqa: E402
from pearlroute.db.session import get_engine, init_db, safe_url, session_scope  # noqa: E402
from pearlroute.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("init_db")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    setup_logging("INFO")
    engine = get_engine()
    log.info("database: %s", safe_url(engine.url.render_as_string(hide_password=False)))
    created = init_db(engine)
    log.info("tables created: %s", created or "none (all exist)")

    models_dir = load_train_config().models_dir
    active = load_model_config().serving.active_model
    with session_scope() as session:
        for folder in sorted(p for p in models_dir.iterdir() if (p / "model.keras").exists()):
            if repo.get_model_version(session, folder.name):
                log.info("model %s already registered", folder.name)
                continue
            metrics_file = folder / "metrics.json"
            metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else None
            repo.add_model_version(session, folder.name, metrics=metrics, notes="trained offline (scripts/train.py)")
            log.info("model %s registered", folder.name)
        if repo.get_active(session) is None and repo.get_model_version(session, active):
            repo.set_active(session, active)
            log.info("active model: %s", active)
        for v in repo.list_model_versions(session):
            log.info("  %-12s active=%-5s val F1 %s  test acc %s", v.name, v.is_active, v.val_f1_macro, v.test_accuracy)


if __name__ == "__main__":
    main()
