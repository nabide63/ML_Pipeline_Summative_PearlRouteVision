"""CLI: collect images from the configured sources.

Safe to interrupt (Ctrl+C) and re-run: already-downloaded and failed URLs are skipped.

Examples:
    python scripts/collect_images.py
    python scripts/collect_images.py --classes waterfalls whitewater_rafting --target 50
    python scripts/collect_images.py --include-fallback
"""

from __future__ import annotations

import argparse

from pearlroute.config import get_settings, load_data_config
from pearlroute.data.downloader import run_collection
from pearlroute.logging_utils import get_logger, setup_logging


def main() -> None:
    cfg = load_data_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="+", choices=list(cfg.classes), help="default: all classes")
    parser.add_argument("--sources", nargs="+", choices=["openverse", "wikimedia"], help="default: both")
    parser.add_argument("--target", type=int, help=f"images per class (default {cfg.collection.target_per_class})")
    parser.add_argument("--include-fallback", action="store_true", help="also use the search terms from outside East Africa")
    parser.add_argument("--log-file", default=str(cfg.paths.raw_dir / "collect.log"))
    parser.add_argument("--verbose", action="store_true", help="log every skipped URL")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO", args.log_file)
    counts = run_collection(cfg, get_settings(), args.classes, args.sources, args.target, args.include_fallback)
    log = get_logger("collect_images")
    for label, n in counts.items():
        log.info("%-26s %d", label, n)


if __name__ == "__main__":
    main()
