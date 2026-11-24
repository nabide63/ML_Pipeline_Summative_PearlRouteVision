"""CLI: apply the label rules in configs/data.yaml to photos already downloaded.

By default it only REPORTS what would change (and writes data/raw/label_rule_changes.csv to review).
Add --apply to move the files to their new class folders and drop the rest via data/exclude.txt.
Nothing is deleted. Run it while the collector is NOT running.

Examples:
    python scripts/apply_label_rules.py --fetch-keywords      # first time: get tags/categories, then report
    python scripts/apply_label_rules.py --apply
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

from pearlroute.config import get_settings, load_data_config
from pearlroute.data.downloader import write_metadata
from pearlroute.data.relabel import apply_changes, fetch_keywords, plan_changes, read_metadata
from pearlroute.logging_utils import get_logger, setup_logging

log = get_logger("apply_label_rules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-keywords", action="store_true",
                        help="first fill in missing tags/categories from the sources (saved to metadata.csv)")
    parser.add_argument("--no-openverse", action="store_true",
                        help="with --fetch-keywords: Commons only (Openverse needs one request per photo)")
    parser.add_argument("--apply", action="store_true", help="make the changes (default: report only)")
    args = parser.parse_args()
    setup_logging("INFO")
    cfg = load_data_config()
    rows = read_metadata(cfg.paths.metadata_csv)

    if args.fetch_keywords:
        n = fetch_keywords(rows, cfg, get_settings(), openverse=not args.no_openverse,
                           checkpoint=lambda: write_metadata(cfg.paths.metadata_csv, rows))
        write_metadata(cfg.paths.metadata_csv, rows)
        log.info("keywords added for %d photos", n)

    changes = plan_changes(rows, cfg)
    report = cfg.paths.raw_dir / "label_rule_changes.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "old_label", "new_label", "reason", "title"])
        writer.writerows([c.filename, c.old_label, c.new_label or "DROP", c.reason, c.title] for c in changes)
    moves = Counter((c.old_label, c.new_label or "DROP") for c in changes)
    for (old, new), n in sorted(moves.items()):
        log.info("%-26s -> %-26s %d", old, new, n)
    log.info("%d of %d photos change; details in %s", len(changes), len(rows), report)

    if args.apply and changes:
        apply_changes(rows, changes, cfg)
    elif changes:
        log.info("report only; re-run with --apply to make these changes")


if __name__ == "__main__":
    main()
