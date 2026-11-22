"""CLI: clean and dedupe downloaded images.

Reads data/metadata.csv and writes the cleaned and dropped manifests (see configs/data.yaml paths).
Nothing in data/raw is changed or deleted. To drop images you spotted by eye, list their
filenames (as in metadata.csv, e.g. waterfalls/openverse_ab12.jpg) in data/exclude.txt and re-run.

Run it after the collector has finished, not while it is still writing metadata.csv.

Example:
    python scripts/clean_images.py
"""

from __future__ import annotations

import argparse

from pearlroute.config import load_data_config
from pearlroute.data.cleaning import run_cleaning
from pearlroute.logging_utils import setup_logging


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    setup_logging("INFO")
    run_cleaning(load_data_config())


if __name__ == "__main__":
    main()
