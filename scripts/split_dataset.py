"""CLI: leakage-safe dataset split.

Run after scripts/clean_images.py. Holds out the demo images and the retrain sample, then splits
the rest into train/val/test with each photographer in a single split, and links the images into
data/processed/<split>/<class>/. Re-running rebuilds those folders from scratch.

Example:
    python scripts/split_dataset.py
"""

from __future__ import annotations

import argparse

from pearlroute.config import load_data_config
from pearlroute.data.splitting import run_split
from pearlroute.logging_utils import setup_logging


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    setup_logging("INFO")
    run_split(load_data_config())


if __name__ == "__main__":
    main()
