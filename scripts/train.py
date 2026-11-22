"""CLI: train a model from configs.

Trains on data/processed/train, picks the best epoch on val, scores it on val and test, and saves
everything to models/<name>/ and reports/figures/ (see pearlroute.training.run).

Examples:
    python scripts/train.py --model baseline
    python scripts/train.py --model baseline --epochs 2      # quick smoke test
    python scripts/train.py --model baseline --no-augment
    python scripts/train.py --model pearlnet --run-name pearlnet_v2   # a variant, kept side by side
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from pearlroute.config import load_data_config, load_model_config, load_train_config  # noqa: E402
from pearlroute.logging_utils import setup_logging  # noqa: E402
from pearlroute.models.registry import available_models  # noqa: E402
from pearlroute.training.run import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--epochs", type=int, help="override configs/train.yaml epochs")
    parser.add_argument("--no-augment", action="store_true", help="train without augmentation")
    parser.add_argument("--run-name", help="save to models/<run-name>/ instead of models/<model>/")
    args = parser.parse_args()

    setup_logging("INFO")
    train_and_evaluate(args.model, load_data_config(), load_model_config(), load_train_config(),
                       epochs=args.epochs, augment=not args.no_augment, run_name=args.run_name)


if __name__ == "__main__":
    main()
