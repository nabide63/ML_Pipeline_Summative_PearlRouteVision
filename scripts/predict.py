"""CLI: classify one or more photos with the active model.

Examples:
    python scripts/predict.py my_photo.jpg
    python scripts/predict.py data/demo/*.jpg
    python scripts/predict.py photo.jpg --json
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pearlroute.inference.predictor import InvalidImageError, get_predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="image files (JPEG, PNG, WebP, ...)")
    parser.add_argument("--json", action="store_true", help="print full results as JSON")
    args = parser.parse_args()

    predictor = get_predictor()
    results = {}
    for path in args.images:
        try:
            results[path] = predictor.predict(path).to_dict()
        except (InvalidImageError, FileNotFoundError) as exc:
            results[path] = {"error": str(exc)}
    if args.json:
        print(json.dumps(results, indent=2))
        return
    print(f"model: {predictor.info['model_dir']}")
    for path, r in results.items():
        if "error" in r:
            print(f"{path}: ERROR {r['error']}")
            continue
        flag = "  (uncertain)" if r["uncertain"] else ""
        others = ", ".join(f"{t['display_name']} {t['confidence']:.0%}" for t in r["top_k"][1:])
        print(f"{path}: {r['display_name']} {r['confidence']:.0%}{flag} | also: {others} "
              f"| where: {', '.join(r['destinations'])}")


if __name__ == "__main__":
    main()
