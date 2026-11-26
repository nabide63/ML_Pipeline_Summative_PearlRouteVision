"""Deploy the Docker image to a Hugging Face Space (the Streamlit app, with the API inside).

Copies the files the Dockerfile needs into a staging folder, adds the Space's README header
(sdk: docker, app_port: 8501) and the demo photos, and uploads it. Hugging Face then builds the
image itself (~10 min). Log in first with:  .venv/Scripts/hf auth login   (a token with write access)

    python scripts/deploy_hf_space.py --space <username>/pearlroute-vision
    python scripts/deploy_hf_space.py --space <username>/pearlroute-vision --set-db-secret

--set-db-secret copies DATABASE_URL from .env into the Space as a secret, so upload and
retraining can use the database. Without it, the hosted app still predicts.
Retraining on the Space has no original training photos (data/processed is not uploaded).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
COPY = ["requirements.txt", "pyproject.toml", "start.sh", "src", "configs", "scripts", "app", "models",
        "data/counts.json", "data/metadata.csv", "data/stats.json", "data/exclude.txt",
        "data/train_exclude.txt", "data/retrain_sample.zip", "data/demo"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "checkpoints", "active.json")

SPACE_HEADER = """---
title: PearlRoute Vision
emoji: 🦍
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
short_description: Classifies East African travel photos into six experiences
---

"""


def read_env_value(key: str) -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def stage(dest: Path) -> None:
    for item in COPY:
        src, out = ROOT / item, dest / item
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, out, ignore=SKIP)
        elif src.exists():
            shutil.copy2(src, out)
        else:
            raise SystemExit(f"missing {item}")
    # The Space shows README.md as its page, so it gets the Space header on top of the project README.
    (dest / "README.md").write_text(SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8"),
                                    encoding="utf-8")
    # Same Dockerfile, plus the demo photos for the Predict page (compose mounts them instead).
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerfile = dockerfile.replace("COPY start.sh .", "COPY data/demo/ data/demo/\nCOPY start.sh .", 1)
    (dest / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    ignore = [line for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
              if line.strip() != "data/demo"]
    (dest / ".dockerignore").write_text("\n".join(ignore) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="<username>/<space-name>")
    parser.add_argument("--set-db-secret", action="store_true", help="copy DATABASE_URL from .env into the Space")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    print(f"logged in as {api.whoami()['name']}")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    if args.set_db_secret:
        url = read_env_value("DATABASE_URL")
        if not url:
            raise SystemExit("DATABASE_URL not found in .env")
        api.add_space_secret(args.space, "DATABASE_URL", url)
        print("DATABASE_URL secret set")

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file()) / 1e6
        print(f"uploading {size:.0f} MB ...")
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp,
                          commit_message="Deploy PearlRoute Vision", delete_patterns=["*"])
    user, name = args.space.split("/")
    print(f"uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
    print(f"app (once built):     https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
