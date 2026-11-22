"""Leakage-safe stratified train/val/test split.

Reads the cleaned manifest and:
  1. holds out `demo_total` images (spread over the classes) into demo_dir, outside every split;
  2. holds out `retrain_per_class` images per class into retrain_dir and retrain_zip;
  3. splits the rest into train/val/test so that every image by one photographer (`group_by`)
     lands in the same split, while keeping each class close to the configured ratios;
  4. links the images into processed_dir/<split>/<class>/ (the layout Keras directory loaders read)
     and writes splits_csv plus the "split" section of counts.json, with class weights when the
     training set is more imbalanced than `max_class_imbalance`.

Same seed and same cleaned manifest give the same split.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pearlroute.config import DataConfig
from pearlroute.data.cleaning import read_manual_excludes, update_counts_json
from pearlroute.logging_utils import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


def group_keys(df: pd.DataFrame, column: str) -> pd.Series:
    """Normalised group id per row; rows without a value get their own singleton group."""
    keys = df[column].fillna("").astype(str).str.strip().str.casefold()
    return keys.where(keys != "", "__single__" + df["filename"])


def select_holdouts(df: pd.DataFrame, demo_total: int, retrain_per_class: int,
                    seed: int) -> tuple[pd.Index, pd.Index]:
    """Pick row indices for the demo set (spread round-robin over classes) and the retrain sample."""
    rng = np.random.default_rng(seed)
    pools = {label: list(rng.permutation(idx.to_numpy()))
             for label, idx in sorted(df.groupby("label").groups.items())}
    demo: list[int] = []
    labels = list(pools)
    i = 0
    while len(demo) < demo_total and any(pools.values()):
        label = labels[i % len(labels)]
        if pools[label]:
            demo.append(pools[label].pop())
        i += 1
    retrain = [pools[label].pop() for label in labels for _ in range(min(retrain_per_class, len(pools[label])))]
    return pd.Index(demo), pd.Index(retrain)


def assign_groups(df: pd.DataFrame, groups: pd.Series, ratios: dict[str, float], seed: int) -> pd.Series:
    """Assign every group to one split, greedily keeping each class near the target ratios.

    Groups are placed largest first. Each goes to the split where adding it brings the per-class
    counts closest to their targets, measured in images (so overfilling counts against a split).
    Big photographers therefore land in train, and the many small groups fill val and test.
    """
    classes = sorted(df["label"].unique())
    totals = df["label"].value_counts()
    target = {s: np.array([ratios[s] * totals[c] for c in classes]) for s in SPLITS}
    current = {s: np.zeros(len(classes)) for s in SPLITS}

    counts = pd.crosstab(groups, df["label"]).reindex(columns=classes, fill_value=0)
    rng = np.random.default_rng(seed)
    shuffled = counts.iloc[rng.permutation(len(counts))]        # random order among equal sizes
    order = shuffled.sum(axis=1).sort_values(ascending=False, kind="stable").index

    assignment: dict[str, str] = {}
    for g in order:
        vec = counts.loc[g].to_numpy(dtype=float)
        def cost(s: str) -> float:
            before = np.abs(current[s] - target[s])
            after = np.abs(current[s] + vec - target[s])
            return float((after - before).sum())
        best = min(SPLITS, key=lambda s: (cost(s), -ratios[s]))
        assignment[g] = best
        current[best] += vec
    return groups.map(assignment)


def class_weights(train_counts: pd.Series) -> dict[str, float]:
    """'Balanced' weights: n_samples / (n_classes * n_class)."""
    n, k = train_counts.sum(), len(train_counts)
    return {c: round(float(n / (k * m)), 4) for c, m in train_counts.items()}


def _place(src: Path, dst: Path) -> None:
    """Hard-link (no extra disk space) or, across drives, copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def run_split(cfg: DataConfig) -> pd.DataFrame:
    """Split the cleaned manifest and lay the images out on disk. Returns the manifest with a 'split' column."""
    p, s, h = cfg.paths, cfg.split, cfg.holdout
    df = pd.read_csv(p.cleaned_csv, dtype=str, keep_default_na=False)
    ratios = {"train": s.train, "val": s.val, "test": s.test}

    demo_idx, retrain_idx = select_holdouts(df, h.demo_total, h.retrain_per_class, cfg.seed)
    df["split"] = ""
    df.loc[demo_idx, "split"] = "demo"
    df.loc[retrain_idx, "split"] = "retrain"
    rest = df[df["split"] == ""]
    groups = group_keys(rest, s.group_by)
    df.loc[rest.index, "split"] = assign_groups(rest, groups, ratios, cfg.seed)
    df["group"] = ""
    df.loc[rest.index, "group"] = groups

    # Leakage check: no photographer in more than one of train/val/test.
    in_splits = df[df["split"].isin(SPLITS)]
    leaks = in_splits.groupby("group")["split"].nunique()
    assert (leaks <= 1).all(), f"groups in several splits: {list(leaks[leaks > 1].index)[:5]}"

    # Wrong labels found in the training set: dropped from train only, AFTER the split is decided,
    # so the split (and therefore val/test) is identical with or without this list.
    train_excluded = read_manual_excludes(p.train_exclude)
    dropped_train = (df["split"] == "train") & df["filename"].isin(train_excluded)
    df.loc[dropped_train, "split"] = "train_excluded"
    if train_excluded:
        log.info("dropped %d wrongly labelled training images (%s)", dropped_train.sum(), p.train_exclude.name)

    # Lay out files. Only the folders this step owns are cleared.
    for split in SPLITS:
        _reset_dir(p.processed_dir / split)
    _reset_dir(p.demo_dir)
    _reset_dir(p.retrain_dir)
    for row in df.itertuples():
        src = p.raw_dir / row.filename
        name = Path(row.filename).name
        if row.split in SPLITS:
            _place(src, p.processed_dir / row.split / row.label / name)
        elif row.split == "retrain":
            _place(src, p.retrain_dir / row.label / name)
        else:  # demo: flat folder; the true label is in demo_labels.csv, not the folder name
            _place(src, p.demo_dir / name)
    demo = df[df["split"] == "demo"].assign(file=lambda d: d["filename"].map(lambda f: Path(f).name))
    demo[["file", "label", "title", "creator", "license", "landing_url"]].to_csv(
        p.demo_dir / "demo_labels.csv", index=False)

    p.retrain_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(p.retrain_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(p.retrain_dir.rglob("*.jpg")):
            zf.write(file, file.relative_to(p.retrain_dir).as_posix())

    p.splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p.splits_csv, index=False)

    table = pd.crosstab(df["label"], df["split"]).reindex(columns=[*SPLITS, "retrain", "demo"], fill_value=0)
    train_counts = table["train"]
    imbalance = float(train_counts.max() / max(train_counts.min(), 1))
    payload = {
        "per_class": {c: {k: int(v) for k, v in row.items()} for c, row in table.iterrows()},
        "totals": {k: int(v) for k, v in table.sum().items()},
        "groups": {sp: int(in_splits[in_splits["split"] == sp]["group"].nunique()) for sp in SPLITS},
        "train_imbalance": round(imbalance, 3),
        "class_weights": class_weights(train_counts) if imbalance > s.max_class_imbalance else None,
        "train_excluded": int(dropped_train.sum()),
    }
    update_counts_json(p.counts_json, "split", payload)

    log.info("split per class:\n%s", table.to_string())
    actual = {sp: round(float(table[sp].sum() / table[list(SPLITS)].values.sum()), 3) for sp in SPLITS}
    log.info("split shares %s (target %s); photographers per split %s", actual, ratios, payload["groups"])
    if payload["class_weights"]:
        log.info("train imbalance %.2f > %.2f: class weights written to %s",
                 imbalance, s.max_class_imbalance, p.counts_json.name)
    else:
        log.info("train imbalance %.2f <= %.2f: no class weights needed", imbalance, s.max_class_imbalance)
    return df
