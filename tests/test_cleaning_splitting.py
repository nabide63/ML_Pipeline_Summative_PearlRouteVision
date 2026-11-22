"""Offline tests for cleaning (filters, dedupe) and the leakage-safe split, on synthetic images."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pearlroute.config import DataConfig, load_data_config
from pearlroute.data.cleaning import clean_metadata, read_manual_excludes, run_cleaning
from pearlroute.data.downloader import METADATA_COLUMNS
from pearlroute.data.splitting import SPLITS, assign_groups, group_keys, run_split, select_holdouts

CLASSES = ["alpha", "beta"]


def _config(tmp: Path) -> DataConfig:
    base = load_data_config()
    paths = base.paths.model_copy(update={
        "raw_dir": tmp / "raw", "processed_dir": tmp / "processed", "metadata_csv": tmp / "metadata.csv",
        "manual_exclude": tmp / "exclude.txt", "train_exclude": tmp / "train_exclude.txt", "cleaned_csv": tmp / "processed/cleaned.csv",
        "dropped_csv": tmp / "processed/dropped.csv", "splits_csv": tmp / "processed/splits.csv",
        "counts_json": tmp / "counts.json", "demo_dir": tmp / "demo", "retrain_dir": tmp / "retrain",
        "retrain_zip": tmp / "retrain.zip"})
    classes = {c: base.classes["waterfalls"] for c in CLASSES}
    return base.model_copy(update={
        "paths": paths, "classes": classes,
        "holdout": base.holdout.model_copy(update={"demo_total": 3, "retrain_per_class": 2})})


def _noise(seed: int, size: int = 200, colour: bool = True) -> Image.Image:
    rng = np.random.default_rng(seed)
    if colour:
        return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    return Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8)).convert("RGB")


def _write_dataset(cfg: DataConfig, images: list[tuple[str, Image.Image, dict]]) -> None:
    rows = []
    for i, (label, img, extra) in enumerate(images):
        name = f"{label}/img_{i:03d}.jpg"
        (cfg.paths.raw_dir / label).mkdir(parents=True, exist_ok=True)
        img.save(cfg.paths.raw_dir / name, quality=95)
        row = dict.fromkeys(METADATA_COLUMNS, "")
        row.update(filename=name, label=label, source="test", source_id=str(i), url=f"https://x/{i}.jpg",
                   title=f"photo {i}", creator=f"creator{i % 7}", license="by",
                   width=img.width, height=img.height)
        row.update(extra)
        rows.append(row)
    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_csv(cfg.paths.metadata_csv, index=False)


def test_cleaning_drops_each_kind_of_bad_image(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    original = _noise(1, 300)
    _write_dataset(cfg, [
        ("alpha", original, {}),                                            # kept (larger copy)
        ("alpha", original.resize((220, 220)), {}),                         # near-duplicate
        ("alpha", _noise(2), {"title": "Rwenzori locator map"}),            # title word
        ("alpha", _noise(3), {"creator": "Luigi Amedeo di Savoia"}),        # excluded creator
        ("beta", _noise(4, colour=False), {}),                              # grayscale
        ("beta", _noise(5, 100), {}),                                       # too small
        ("beta", _noise(6), {}),                                            # manual exclude
        ("beta", _noise(7), {}),                                            # kept
    ])
    cfg.paths.manual_exclude.write_text("# by eye\nbeta/img_006.jpg  # blurry\n", encoding="utf-8")

    kept = run_cleaning(cfg)
    dropped = pd.read_csv(cfg.paths.dropped_csv).set_index("filename")["drop_reason"]
    assert sorted(kept["filename"]) == ["alpha/img_000.jpg", "beta/img_007.jpg"]
    assert dropped["alpha/img_001.jpg"].startswith("near-duplicate of alpha/img_000.jpg")
    assert dropped["alpha/img_002.jpg"].startswith("excluded title word")
    assert dropped["alpha/img_003.jpg"].startswith("excluded creator")
    assert dropped["beta/img_004.jpg"] == "grayscale"
    assert dropped["beta/img_005.jpg"].startswith("too small")
    assert dropped["beta/img_006.jpg"] == "manual exclude"
    counts = json.loads(cfg.paths.counts_json.read_text())["cleaning"]
    assert counts["clean"] == {"alpha": 1, "beta": 1}
    assert len(list(cfg.paths.raw_dir.rglob("*.jpg"))) == 8              # raw data untouched


def test_cross_class_duplicates_are_both_dropped(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    img = _noise(9)
    _write_dataset(cfg, [("alpha", img, {}), ("beta", img, {})])
    kept, dropped = clean_metadata(pd.read_csv(cfg.paths.metadata_csv, dtype=str), cfg)
    assert kept.empty and len(dropped) == 2


def test_read_manual_excludes_ignores_comments_and_backslashes(tmp_path: Path) -> None:
    path = tmp_path / "exclude.txt"
    path.write_text("# header\n\nwaterfalls\\a.jpg\n  b/c.jpg # why\n", encoding="utf-8")
    assert read_manual_excludes(path) == {"waterfalls/a.jpg", "b/c.jpg"}
    assert read_manual_excludes(tmp_path / "missing.txt") == set()


def _manifest(sizes: dict[str, dict[str, int]]) -> pd.DataFrame:
    """sizes: creator -> {label: n images}."""
    rows = [{"filename": f"{label}/{creator}_{i}.jpg", "label": label, "creator": creator}
            for creator, per_label in sizes.items() for label, n in per_label.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_assign_groups_keeps_photographers_together_and_near_ratios() -> None:
    rng = np.random.default_rng(0)
    sizes = {"big": {"alpha": 60, "beta": 10}}
    sizes |= {f"c{i}": {CLASSES[i % 2]: int(rng.integers(1, 8))} for i in range(120)}
    df = _manifest(sizes)
    groups = group_keys(df, "creator")
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    split = assign_groups(df, groups, ratios, seed=1)

    assert (df.assign(split=split).groupby("creator")["split"].nunique() == 1).all()
    assert split[df["creator"] == "big"].iloc[0] == "train"               # big group never fills val/test
    share = pd.crosstab(df["label"], split, normalize="index")
    for s, r in ratios.items():
        assert share[s].between(r - 0.05, r + 0.05).all(), share


def test_group_keys_normalise_and_give_blank_creators_their_own_group() -> None:
    df = pd.DataFrame({"filename": ["a", "b", "c", "d"], "creator": ["Ann ", "ann", "", ""]})
    keys = group_keys(df, "creator")
    assert keys[0] == keys[1] and keys[2] != keys[3]


def test_select_holdouts_spreads_demo_over_classes() -> None:
    df = _manifest({"x": {"alpha": 20, "beta": 20}})
    demo, retrain = select_holdouts(df, demo_total=3, retrain_per_class=2, seed=0)
    assert set(demo).isdisjoint(retrain)
    assert sorted(df.loc[demo, "label"].value_counts().tolist()) == [1, 2]
    assert df.loc[retrain, "label"].value_counts().to_dict() == {"alpha": 2, "beta": 2}


def test_run_split_lays_out_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_dataset(cfg, [(CLASSES[i % 2], _noise(100 + i), {"creator": f"p{i % 15}"}) for i in range(80)])
    run_cleaning(cfg)
    df = run_split(cfg)

    assert (df["split"].value_counts()[["demo", "retrain"]] == [3, 4]).all()
    for split in SPLITS:
        on_disk = {p.relative_to(cfg.paths.processed_dir / split).as_posix()
                   for p in (cfg.paths.processed_dir / split).rglob("*.jpg")}
        expected = {f"{r.label}/{Path(r.filename).name}" for r in df[df["split"] == split].itertuples()}
        assert on_disk == expected
    assert len(pd.read_csv(cfg.paths.demo_dir / "demo_labels.csv")) == 3
    with zipfile.ZipFile(cfg.paths.retrain_zip) as zf:
        assert sorted(n.split("/")[0] for n in zf.namelist()) == ["alpha", "alpha", "beta", "beta"]
    counts = json.loads(cfg.paths.counts_json.read_text())
    assert {"cleaning", "split"} <= counts.keys()

    again = run_split(cfg)                                                  # same seed, same split
    assert again["split"].tolist() == df["split"].tolist()


@pytest.fixture(autouse=True)
def _quiet_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")


def test_train_exclude_only_touches_train_and_keeps_the_split(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_dataset(cfg, [(CLASSES[i % 2], _noise(300 + i), {"creator": f"p{i % 15}"}) for i in range(80)])
    run_cleaning(cfg)
    before = run_split(cfg).set_index("filename")["split"]
    train_files = before[before == "train"].index[:3].tolist()
    test_file = before[before == "test"].index[0]
    cfg.paths.train_exclude.write_text("\n".join(train_files + [test_file]) + "\n", encoding="utf-8")
    after = run_split(cfg).set_index("filename")["split"]
    assert (after[train_files] == "train_excluded").all()
    assert after[test_file] == "test"                                       # never removed from test
    unchanged = after.index.difference(train_files)
    assert (after[unchanged] == before[unchanged]).all()                     # same split otherwise
    on_disk = {q.name for q in (cfg.paths.processed_dir / "train").rglob("*.jpg")}
    assert not on_disk & {Path(f).name for f in train_files}
