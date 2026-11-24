"""Data insights page: three features of the dataset, each with a chart and what it means for the model.

Reads files kept in git (data/counts.json, data/metadata.csv), so it works without the API or the images.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ROOT / "data" / "counts.json"
METADATA = ROOT / "data" / "metadata.csv"


def nice(label: str) -> str:
    return label.replace("_", " ").capitalize()


@st.cache_data
def load() -> tuple[dict, pd.DataFrame]:
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    meta = pd.read_csv(METADATA)
    meta["Class"] = meta["label"].map(nice)
    meta["aspect"] = meta["orig_width"] / meta["orig_height"]
    meta["orientation"] = pd.cut(meta["aspect"], [0, 0.95, 1.05, 100],
                                 labels=["Portrait", "Square", "Landscape"])
    return counts, meta


st.title("Data insights")
if not COUNTS.exists() or not METADATA.exists():
    st.error("data/counts.json or data/metadata.csv is missing; run the data pipeline first.")
    st.stop()
counts, meta = load()
split = counts["split"]

a, b, c, d = st.columns(4)
a.metric("Photos downloaded", f"{len(meta):,}")
b.metric("Kept after cleaning", f"{sum(counts['cleaning']['clean'].values()):,}")
c.metric("Train / val / test", f"{split['totals']['train']:,} / {split['totals']['val']} / {split['totals']['test']}")
d.metric("Photographers", f"{meta['creator'].nunique():,}")

# 1. Class balance ---------------------------------------------------------------------------------
st.header("1. How many photos each class has")
rows = [{"Class": nice(cls), "Split": s, "Photos": n}
        for cls, per in split["per_class"].items() for s, n in per.items() if s in ("train", "val", "test")]
balance = pd.DataFrame(rows)
st.altair_chart(
    alt.Chart(balance).mark_bar().encode(
        x=alt.X("sum(Photos):Q", title="Photos"),
        y=alt.Y("Class:N", sort="-x", title=None),
        color=alt.Color("Split:N", sort=["train", "val", "test"],
                        scale=alt.Scale(range=["#2563eb", "#f59e0b", "#14b8a6"])),
        order=alt.Order("split_order:Q"),
        tooltip=["Class", "Split", "Photos"],
    ).transform_calculate(split_order="indexof(['train','val','test'], datum.Split)").properties(height=260),
    width="stretch")
train = {nice(k): v["train"] for k, v in split["per_class"].items()}
small, big = min(train, key=train.get), max(train, key=train.get)
st.markdown(
    f"**What it shows.** Five classes have about 450-475 training photos; **{small}** has only "
    f"{train[small]}, because fewer usable mountain photos exist under open licences. The largest class "
    f"({big}) has {split['train_imbalance']:.2f}x as many training photos as the smallest. "
    f"That is under the 1.5x limit set in `configs/data.yaml`, so no class weights were used.\n\n"
    f"**What it means for the model.** {small} is the class the model sees least, and it is also its "
    "weakest class on validation. That's why models are compared on **macro F1**, which counts each "
    "class equally, instead of plain accuracy. Every class is split about 70/15/15, so the val and "
    "test scores cover every class.")

# 2. Photo shape -----------------------------------------------------------------------------------
st.header("2. Photo shape: portrait, square or landscape")
shape = (meta.groupby(["Class", "orientation"], observed=False).size().rename("Photos").reset_index())
shape["Share"] = shape["Photos"] / shape.groupby("Class")["Photos"].transform("sum")
st.altair_chart(
    alt.Chart(shape).mark_bar().encode(
        x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="Share of the class's photos"),
        y=alt.Y("Class:N", sort=alt.EncodingSortField("Share", op="max", order="descending"), title=None),
        color=alt.Color("orientation:N", title="Orientation", sort=["Portrait", "Square", "Landscape"],
                        scale=alt.Scale(range=["#db2777", "#a3a3a3", "#4f46e5"])),
        order=alt.Order("orientation:N"),
        tooltip=["Class", "orientation", "Photos", alt.Tooltip("Share:Q", format=".0%")],
    ).properties(height=260),
    width="stretch")
portrait = meta.assign(p=meta["orientation"] == "Portrait").groupby("Class")["p"].mean()
st.markdown(
    f"**What it shows.** Most photos are landscape (median width/height {meta['aspect'].median():.2f}). "
    f"But the share of portrait photos differs a lot by class: **{portrait.idxmax()}** "
    f"{portrait.max():.0%} (tall falls are photographed upright), against **{portrait.idxmin()}** "
    f"{portrait.min():.0%} (rapids run sideways across the frame).\n\n"
    "**What it means for the model.** Every photo is squashed to a 160x160 square without cropping, "
    "so a portrait photo comes out stretched sideways. The model can learn that stretching as a weak "
    "hint for waterfalls, which is a shortcut rather than a real feature. That's one reason the "
    "augmentation only uses mild zoom and shifts, and why a tall photo of something else can get a "
    "waterfall vote.")

# 3. Photographers ---------------------------------------------------------------------------------
st.header("3. How many photographers each class comes from")
creators = (meta.groupby("Class")["creator"]
            .agg(photographers="nunique",
                 top5=lambda s: s.value_counts().head(5).sum() / len(s))
            .reset_index())
base = alt.Chart(creators).encode(y=alt.Y("Class:N", sort="-x", title=None))
st.altair_chart(
    alt.hconcat(
        base.mark_bar(color="#0d9488").encode(x=alt.X("photographers:Q", title="Different photographers"),
                                              tooltip=["Class", "photographers"]).properties(height=240),
        base.mark_bar(color="#ea580c").encode(
            x=alt.X("top5:Q", axis=alt.Axis(format="%"), title="Share taken by the top 5 photographers"),
            y=alt.Y("Class:N", sort=alt.EncodingSortField("top5", order="descending"), title=None),
            tooltip=["Class", alt.Tooltip("top5:Q", format=".0%")]).properties(height=240),
    ).resolve_scale(y="independent"),
    width="stretch")
top = creators.set_index("Class")["top5"]
st.markdown(
    f"**What it shows.** {top.idxmax()} is the most concentrated: five photographers took "
    f"{top.max():.0%} of its photos. {top.idxmin()} is the most varied ({top.min():.0%}). "
    "A few photographers uploaded whole albums of the same event or trail.\n\n"
    "**What it means for the model.** Photos from one album look alike: same light, same place, same "
    "camera. If an album were split between train and test, the test would be too easy. So the split "
    f"keeps each photographer in one split only ({split['groups']['train']} photographer groups in train, "
    f"{split['groups']['val']} in val, {split['groups']['test']} in test). It also explains part of the "
    "overfitting: a class built from a few albums has less real variety than its photo count suggests.")

with st.expander("Where the photos come from (source and licence)"):
    src = pd.crosstab(meta["Class"], meta["source"], normalize="index").round(2)
    st.dataframe(src.style.format("{:.0%}"), width="stretch")
    st.dataframe(meta["license"].value_counts().rename("Photos"), width="stretch")
    st.caption("Every photo has a licence that allows reuse (CC BY, CC BY-SA, CC0 or public domain). "
               "Credits are in data/metadata.csv.")
