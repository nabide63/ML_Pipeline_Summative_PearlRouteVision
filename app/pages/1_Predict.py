"""Predict page: one photo (upload or camera) or several at once, with destinations for the answer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import api_client

DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo"   # git-ignored; may be missing


def show_prediction(p: dict, image: bytes | None = None) -> None:
    """One prediction: the photo, the answer, the top classes and where to go in East Africa."""
    left, right = st.columns([1, 1.3], gap="large")
    if image is not None:
        left.image(image, width="stretch", caption=p.get("filename"))
    with right:
        st.subheader(p["display_name"])
        st.progress(p["confidence"], text=f"Confidence {p['confidence']:.0%}")
        if p["uncertain"]:
            st.warning("Low confidence: the photo may show something outside the six experiences, "
                       "or a mix of them. Check the runner-up classes.", icon=":material/help:")
        st.markdown("**Where to have this experience:** " + ", ".join(p["destinations"]))
        top = pd.DataFrame(p["top_k"])[["display_name", "confidence"]]
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"display_name": "Class",
                                    "confidence": st.column_config.ProgressColumn(
                                        "Probability", format="percent", min_value=0, max_value=1)})
        st.caption(f"Model: {p['model']}")


st.title("Predict the experience in a photo")
st.write("PearlNet recognises six East African tourism experiences: gorilla & primate trekking, wildlife safari, "
         "whitewater rafting, waterfalls, mountain hiking and cultural heritage.")

single, batch = st.tabs(["One photo", "Several photos"])

with single:
    source = st.radio("Photo from", ["Upload", "Camera", "Demo photos"], horizontal=True,
                      label_visibility="collapsed")
    photo: tuple[str, bytes, str] | None = None
    if source == "Upload":
        f = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"])
        if f:
            photo = (f.name, f.getvalue(), f.type or "image/jpeg")
    elif source == "Camera":
        shot = st.camera_input("Take a photo (or hold a printed photo up to the camera)")
        if shot:
            photo = ("camera.jpg", shot.getvalue(), "image/jpeg")
    else:
        demo = sorted(DEMO_DIR.glob("*.jpg")) if DEMO_DIR.exists() else []
        if not demo:
            st.info("No demo photos here (data/demo is created by scripts/split_dataset.py).")
        else:
            labels = {}
            if (DEMO_DIR / "demo_labels.csv").exists():
                labels = pd.read_csv(DEMO_DIR / "demo_labels.csv").set_index("file")["label"].to_dict()
            name = st.selectbox("Demo photo (held out from every split)", [p.name for p in demo])
            if labels.get(name):
                st.caption(f"True label: **{labels[name].replace('_', ' ')}**")
            photo = (name, (DEMO_DIR / name).read_bytes(), "image/jpeg")

    if photo:
        try:
            with st.spinner("Classifying..."):
                result = api_client.predict(photo)
            show_prediction(result, photo[1])
        except api_client.ApiError as exc:
            st.error(str(exc))

with batch:
    files = st.file_uploader("Choose up to 20 photos", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key="batch")
    if files and st.button("Classify all", type="primary"):
        try:
            with st.spinner(f"Classifying {len(files)} photos..."):
                out = api_client.predict_batch([(f.name, f.getvalue(), f.type or "image/jpeg") for f in files])
        except api_client.ApiError as exc:
            st.error(str(exc))
        else:
            for err in out["errors"]:
                st.error(f"{err['filename']}: {err['error']}")
            preds = out["predictions"]
            if preds:
                table = pd.DataFrame([{"Photo": p["filename"], "Experience": p["display_name"],
                                       "Confidence": p["confidence"],
                                       "Uncertain": p["uncertain"],
                                       "Destinations": ", ".join(p["destinations"])} for p in preds])
                st.dataframe(table, hide_index=True, width="stretch",
                             column_config={"Confidence": st.column_config.ProgressColumn(
                                 format="percent", min_value=0, max_value=1)})
                images = {f.name: f.getvalue() for f in files}
                cols = st.columns(4)
                for i, p in enumerate(preds):
                    cols[i % 4].image(images.get(p["filename"]), width="stretch",
                                      caption=f"{p['display_name']} ({p['confidence']:.0%})")
