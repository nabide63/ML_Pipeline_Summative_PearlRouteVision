"""Model status page: uptime, the model being served, its scores per class, and every model version."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import api_client


def nice(label: str) -> str:
    return label.replace("_", " ").capitalize()


st.title("Model status")
try:
    health = api_client.health()
    info = api_client.model_info()
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

up = health["uptime_seconds"]
a, b, c, d = st.columns(4)
a.metric("API", "Up" if health["status"] == "ok" else health["status"])
b.metric("Uptime", f"{up / 3600:.1f} h" if up >= 3600 else f"{up / 60:.0f} min")
c.metric("Serving", health["model"] or info["model_dir"].replace("\\", "/").rsplit("/", 1)[-1])
d.metric("Parameters", f"{info['parameters']:,}" if info.get("parameters") else "?")

metrics = info.get("metrics") or {}
test, val = metrics.get("test", {}), metrics.get("val", {})
if test:
    st.header("Scores on the test set")
    cols = st.columns(5)
    for col, (key, title) in zip(cols, [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"),
                                        ("precision_macro", "Precision"), ("recall_macro", "Recall"),
                                        ("roc_auc_macro", "ROC-AUC")]):
        if key in test:
            col.metric(title, f"{test[key]:.3f}",
                       f"{test[key] - val[key]:+.3f} vs val" if key in val else None, delta_color="off")
    st.caption(f"{test.get('n', '?')} test photos the model never trained on. Six classes, so guessing "
               "would score about 0.17 accuracy and 0.5 ROC-AUC.")
    if metrics.get("decision"):
        st.caption(f"Promotion decision: {metrics['decision']}")

    per_class = pd.DataFrame([{"Class": nice(k), "Metric": m, "Score": v[m]}
                              for k, v in test.get("per_class", {}).items()
                              for m in ("precision", "recall", "f1-score")])
    cm = test.get("confusion_matrix")
    left, right = st.columns(2, gap="large")
    if not per_class.empty:
        left.subheader("Per class")
        left.altair_chart(
            alt.Chart(per_class).mark_bar().encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Metric:N", title=None, axis=None),
                row=alt.Row("Class:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#2563eb", "#f59e0b", "#14b8a6"])),
                tooltip=["Class", "Metric", alt.Tooltip("Score:Q", format=".3f")],
            ).properties(height=48, width=260),
            width="content")
    if cm:
        names = [nice(n) for n in info["class_names"]]
        cells = pd.DataFrame([{"True": names[i], "Predicted": names[j], "Photos": cm[i][j]}
                              for i in range(len(names)) for j in range(len(names))])
        right.subheader("Confusion matrix")
        heat = alt.Chart(cells).encode(x=alt.X("Predicted:N", sort=names, axis=alt.Axis(labelAngle=-35)),
                                       y=alt.Y("True:N", sort=names))
        right.altair_chart(
            (heat.mark_rect().encode(color=alt.Color("Photos:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                     tooltip=["True", "Predicted", "Photos"])
             + heat.mark_text().encode(text="Photos:Q",
                                       color=alt.condition("datum.Photos > 50", alt.value("white"),
                                                           alt.value("black"))))
            .properties(height=340),
            width="stretch")
        right.caption("Rows are the true class, columns the prediction; the diagonal is correct.")

st.header("Model versions")
try:
    versions = api_client.model_versions()
except api_client.ApiError as exc:
    st.warning(f"Model history unavailable: {exc}")
    versions = []
if versions:
    table = pd.DataFrame(versions)
    table["created_at"] = pd.to_datetime(table["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(table[["name", "is_active", "parent", "created_at", "n_new_images", "val_f1_macro",
                        "test_accuracy", "test_f1_macro", "notes"]],
                 hide_index=True, width="stretch",
                 column_config={"name": "Model", "is_active": "Live", "parent": "Fine-tuned from",
                                "created_at": "Created", "n_new_images": "New photos",
                                "val_f1_macro": st.column_config.NumberColumn("Val F1", format="%.3f"),
                                "test_accuracy": st.column_config.NumberColumn("Test acc", format="%.3f"),
                                "test_f1_macro": st.column_config.NumberColumn("Test F1", format="%.3f"),
                                "notes": "Notes"})
elif not versions:
    st.caption("No versions registered yet.")
