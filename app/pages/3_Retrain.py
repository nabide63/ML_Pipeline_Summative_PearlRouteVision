"""Upload and retrain page: add labelled photos, then fine-tune the active model on them and watch it run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import api_client
from pearlroute.config import load_data_config, load_train_config

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ZIP = ROOT / "data" / "retrain_sample.zip"
CLASSES = load_data_config().class_names
MIN_NEW = load_train_config().retrain.min_new_images


def nice(label: str) -> str:
    return label.replace("_", " ").capitalize()


def show_upload_result(r: dict) -> None:
    st.success(f"Batch {r['batch_id']}: {r['accepted']} photos stored, {r['rejected']} rejected.")
    if r["accepted_per_label"]:
        st.caption("Stored: " + ", ".join(f"{nice(k)} {v}" for k, v in r["accepted_per_label"].items()))
    if r["rejections"]:
        with st.expander(f"Why {r['rejected']} photos were rejected"):
            st.dataframe(pd.DataFrame(r["rejections"]), hide_index=True, width="stretch")


st.title("Upload photos and retrain")
st.write("New labelled photos are stored in the database. Retraining fine-tunes the model being served "
         "on them, scores it on the same validation set, and only puts it live if it is at least as good.")

# 1. Upload ----------------------------------------------------------------------------------------
st.header("1. Add labelled photos")
photos_tab, zip_tab = st.tabs(["Photos of one class", "Zip with a folder per class"])
with photos_tab:
    label = st.selectbox("They all show", CLASSES, format_func=nice)
    files = st.file_uploader("Photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if st.button("Upload photos", type="primary", disabled=not files):
        try:
            with st.spinner("Checking and storing..."):
                show_upload_result(api_client.upload(
                    [(f.name, f.getvalue(), f.type or "image/jpeg") for f in files], label))
        except api_client.ApiError as exc:
            st.error(str(exc))
with zip_tab:
    st.caption("Folders named after the classes, e.g. `waterfalls/a.jpg`, `wildlife_safari/b.jpg`: "
               + ", ".join(f"`{c}`" for c in CLASSES))
    archive = st.file_uploader("Zip file", type=["zip"])
    use_sample = SAMPLE_ZIP.exists() and st.checkbox(
        f"Or use the sample zip in the repo ({SAMPLE_ZIP.stat().st_size / 1e6:.1f} MB, 10 held-out photos per class)")
    if st.button("Upload zip", type="primary", disabled=not (archive or use_sample)):
        payload = ((archive.name, archive.getvalue(), "application/zip") if archive
                   else (SAMPLE_ZIP.name, SAMPLE_ZIP.read_bytes(), "application/zip"))
        try:
            with st.spinner("Unzipping, checking and storing..."):
                show_upload_result(api_client.upload([payload]))
        except api_client.ApiError as exc:
            st.error(str(exc))

# 2. Waiting photos --------------------------------------------------------------------------------
st.header("2. Photos waiting for retraining")
try:
    stats = api_client.upload_stats()
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()
a, b, c = st.columns(3)
a.metric("Waiting", stats["unused"])
b.metric("Already used", stats["used"])
c.metric("Upload batches", stats["batches"])
waiting = pd.DataFrame({"Class": [nice(k) for k in CLASSES],
                        "Photos": [stats["unused_per_label"].get(k, 0) for k in CLASSES]})
if stats["unused"]:
    st.bar_chart(waiting, x="Class", y="Photos", horizontal=True, height=220)
st.caption("Photos already in the training data, or uploaded before, are rejected as duplicates.")

# 3. Retrain ---------------------------------------------------------------------------------------
st.header("3. Retrain")
ready = stats["unused"] >= MIN_NEW
if not ready:
    st.info(f"Upload at least {MIN_NEW} photos to retrain ({stats['unused']} waiting).")
if st.button("Start retraining", type="primary", disabled=not ready):
    try:
        st.session_state["job_id"] = api_client.start_retrain()["id"]
    except api_client.ApiError as exc:
        st.error(str(exc))

if "job_id" not in st.session_state:
    try:
        latest = api_client.latest_retrain_job()
        if latest:
            st.session_state["job_id"] = latest["id"]
    except api_client.ApiError:
        pass


def show_job(job: dict) -> None:
    status = job["status"]
    step_text = f"Step {job['step']}/{job['total_steps']}: {job['step_name']}"
    if job.get("epochs") and job["step_name"] == "fine-tune" and status == "running":
        step_text += f" (epoch {job.get('epoch') or 0}/{job['epochs']})"
    st.progress(job["step"] / job["total_steps"] if status != "succeeded" else 1.0, text=step_text)
    if job.get("last_epoch_metrics") and status == "running":
        m = job["last_epoch_metrics"]
        st.caption(" · ".join(f"{k} {v:.3f}" for k, v in m.items()))
    if status == "failed":
        st.error(f"Retraining stopped: {job['error']}")
    if status == "succeeded" and job.get("result"):
        r = job["result"]
        (st.success if r["promoted"] else st.warning)(
            f"**{r['model']}** {'is now live' if r['promoted'] else 'was kept aside; ' + r['parent'] + ' stays live'}. "
            f"{r['decision']}.")
        x, y, z = st.columns(3)
        x.metric("Val macro F1", f"{r['val_f1_macro']:.3f}",
                 f"{r['val_f1_macro'] - r['parent_val_f1_macro']:+.3f} vs {r['parent']}"
                 if r.get("parent_val_f1_macro") is not None else None)
        y.metric("Test accuracy", f"{r['test_accuracy']:.3f}",
                 f"{r['test_accuracy'] - r['parent_test_accuracy']:+.3f} vs {r['parent']}"
                 if r.get("parent_test_accuracy") is not None else None)
        z.metric("New photos used", r["new_images"])
    with st.expander("Log", expanded=status == "running"):
        st.code("\n".join(job["log"]) or "(starting)", language=None)


@st.fragment(run_every=3)
def job_panel() -> None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return
    try:
        job = api_client.retrain_job(job_id)
    except api_client.ApiError as exc:
        if exc.status == 404:          # the API restarted and forgot the job
            st.session_state.pop("job_id")
            return
        st.error(str(exc))
        return
    st.subheader(f"Job {job['id']} · {job['status']}")
    show_job(job)
    if job["status"] in ("succeeded", "failed") and not st.session_state.get(f"done_{job_id}"):
        st.session_state[f"done_{job_id}"] = True
        st.rerun()                      # refresh the waiting counts and the sidebar model name


job_panel()
