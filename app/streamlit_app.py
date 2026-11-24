"""Streamlit entry point: navigation only.

Run (with the API already running on port 8000):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))          # so pages can `import api_client` and `import ui`

import api_client  # noqa: E402

st.set_page_config(page_title="PearlRoute Vision", page_icon=":earth_africa:", layout="wide")

pages = st.navigation([
    st.Page(APP_DIR / "pages" / "1_Predict.py", title="Predict", icon=":material/photo_camera:", default=True),
    st.Page(APP_DIR / "pages" / "2_Insights.py", title="Data insights", icon=":material/bar_chart:"),
    st.Page(APP_DIR / "pages" / "3_Retrain.py", title="Upload & retrain", icon=":material/model_training:"),
    st.Page(APP_DIR / "pages" / "4_Model_Status.py", title="Model status", icon=":material/monitor_heart:"),
])

with st.sidebar:
    st.markdown("### PearlRoute Vision")
    st.caption("Which East African experience is in this photo?")
    try:
        h = api_client.health()
        if h["model_loaded"]:
            st.success(f"API up · serving **{h['model']}**", icon=":material/check_circle:")
        else:
            st.warning("API up, but no model loaded yet", icon=":material/hourglass_empty:")
        st.caption(f"Uptime {h['uptime_seconds'] / 60:.0f} min · v{h['version']}")
    except api_client.ApiError as exc:
        st.error(f"API offline: {exc}", icon=":material/error:")
    st.caption(f"API: `{api_client.API_URL}`")

pages.run()
