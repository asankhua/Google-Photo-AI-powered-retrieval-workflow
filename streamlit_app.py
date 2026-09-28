"""
One app for both artefacts: the AI discovery engine and the Moment Finder MVP.
Switch between them in the top bar. Each still runs on its own (streamlit run mvp/app.py).

Run:  streamlit run streamlit_app.py      (Streamlit >= 1.46)
"""
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
# Both apps import their own modules by bare name (agent, library, classify, _env).
for d in ("mvp", "discovery_engine"):
    sys.path.insert(0, os.path.join(ROOT, d))

st.set_page_config(page_title="Google Photos retrieval: discovery engine + Moment Finder",
                   page_icon="🔎", layout="wide")

nav = st.navigation([
    st.Page(os.path.join(ROOT, "discovery_engine", "app.py"), title="AI discovery engine", icon="🔎",
            url_path="discovery-engine", default=True),
    st.Page(os.path.join(ROOT, "mvp", "app.py"), title="Moment Finder (MVP)", icon="📷",
            url_path="moment-finder"),
], position="top")
nav.run()
