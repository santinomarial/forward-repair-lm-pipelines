"""Offline-first experiment explorer, with paid inference behind explicit mode selection."""

from pathlib import Path
import runpy
import sys

import streamlit as st

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parent / "src"))
sys.path.insert(0, str(DEMO_DIR))

st.set_page_config(page_title="Forward Repair · Experiment Explorer", page_icon="🔎", layout="wide")
st.sidebar.title("Forward Repair")
st.sidebar.caption("RAG reliability, case by case.")
mode = st.sidebar.radio("Mode", ["Saved experiments · free", "Live playground · API calls"], key="app_mode")

if mode == "Saved experiments · free":
    from replay import render
    render()
else:
    st.warning("Live mode can incur model charges. Saved-experiment mode never calls an LLM. "
               "The study's $3 collection cap does not apply to this live playground.")
    runpy.run_path(str(DEMO_DIR / "live_app.py"), run_name="__main__")
