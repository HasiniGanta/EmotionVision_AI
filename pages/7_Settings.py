"""pages/7_Settings.py - App-wide configurable settings, stored in session_state."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header

st.set_page_config(page_title=f"{config.APP_NAME} — Settings", page_icon="⚙️", layout="wide")
inject_css()
page_header(subtitle="Settings")

for k, v in {
    "camera_index": 0, "processing_fps": config.DEFAULT_PROCESSING_FPS,
    "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
    "detection_interval": config.DEFAULT_DETECTION_INTERVAL_SEC,
    "theme": "Light", "sound_alerts": False, "save_history": config.SAVE_HISTORY_DEFAULT,
    "demo_mode": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown('<div class="evai-section-title">Camera</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.session_state.camera_index = st.number_input(
        "Camera Index", min_value=0, max_value=5, value=st.session_state.camera_index,
        help="0 is usually the default built-in webcam. Try 1, 2... for external cameras."
    )
with c2:
    st.session_state.processing_fps = st.slider(
        "Processing FPS", min_value=1, max_value=15, value=st.session_state.processing_fps,
        help="How many frames per second are analyzed. Lower = less CPU usage."
    )

st.markdown('<div class="evai-section-title">Detection</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0,
        value=float(st.session_state.confidence_threshold), step=0.01,
        help="Predictions below this confidence show as 'Low Confidence' instead of a specific emotion."
    )
with c4:
    st.session_state.detection_interval = st.number_input(
        "Detection Interval (seconds)", min_value=0.0, max_value=5.0,
        value=float(st.session_state.detection_interval), step=0.1,
        help="Optional extra pause between detections, on top of Processing FPS."
    )

st.markdown('<div class="evai-section-title">Interface & Alerts</div>', unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
with c5:
    st.session_state.theme = st.selectbox("Theme", ["Light", "Purple (default)"],
                                           index=["Light", "Purple (default)"].index(st.session_state.theme)
                                           if st.session_state.theme in ["Light", "Purple (default)"] else 0)
with c6:
    st.session_state.sound_alerts = st.toggle("Enable Sound Alert", value=st.session_state.sound_alerts)
with c7:
    st.session_state.save_history = st.toggle("Save History", value=st.session_state.save_history)
with c8:
    st.session_state.demo_mode = st.toggle("Demo Mode", value=st.session_state.demo_mode)

st.success("Settings are applied immediately across the app for this session.")

st.divider()
st.caption(
    "These settings live in the current browser session. Restarting the Streamlit server "
    "resets them back to the defaults defined in config.py."
)
