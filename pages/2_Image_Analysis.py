"""pages/2_Image_Analysis.py - Upload a JPG/PNG and detect emotions on all faces."""

import os
import sys

import streamlit as st
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header, ethics_disclaimer
from core.inference import InferenceEngine
from core.session_manager import SessionManager
from utils.model_manager import ModelManagerError
from utils.image_utils import bgr_to_rgb

st.set_page_config(page_title=f"{config.APP_NAME} — Image Analysis", page_icon="🖼️", layout="wide")
inject_css()
page_header(subtitle="Image Analysis")

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD
if "save_history" not in st.session_state:
    st.session_state.save_history = True

session_mgr = SessionManager()

uploaded = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        pil_image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"❌ Invalid or corrupt image: {e}")
        st.stop()

    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    try:
        engine = InferenceEngine.get_instance(progress_callback=lambda m: st.info(m))
    except ModelManagerError as e:
        st.error(f"❌ Model unavailable: {e}")
        st.stop()

    with st.spinner("Analyzing image..."):
        annotated, results, inference_ms = engine.process_frame(
            frame_bgr, confidence_threshold=st.session_state.confidence_threshold
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True,
                  caption=f"Processed in {inference_ms:.1f} ms")
    with col2:
        st.markdown(f"### Detected Faces: {len(results)}")
        if results:
            session_id = session_mgr.start_session()
            for i, r in enumerate(results, start=1):
                disp = config.EMOTION_DISPLAY.get(r["emotion"], {})
                label = "Low Confidence" if r["low_confidence"] else disp.get("label", r["emotion"])
                st.markdown(f"**Face {i}** — {disp.get('emoji','')} {label} — "
                            f"{r['confidence']*100:.0f}%")
                if st.session_state.save_history:
                    session_mgr.log_detection(session_id, r["emotion"], r["confidence"],
                                               inference_ms, len(results))
            session_mgr.stop_session(session_id)
        else:
            st.info("No faces detected in this image.")

st.divider()
ethics_disclaimer()
