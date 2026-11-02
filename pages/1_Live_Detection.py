"""
pages/1_Live_Detection.py

Real-time webcam emotion detection: START/STOP camera, live bounding
boxes, emotion + confidence, live stat updates, snapshot saving,
and a clearly-labeled Demo Mode for when no webcam is available.
"""

import os
import sys
import time
import random

import streamlit as st
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header, metric_card, status_badge, privacy_notice, ethics_disclaimer
from core.webcam import Webcam, WebcamError
from core.inference import InferenceEngine
from core.session_manager import SessionManager
from utils.model_manager import ModelManagerError
from utils.image_utils import save_snapshot, bgr_to_rgb, resize_keep_aspect, draw_detection_box

st.set_page_config(page_title=f"{config.APP_NAME} — Live Detection", page_icon="🎥", layout="wide")
inject_css()

for k, v in {
    "camera_running": False, "demo_mode": False,
    "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
    "processing_fps": config.DEFAULT_PROCESSING_FPS, "camera_index": 0,
    "save_history": True, "current_session_id": None, "sound_alerts": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

session_mgr = SessionManager()

page_header(subtitle="Live Detection")
privacy_notice()

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
with ctrl1:
    start_clicked = st.button("▶ START CAMERA", use_container_width=True, type="primary",
                               disabled=st.session_state.camera_running)
with ctrl2:
    stop_clicked = st.button("⏹ STOP CAMERA", use_container_width=True,
                              disabled=not st.session_state.camera_running)
with ctrl3:
    snapshot_clicked = st.button("📸 TAKE SNAPSHOT", use_container_width=True,
                                  disabled=not st.session_state.camera_running)
with ctrl4:
    st.session_state.demo_mode = st.toggle(
        "Demo Mode (simulated data — no webcam needed)",
        value=st.session_state.demo_mode, disabled=st.session_state.camera_running,
    )

status_col1, status_col2 = st.columns([1, 3])
with status_col1:
    st.markdown(status_badge(st.session_state.camera_running, st.session_state.demo_mode),
                unsafe_allow_html=True)

if start_clicked:
    st.session_state.camera_running = True
    st.session_state.current_session_id = session_mgr.start_session()
    st.rerun()

if stop_clicked:
    st.session_state.camera_running = False
    if st.session_state.current_session_id:
        session_mgr.stop_session(st.session_state.current_session_id)
    st.rerun()

st.divider()

video_col, stats_col = st.columns([2, 1])
video_placeholder = video_col.empty()
info_placeholder = video_col.empty()
stats_placeholder = stats_col.container()


def render_stats(container, total, avg_ms):
    with container:
        st.markdown('<div class="evai-section-title">Session Stats</div>', unsafe_allow_html=True)
        metric_card("Detections This Run", f"{total:,}")
        st.write("")
        metric_card("Avg Response Time", f"{avg_ms:.1f} ms" if total else "—")


# --------------------------------------------------------------------------
# DEMO MODE — clearly simulated, never mixed with real stats
# --------------------------------------------------------------------------
def run_demo_frame():
    frame = np.full((360, 640, 3), 245, dtype=np.uint8)
    cv2.putText(frame, "DEMO MODE - SIMULATED DATA", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 58, 237), 2, cv2.LINE_AA)
    box = (220, 90, 200, 200)
    emotion = random.choice(config.EMOTION_LABELS)
    confidence = round(random.uniform(0.55, 0.97), 2)
    frame = draw_detection_box(frame, box, emotion, confidence)
    return frame, emotion, confidence


if st.session_state.camera_running and st.session_state.demo_mode:
    st.warning("⚠️ DEMO MODE — SIMULATED DATA. These values are randomly generated for UI "
               "demonstration only and are not real predictions.")
    frame, emotion, confidence = run_demo_frame()
    video_placeholder.image(bgr_to_rgb(frame), channels="RGB", use_container_width=True)
    disp = config.EMOTION_DISPLAY.get(emotion, {})
    info_placeholder.markdown(
        f"### {disp.get('emoji','')} {disp.get('label', emotion)} — {confidence*100:.0f}% *(simulated)*"
    )
    render_stats(stats_placeholder, total=0, avg_ms=0)
    time.sleep(1.0 / max(st.session_state.processing_fps, 1))
    st.rerun()

# --------------------------------------------------------------------------
# REAL WEBCAM MODE
# --------------------------------------------------------------------------
elif st.session_state.camera_running and not st.session_state.demo_mode:
    try:
        engine = InferenceEngine.get_instance(
            progress_callback=lambda msg: info_placeholder.info(msg)
        )
    except ModelManagerError as e:
        st.error(f"❌ Model unavailable: {e}")
        st.session_state.camera_running = False
        st.stop()

    try:
        with Webcam(st.session_state.camera_index) as cam:
            frame = cam.read_frame()
            frame = resize_keep_aspect(frame, config.FRAME_RESIZE_WIDTH)
            annotated, results, inference_ms = engine.process_frame(
                frame, confidence_threshold=st.session_state.confidence_threshold
            )

            video_placeholder.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

            if results:
                lines = []
                for i, r in enumerate(results, start=1):
                    disp = config.EMOTION_DISPLAY.get(r["emotion"], {})
                    label = "Low Confidence" if r["low_confidence"] else disp.get("label", r["emotion"])
                    emoji = "" if r["low_confidence"] else disp.get("emoji", "")
                    lines.append(f"**Face {i} → {emoji} {label} {r['confidence']*100:.0f}%**")
                    if st.session_state.save_history:
                        session_mgr.log_detection(
                            st.session_state.current_session_id, r["emotion"], r["confidence"],
                            inference_ms, len(results), save_csv=True,
                        )
                info_placeholder.markdown("  \n".join(lines))

                if st.session_state.sound_alerts and any(not r["low_confidence"] for r in results):
                    st.toast(f"Emotion detected: {results[0]['emotion'].upper()}", icon="🔔")
            else:
                info_placeholder.markdown("_No face detected in frame._")

            if snapshot_clicked:
                path = save_snapshot(annotated)
                st.success(f"Snapshot saved: {path}")

            stats = session_mgr.stats(st.session_state.current_session_id)
            render_stats(stats_placeholder, stats["total_detections"], stats["average_inference_time_ms"])

    except WebcamError as e:
        st.error(f"❌ Camera error: {e}")
        st.session_state.camera_running = False
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error during live detection: {e}")
        st.session_state.camera_running = False
        st.stop()

    time.sleep(1.0 / max(st.session_state.processing_fps, 1))
    st.rerun()

else:
    video_placeholder.markdown(
        """
        <div class="evai-card" style="text-align:center; padding:80px 20px; color:#999;">
            <h3>Camera is idle</h3>
            <p>Click <b>START CAMERA</b> above to begin real-time emotion detection,<br>
            or enable <b>Demo Mode</b> to preview the UI without a webcam.</p>
        </div>
        """, unsafe_allow_html=True,
    )
    stats = session_mgr.stats()
    render_stats(stats_placeholder, stats["total_detections"], stats["average_inference_time_ms"])

st.divider()
ethics_disclaimer()
