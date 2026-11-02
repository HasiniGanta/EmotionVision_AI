"""pages/3_Video_Analysis.py - Upload a video and process it frame-by-frame with FPS control."""

import os
import sys
import tempfile

import streamlit as st
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header, ethics_disclaimer
from core.inference import InferenceEngine
from core.session_manager import SessionManager
from utils.model_manager import ModelManagerError
from utils.image_utils import bgr_to_rgb, resize_keep_aspect

st.set_page_config(page_title=f"{config.APP_NAME} — Video Analysis", page_icon="🎬", layout="wide")
inject_css()
page_header(subtitle="Video Analysis")

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD
if "save_history" not in st.session_state:
    st.session_state.save_history = True

session_mgr = SessionManager()

st.caption("Video is processed at a configurable rate to keep CPU usage reasonable "
           "(not every frame is analyzed).")
processing_fps = st.slider("PROCESSING FPS", min_value=1, max_value=15,
                            value=config.DEFAULT_PROCESSING_FPS)

uploaded = st.file_uploader("Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded is not None:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp_file.write(uploaded.read())
    tmp_file.close()

    cap = cv2.VideoCapture(tmp_file.name)
    if not cap.isOpened():
        st.error("❌ Could not open this video file. It may be corrupt or in an unsupported format.")
        st.stop()

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(native_fps / processing_fps), 1)

    try:
        engine = InferenceEngine.get_instance(progress_callback=lambda m: st.info(m))
    except ModelManagerError as e:
        st.error(f"❌ Model unavailable: {e}")
        st.stop()

    if st.button("▶ Process Video", type="primary"):
        session_id = session_mgr.start_session()
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        summary_placeholder = st.empty()

        frame_idx = 0
        processed_count = 0
        emotion_tally = {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % frame_interval == 0 and processed_count < config.VIDEO_MAX_FRAMES_PROCESSED:
                frame = resize_keep_aspect(frame, config.FRAME_RESIZE_WIDTH)
                annotated, results, inference_ms = engine.process_frame(
                    frame, confidence_threshold=st.session_state.confidence_threshold
                )
                video_placeholder.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

                for r in results:
                    if st.session_state.save_history:
                        session_mgr.log_detection(session_id, r["emotion"], r["confidence"],
                                                   inference_ms, len(results))
                    emotion_tally[r["emotion"]] = emotion_tally.get(r["emotion"], 0) + 1
                processed_count += 1

            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        session_mgr.stop_session(session_id)

        with summary_placeholder.container():
            st.success(f"✅ Processed {processed_count} sampled frames from the video.")
            if emotion_tally:
                st.markdown("**Emotion tally across processed frames:**")
                for emo, count in sorted(emotion_tally.items(), key=lambda x: -x[1]):
                    disp = config.EMOTION_DISPLAY.get(emo, {})
                    st.write(f"{disp.get('emoji','')} {disp.get('label', emo)}: {count}")
            else:
                st.info("No faces were detected in the processed frames.")

    os.unlink(tmp_file.name) if False else None  # keep temp file until app cleans up on restart

st.divider()
ethics_disclaimer()
