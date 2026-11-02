"""pages/4_History.py - Browse, filter, and export detection history."""

import os
import sys
import json
from datetime import datetime

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header
from utils import database

st.set_page_config(page_title=f"{config.APP_NAME} — History", page_icon="🗂️", layout="wide")
inject_css()
page_header(subtitle="Detection History")

database.init_db()
sessions = database.get_all_sessions()
session_options = ["All Sessions"] + [s["id"] for s in sessions]

st.markdown('<div class="evai-section-title">Filters</div>', unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns(4)
with f1:
    session_filter = st.selectbox("Session", session_options)
with f2:
    emotion_filter = st.selectbox("Emotion", ["All"] + config.EMOTION_LABELS)
with f3:
    conf_range = st.slider("Confidence Range", 0.0, 1.0, (0.0, 1.0))
with f4:
    date_range = st.date_input("Date Range", value=())

sid = None if session_filter == "All Sessions" else session_filter
date_from = date_to = None
if isinstance(date_range, tuple) and len(date_range) == 2:
    date_from = datetime.combine(date_range[0], datetime.min.time()).isoformat()
    date_to = datetime.combine(date_range[1], datetime.max.time()).isoformat()

rows = database.get_all_detections(
    session_id=sid, emotion=emotion_filter,
    min_confidence=conf_range[0], max_confidence=conf_range[1],
    date_from=date_from, date_to=date_to,
)

st.markdown(f'<div class="evai-section-title">Results ({len(rows)})</div>', unsafe_allow_html=True)

if rows:
    df = pd.DataFrame(rows)
    df["confidence_%"] = (df["confidence"] * 100).round(1)
    display_df = df[["timestamp", "session_id", "emotion", "confidence_%",
                      "inference_time_ms", "face_count"]]
    st.dataframe(display_df, use_container_width=True, height=420)

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ DOWNLOAD CSV", data=csv_bytes,
                            file_name="emotionvision_detections.csv", mime="text/csv",
                            use_container_width=True)
    with col2:
        if sid:
            stats = database.get_session_stats(sid)
            report = {
                "session_id": sid,
                "generated_at": datetime.now().isoformat(),
                "total_detections": stats["total_detections"],
                "average_confidence": stats["average_confidence"],
                "average_inference_time_ms": stats["average_inference_time"],
                "emotion_distribution": database.get_emotion_distribution(sid),
            }
            st.download_button("⬇️ DOWNLOAD SESSION REPORT", data=json.dumps(report, indent=2),
                                file_name=f"session_{sid}_report.json", mime="application/json",
                                use_container_width=True)
        else:
            st.caption("Select a specific session above to download its report.")
else:
    st.info("No detections match the current filters.")
