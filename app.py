"""
EmotionVision AI - app.py
Main entry point. Run with:  streamlit run app.py

This page is the Dashboard: live status, aggregate stats, emotion
distribution, recent detections, trend chart, applications. It reads
real data from the local SQLite database (data/emotionvision.db).
Use the sidebar to navigate to Live Detection, Image/Video Analysis, etc.
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from ui_theme import inject_css, page_header, metric_card, ethics_disclaimer, privacy_notice
from utils import database
from utils.model_manager import get_model_status

st.set_page_config(
    page_title=f"{config.APP_NAME} — Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
database.init_db()

# --------------------------------------------------------------------------
# Session state defaults (shared across all pages of the app)
# --------------------------------------------------------------------------
defaults = {
    "camera_running": False,
    "demo_mode": False,
    "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
    "processing_fps": config.DEFAULT_PROCESSING_FPS,
    "camera_index": 0,
    "theme": "Light",
    "sound_alerts": False,
    "save_history": config.SAVE_HISTORY_DEFAULT,
    "current_session_id": None,
    "engine_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------------------------------
# Sidebar navigation info + quick status
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🧠 {config.APP_NAME}")
    st.caption(config.APP_TAGLINE)
    st.divider()
    st.markdown("**Navigate using the pages above** ⬆️")
    st.divider()
    model_status = get_model_status()
    if model_status["exists_locally"]:
        st.success(f"Model ready ({model_status['size_bytes']/1024:.0f} KB)")
    else:
        st.warning("Model not downloaded yet — visit Live Detection or Model page to fetch it.")

page_header(subtitle="Dashboard")

# --------------------------------------------------------------------------
# Top statistics cards (REAL data only)
# --------------------------------------------------------------------------
total_detections = database.get_total_detections()
avg_time = database.get_average_inference_time()
is_live = st.session_state.get("camera_running", False)

if model_status["reported_val_accuracy"] is not None:
    acc_display = f"{model_status['reported_val_accuracy']*100:.1f}%"
    acc_help = "Model's documented benchmark accuracy (FER2013)"
else:
    acc_display = "Not Provided"
    acc_help = "No official benchmark accuracy available"

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Live Status", "🟢 LIVE" if is_live else "⚪ IDLE",
                "Camera currently running" if is_live else "Start on Live Detection page")
with c2:
    metric_card("Total Detections", f"{total_detections:,}", "All-time, from real inference")
with c3:
    metric_card("Model Accuracy", acc_display, acc_help)
with c4:
    metric_card("Avg Response Time", f"{avg_time:.1f} ms" if total_detections else "—",
                "Measured inference latency")

st.write("")

# --------------------------------------------------------------------------
# Emotion Distribution + Recent Detections
# --------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="evai-section-title">Emotion Distribution</div>', unsafe_allow_html=True)
    distribution = database.get_emotion_distribution()
    if distribution:
        labels = [config.EMOTION_DISPLAY.get(e, {}).get("label", e) for e in distribution.keys()]
        colors = [config.EMOTION_DISPLAY.get(e, {}).get("color", "#999") for e in distribution.keys()]
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=list(distribution.values()), hole=0.55,
            marker=dict(colors=colors), textinfo="label+percent",
        )])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340,
                           showlegend=True, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections yet. Run Live Detection or analyze an image to populate this chart.")

with col_right:
    st.markdown('<div class="evai-section-title">Recent Detections</div>', unsafe_allow_html=True)
    recent = database.get_recent_detections(limit=config.RECENT_DETECTIONS_DISPLAY_COUNT)
    if recent:
        for row in recent:
            disp = config.EMOTION_DISPLAY.get(row["emotion"], {"label": row["emotion"], "emoji": ""})
            ts = row["timestamp"].split("T")[1][:8] if "T" in row["timestamp"] else row["timestamp"]
            st.markdown(f"""
            <div class="evai-detection-row">
                <span>{ts} &nbsp; {disp.get('emoji','')} {disp.get('label', row['emotion'])}</span>
                <span><b>{row['confidence']*100:.0f}%</b></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent detections yet.")

st.write("")

# --------------------------------------------------------------------------
# Real-Time Emotion Trend
# --------------------------------------------------------------------------
st.markdown('<div class="evai-section-title">Real-Time Emotion Trend</div>', unsafe_allow_html=True)
trend_rows = database.get_trend_data(limit=150)
if trend_rows:
    df = pd.DataFrame(trend_rows)
    df["time"] = pd.to_datetime(df["timestamp"])
    fig2 = px.line(df, x="time", y="confidence", color="emotion",
                    color_discrete_map={e: config.EMOTION_DISPLAY.get(e, {}).get("color", "#888")
                                         for e in config.EMOTION_LABELS},
                    markers=True)
    fig2.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Emotion",
                        yaxis_title="Confidence", xaxis_title="Time")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Trend chart will populate as real detections are recorded.")

st.write("")

# --------------------------------------------------------------------------
# Applications
# --------------------------------------------------------------------------
st.markdown('<div class="evai-section-title">Applications</div>', unsafe_allow_html=True)
app_cols = st.columns(4)
for i, (name, desc) in enumerate(config.APPLICATIONS):
    with app_cols[i % 4]:
        st.markdown(f"""
        <div class="evai-card" style="min-height:120px;margin-bottom:14px;">
            <b>{name}</b>
            <p style="font-size:13px;color:#666;margin-top:6px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.write("")
ethics_disclaimer()
