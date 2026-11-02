"""pages/5_Analytics.py - Deeper aggregate analytics across all recorded detections."""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header, metric_card
from utils import database

st.set_page_config(page_title=f"{config.APP_NAME} — Analytics", page_icon="📊", layout="wide")
inject_css()
page_header(subtitle="Analytics")

database.init_db()
rows = database.get_all_detections()

if not rows:
    st.info("No detection data yet. Use Live Detection, Image Analysis, or Video Analysis first.")
    st.stop()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.floor("h")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Records", f"{len(df):,}")
with c2:
    metric_card("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
with c3:
    metric_card("Avg Inference Time", f"{df['inference_time_ms'].mean():.1f} ms")
with c4:
    metric_card("Sessions Recorded", f"{df['session_id'].nunique():,}")

st.write("")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="evai-section-title">Detections Over Time</div>', unsafe_allow_html=True)
    ts_counts = df.groupby("hour").size().reset_index(name="count")
    fig = px.bar(ts_counts, x="hour", y="count",
                 color_discrete_sequence=[config.PRIMARY_COLOR])
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="evai-section-title">Average Confidence by Emotion</div>', unsafe_allow_html=True)
    conf_by_emotion = df.groupby("emotion")["confidence"].mean().reset_index()
    conf_by_emotion["confidence"] = (conf_by_emotion["confidence"] * 100).round(1)
    fig2 = px.bar(conf_by_emotion, x="emotion", y="confidence",
                  color="emotion",
                  color_discrete_map={e: config.EMOTION_DISPLAY.get(e, {}).get("color", "#888")
                                       for e in config.EMOTION_LABELS})
    fig2.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False, yaxis_title="Avg Confidence (%)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="evai-section-title">Face Count Distribution</div>', unsafe_allow_html=True)
fc_counts = df["face_count"].value_counts().sort_index().reset_index()
fc_counts.columns = ["faces_in_frame", "occurrences"]
fig3 = px.bar(fc_counts, x="faces_in_frame", y="occurrences",
              color_discrete_sequence=[config.PRIMARY_LIGHT])
fig3.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)
