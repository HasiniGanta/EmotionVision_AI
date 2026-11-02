"""
ui_theme.py

Shared Streamlit styling and small UI components reused across every page,
so all pages of EmotionVision AI look consistent (purple/white dashboard).
"""

import streamlit as st
import config


def inject_css():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {config.BACKGROUND_COLOR};
        }}
        #MainMenu, footer, header {{visibility: hidden;}}

        .evai-header {{
            background: linear-gradient(135deg, {config.PRIMARY_COLOR} 0%, {config.PRIMARY_LIGHT} 100%);
            padding: 28px 32px;
            border-radius: 18px;
            margin-bottom: 24px;
            color: white;
        }}
        .evai-header h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}
        .evai-header p {{
            margin: 4px 0 0 0;
            font-size: 15px;
            opacity: 0.9;
        }}

        .evai-card {{
            background: {config.CARD_COLOR};
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            border: 1px solid {config.BORDER_COLOR};
        }}

        .evai-metric-label {{
            font-size: 13px;
            color: {config.TEXT_MUTED};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .evai-metric-value {{
            font-size: 26px;
            font-weight: 800;
            color: {config.TEXT_DARK};
            margin-top: 4px;
        }}
        .evai-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .evai-badge-live {{ background: #dcfce7; color: #15803d; }}
        .evai-badge-off {{ background: #fee2e2; color: #b91c1c; }}
        .evai-badge-demo {{ background: #fef3c7; color: #92400e; }}

        .evai-section-title {{
            font-size: 20px;
            font-weight: 800;
            color: {config.TEXT_DARK};
            margin: 8px 0 14px 0;
        }}

        .evai-detection-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 4px;
            border-bottom: 1px solid {config.BORDER_COLOR};
            font-size: 14px;
        }}

        div[data-testid="stSidebar"] {{
            background-color: {config.SIDEBAR_BG};
            border-right: 1px solid {config.BORDER_COLOR};
        }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str = None, subtitle: str = None):
    st.markdown(f"""
    <div class="evai-header">
        <h1>🧠 {config.APP_NAME}</h1>
        <p>{title or config.APP_TAGLINE}{(' — ' + subtitle) if subtitle else ''}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(f"""
    <div class="evai-card">
        <div class="evai-metric-label">{label}</div>
        <div class="evai-metric-value">{value}</div>
        <div style="font-size:12px;color:{config.TEXT_MUTED};margin-top:2px;">{help_text}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(is_live: bool, is_demo: bool = False) -> str:
    if is_demo:
        return '<span class="evai-badge evai-badge-demo">DEMO MODE</span>'
    if is_live:
        return '<span class="evai-badge evai-badge-live">● LIVE</span>'
    return '<span class="evai-badge evai-badge-off">OFFLINE</span>'


def privacy_notice():
    st.info(
        "🔒 **Privacy:** Camera frames are processed locally by the application and "
        "are not uploaded to a remote service. Only timestamp, prediction, and confidence "
        "are stored, unless you explicitly enable snapshot saving.",
        icon="🔒",
    )


def ethics_disclaimer():
    st.caption(
        "⚠️ Facial emotion recognition is an AI prediction and may be inaccurate or biased "
        "across different people, lighting conditions, expressions, and demographics. "
        "This tool is for educational, research, demonstration, and low-risk use only — "
        "it must not be used for medical/mental-health diagnosis, employment, law "
        "enforcement, or other high-stakes decisions."
    )
