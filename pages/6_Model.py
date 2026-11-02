"""pages/6_Model.py - Model Information page. No invented facts; everything from config/model_manager."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header, metric_card
from utils.model_manager import get_model_status, ensure_model_downloaded, ModelManagerError

st.set_page_config(page_title=f"{config.APP_NAME} — Model", page_icon="🧩", layout="wide")
inject_css()
page_header(subtitle="Model Information")

status = get_model_status()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Model Name", status["name"])
with c2:
    metric_card("Framework", status["framework"])
with c3:
    metric_card("Input Size", f"{status['input_size'][0]}×{status['input_size'][1]} (grayscale)")
with c4:
    metric_card("Status", "✅ Downloaded" if status["exists_locally"] else "⬇️ Not downloaded")

st.write("")
st.markdown('<div class="evai-section-title">Details</div>', unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.markdown(f"""
    - **Model type:** {status['type']}
    - **Supported emotions:** {", ".join(config.EMOTION_LABELS)}
    - **Inference framework:** {status['framework']}
    - **CPU/GPU status:** Running on CPU by default (GPU used automatically if TensorFlow detects one)
    - **Local path:** `{status['path']}`
    - **File size:** {status['size_bytes']/1024:.0f} KB (0 if not yet downloaded)
    """)
with right:
    acc = status["reported_val_accuracy"]
    acc_text = f"{acc*100:.1f}% (FER2013 validation set)" if acc is not None else "Not Provided"
    st.markdown(f"""
    - **Model source:** [{status['source_url']}]({status['source_url']})
    - **License:** {config.MODEL_LICENSE}
    - **Model validation/benchmark accuracy:** {acc_text}
    - **Acquisition method:** Automatic download on first run (no manual steps, no API key)
    - **Offline behavior:** Cached locally after first download; no repeat downloads
    """)

st.info(
    "This is the model author's own documented benchmark figure from the source repository, "
    "shown here for transparency. It is not measured by this application and does not "
    "reflect performance on any specific person, camera, or lighting condition.",
    icon="ℹ️",
)

st.write("")
if not status["exists_locally"]:
    if st.button("⬇️ Download Model Now", type="primary"):
        progress = st.empty()
        try:
            ensure_model_downloaded(progress_callback=lambda m: progress.info(m))
            st.success("Model downloaded successfully. Reload this page to see updated status.")
        except ModelManagerError as e:
            st.error(f"❌ {e}")

st.divider()
st.caption(
    "To use a different model, update MODEL_SOURCES and related fields in config.py. "
    "The application will automatically download and validate the new file on next run."
)
