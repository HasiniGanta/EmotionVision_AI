"""pages/8_Learn.py - 'How It Works' educational page with a simple diagram and glossary."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ui_theme import inject_css, page_header

st.set_page_config(page_title=f"{config.APP_NAME} — Learn", page_icon="📘", layout="wide")
inject_css()
page_header(subtitle="Learn — How It Works")

st.markdown('<div class="evai-section-title">Pipeline Overview</div>', unsafe_allow_html=True)
steps = [
    "1. Capture frame", "2. Detect face", "3. Crop face", "4. Preprocess image",
    "5. Run neural network", "6. Generate emotion probabilities",
    "7. Select predicted emotion", "8. Draw result", "9. Store statistics",
]
cols = st.columns(3)
for i, step in enumerate(steps):
    with cols[i % 3]:
        st.markdown(f'<div class="evai-card" style="margin-bottom:12px;">{step}</div>',
                    unsafe_allow_html=True)

st.write("")
st.markdown('<div class="evai-section-title">Simple Diagram</div>', unsafe_allow_html=True)
st.code(
    "Camera\n  ↓\nFace Detection\n  ↓\nFace Crop\n  ↓\nPreprocessing\n  ↓\n"
    "Deep Learning Model\n  ↓\nEmotion Probabilities\n  ↓\nPrediction\n  ↓\nDashboard",
    language=None,
)

st.write("")
st.markdown('<div class="evai-section-title">Glossary</div>', unsafe_allow_html=True)

glossary = {
    "CNN (Convolutional Neural Network)":
        "A type of neural network especially good at finding visual patterns in images, "
        "such as edges, textures, and shapes, by sliding small filters across the image.",
    "Face detection":
        "The process of locating where a face is within an image, before we try to "
        "understand anything about the expression on it.",
    "Image preprocessing":
        "Preparing an image for a model — resizing, converting to grayscale, and "
        "normalizing pixel values so they match what the model was trained on.",
    "Classification":
        "Assigning an input (like a face) to one of several predefined categories — "
        "here, one of the seven emotion labels.",
    "Softmax":
        "A math function that turns raw model outputs into probabilities that sum to 100%, "
        "so we can say 'Happy 87%' instead of an arbitrary number.",
    "Confidence":
        "How strongly the model believes in its top prediction, expressed as a probability.",
    "Inference":
        "Running a trained model on new data to get a prediction — as opposed to training, "
        "which is how the model learns in the first place.",
    "Model training":
        "The process of teaching a model by showing it many labeled examples so it learns "
        "to recognize patterns associated with each emotion.",
    "Validation accuracy":
        "How often a trained model's predictions were correct on data it did not see "
        "during training — a standard way to benchmark performance.",
    "Overfitting":
        "When a model memorizes its training data too closely and performs poorly on new, "
        "unseen data.",
    "Transfer learning":
        "Reusing a model that was already trained on a related task, instead of training "
        "one from scratch, to save time and data.",
}

for term, definition in glossary.items():
    with st.expander(term):
        st.write(definition)

st.write("")
st.info(
    "This project uses a **pretrained** model (mini-XCEPTION trained on FER2013) rather than "
    "training a large model from scratch. Training from scratch requires large datasets and "
    "significant compute — an optional, advanced local training module is included in the "
    "`training/` folder for anyone who wants to explore that separately.",
    icon="📘",
)
