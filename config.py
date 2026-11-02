"""
EmotionVision AI - Central Configuration
All tunable parameters live here. Nothing in this project hardcodes
statistics, emotions, or model info outside of this file.
"""

import os

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "emotionvision.log")

DATABASE_PATH = os.path.join(DATA_DIR, "emotionvision.db")
DETECTIONS_CSV_PATH = os.path.join(DATA_DIR, "detections.csv")

# --------------------------------------------------------------------------
# EMOTION MODEL (pretrained, auto-downloaded on first run)
# --------------------------------------------------------------------------
# Model: mini-XCEPTION trained on FER2013
# Source: oarriaga/face_classification (MIT License), a widely used,
#         legitimately public, lightweight CPU-friendly Keras model.
# Reported benchmark accuracy in the source repository/paper: ~66%
#         (FER2013 validation set). This is the model author's own
#         reported/documented figure, not fabricated by this project.
MODEL_FILENAME = "emotion_model.hdf5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# Primary + fallback mirrors. Configure/replace if a URL goes stale.
MODEL_SOURCES = [
    "https://raw.githubusercontent.com/oarriaga/face_classification/master/"
    "trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5",
    "https://github.com/oarriaga/face_classification/raw/master/"
    "trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5",
]

MODEL_NAME = "FER2013 Mini-XCEPTION"
MODEL_TYPE = "Convolutional Neural Network (Depthwise-Separable / Xception-style)"
MODEL_INPUT_SIZE = (64, 64)  # grayscale
MODEL_FRAMEWORK = "TensorFlow / Keras"
MODEL_SOURCE_URL = "https://github.com/oarriaga/face_classification"
MODEL_LICENSE = "MIT License"
MODEL_REPORTED_VAL_ACCURACY = 0.66  # As documented by the model author (FER2013). None if unknown.
MODEL_MIN_EXPECTED_BYTES = 500_000  # sanity check for a valid download

# Emotion label order MUST match the model's training order (FER2013 standard order)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_DISPLAY = {
    "happy": {"label": "Happy", "emoji": "😊", "color": "#22c55e"},
    "sad": {"label": "Sad", "emoji": "😢", "color": "#3b82f6"},
    "angry": {"label": "Angry", "emoji": "😠", "color": "#ef4444"},
    "surprise": {"label": "Surprise", "emoji": "😮", "color": "#f59e0b"},
    "fear": {"label": "Fear", "emoji": "😨", "color": "#8b5cf6"},
    "disgust": {"label": "Disgust", "emoji": "🤢", "color": "#10b981"},
    "neutral": {"label": "Neutral", "emoji": "😐", "color": "#94a3b8"},
}

# --------------------------------------------------------------------------
# FACE DETECTION
# --------------------------------------------------------------------------
# OpenCV built-in Haar Cascade -> zero extra downloads (ships with opencv-python)
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
FACE_DETECTOR_SCALE_FACTOR = 1.1
FACE_DETECTOR_MIN_NEIGHBORS = 5
FACE_DETECTOR_MIN_SIZE = (60, 60)

# --------------------------------------------------------------------------
# INFERENCE / PERFORMANCE
# --------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_PROCESSING_FPS = 5           # frames per second actually processed
DEFAULT_DETECTION_INTERVAL_SEC = 0.0  # extra throttle between detections
FRAME_RESIZE_WIDTH = 640              # resize live frames for CPU speed
VIDEO_MAX_FRAMES_PROCESSED = 600      # safety cap for uploaded videos

# --------------------------------------------------------------------------
# UI / THEME
# --------------------------------------------------------------------------
APP_NAME = "EmotionVision AI"
APP_TAGLINE = "Real-Time Facial Emotion Intelligence"
PRIMARY_COLOR = "#5b21b6"      # stronger purple for header and accents
PRIMARY_LIGHT = "#8b5cf6"      # lighter purple for gradients and highlights
BACKGROUND_COLOR = "#f3f4f6"   # light neutral background for better contrast
CARD_COLOR = "#ffffff"
BOX_COLOR_BGR = (37, 99, 235)   # OpenCV draws in BGR - vivid box color
TEXT_DARK = "#0f172a"         # very dark slate for high contrast text
TEXT_MUTED = "#4b5563"        # muted text for secondary labels
BORDER_COLOR = "#e6e7f2"      # soft border/divider color
SIDEBAR_BG = "#ffffff"         # explicit sidebar background

# --------------------------------------------------------------------------
# HISTORY / SESSIONS
# --------------------------------------------------------------------------
RECENT_DETECTIONS_DISPLAY_COUNT = 10
SAVE_HISTORY_DEFAULT = True

# --------------------------------------------------------------------------
# APPLICATIONS SHOWN IN UI (informational only)
# --------------------------------------------------------------------------
APPLICATIONS = [
    ("Customer Experience", "Understand aggregate reaction trends in retail or service settings."),
    ("Market Research", "Gauge audience response to ads, products, or content prototypes."),
    ("Education & E-Learning", "Explore engagement signals during self-paced learning (research use)."),
    ("Human-Computer Interaction", "Prototype emotion-aware interfaces and accessibility tools."),
    ("Entertainment", "Build interactive filters, games, and reactive media experiences."),
    ("User Research", "Support qualitative UX studies with supplementary signal (not a sole metric)."),
    ("Accessibility Research", "Explore assistive technology concepts for expression-based feedback."),
]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
