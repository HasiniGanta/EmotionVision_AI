"""utils/image_utils.py - shared image helper functions."""

import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    new_size = (target_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def preprocess_face_for_model(face_bgr: np.ndarray, target_size=(64, 64)) -> np.ndarray:
    """Converts a cropped BGR face into the normalized grayscale tensor the model expects."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    normalized = normalized - 0.5  # center, matches mini-XCEPTION training preprocessing
    normalized = normalized * 2.0
    tensor = np.expand_dims(normalized, axis=-1)   # (H, W, 1)
    tensor = np.expand_dims(tensor, axis=0)         # (1, H, W, 1)
    return tensor


def draw_detection_box(frame: np.ndarray, box, emotion_label: str, confidence: float,
                        color_bgr=None, low_confidence: bool = False) -> np.ndarray:
    x, y, w, h = box
    color = color_bgr or config.BOX_COLOR_BGR
    if low_confidence:
        color = (148, 148, 148)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    display = config.EMOTION_DISPLAY.get(emotion_label, {"label": emotion_label})
    text = f"{display.get('label', emotion_label)} {confidence * 100:.0f}%" if not low_confidence else "Low Confidence"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y - 10, th + 5)
    cv2.rectangle(frame, (x, label_y - th - 8), (x + tw + 8, label_y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def save_snapshot(frame_bgr: np.ndarray) -> str:
    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
    filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(config.SNAPSHOTS_DIR, filename)
    cv2.imwrite(path, frame_bgr)
    return path


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
