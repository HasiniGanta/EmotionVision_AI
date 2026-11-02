"""
core/inference.py

The complete inference pipeline:
frame -> face detection -> crop -> preprocess -> emotion model -> results

Combines FaceDetector + EmotionClassifier and produces annotated frames
plus structured per-face results, with real timing (no fake numbers).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.face_detector import FaceDetector
from core.emotion_classifier import EmotionClassifier
from utils.image_utils import draw_detection_box
from utils.metrics import timer
from utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Loads models once and reuses them for every subsequent frame/image."""

    _instance = None

    def __init__(self, progress_callback=None):
        self.detector = FaceDetector()
        self.classifier = EmotionClassifier(progress_callback=progress_callback)

    @classmethod
    def get_instance(cls, progress_callback=None):
        if cls._instance is None:
            cls._instance = cls(progress_callback=progress_callback)
        return cls._instance

    def process_frame(self, frame_bgr: np.ndarray, confidence_threshold: float = 0.5):
        """
        Runs the full pipeline on a single BGR frame.

        Returns:
            annotated_frame (np.ndarray),
            results (list of dicts: {box, emotion, confidence, probabilities, low_confidence}),
            inference_time_ms (float, real measured time for this frame)
        """
        results = []
        annotated = frame_bgr.copy()

        with timer() as t:
            boxes = self.detector.detect(frame_bgr)
            for (x, y, w, h) in boxes:
                face_crop = frame_bgr[max(y, 0):y + h, max(x, 0):x + w]
                if face_crop.size == 0:
                    continue
                try:
                    emotion, confidence, probs = self.classifier.predict(face_crop)
                except Exception as e:
                    logger.warning(f"Emotion inference failed for a face: {e}")
                    continue

                low_conf = confidence < confidence_threshold
                results.append({
                    "box": (x, y, w, h),
                    "emotion": emotion,
                    "confidence": confidence,
                    "probabilities": probs,
                    "low_confidence": low_conf,
                })
                annotated = draw_detection_box(
                    annotated, (x, y, w, h), emotion, confidence, low_confidence=low_conf
                )

        return annotated, results, t["elapsed_ms"]
