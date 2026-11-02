"""
core/emotion_classifier.py

Wraps the pretrained mini-XCEPTION model to turn a cropped face image
into an emotion label + confidence score + full probability vector.
The model is loaded exactly once via utils.model_manager (cached).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.model_manager import load_model, ModelManagerError
from utils.image_utils import preprocess_face_for_model
from utils.logger import get_logger

logger = get_logger(__name__)


class EmotionClassifier:
    def __init__(self, progress_callback=None):
        self.model = load_model(progress_callback=progress_callback)
        self.labels = config.EMOTION_LABELS

    def predict(self, face_bgr: np.ndarray):
        """
        Returns (emotion_label: str, confidence: float, probabilities: dict).
        Raises ModelManagerError only if the underlying model is unusable.
        """
        tensor = preprocess_face_for_model(face_bgr, target_size=config.MODEL_INPUT_SIZE)
        preds = self.model.predict(tensor, verbose=0)[0]

        # Defensive: make sure we always return a valid softmax-like vector
        preds = np.asarray(preds, dtype="float32")
        if preds.shape[0] != len(self.labels):
            raise ModelManagerError(
                f"Model output size {preds.shape[0]} does not match configured "
                f"emotion labels ({len(self.labels)}). Check config.EMOTION_LABELS."
            )

        probabilities = {label: float(p) for label, p in zip(self.labels, preds)}
        best_idx = int(np.argmax(preds))
        best_label = self.labels[best_idx]
        best_conf = float(preds[best_idx])
        return best_label, best_conf, probabilities
