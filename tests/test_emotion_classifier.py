import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.emotion_classifier import EmotionClassifier


def _sample_face():
    rng = np.random.default_rng(42)
    return (rng.random((120, 120, 3)) * 255).astype("uint8")


def test_classifier_loads_model():
    clf = EmotionClassifier()
    assert clf.model is not None


def test_predict_returns_valid_emotion_and_confidence():
    clf = EmotionClassifier()
    face = _sample_face()
    emotion, confidence, probs = clf.predict(face)
    assert emotion in config.EMOTION_LABELS
    assert 0.0 <= confidence <= 1.0
    assert set(probs.keys()) == set(config.EMOTION_LABELS)
    assert abs(sum(probs.values()) - 1.0) < 0.05  # softmax should sum ~1
