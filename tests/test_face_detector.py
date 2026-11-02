import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.face_detector import FaceDetector


def test_detector_initializes():
    detector = FaceDetector()
    assert detector is not None


def test_detect_on_blank_frame_returns_no_crash():
    detector = FaceDetector()
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = detector.detect(blank)
    assert isinstance(boxes, list)


def test_detect_on_none_returns_empty_list():
    detector = FaceDetector()
    assert detector.detect(None) == []


def test_detect_on_empty_array_returns_empty_list():
    detector = FaceDetector()
    empty = np.array([])
    assert detector.detect(empty) == []
