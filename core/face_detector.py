"""
core/face_detector.py

Lightweight, dependency-free face detection using OpenCV's built-in
Haar Cascade classifier (ships with opencv-python, no extra download).
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    def __init__(self):
        cascade_path = os.path.join(cv2.data.haarcascades, config.HAAR_CASCADE_FILE)
        if not os.path.exists(cascade_path):
            raise RuntimeError(f"Haar cascade file not found at {cascade_path}")
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier.")

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns a list of bounding boxes (x, y, w, h) for detected faces.
        Never raises on a frame with zero faces or an unusual frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=config.FACE_DETECTOR_SCALE_FACTOR,
                minNeighbors=config.FACE_DETECTOR_MIN_NEIGHBORS,
                minSize=config.FACE_DETECTOR_MIN_SIZE,
            )
            return [tuple(int(v) for v in box) for box in faces]
        except Exception as e:
            logger.warning(f"Face detection failed on a frame: {e}")
            return []
