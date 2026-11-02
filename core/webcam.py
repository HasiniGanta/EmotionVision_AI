"""
core/webcam.py

Thin wrapper around cv2.VideoCapture with graceful error handling for
unavailable cameras, permission issues, and invalid frames.
"""

import sys
import os

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


class WebcamError(Exception):
    pass


class Webcam:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        # CAP_DSHOW improves reliability/speed on Windows; harmless elsewhere via fallback
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            raise WebcamError(
                "Could not access the webcam. It may be unavailable, in use by "
                "another application, or camera permission was denied."
            )
        logger.info(f"Webcam opened at index {self.camera_index}")

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            raise WebcamError("Webcam is not open.")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise WebcamError("Failed to read a valid frame from the webcam.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Webcam released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
