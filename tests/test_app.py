import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import InferenceEngine
from utils.image_utils import preprocess_face_for_model, resize_keep_aspect, save_snapshot
import config


def _blank_frame():
    return np.full((480, 640, 3), 200, dtype=np.uint8)


def test_inference_engine_singleton():
    e1 = InferenceEngine.get_instance()
    e2 = InferenceEngine.get_instance()
    assert e1 is e2


def test_process_frame_no_face_does_not_crash():
    engine = InferenceEngine.get_instance()
    frame = _blank_frame()
    annotated, results, ms = engine.process_frame(frame, confidence_threshold=0.5)
    assert annotated.shape == frame.shape
    assert isinstance(results, list)
    assert ms >= 0


def test_process_frame_with_synthetic_face_like_pattern():
    engine = InferenceEngine.get_instance()
    frame = _blank_frame()
    # Not guaranteed to be detected as a face (synthetic), but must not crash.
    cv2.rectangle(frame, (250, 150), (400, 320), (180, 160, 140), -1)
    annotated, results, ms = engine.process_frame(frame, confidence_threshold=0.5)
    assert annotated is not None
    assert isinstance(results, list)


def test_preprocess_face_shape():
    face = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = preprocess_face_for_model(face, target_size=config.MODEL_INPUT_SIZE)
    assert tensor.shape == (1, config.MODEL_INPUT_SIZE[0], config.MODEL_INPUT_SIZE[1], 1)


def test_resize_keep_aspect_shrinks_wide_frame():
    frame = np.zeros((100, 1000, 3), dtype=np.uint8)
    resized = resize_keep_aspect(frame, target_width=500)
    assert resized.shape[1] == 500


def test_resize_keep_aspect_noop_when_already_smaller():
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    resized = resize_keep_aspect(frame, target_width=500)
    assert resized.shape[1] == 300


def test_save_snapshot_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOTS_DIR", str(tmp_path))
    frame = _blank_frame()
    path = save_snapshot(frame)
    assert os.path.exists(path)
