"""
core/session_manager.py

Manages detection sessions (start/stop) and records real detections
to the local SQLite database + CSV. No cloud dependency.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import database
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    def __init__(self):
        database.init_db()

    def start_session(self) -> str:
        return database.create_session()

    def stop_session(self, session_id: str):
        if session_id:
            database.end_session(session_id)

    def log_detection(self, session_id: str, emotion: str, confidence: float,
                       inference_time_ms: float, face_count: int, save_csv: bool = True):
        return database.record_detection(
            session_id, emotion, confidence, inference_time_ms, face_count, save_csv=save_csv
        )

    def stats(self, session_id: str = None) -> dict:
        total = database.get_total_detections(session_id)
        avg_time = database.get_average_inference_time(session_id)
        distribution = database.get_emotion_distribution(session_id)
        recent = database.get_recent_detections(limit=10, session_id=session_id)
        return {
            "total_detections": total,
            "average_inference_time_ms": avg_time,
            "distribution": distribution,
            "recent": recent,
        }
