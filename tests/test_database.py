import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@pytest.fixture()
def temp_db(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test.db")
    csv_path = os.path.join(tmp_dir, "test.csv")
    monkeypatch.setattr(config, "DATABASE_PATH", db_path)
    monkeypatch.setattr(config, "DETECTIONS_CSV_PATH", csv_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_dir)

    from utils import database
    database.init_db()
    return database


def test_init_db_creates_file(temp_db):
    assert os.path.exists(config.DATABASE_PATH)


def test_create_and_end_session(temp_db):
    sid = temp_db.create_session()
    assert isinstance(sid, str) and len(sid) > 0
    temp_db.end_session(sid)


def test_record_and_read_detection(temp_db):
    sid = temp_db.create_session()
    temp_db.record_detection(sid, "happy", 0.87, 42.5, 1, save_csv=False)
    rows = temp_db.get_recent_detections(limit=5, session_id=sid)
    assert len(rows) == 1
    assert rows[0]["emotion"] == "happy"
    assert rows[0]["confidence"] == 0.87


def test_emotion_distribution(temp_db):
    sid = temp_db.create_session()
    temp_db.record_detection(sid, "happy", 0.9, 10, 1, save_csv=False)
    temp_db.record_detection(sid, "happy", 0.8, 12, 1, save_csv=False)
    temp_db.record_detection(sid, "sad", 0.7, 11, 1, save_csv=False)
    dist = temp_db.get_emotion_distribution(sid)
    assert dist["happy"] == 2
    assert dist["sad"] == 1


def test_total_detections_and_avg_time(temp_db):
    sid = temp_db.create_session()
    temp_db.record_detection(sid, "neutral", 0.6, 20, 1, save_csv=False)
    temp_db.record_detection(sid, "neutral", 0.6, 40, 1, save_csv=False)
    assert temp_db.get_total_detections(sid) == 2
    assert temp_db.get_average_inference_time(sid) == 30.0
