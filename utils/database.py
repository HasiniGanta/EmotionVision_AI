"""
utils/database.py

Local SQLite storage - no cloud database, no manual setup required.
Creates data/emotionvision.db automatically with `detections` and
`sessions` tables on first use.
"""

import os
import sqlite3
import sys
import uuid
import csv
from datetime import datetime, timezone
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Creates database and tables automatically. Safe to call repeatedly."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_detections INTEGER DEFAULT 0,
                average_confidence REAL DEFAULT 0,
                average_inference_time REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                inference_time_ms REAL NOT NULL,
                face_count INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_det_timestamp ON detections(timestamp)")
    logger.info("Database initialized (tables ensured).")

    if not os.path.exists(config.DETECTIONS_CSV_PATH):
        with open(config.DETECTIONS_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emotion", "confidence", "inference_time_ms", "face_count", "session_id"])


def create_session() -> str:
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, start_time) VALUES (?, ?)",
            (session_id, now),
        )
    logger.info(f"Session started: {session_id}")
    return session_id


def end_session(session_id: str):
    stats = get_session_stats(session_id)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """UPDATE sessions SET end_time = ?, total_detections = ?,
               average_confidence = ?, average_inference_time = ? WHERE id = ?""",
            (now, stats["total_detections"], stats["average_confidence"],
             stats["average_inference_time"], session_id),
        )
    logger.info(f"Session ended: {session_id}")


def record_detection(session_id: str, emotion: str, confidence: float,
                      inference_time_ms: float, face_count: int, save_csv: bool = True):
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO detections
               (session_id, timestamp, emotion, confidence, inference_time_ms, face_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, timestamp, emotion, confidence, inference_time_ms, face_count),
        )
    if save_csv:
        try:
            with open(config.DETECTIONS_CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, emotion, f"{confidence:.4f}",
                                  f"{inference_time_ms:.2f}", face_count, session_id])
        except Exception as e:
            logger.warning(f"Could not append to CSV: {e}")
    return timestamp


def get_recent_detections(limit: int = 10, session_id: str = None):
    query = "SELECT * FROM detections"
    params = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    query += " ORDER BY id DESC LIMIT ?"
    params = params + (limit,)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_all_detections(session_id: str = None, emotion: str = None,
                        min_confidence: float = None, max_confidence: float = None,
                        date_from: str = None, date_to: str = None):
    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if emotion and emotion != "All":
        query += " AND emotion = ?"
        params.append(emotion)
    if min_confidence is not None:
        query += " AND confidence >= ?"
        params.append(min_confidence)
    if max_confidence is not None:
        query += " AND confidence <= ?"
        params.append(max_confidence)
    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to)
    query += " ORDER BY id DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_emotion_distribution(session_id: str = None):
    query = "SELECT emotion, COUNT(*) as count FROM detections"
    params = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    query += " GROUP BY emotion"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return {r["emotion"]: r["count"] for r in rows}


def get_total_detections(session_id: str = None) -> int:
    query = "SELECT COUNT(*) as c FROM detections"
    params = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    return row["c"] if row else 0


def get_average_inference_time(session_id: str = None) -> float:
    query = "SELECT AVG(inference_time_ms) as avg_t FROM detections"
    params = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    return round(row["avg_t"], 2) if row and row["avg_t"] is not None else 0.0


def get_session_stats(session_id: str) -> dict:
    total = get_total_detections(session_id)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT AVG(confidence) as avg_c, AVG(inference_time_ms) as avg_t FROM detections WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return {
        "total_detections": total,
        "average_confidence": round(row["avg_c"], 4) if row and row["avg_c"] is not None else 0.0,
        "average_inference_time": round(row["avg_t"], 2) if row and row["avg_t"] is not None else 0.0,
    }


def get_all_sessions():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY start_time DESC").fetchall()
    return [dict(r) for r in rows]


def get_trend_data(session_id: str = None, limit: int = 200):
    """Returns chronological rows for trend charting."""
    query = "SELECT timestamp, emotion, confidence FROM detections"
    params = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    query += " ORDER BY id DESC LIMIT ?"
    params = params + (limit,)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return list(reversed([dict(r) for r in rows]))
