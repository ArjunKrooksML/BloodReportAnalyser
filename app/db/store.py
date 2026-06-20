import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                detected_panels TEXT,
                extracted_markers TEXT,
                anomaly_tags    TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                question    TEXT NOT NULL,
                response    TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_session(session_id: str, detected_panels: list, extracted_markers: list, anomaly_tags: list):
    with _conn() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, detected_panels, extracted_markers, anomaly_tags)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                detected_panels   = excluded.detected_panels,
                extracted_markers = excluded.extracted_markers,
                anomaly_tags      = excluded.anomaly_tags
        """, (
            session_id,
            json.dumps(detected_panels),
            json.dumps(extracted_markers),
            json.dumps(anomaly_tags)
        ))


def load_session(session_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    if not row:
        return None

    return {
        "session_id": row["session_id"],
        "detected_panels": json.loads(row["detected_panels"]),
        "extracted_markers": json.loads(row["extracted_markers"]),
        "anomaly_tags": json.loads(row["anomaly_tags"])
    }


def load_chat_history(session_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT question, response FROM chat_history WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    return [{"question": r["question"], "response": r["response"]} for r in rows]


def save_chat_turn(session_id: str, question: str, response: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, question, response) VALUES (?, ?, ?)",
            (session_id, question, response)
        )
