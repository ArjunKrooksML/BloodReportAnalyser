import sqlite3
import json
import logging
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data.db"
log = logging.getLogger(__name__)


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id        TEXT PRIMARY KEY,
                patient_id        TEXT,
                detected_panels   TEXT,
                extracted_markers TEXT,
                anomaly_tags      TEXT,
                pattern_matches   TEXT,
                trend_results     TEXT,
                doctor_briefing   TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        # migrate existing DBs that lack newer columns
        for col, definition in [
            ("patient_id",      "TEXT"),
            ("pattern_matches", "TEXT"),
            ("trend_results",   "TEXT"),
            ("doctor_briefing", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {definition}")
            except Exception:
                pass


def save_session(
    session_id: str,
    detected_panels: list,
    extracted_markers: list,
    anomaly_tags: list,
    patient_id: str = "",
    pattern_matches: list = None,
    trend_results: list = None,
    doctor_briefing: dict = None,
):
    with _conn() as conn:
        conn.execute("""
            INSERT INTO sessions
                (session_id, patient_id, detected_panels, extracted_markers,
                 anomaly_tags, pattern_matches, trend_results, doctor_briefing)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                patient_id        = excluded.patient_id,
                detected_panels   = excluded.detected_panels,
                extracted_markers = excluded.extracted_markers,
                anomaly_tags      = excluded.anomaly_tags,
                pattern_matches   = excluded.pattern_matches,
                trend_results     = excluded.trend_results,
                doctor_briefing   = excluded.doctor_briefing
        """, (
            session_id,
            patient_id or "",
            json.dumps(detected_panels),
            json.dumps(extracted_markers),
            json.dumps(anomaly_tags),
            json.dumps(pattern_matches or []),
            json.dumps(trend_results or []),
            json.dumps(doctor_briefing or {}),
        ))


def load_session(session_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    if not row:
        return None

    return {
        "session_id":       row["session_id"],
        "patient_id":       row["patient_id"] or "",
        "detected_panels":  json.loads(row["detected_panels"] or "[]"),
        "extracted_markers": json.loads(row["extracted_markers"] or "[]"),
        "anomaly_tags":     json.loads(row["anomaly_tags"] or "[]"),
        "pattern_matches":  json.loads(row["pattern_matches"] or "[]"),
        "trend_results":    json.loads(row["trend_results"] or "[]"),
        "doctor_briefing":  json.loads(row["doctor_briefing"] or "{}"),
    }


def get_patient_history(patient_id: str, exclude_session: str = "") -> list[dict]:
    """Return all previous sessions for a patient ordered oldest → newest."""
    if not patient_id:
        return []
    with _conn() as conn:
        rows = conn.execute("""
            SELECT session_id, extracted_markers, created_at
            FROM sessions
            WHERE patient_id = ? AND session_id != ?
            ORDER BY created_at ASC
        """, (patient_id, exclude_session)).fetchall()

    return [
        {
            "session_id": r["session_id"],
            "markers": json.loads(r["extracted_markers"] or "[]"),
            "date": r["created_at"],
        }
        for r in rows
    ]


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
