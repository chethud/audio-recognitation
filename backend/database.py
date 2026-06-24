"""
ALM Database - SQLite schema and operations. WAL mode for reliability.
Default store: ``data/alm.db`` (no external DB required).
"""
import base64
import hashlib
import json
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

_PBKDF2_ITERATIONS = 310_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return (
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
    except (ValueError, OSError):
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return secrets.compare_digest(dk, expected)

DB_PATH = Path(__file__).parent.parent / "data" / "alm.db"


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


def get_conn():
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_filename TEXT NOT NULL,
            audio_duration_sec REAL,
            audio_mime TEXT,
            audio_data BLOB,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dataset_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT UNIQUE NOT NULL,
            audio_path TEXT NOT NULL,
            language TEXT,
            data_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_inferences_created ON inferences(created_at);
        CREATE INDEX IF NOT EXISTS idx_dataset_sample_id ON dataset_samples(sample_id);

        CREATE TABLE IF NOT EXISTS analyze_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_filename TEXT NOT NULL,
            audio_mime TEXT,
            audio_data BLOB,
            question TEXT NOT NULL,
            transcript TEXT,
            sounds_json TEXT,
            emotion TEXT,
            answer TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_analyze_logs_created ON analyze_logs(created_at);

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
    """)
    _ensure_column(conn, "inferences", "audio_mime", "TEXT")
    _ensure_column(conn, "inferences", "audio_data", "BLOB")
    _ensure_column(conn, "analyze_logs", "audio_mime", "TEXT")
    _ensure_column(conn, "analyze_logs", "audio_data", "BLOB")
    conn.commit()
    conn.close()


def save_inference(
    audio_filename: str,
    question: str,
    answer: str,
    audio_duration_sec: Optional[float] = None,
    audio_data: Optional[bytes] = None,
    audio_mime: Optional[str] = None,
) -> int:
    """Save inference record, return row id. Retries on busy."""
    for attempt in range(3):
        try:
            conn = get_conn()
            cur = conn.execute(
                """INSERT INTO inferences
                   (audio_filename, audio_duration_sec, audio_mime, audio_data, question, answer)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (audio_filename, audio_duration_sec, audio_mime, audio_data, question, answer),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            return row_id
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.1 * (attempt + 1))
                continue
            raise
    return 0


def get_inferences(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """Get inference history."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, audio_filename, audio_duration_sec, question, answer, created_at
           FROM inferences ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_inference_by_id(inference_id: int) -> Optional[dict[str, Any]]:
    """Get single inference by id."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM inferences WHERE id = ?", (inference_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_analyze_log(
    *,
    audio_filename: str,
    question: str,
    transcript: str,
    sounds: list[str],
    emotion: str,
    answer: str,
    audio_data: Optional[bytes] = None,
    audio_mime: Optional[str] = None,
) -> int:
    """Persist full ALM-Lite /analyze result. Returns row id."""
    sounds_json = json.dumps(sounds, ensure_ascii=False)
    for attempt in range(3):
        try:
            conn = get_conn()
            cur = conn.execute(
                """INSERT INTO analyze_logs
                   (audio_filename, audio_mime, audio_data, question, transcript, sounds_json, emotion, answer)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    audio_filename,
                    audio_mime,
                    audio_data,
                    question,
                    transcript or "",
                    sounds_json,
                    emotion or "",
                    answer,
                ),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            return int(row_id)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(0.1 * (attempt + 1))
                continue
            raise
    return 0


def get_analyze_logs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, audio_filename, question, transcript, sounds_json, emotion, answer, created_at
           FROM analyze_logs ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["sounds"] = json.loads(d.pop("sounds_json") or "[]")
        except json.JSONDecodeError:
            d["sounds"] = []
        out.append(d)
    return out


def get_analyze_by_id(log_id: int) -> Optional[dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        """SELECT id, audio_filename, question, transcript, sounds_json, emotion, answer, created_at
           FROM analyze_logs WHERE id = ?""",
        (log_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["sounds"] = json.loads(d.pop("sounds_json") or "[]")
    except json.JSONDecodeError:
        d["sounds"] = []
    return d


def get_analyze_audio(log_id: int) -> Optional[tuple[bytes, str, str]]:
    """Return (audio_bytes, mime_type, filename) for a stored analyze log."""
    conn = get_conn()
    row = conn.execute(
        "SELECT audio_data, audio_mime, audio_filename FROM analyze_logs WHERE id = ?",
        (log_id,),
    ).fetchone()
    conn.close()
    if not row or not row["audio_data"]:
        return None
    mime = row["audio_mime"] or "application/octet-stream"
    return bytes(row["audio_data"]), mime, row["audio_filename"] or "audio"


def create_user(email: str, password: str) -> int:
    """Register user; raises sqlite3.IntegrityError if email exists."""
    email = email.strip().lower()
    salt_b64, hash_b64 = _hash_password(password)
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO users (email, password_hash, password_salt)
               VALUES (?, ?, ?)""",
            (email, hash_b64, salt_b64),
        )
        uid = int(cur.lastrowid)
        conn.commit()
        return uid
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        """SELECT id, email, password_hash, password_salt FROM users
           WHERE email = ? COLLATE NOCASE""",
        (email.strip().lower(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, email FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_session(user_id: int, days_valid: int = 30) -> str:
    token = secrets.token_urlsafe(32)
    exp = (datetime.now(timezone.utc) + timedelta(days=days_valid)).isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user_id, token, exp),
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> Optional[dict[str, Any]]:
    if not token or not token.strip():
        return None
    token = token.strip()
    now = _utc_now_iso()
    conn = get_conn()
    row = conn.execute(
        """SELECT u.id, u.email FROM users u
           INNER JOIN sessions s ON s.user_id = u.id
           WHERE s.token = ? AND s.expires_at > ?""",
        (token, now),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    if not token or not token.strip():
        return
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token.strip(),))
    conn.commit()
    conn.close()
