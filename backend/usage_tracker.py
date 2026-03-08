"""Roboflow API usage tracker backed by SQLite.

Replaces the previous JSON file-based tracker for thread safety
and reduced I/O overhead on each call.
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent / 'usage.db'
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return a module-level SQLite connection, creating the table on first use."""
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
                _conn.execute('PRAGMA journal_mode=WAL')
                _conn.execute('''
                    CREATE TABLE IF NOT EXISTS api_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'success'
                    )
                ''')
                _conn.commit()
    return _conn


def log_api_call(model_id: str, status: str = "success"):
    """Logs a Roboflow API call to track quota usage."""
    conn = _get_conn()
    with _lock:
        conn.execute(
            'INSERT INTO api_calls (timestamp, model, status) VALUES (?, ?, ?)',
            (datetime.now().isoformat(), model_id, status),
        )
        conn.commit()


def get_usage_stats() -> int:
    """Returns total API calls recorded."""
    conn = _get_conn()
    with _lock:
        row = conn.execute('SELECT COUNT(*) FROM api_calls').fetchone()
        return row[0] if row else 0
