"""SQLite database for users, sessions, and messages.

Schema:
  users    — login credentials (bcrypt-hashed passwords)
  sessions — chat sessions per user
  messages — individual messages per session
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = os.environ.get("TQ_DB_PATH", "data/turboquant.db")


def _ensure_dir() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a new connection with WAL mode and row_factory."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                username    TEXT NOT NULL UNIQUE,
                password    TEXT NOT NULL,
                created_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('system','user','assistant','tool')),
                content     TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        """)
        conn.commit()
    finally:
        conn.close()


# ── User CRUD ────────────────────────────────────────────────────────

def create_user(username: str, hashed_pw: str) -> dict:
    """Insert a new user. Returns user dict."""
    conn = get_connection()
    try:
        uid = uuid.uuid4().hex[:16]
        now = time.time()
        conn.execute(
            "INSERT INTO users (id, username, password, created_at) VALUES (?, ?, ?, ?)",
            (uid, username, hashed_pw, now),
        )
        conn.commit()
        return {"id": uid, "username": username, "created_at": now}
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    """Fetch user by username. Returns dict or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, password, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Session CRUD ─────────────────────────────────────────────────────

def create_session(user_id: str, title: str = "New Chat") -> dict:
    sid = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (sid, user_id, title, now, now),
        )
        conn.commit()
        return {"id": sid, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}
    finally:
        conn.close()


def list_sessions(user_id: str, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_session_title(session_id: str, user_id: str, title: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, time.time(), session_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_session(session_id: str, user_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Message CRUD ─────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str) -> dict:
    mid = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (mid, session_id, role, content, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()
        return {"id": mid, "session_id": session_id, "role": role, "content": content, "created_at": now}
    finally:
        conn.close()


def get_messages(session_id: str, limit: int = 200) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
