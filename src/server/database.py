"""SQLite database for users, sessions, messages, attachments, and workspaces.

Schema:
  users            — login credentials (bcrypt-hashed passwords)
  sessions         — chat sessions per user
  messages         — individual messages per session
  attachments      — file attachments linked to sessions/messages
  workspaces       — n8n workflow workspaces per user
  workspace_designs— design requests and results for workspaces
"""

from __future__ import annotations

import os
import shutil
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

            CREATE TABLE IF NOT EXISTS attachments (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_id      TEXT REFERENCES messages(id) ON DELETE SET NULL,
                original_name   TEXT NOT NULL,
                stored_path     TEXT NOT NULL,
                mime_type       TEXT NOT NULL,
                size_bytes      INTEGER NOT NULL,
                content_hash    TEXT NOT NULL,
                extracted_text  TEXT,
                created_at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_attachments_session ON attachments(session_id);
            CREATE INDEX IF NOT EXISTS idx_attachments_message ON attachments(message_id);
            CREATE INDEX IF NOT EXISTS idx_attachments_hash ON attachments(content_hash);

            CREATE TABLE IF NOT EXISTS workspaces (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title            TEXT NOT NULL DEFAULT 'New Workspace',
                n8n_workflow_id  TEXT,
                status           TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','designing','designed','approved','active','rejected','failed','inactive')),
                config           TEXT DEFAULT '{}',
                created_at       REAL NOT NULL,
                updated_at       REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_workspaces_user ON workspaces(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_designs (
                id               TEXT PRIMARY KEY,
                workspace_id     TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                prompt           TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','building','complete','approved','rejected')),
                n8n_session_id   TEXT,
                n8n_workflow_id  TEXT,
                result_data      TEXT,
                created_at       REAL NOT NULL,
                updated_at       REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_designs_workspace ON workspace_designs(workspace_id, created_at DESC);
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
        deleted = cur.rowcount > 0
        
        # Clean up upload directory
        if deleted:
            upload_dir = Path(f"data/uploads/{session_id}")
            if upload_dir.exists():
                shutil.rmtree(upload_dir, ignore_errors=True)
        
        return deleted
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


# ── Attachment CRUD ──────────────────────────────────────────────────

def create_attachment(session_id: str, original_name: str, stored_path: str, mime_type: str, size_bytes: int, content_hash: str, extracted_text: str | None = None) -> dict:
    """Create attachment record. Returns dict with id, session_id, etc."""
    aid = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO attachments (id, session_id, message_id, original_name, stored_path, mime_type, size_bytes, content_hash, extracted_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, session_id, None, original_name, stored_path, mime_type, size_bytes, content_hash, extracted_text, now),
        )
        conn.commit()
        return {
            "id": aid,
            "session_id": session_id,
            "message_id": None,
            "original_name": original_name,
            "stored_path": stored_path,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "extracted_text": extracted_text,
            "created_at": now,
        }
    finally:
        conn.close()


def get_attachment(attachment_id: str) -> dict | None:
    """Get attachment by ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, session_id, message_id, original_name, stored_path, mime_type, size_bytes, content_hash, extracted_text, created_at FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_attachments(session_id: str) -> list[dict]:
    """List all attachments in a session."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, session_id, message_id, original_name, stored_path, mime_type, size_bytes, content_hash, extracted_text, created_at FROM attachments WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def link_attachment_to_message(attachment_id: str, message_id: str) -> bool:
    """Set message_id on an attachment. Returns True if updated."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE attachments SET message_id = ? WHERE id = ?",
            (message_id, attachment_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_attachment(attachment_id: str) -> bool:
    """Delete attachment record. Returns True if deleted. Caller must delete file."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM attachments WHERE id = ?",
            (attachment_id,),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_orphan_attachments(max_age_seconds: int = 3600) -> list[dict]:
    """Find attachments with no message_id older than max_age_seconds."""
    cutoff = time.time() - max_age_seconds
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, session_id, message_id, original_name, stored_path, mime_type, size_bytes, content_hash, extracted_text, created_at FROM attachments WHERE message_id IS NULL AND created_at < ?",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_messages_with_attachments(session_id: str, limit: int = 200) -> list[dict]:
    """Get messages with their attachments via LEFT JOIN. Each message dict has 'attachments' key."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT 
                m.id as message_id,
                m.role,
                m.content,
                m.created_at,
                a.id as attachment_id,
                a.original_name,
                a.mime_type,
                a.size_bytes
            FROM messages m
            LEFT JOIN attachments a ON m.id = a.message_id
            WHERE m.session_id = ?
            ORDER BY m.created_at, a.created_at
            LIMIT ?
        """, (session_id, limit * 5)).fetchall()  # *5 to account for multiple attachments per message
        
        # Group by message
        messages = {}
        for row in rows:
            msg_id = row["message_id"]
            if msg_id not in messages:
                messages[msg_id] = {
                    "id": msg_id,
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "attachments": []
                }
            
            # Add attachment if present
            if row["attachment_id"]:
                messages[msg_id]["attachments"].append({
                    "id": row["attachment_id"],
                    "original_name": row["original_name"],
                    "mime_type": row["mime_type"],
                    "size_bytes": row["size_bytes"]
                })
        
        # Return as sorted list
        return sorted(messages.values(), key=lambda x: x["created_at"])[:limit]
    finally:
        conn.close()


# ── Workspace CRUD ───────────────────────────────────────────────────

def create_workspace(user_id: str, title: str) -> dict:
    wid = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO workspaces (id, user_id, title, n8n_workflow_id, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (wid, user_id, title, None, "draft", "{}", now, now),
        )
        conn.commit()
        return {
            "id": wid,
            "user_id": user_id,
            "title": title,
            "n8n_workflow_id": None,
            "status": "draft",
            "config": "{}",
            "created_at": now,
            "updated_at": now,
        }
    finally:
        conn.close()


def list_workspaces(user_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, user_id, title, n8n_workflow_id, status, config, created_at, updated_at FROM workspaces WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_workspace(workspace_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, user_id, title, n8n_workflow_id, status, config, created_at, updated_at FROM workspaces WHERE id = ?",
            (workspace_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_workspace(workspace_id: str, user_id: str, **kwargs) -> bool:
    """Update workspace fields. Only allows updates by owner."""
    if not kwargs:
        return False
    
    # Build dynamic SET clause
    allowed_fields = {"title", "n8n_workflow_id", "status", "config"}
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return False
    
    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [workspace_id, user_id]
    
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE workspaces SET {set_clause} WHERE id = ? AND user_id = ?",
            values,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_workspace(workspace_id: str, user_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def create_workspace_design(workspace_id: str, prompt: str, n8n_session_id: str | None = None) -> dict:
    did = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO workspace_designs (id, workspace_id, prompt, status, n8n_session_id, n8n_workflow_id, result_data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (did, workspace_id, prompt, "pending", n8n_session_id, None, None, now, now),
        )
        conn.commit()
        return {
            "id": did,
            "workspace_id": workspace_id,
            "prompt": prompt,
            "status": "pending",
            "n8n_session_id": n8n_session_id,
            "n8n_workflow_id": None,
            "result_data": None,
            "created_at": now,
            "updated_at": now,
        }
    finally:
        conn.close()


def list_workspace_designs(workspace_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, workspace_id, prompt, status, n8n_session_id, n8n_workflow_id, result_data, created_at, updated_at FROM workspace_designs WHERE workspace_id = ? ORDER BY created_at DESC",
            (workspace_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_workspace_design(design_id: str, **kwargs) -> bool:
    """Update workspace design fields."""
    if not kwargs:
        return False
    
    # Build dynamic SET clause
    allowed_fields = {"status", "n8n_workflow_id", "result_data"}
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return False
    
    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [design_id]
    
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE workspace_designs SET {set_clause} WHERE id = ?",
            values,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
