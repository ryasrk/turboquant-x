"""Agent memory tools: persistent note storage and recall."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from src.agent.base import Tool

_DB_PATH = os.path.join("data", "agent_memory.db")
_lock = threading.Lock()
_MAX_KEY_LEN = 128
_MAX_VALUE_LEN = 8_000
_MAX_NOTES = 500


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  created_at TEXT NOT NULL,"
        "  updated_at TEXT NOT NULL"
        ")"
    )
    conn.commit()
    return conn


class SaveNoteTool(Tool):
    """Save a note to persistent memory (key-value store)."""

    @property
    def name(self) -> str:
        return "save_note"

    @property
    def description(self) -> str:
        return (
            "Save a note to persistent memory. Use this to remember facts, "
            "user preferences, or context across conversations. "
            "Overwrites if the key already exists."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short identifier for the note (e.g. 'user_name', 'project_goal')",
                },
                "value": {
                    "type": "string",
                    "description": "The content to store",
                },
            },
            "required": ["key", "value"],
        }

    async def execute(self, **kwargs: Any) -> str:
        key: str = kwargs["key"].strip()
        value: str = kwargs["value"].strip()

        if not key or len(key) > _MAX_KEY_LEN:
            return f"Error: key must be 1-{_MAX_KEY_LEN} characters."
        if not value or len(value) > _MAX_VALUE_LEN:
            return f"Error: value must be 1-{_MAX_VALUE_LEN} characters."

        now = datetime.now(timezone.utc).isoformat()
        with _lock:
            conn = _get_conn()
            try:
                # Check note limit
                count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                existing = conn.execute(
                    "SELECT 1 FROM notes WHERE key = ?", (key,)
                ).fetchone()
                if count >= _MAX_NOTES and not existing:
                    return f"Error: memory full ({_MAX_NOTES} notes). Delete old notes first."

                conn.execute(
                    "INSERT INTO notes (key, value, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                    (key, value, now, now, value, now),
                )
                conn.commit()
                action = "updated" if existing else "saved"
                return f"Note '{key}' {action} successfully."
            finally:
                conn.close()


class RecallNoteTool(Tool):
    """Recall a note from persistent memory."""

    @property
    def name(self) -> str:
        return "recall_note"

    @property
    def description(self) -> str:
        return (
            "Recall a note from persistent memory by key. "
            "Use key='*' to list all stored note keys."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The note key to recall, or '*' to list all keys",
                },
            },
            "required": ["key"],
        }

    async def execute(self, **kwargs: Any) -> str:
        key: str = kwargs["key"].strip()

        with _lock:
            conn = _get_conn()
            try:
                if key == "*":
                    rows = conn.execute(
                        "SELECT key, updated_at FROM notes ORDER BY updated_at DESC"
                    ).fetchall()
                    if not rows:
                        return "Memory is empty — no notes stored yet."
                    lines = [f"- {r[0]}  (updated {r[1][:10]})" for r in rows]
                    return f"Stored notes ({len(rows)}):\n" + "\n".join(lines)

                row = conn.execute(
                    "SELECT value, updated_at FROM notes WHERE key = ?", (key,)
                ).fetchone()
                if not row:
                    return f"No note found with key '{key}'."
                return f"[{key}] (updated {row[1][:10]})\n{row[0]}"
            finally:
                conn.close()


class DeleteNoteTool(Tool):
    """Delete a note from persistent memory."""

    @property
    def name(self) -> str:
        return "delete_note"

    @property
    def description(self) -> str:
        return "Delete a note from persistent memory by key."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The note key to delete",
                },
            },
            "required": ["key"],
        }

    async def execute(self, **kwargs: Any) -> str:
        key: str = kwargs["key"].strip()

        with _lock:
            conn = _get_conn()
            try:
                cursor = conn.execute("DELETE FROM notes WHERE key = ?", (key,))
                conn.commit()
                if cursor.rowcount == 0:
                    return f"No note found with key '{key}'."
                return f"Note '{key}' deleted."
            finally:
                conn.close()
