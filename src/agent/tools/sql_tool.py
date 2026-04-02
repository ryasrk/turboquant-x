"""SQL query tool: execute read-only queries against a local SQLite database."""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

from src.agent.base import Tool

_MAX_ROWS = 200
_MAX_RESULT_CHARS = 8_000
# Only allow read-only statements
_ALLOWED_PREFIXES = ("SELECT", "PRAGMA", "EXPLAIN")
_DANGEROUS_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|DETACH|REPLACE|VACUUM)\b",
    re.IGNORECASE,
)


class SqlQueryTool(Tool):
    """Execute read-only SQL queries against a local SQLite database."""

    @property
    def name(self) -> str:
        return "sql_query"

    @property
    def description(self) -> str:
        return (
            "Execute a read-only SQL query against a local SQLite database. "
            "Only SELECT, PRAGMA, and EXPLAIN statements are allowed. "
            "Use db_path to specify which database file to query."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute (SELECT only)",
                },
                "db_path": {
                    "type": "string",
                    "description": "Path to the SQLite database file (default: data/turboquant.db)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"].strip()
        db_path: str = kwargs.get("db_path", "data/turboquant.db")

        # Security: validate query
        upper_query = query.lstrip().upper()
        if not any(upper_query.startswith(p) for p in _ALLOWED_PREFIXES):
            return "Error: only SELECT, PRAGMA, and EXPLAIN queries are allowed."
        if _DANGEROUS_RE.search(query):
            return "Error: query contains forbidden keywords (write operations are not allowed)."

        # Security: validate path
        if ".." in db_path or db_path.startswith("/"):
            return "Error: db_path must be a relative path without '..'."
        if not os.path.exists(db_path):
            return f"Error: database file '{db_path}' not found."

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(query)
                rows = cursor.fetchmany(_MAX_ROWS + 1)
                if not rows:
                    return "Query returned 0 rows."

                columns = [desc[0] for desc in cursor.description]
                truncated = len(rows) > _MAX_ROWS
                display_rows = rows[:_MAX_ROWS]

                # Format as table
                lines = [" | ".join(columns)]
                lines.append("-+-".join("-" * max(len(c), 5) for c in columns))
                for row in display_rows:
                    vals = [str(row[c]) if row[c] is not None else "NULL" for c in columns]
                    lines.append(" | ".join(vals))

                result = "\n".join(lines)
                if truncated:
                    result += f"\n\n... ({_MAX_ROWS}+ rows, showing first {_MAX_ROWS})"

                if len(result) > _MAX_RESULT_CHARS:
                    result = result[:_MAX_RESULT_CHARS] + "\n...[truncated]"

                return f"Rows: {min(len(display_rows), _MAX_ROWS)}\n\n{result}"
            finally:
                conn.close()
        except sqlite3.Error as e:
            return f"SQL Error: {e}"
