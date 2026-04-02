"""Diff tool: compare two files and show differences."""
from __future__ import annotations

import difflib
import os
from typing import Any

from src.agent.base import Tool

_MAX_FILE_SIZE = 256_000  # 256KB per file
_MAX_OUTPUT_CHARS = 12_000


class DiffFilesTool(Tool):
    """Compare two files and show a unified diff."""

    @property
    def name(self) -> str:
        return "diff_files"

    @property
    def description(self) -> str:
        return (
            "Compare two files and show a unified diff. "
            "Shows additions (+), deletions (-), and context lines. "
            "Useful for reviewing changes between file versions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_a": {
                    "type": "string",
                    "description": "Path to the first (original) file",
                },
                "file_b": {
                    "type": "string",
                    "description": "Path to the second (modified) file",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines around changes (default: 3)",
                },
            },
            "required": ["file_a", "file_b"],
        }

    async def execute(self, **kwargs: Any) -> str:
        file_a: str = kwargs["file_a"]
        file_b: str = kwargs["file_b"]
        context: int = int(kwargs.get("context_lines", 3))

        # Security: prevent path traversal
        for path in (file_a, file_b):
            if ".." in path:
                return "Error: path must not contain '..'."

        for path in (file_a, file_b):
            if not os.path.isfile(path):
                return f"Error: file not found: {path}"
            if os.path.getsize(path) > _MAX_FILE_SIZE:
                return f"Error: file too large (>{_MAX_FILE_SIZE // 1024}KB): {path}"

        try:
            with open(file_a, encoding="utf-8", errors="replace") as f:
                lines_a = f.readlines()
            with open(file_b, encoding="utf-8", errors="replace") as f:
                lines_b = f.readlines()
        except OSError as e:
            return f"Error reading file: {e}"

        diff = list(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=file_a, tofile=file_b,
            n=context,
        ))

        if not diff:
            return f"Files are identical ({len(lines_a)} lines)."

        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

        result = f"Changes: +{added} -{removed} lines\n\n" + "".join(diff)
        if len(result) > _MAX_OUTPUT_CHARS:
            result = result[:_MAX_OUTPUT_CHARS] + "\n...[diff truncated]"
        return result
