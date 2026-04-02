"""File system tools with workspace sandboxing."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from src.agent.base import Tool

MAX_READ_CHARS = 16_000


def _resolve_safe(path: str, workspace_root: str) -> Path:
    """Resolve *path* and verify it stays inside *workspace_root*.

    Raises ``PermissionError`` on path-traversal attempts.
    """
    root = Path(workspace_root).resolve()
    resolved = (root / path).resolve()
    if not (resolved == root or str(resolved).startswith(str(root) + os.sep)):
        raise PermissionError(f"Path escapes workspace root: {path}")
    return resolved


class ReadFileTool(Tool):
    """Read file contents, optionally a line range."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read file contents. Optionally specify line range."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-based, optional)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (1-based, inclusive, optional)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        start_line: int | None = kwargs.get("start_line")
        end_line: int | None = kwargs.get("end_line")

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _read() -> str:
            with open(resolved, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()

            if start_line is not None or end_line is not None:
                s = (start_line or 1) - 1
                e = end_line if end_line is not None else len(lines)
                lines = lines[s:e]

            content = "".join(lines)
            if len(content) > MAX_READ_CHARS:
                content = content[:MAX_READ_CHARS] + "\n... [truncated]"
            return content

        try:
            return await asyncio.to_thread(_read)
        except FileNotFoundError:
            return f"Error: file not found – {path_str}"
        except Exception as exc:  # noqa: BLE001
            return f"Error reading file: {exc}"


class WriteFileTool(Tool):
    """Create or overwrite a file."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Create or overwrite a file with given content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        content: str = kwargs["content"]

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _write() -> str:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as fh:
                fh.write(content)
            return f"Wrote {len(content)} bytes to {resolved.relative_to(Path(self._workspace_root).resolve())}"

        try:
            return await asyncio.to_thread(_write)
        except Exception as exc:  # noqa: BLE001
            return f"Error writing file: {exc}"


class ListDirTool(Tool):
    """List directory contents."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: current dir)",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs.get("path", ".")

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _list() -> str:
            if not resolved.is_dir():
                return f"Error: not a directory – {path_str}"
            entries: list[str] = []
            for entry in sorted(resolved.iterdir()):
                name = entry.name + ("/" if entry.is_dir() else "")
                entries.append(name)
            return "\n".join(entries) if entries else "(empty directory)"

        try:
            return await asyncio.to_thread(_list)
        except Exception as exc:  # noqa: BLE001
            return f"Error listing directory: {exc}"


_SKIP_DIRS = {".git", "node_modules", "__pycache__", "env", ".venv"}


class FindFilesTool(Tool):
    """Search for files matching a glob or regex pattern in workspace."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return "Search for files matching a glob or regex pattern in workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob or regex pattern to match files",
                },
                "regex": {
                    "type": "boolean",
                    "description": "Use regex instead of glob (default: false)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 50)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs: Any) -> str:
        pattern: str = kwargs["pattern"]
        use_regex: bool = kwargs.get("regex", False)
        max_results: int = kwargs.get("max_results", 50)

        root = Path(self._workspace_root).resolve()

        def _should_skip(p: Path) -> bool:
            return any(part in _SKIP_DIRS for part in p.parts)

        def _search() -> str:
            results: list[str] = []
            truncated = False

            if use_regex:
                compiled = re.compile(pattern)
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [
                        d for d in dirnames if d not in _SKIP_DIRS
                    ]
                    for fname in filenames:
                        full = Path(dirpath) / fname
                        rel = full.relative_to(root)
                        if compiled.search(str(rel)):
                            if len(results) >= max_results:
                                truncated = True
                                break
                            results.append(str(rel))
                    if truncated:
                        break
            else:
                for full in root.rglob(pattern):
                    if _should_skip(full.relative_to(root)):
                        continue
                    if len(results) >= max_results:
                        truncated = True
                        break
                    results.append(str(full.relative_to(root)))

            if not results:
                return "No files found."
            output = "\n".join(results)
            if truncated:
                output += f"\n[truncated – showing first {max_results} results]"
            return output

        try:
            return await asyncio.to_thread(_search)
        except re.error as exc:
            return f"Error: invalid regex – {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error searching files: {exc}"


class ReplaceInFileTool(Tool):
    """Search and replace text in a file."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "replace_in_file"

    @property
    def description(self) -> str:
        return "Search and replace text in a file. Returns count of replacements."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to modify"},
                "old_text": {
                    "type": "string",
                    "description": "Text to search for",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "max_replacements": {
                    "type": "integer",
                    "description": "Max replacements to make (default: 1)",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        old_text: str = kwargs["old_text"]
        new_text: str = kwargs["new_text"]
        max_replacements: int = kwargs.get("max_replacements", 1)

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _replace() -> str:
            content = resolved.read_text(encoding="utf-8")
            count = content.count(old_text)
            if count == 0:
                return "No matches found."

            if max_replacements == 1:
                new_content = content.replace(old_text, new_text, 1)
                actual = 1
            else:
                new_content = content.replace(old_text, new_text)
                actual = count

            resolved.write_text(new_content, encoding="utf-8")
            rel = resolved.relative_to(Path(self._workspace_root).resolve())
            return f"Replaced {actual} occurrence(s) in {rel}"

        try:
            return await asyncio.to_thread(_replace)
        except FileNotFoundError:
            return f"Error: file not found – {path_str}"
        except Exception as exc:  # noqa: BLE001
            return f"Error replacing in file: {exc}"
