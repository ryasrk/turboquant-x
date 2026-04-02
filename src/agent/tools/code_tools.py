"""Code analysis and evaluation tools with workspace sandboxing."""
from __future__ import annotations

import ast
import asyncio
import fnmatch
import math
import os
import re
from pathlib import Path
from typing import Any

from src.agent.base import Tool

_SKIP_DIRS = {".git", "node_modules", "__pycache__", "env", ".venv"}

_BLOCKED_CALLS = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
    }
)

_ALLOWED_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "reversed": reversed,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "type": type,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


def _resolve_safe(path: str, workspace_root: str) -> Path:
    """Resolve *path* and verify it stays inside *workspace_root*."""
    root = Path(workspace_root).resolve()
    resolved = (root / path).resolve()
    if not (resolved == root or str(resolved).startswith(str(root) + os.sep)):
        raise PermissionError(f"Path escapes workspace root: {path}")
    return resolved


def _is_binary(filepath: Path) -> bool:
    """Return True if the file appears to be binary (null bytes in first 8KB)."""
    try:
        with open(filepath, "rb") as fh:
            chunk = fh.read(8192)
        return b"\x00" in chunk
    except (OSError, PermissionError):
        return True


class GrepCodeTool(Tool):
    """Search for a regex pattern across workspace files."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "grep_code"

    @property
    def description(self) -> str:
        return (
            "Search for a pattern across files in the workspace. "
            "Returns matching lines with file paths and line numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search (default: workspace root)",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Only search files matching this glob, e.g. '*.py'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default 50)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs: Any) -> str:
        pattern: str = kwargs["pattern"]
        sub_path: str = kwargs.get("path", ".")
        file_glob: str | None = kwargs.get("file_glob")
        max_results: int = kwargs.get("max_results", 50)

        try:
            search_root = _resolve_safe(sub_path, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex – {exc}"

        workspace = Path(self._workspace_root).resolve()

        def _search() -> str:
            matches: list[str] = []
            for dirpath, dirnames, filenames in os.walk(search_root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                for fname in filenames:
                    if file_glob and not fnmatch.fnmatch(fname, file_glob):
                        continue
                    fpath = Path(dirpath) / fname
                    if _is_binary(fpath):
                        continue
                    try:
                        with open(fpath, encoding="utf-8", errors="replace") as fh:
                            for lineno, line in enumerate(fh, 1):
                                if regex.search(line):
                                    rel = fpath.relative_to(workspace)
                                    matches.append(
                                        f"{rel}:{lineno}: {line.rstrip()}"
                                    )
                                    if len(matches) >= max_results:
                                        return "\n".join(matches)
                    except (OSError, PermissionError):
                        continue
            if not matches:
                return "No matches found."
            return "\n".join(matches)

        try:
            return await asyncio.to_thread(_search)
        except Exception as exc:  # noqa: BLE001
            return f"Error searching files: {exc}"


class PythonEvalTool(Tool):
    """Evaluate a pure Python expression safely."""

    @property
    def name(self) -> str:
        return "python_eval"

    @property
    def description(self) -> str:
        return (
            "Evaluate a Python expression and return the result. "
            "Only pure expressions allowed, no imports or statements."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python expression to evaluate",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs: Any) -> str:
        expression: str = kwargs["expression"]

        # Validate it's a pure expression via AST
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return f"Error: not a valid expression – {exc}"

        # Walk AST to block dangerous nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Error: imports are not allowed"
            if isinstance(node, ast.Call):
                func = node.func
                func_name: str | None = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                if func_name in _BLOCKED_CALLS:
                    return f"Error: call to '{func_name}' is not allowed"

        safe_globals: dict[str, Any] = {"__builtins__": {}}
        safe_globals.update(_ALLOWED_BUILTINS)
        safe_globals["math"] = math

        def _eval() -> str:
            result = eval(expression, safe_globals)  # noqa: S307
            return str(result)

        try:
            return await asyncio.wait_for(asyncio.to_thread(_eval), timeout=5)
        except asyncio.TimeoutError:
            return "Error: expression timed out (5s limit)"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"


class CountLinesTool(Tool):
    """Count lines, words, and characters in files."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "count_lines"

    @property
    def description(self) -> str:
        return "Count lines, words, and characters in a file or directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to count",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Only count files matching this glob, e.g. '*.py'",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        file_glob: str | None = kwargs.get("file_glob")

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        workspace = Path(self._workspace_root).resolve()

        def _count_file(fpath: Path) -> tuple[str, int, int, int]:
            lines = words = chars = 0
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    lines += 1
                    words += len(line.split())
                    chars += len(line)
            rel = str(fpath.relative_to(workspace))
            return rel, lines, words, chars

        def _count() -> str:
            if resolved.is_file():
                rel, lines, words, chars = _count_file(resolved)
                return f"{rel}: {lines} lines, {words} words, {chars} chars"

            results: list[str] = []
            total_lines = total_words = total_chars = 0
            for dirpath, dirnames, filenames in os.walk(resolved):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                for fname in filenames:
                    if file_glob and not fnmatch.fnmatch(fname, file_glob):
                        continue
                    fpath = Path(dirpath) / fname
                    if _is_binary(fpath):
                        continue
                    try:
                        rel, lines, words, chars = _count_file(fpath)
                        results.append(
                            f"{rel}: {lines} lines, {words} words, {chars} chars"
                        )
                        total_lines += lines
                        total_words += words
                        total_chars += chars
                    except (OSError, PermissionError):
                        continue
            if not results:
                return "No matching files found."
            results.append(
                f"Total: {total_lines} lines, {total_words} words, {total_chars} chars"
            )
            return "\n".join(results)

        try:
            return await asyncio.to_thread(_count)
        except FileNotFoundError:
            return f"Error: path not found – {path_str}"
        except Exception as exc:  # noqa: BLE001
            return f"Error counting: {exc}"
