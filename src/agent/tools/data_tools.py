"""Data tools: JSON query, calculator, CSV reader with workspace sandboxing."""
from __future__ import annotations

import ast
import asyncio
import csv
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from src.agent.base import Tool

MAX_RESULT_CHARS = 16_000


def _resolve_safe(path: str, workspace_root: str) -> Path:
    """Resolve *path* and verify it stays inside *workspace_root*.

    Raises ``PermissionError`` on path-traversal attempts.
    """
    root = Path(workspace_root).resolve()
    resolved = (root / path).resolve()
    if not (resolved == root or str(resolved).startswith(str(root) + os.sep)):
        raise PermissionError(f"Path escapes workspace root: {path}")
    return resolved


# ---------------------------------------------------------------------------
# JsonQueryTool
# ---------------------------------------------------------------------------

class JsonQueryTool(Tool):
    """Read a JSON file and optionally extract data via dot-notation."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "json_query"

    @property
    def description(self) -> str:
        return (
            "Read a JSON file and extract data using a dot-notation path "
            "(e.g. 'data.items[0].name')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to JSON file",
                },
                "query": {
                    "type": "string",
                    "description": "Dot-notation query path e.g. 'users[0].name'",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        query: str | None = kwargs.get("query")

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _read_and_query() -> str:
            with open(resolved, encoding="utf-8") as fh:
                data = json.load(fh)

            if query:
                current = data
                for segment in query.split("."):
                    m = re.match(r"(\w+)(?:\[(\d+)\])?", segment)
                    if not m:
                        return f"Error: invalid query segment '{segment}'"
                    key, idx = m.group(1), m.group(2)
                    if isinstance(current, dict):
                        if key not in current:
                            return f"Error: key '{key}' not found in object"
                        current = current[key]
                    elif isinstance(current, list):
                        # key may itself be an index-only segment handled below
                        return f"Error: expected object for key '{key}', got list"
                    else:
                        return f"Error: cannot navigate into {type(current).__name__}"
                    if idx is not None:
                        idx_int = int(idx)
                        if not isinstance(current, list):
                            return f"Error: '{key}' is not an array"
                        if idx_int >= len(current):
                            return (
                                f"Error: index {idx_int} out of range "
                                f"(length {len(current)})"
                            )
                        current = current[idx_int]
                result = current
            else:
                result = data

            text = json.dumps(result, indent=2)
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "\n... [truncated]"
            return text

        try:
            return await asyncio.to_thread(_read_and_query)
        except FileNotFoundError:
            return f"Error: file not found – {path_str}"
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON – {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"


# ---------------------------------------------------------------------------
# CalculateTool
# ---------------------------------------------------------------------------

_SAFE_NAMES: dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
}

_ALLOWED_AST_NODES = frozenset({
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Name,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Tuple,
    ast.List,
    # operators / contexts
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or,
    ast.Load,
})


class CalculateTool(Tool):
    """Evaluate a mathematical expression safely."""

    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression safely. Supports arithmetic, "
            "trig, log, and common math functions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs: Any) -> str:
        expression: str = kwargs["expression"]

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return f"Error: invalid expression – {exc}"

        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_AST_NODES:
                return (
                    f"Error: disallowed expression node {type(node).__name__}"
                )

        try:
            result = eval(  # noqa: S307
                compile(tree, "<calc>", "eval"),
                {"__builtins__": {}},
                _SAFE_NAMES,
            )
        except ZeroDivisionError:
            return "Error: division by zero"
        except OverflowError:
            return "Error: numerical overflow"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"

        return str(result)


# ---------------------------------------------------------------------------
# CsvReadTool
# ---------------------------------------------------------------------------

class CsvReadTool(Tool):
    """Read a CSV file with optional column/row filtering."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    @property
    def name(self) -> str:
        return "csv_read"

    @property
    def description(self) -> str:
        return "Read a CSV file and optionally filter/select columns. Returns formatted table."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to CSV file",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names to select",
                },
                "filter_column": {
                    "type": "string",
                    "description": "Column to filter on",
                },
                "filter_value": {
                    "type": "string",
                    "description": "Value to match in filter_column",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to return (default 50)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str: str = kwargs["path"]
        columns: list[str] | None = kwargs.get("columns")
        filter_column: str | None = kwargs.get("filter_column")
        filter_value: str | None = kwargs.get("filter_value")
        max_rows: int = kwargs.get("max_rows", 50)

        try:
            resolved = _resolve_safe(path_str, self._workspace_root)
        except PermissionError as exc:
            return f"Error: {exc}"

        def _read_csv() -> str:
            with open(resolved, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    return "Error: CSV file has no headers"

                headers = list(columns) if columns else list(reader.fieldnames)

                rows: list[dict[str, str]] = []
                total_count = 0
                for row in reader:
                    # Apply row filter
                    if filter_column and filter_value:
                        if row.get(filter_column, "") != filter_value:
                            continue
                    total_count += 1
                    if len(rows) < max_rows:
                        rows.append(
                            {h: row.get(h, "") for h in headers}
                        )

            # Build text table
            col_widths = {
                h: max(len(h), *(len(r.get(h, "")) for r in rows) if rows else 0)
                for h in headers
            }
            # Ensure minimum width equals header width when no rows
            col_widths = {h: max(len(h), w) for h, w in col_widths.items()}

            buf = io.StringIO()
            # Header
            buf.write(" | ".join(h.ljust(col_widths[h]) for h in headers))
            buf.write("\n")
            buf.write("-+-".join("-" * col_widths[h] for h in headers))
            buf.write("\n")
            # Rows
            for row in rows:
                buf.write(
                    " | ".join(
                        row.get(h, "").ljust(col_widths[h]) for h in headers
                    )
                )
                buf.write("\n")

            if total_count > max_rows:
                buf.write(
                    f"\n... truncated ({total_count} total rows, "
                    f"showing {max_rows})\n"
                )

            return buf.getvalue()

        try:
            return await asyncio.to_thread(_read_csv)
        except FileNotFoundError:
            return f"Error: file not found – {path_str}"
        except Exception as exc:  # noqa: BLE001
            return f"Error reading CSV: {exc}"
