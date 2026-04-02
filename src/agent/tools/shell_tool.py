"""Shell execution tool with deny-list guard."""
from __future__ import annotations

import asyncio
import re
from typing import Any

from src.agent.base import Tool

MAX_OUTPUT_CHARS = 10_000

DEFAULT_DENY_PATTERNS: list[str] = [
    r"\brm\s+-[rf]{1,2}\b",
    r"\bdel\s+/[fq]\b",
    r"\brmdir\s+/s\b",
    r"(?:^|[;&|]\s*)format\b",
    r"\b(mkfs|diskpart)\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r"\b(shutdown|reboot|poweroff)\b",
    r":\(\)\s*\{.*\};\s*:",
]


class ExecTool(Tool):
    """Execute a shell command and return output."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
    ) -> None:
        self._timeout = min(timeout, 300)
        self._working_dir = working_dir
        self._deny_re = [
            re.compile(p) for p in (deny_patterns if deny_patterns is not None else DEFAULT_DENY_PATTERNS)
        ]

    @property
    def name(self) -> str:
        return "exec_shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 60, max 300)",
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            "required": ["command"],
        }

    def _guard_command(self, command: str) -> str | None:
        """Return an error string if *command* matches a deny pattern, else ``None``."""
        for pattern in self._deny_re:
            if pattern.search(command):
                return f"Error: command blocked by security policy (matched: {pattern.pattern})"
        return None

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        return text[:half] + "\n\n... [truncated] ...\n\n" + text[-half:]

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs["command"]
        timeout: int = min(kwargs.get("timeout", self._timeout), 300)

        blocked = self._guard_command(command)
        if blocked:
            return blocked

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()  # type: ignore[union-attr]
            return f"Error: command timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return f"Error executing command: {exc}"

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        output = stdout
        if stderr:
            output = output + ("\n" if output else "") + stderr

        output = self._truncate(output)
        return f"[exit code: {proc.returncode}]\n{output}"
