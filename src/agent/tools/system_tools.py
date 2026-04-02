"""System utility tools: environment variables and current time."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.agent.base import Tool

_SENSITIVE_PATTERN = re.compile(
    r"SECRET|PASSWORD|TOKEN|KEY|PRIVATE|CREDENTIAL", re.IGNORECASE
)


class GetEnvTool(Tool):
    """Read an environment variable value."""

    @property
    def name(self) -> str:
        return "get_env"

    @property
    def description(self) -> str:
        return "Read an environment variable value."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the environment variable to read",
                },
            },
            "required": ["name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs["name"]
        if _SENSITIVE_PATTERN.search(name):
            return "Error: access to sensitive environment variable denied."
        value = os.environ.get(name)
        if value is None:
            return f"Environment variable '{name}' is not set."
        return value


class CurrentTimeTool(Tool):
    """Get current date, time, and timezone information."""

    @property
    def name(self) -> str:
        return "current_time"

    @property
    def description(self) -> str:
        return "Get current date, time, and timezone information."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name e.g. 'America/New_York'",
                    "default": "UTC",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        tz_name: str = kwargs.get("timezone", "UTC")
        try:
            tz = ZoneInfo(tz_name) if tz_name != "UTC" else timezone.utc
        except (ZoneInfoNotFoundError, KeyError):
            return f"Error: unknown timezone '{tz_name}'."

        now = datetime.now(tz)
        day_of_week = now.strftime("%A")
        formatted = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        # unix_ts = int(now.timestamp())
        return f"{formatted} ({day_of_week})"
