"""Tests for system_tools: GetEnvTool and CurrentTimeTool."""
from __future__ import annotations

import pytest

from src.agent.tools.system_tools import CurrentTimeTool, GetEnvTool


# ── GetEnvTool ────────────────────────────────────────────────


class TestGetEnvTool:
    tool = GetEnvTool()

    @pytest.mark.asyncio
    async def test_reads_existing_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_HELLO", "world")
        result = await self.tool.execute(name="TEST_HELLO")
        assert result == "world"

    @pytest.mark.asyncio
    async def test_missing_var(self) -> None:
        result = await self.tool.execute(name="DEFINITELY_NOT_SET_XYZ_12345")
        assert "is not set" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var_name", [
        "MY_SECRET",
        "DB_PASSWORD",
        "AUTH_TOKEN",
        "API_KEY",
        "PRIVATE_STUFF",
        "aws_credential_file",
    ])
    async def test_blocks_sensitive(self, var_name: str) -> None:
        result = await self.tool.execute(name=var_name)
        assert result == "Error: access to sensitive environment variable denied."


# ── CurrentTimeTool ───────────────────────────────────────────


class TestCurrentTimeTool:
    tool = CurrentTimeTool()

    @pytest.mark.asyncio
    async def test_utc(self) -> None:
        result = await self.tool.execute(timezone="UTC")
        assert "UTC" in result
        assert "unix=" in result
        # Day of week should be present in parentheses
        assert "(" in result and ")" in result

    @pytest.mark.asyncio
    async def test_specific_timezone(self) -> None:
        result = await self.tool.execute(timezone="America/New_York")
        # Should contain a timezone abbreviation (EST or EDT)
        assert "unix=" in result
        assert "(" in result

    @pytest.mark.asyncio
    async def test_invalid_timezone(self) -> None:
        result = await self.tool.execute(timezone="Not/A/Timezone")
        assert result.startswith("Error:")
        assert "Not/A/Timezone" in result

    @pytest.mark.asyncio
    async def test_default_timezone(self) -> None:
        """When no timezone kwarg is given, defaults to UTC."""
        result = await self.tool.execute()
        assert "UTC" in result
