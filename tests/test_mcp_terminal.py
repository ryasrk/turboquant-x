"""Tests for terminal tool, approval store, and MCP adapter."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.approval import ApprovalStore, get_approval_store
from src.agent.tools.terminal_tool import (
    TerminalTool,
    CommandIntent,
    RiskLevel,
    classify_command,
    compute_risk_level,
    check_destructive_warning,
    check_path_safety,
    check_sed_safety,
    validate_command,
)
from src.agent.mcp_client import McpClient, McpStdioTransport, McpSseTransport, McpError
from src.agent.tools.mcp_bridge_tool import McpBridgeTool
from src.agent.mcp_loader import load_mcp_config, _expand_env
from src.agent.registry import ToolRegistry


# ── Command Classification ────────────────────────────────────


class TestCommandClassification:
    @pytest.mark.parametrize("cmd,expected", [
        ("ls -la", CommandIntent.READ_ONLY),
        ("cat /etc/hosts", CommandIntent.READ_ONLY),
        ("grep -r foo .", CommandIntent.READ_ONLY),
        ("find . -name '*.py'", CommandIntent.READ_ONLY),
        ("python --version", CommandIntent.READ_ONLY),
        ("echo hello", CommandIntent.READ_ONLY),
        ("git status", CommandIntent.READ_ONLY),
        ("git log --oneline", CommandIntent.READ_ONLY),
        ("git diff HEAD", CommandIntent.READ_ONLY),
    ])
    def test_read_only(self, cmd: str, expected: CommandIntent) -> None:
        assert classify_command(cmd) == expected

    @pytest.mark.parametrize("cmd,expected", [
        ("cp a.txt b.txt", CommandIntent.WRITE),
        ("mv a.txt b.txt", CommandIntent.WRITE),
        ("mkdir new_dir", CommandIntent.WRITE),
        ("touch somefile", CommandIntent.WRITE),
        ("git push origin main", CommandIntent.WRITE),
        ("git commit -m 'msg'", CommandIntent.WRITE),
    ])
    def test_write(self, cmd: str, expected: CommandIntent) -> None:
        assert classify_command(cmd) == expected

    @pytest.mark.parametrize("cmd,expected", [
        ("rm somefile.txt", CommandIntent.DESTRUCTIVE),
        ("shred secret.key", CommandIntent.DESTRUCTIVE),
    ])
    def test_destructive(self, cmd: str, expected: CommandIntent) -> None:
        assert classify_command(cmd) == expected

    @pytest.mark.parametrize("cmd,expected", [
        ("curl https://example.com", CommandIntent.NETWORK),
        ("wget https://example.com/file.zip", CommandIntent.NETWORK),
        ("ssh user@host", CommandIntent.NETWORK),
        ("ping 8.8.8.8", CommandIntent.NETWORK),
    ])
    def test_network(self, cmd: str, expected: CommandIntent) -> None:
        assert classify_command(cmd) == expected

    @pytest.mark.parametrize("cmd,expected", [
        ("pip install pandas", CommandIntent.PACKAGE_MANAGEMENT),
        ("npm install express", CommandIntent.PACKAGE_MANAGEMENT),
        ("apt install vim", CommandIntent.PACKAGE_MANAGEMENT),
        ("brew install jq", CommandIntent.PACKAGE_MANAGEMENT),
    ])
    def test_package_management(self, cmd: str, expected: CommandIntent) -> None:
        assert classify_command(cmd) == expected

    @pytest.mark.parametrize("cmd,expected", [
        ("kill -9 1234", CommandIntent.PROCESS_MANAGEMENT),
        ("pkill python", CommandIntent.PROCESS_MANAGEMENT),
    ])
    def test_process_management(self, cmd: str, expected: CommandIntent) -> None:
        assert classify_command(cmd) == expected

    def test_system_admin_systemctl(self) -> None:
        assert classify_command("systemctl restart nginx") == CommandIntent.SYSTEM_ADMIN

    def test_sudo_strips_to_inner_command(self) -> None:
        # sudo apt -> extracts apt (package_management)
        assert classify_command("sudo apt update") == CommandIntent.PACKAGE_MANAGEMENT

    def test_write_redirection(self) -> None:
        assert classify_command("echo x > output.txt") == CommandIntent.WRITE

    def test_unknown(self) -> None:
        assert classify_command("some_custom_bin arg") == CommandIntent.UNKNOWN

    def test_empty(self) -> None:
        assert classify_command("") == CommandIntent.UNKNOWN


class TestRiskLevel:
    def test_read_only_is_low(self) -> None:
        assert compute_risk_level("ls", CommandIntent.READ_ONLY) == RiskLevel.LOW

    def test_destructive_is_critical(self) -> None:
        assert compute_risk_level("rm file", CommandIntent.DESTRUCTIVE) == RiskLevel.CRITICAL

    def test_package_is_medium(self) -> None:
        assert compute_risk_level("pip install x", CommandIntent.PACKAGE_MANAGEMENT) == RiskLevel.MEDIUM

    def test_network_is_high(self) -> None:
        assert compute_risk_level("curl http://x", CommandIntent.NETWORK) == RiskLevel.HIGH

    def test_system_admin_is_high(self) -> None:
        assert compute_risk_level("sudo x", CommandIntent.SYSTEM_ADMIN) == RiskLevel.HIGH

    def test_unknown_with_system_path_is_high(self) -> None:
        assert compute_risk_level("some_cmd /etc/passwd", CommandIntent.UNKNOWN) == RiskLevel.HIGH


class TestDestructiveWarning:
    def test_rm_rf_star_warns(self) -> None:
        assert check_destructive_warning("rm -rf *") is not None

    def test_shred_warns(self) -> None:
        assert check_destructive_warning("shred secret") is not None

    def test_safe_no_warning(self) -> None:
        assert check_destructive_warning("ls -la") is None

    def test_rm_rf_warns(self) -> None:
        assert check_destructive_warning("rm -rf somedir") is not None


class TestPathSafety:
    def test_rm_in_etc_warns(self) -> None:
        assert check_path_safety("rm /etc/hosts") is not None

    def test_cp_to_usr_warns(self) -> None:
        assert check_path_safety("cp file /usr/local/bin/") is not None

    def test_ls_no_warning(self) -> None:
        assert check_path_safety("ls /etc/") is None

    def test_safe_write_no_warning(self) -> None:
        assert check_path_safety("cp a.txt b.txt") is None


class TestSedSafety:
    def test_sed_in_place_warns(self) -> None:
        assert check_sed_safety("sed -i 's/old/new/' file.txt") is not None

    def test_sed_read_only_ok(self) -> None:
        assert check_sed_safety("sed 's/old/new/' file.txt") is None

    def test_non_sed_ok(self) -> None:
        assert check_sed_safety("grep foo bar") is None


class TestValidateCommand:
    def test_safe_returns_low_risk(self) -> None:
        result = validate_command("ls -la")
        assert result["risk_level"] == "low"
        assert result["intent"] == "read_only"
        assert result["blocked"] is None

    def test_destructive_returns_critical(self) -> None:
        result = validate_command("rm -rf somedir")
        assert result["risk_level"] == "critical"
        assert result["intent"] == "destructive"
        assert len(result["warnings"]) > 0  # rm -rf triggers destructive warning

    def test_package_install_returns_medium(self) -> None:
        result = validate_command("pip install pandas")
        assert result["risk_level"] == "medium"
        assert result["intent"] == "package_management"


class TestTerminalToolValidate:
    tool = TerminalTool()

    def test_validate_hard_deny(self) -> None:
        result = self.tool.validate("rm -rf /")
        assert result["blocked"] is not None
        assert result["risk_level"] == "critical"

    def test_validate_safe(self) -> None:
        result = self.tool.validate("echo hello")
        assert result["blocked"] is None
        assert result["risk_level"] == "low"

    def test_validate_network(self) -> None:
        result = self.tool.validate("curl https://example.com")
        assert result["risk_level"] == "high"
        assert result["intent"] == "network"


# ── TerminalTool ──────────────────────────────────────────────


class TestTerminalTool:
    tool = TerminalTool()

    def test_name(self) -> None:
        assert self.tool.name == "terminal_exec"

    def test_requires_approval(self) -> None:
        assert self.tool.requires_approval is True

    def test_schema_has_command_param(self) -> None:
        params = self.tool.parameters
        assert "command" in params["properties"]
        assert "command" in params["required"]

    def test_schema_has_optional_params(self) -> None:
        params = self.tool.parameters
        assert "working_dir" in params["properties"]
        assert "timeout" in params["properties"]
        assert "reason" in params["properties"]

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf /home",
        "shutdown now",
        "reboot",
        "poweroff",
        "dd if=/dev/zero of=/dev/sda",
        "chmod -R 777 /",
    ])
    def test_hard_deny_blocks_dangerous(self, cmd: str) -> None:
        result = self.tool.is_hard_denied(cmd)
        assert result is not None, f"Expected '{cmd}' to be blocked"

    @pytest.mark.parametrize("cmd", [
        "pip install pandas",
        "npm install express",
        "apt install vim",
        "ls -la",
        "python --version",
        "git status",
        "cat /etc/hosts",
    ])
    def test_hard_deny_allows_safe(self, cmd: str) -> None:
        result = self.tool.is_hard_denied(cmd)
        assert result is None, f"Expected '{cmd}' to be allowed"

    @pytest.mark.asyncio
    async def test_execute_echo(self) -> None:
        result = await self.tool.execute(command="echo hello world")
        assert "hello world" in result
        assert "[exit code: 0]" in result

    @pytest.mark.asyncio
    async def test_execute_hard_denied(self) -> None:
        result = await self.tool.execute(command="rm -rf /")
        assert "Error" in result
        assert "blocked" in result

    @pytest.mark.asyncio
    async def test_execute_timeout(self) -> None:
        tool = TerminalTool(timeout=1)
        result = await tool.execute(command="sleep 10", timeout=1)
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_execute_bad_command(self) -> None:
        result = await self.tool.execute(command="nonexistent_command_xyz_12345")
        assert "[exit code:" in result  # command not found still has exit code

    def test_openai_schema(self) -> None:
        schema = self.tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "terminal_exec"


# ── ApprovalStore ─────────────────────────────────────────────


class TestApprovalStore:
    def test_create_and_resolve_approved(self) -> None:
        store = ApprovalStore()
        future = store.create("req_1")
        assert store.has_pending("req_1")
        store.resolve("req_1", True)
        assert future.result() is True
        assert not store.has_pending("req_1")

    def test_create_and_resolve_denied(self) -> None:
        store = ApprovalStore()
        future = store.create("req_2")
        store.resolve("req_2", False)
        assert future.result() is False

    def test_resolve_nonexistent(self) -> None:
        store = ApprovalStore()
        assert store.resolve("nonexistent", True) is False

    def test_cancel(self) -> None:
        store = ApprovalStore()
        future = store.create("req_3")
        store.cancel("req_3")
        assert future.cancelled()
        assert not store.has_pending("req_3")

    @pytest.mark.asyncio
    async def test_wait_for_approval_approved(self) -> None:
        store = ApprovalStore()
        store.create("req_4")

        async def approve_later():
            await asyncio.sleep(0.05)
            store.resolve("req_4", True)

        asyncio.create_task(approve_later())
        result = await store.wait_for_approval("req_4")
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_approval_denied(self) -> None:
        store = ApprovalStore()
        store.create("req_5")

        async def deny_later():
            await asyncio.sleep(0.05)
            store.resolve("req_5", False)

        asyncio.create_task(deny_later())
        result = await store.wait_for_approval("req_5")
        assert result is False

    def test_global_singleton(self) -> None:
        store1 = get_approval_store()
        store2 = get_approval_store()
        assert store1 is store2


# ── McpBridgeTool ─────────────────────────────────────────────


class TestMcpBridgeTool:
    def _make_client(self, name: str = "test") -> McpClient:
        transport = McpSseTransport("http://localhost:8080/mcp")
        return McpClient(name, transport)

    def test_name_prefixed(self) -> None:
        client = self._make_client("github")
        tool = McpBridgeTool(client, {"name": "create_issue", "description": "Create issue"})
        assert tool.name == "mcp_github_create_issue"

    def test_description(self) -> None:
        client = self._make_client("fs")
        tool = McpBridgeTool(client, {"name": "read_file", "description": "Read a file"})
        assert "[MCP:fs]" in tool.description
        assert "Read a file" in tool.description

    def test_parameters_from_input_schema(self) -> None:
        client = self._make_client()
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        tool = McpBridgeTool(client, {"name": "read", "inputSchema": schema})
        assert tool.parameters == schema

    def test_parameters_default_empty(self) -> None:
        client = self._make_client()
        tool = McpBridgeTool(client, {"name": "ping"})
        assert tool.parameters["type"] == "object"

    def test_requires_approval_default_false(self) -> None:
        client = self._make_client()
        tool = McpBridgeTool(client, {"name": "read", "description": ""})
        assert tool.requires_approval is False

    def test_requires_approval_when_set(self) -> None:
        client = self._make_client()
        tool = McpBridgeTool(client, {"name": "write", "description": ""}, require_approval=True)
        assert tool.requires_approval is True

    def test_openai_schema(self) -> None:
        client = self._make_client("sqlite")
        tool = McpBridgeTool(
            client,
            {"name": "query", "description": "Run SQL", "inputSchema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            }},
        )
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mcp_sqlite_query"
        assert "sql" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_delegates_to_client(self) -> None:
        client = self._make_client()
        client.call_tool = AsyncMock(return_value="result text")
        tool = McpBridgeTool(client, {"name": "echo", "description": ""})
        result = await tool.execute(message="hello")
        client.call_tool.assert_called_once_with("echo", {"message": "hello"})
        assert result == "result text"

    @pytest.mark.asyncio
    async def test_execute_handles_error(self) -> None:
        client = self._make_client()
        client.call_tool = AsyncMock(side_effect=McpError("connection lost"))
        tool = McpBridgeTool(client, {"name": "fail", "description": ""})
        result = await tool.execute()
        assert "Error" in result
        assert "connection lost" in result


# ── McpClient ─────────────────────────────────────────────────


class TestMcpClient:
    def test_is_connected_default_false(self) -> None:
        transport = McpSseTransport("http://localhost:8080")
        client = McpClient("test", transport)
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_call_tool_parses_text_content(self) -> None:
        transport = MagicMock()
        transport.start = AsyncMock()
        transport.stop = AsyncMock()
        transport.send_request = AsyncMock()
        transport.send_notification = AsyncMock()

        # Simulate initialize
        transport.send_request.side_effect = [
            # initialize response
            {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "test"}},
            # tools/list response
            {"tools": [{"name": "echo", "description": "Echo", "inputSchema": {}}]},
            # tools/call response
            {"content": [{"type": "text", "text": "Hello World"}], "isError": False},
        ]

        client = McpClient("test", transport)
        await client.connect()
        assert client.is_connected

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

        result = await client.call_tool("echo", {"message": "hi"})
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self) -> None:
        transport = MagicMock()
        transport.start = AsyncMock()
        transport.stop = AsyncMock()
        transport.send_request = AsyncMock()
        transport.send_notification = AsyncMock()

        transport.send_request.side_effect = [
            {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "test"}},
            {"tools": []},
            {"content": [{"type": "text", "text": "file not found"}], "isError": True},
        ]

        client = McpClient("test", transport)
        await client.connect()
        await client.list_tools()

        result = await client.call_tool("read_file", {"path": "/nope"})
        assert "Error" in result
        assert "file not found" in result


# ── MCP Config ────────────────────────────────────────────────


class TestMcpConfig:
    def test_load_default_config(self) -> None:
        config = load_mcp_config()
        assert "enabled" in config

    def test_load_missing_config(self, tmp_path) -> None:
        config = load_mcp_config(str(tmp_path / "nonexistent.yaml"))
        assert config["enabled"] is False
        assert config["servers"] == []

    def test_expand_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TOKEN", "secret123")
        assert _expand_env("Bearer ${MY_TOKEN}") == "Bearer secret123"

    def test_expand_env_missing_var(self) -> None:
        result = _expand_env("value=${NONEXISTENT_VAR_XYZ}")
        assert result == "value="


# ── Registry Integration ──────────────────────────────────────


class TestMcpRegistryIntegration:
    def test_mcp_tool_registers_and_executes(self) -> None:
        registry = ToolRegistry()
        transport = McpSseTransport("http://localhost")
        client = McpClient("test", transport)
        bridge = McpBridgeTool(
            client,
            {"name": "ping", "description": "Ping server", "inputSchema": {
                "type": "object", "properties": {}, "required": [],
            }},
        )
        registry.register(bridge)
        assert "mcp_test_ping" in registry.list_tools()

        defns = registry.get_definitions()
        names = [d["function"]["name"] for d in defns]
        assert "mcp_test_ping" in names

    def test_terminal_tool_registers(self) -> None:
        registry = ToolRegistry()
        registry.register(TerminalTool())
        assert "terminal_exec" in registry.list_tools()

        tool = registry.get("terminal_exec")
        assert tool is not None
        assert tool.requires_approval is True
