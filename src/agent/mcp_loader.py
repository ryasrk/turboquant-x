"""MCP server loader — reads config, connects to servers, registers bridge tools."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from src.agent.mcp_client import McpClient, McpStdioTransport, McpSseTransport, McpError
from src.agent.registry import ToolRegistry
from src.agent.tools.mcp_bridge_tool import McpBridgeTool

logger = logging.getLogger(__name__)

# Active MCP clients for graceful shutdown
_active_clients: list[McpClient] = []

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} references in a string."""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return _ENV_VAR_RE.sub(_replace, value)


def _expand_env_dict(d: dict[str, str]) -> dict[str, str]:
    """Expand env vars in all dict values."""
    return {k: _expand_env(v) if isinstance(v, str) else v for k, v in d.items()}


def load_mcp_config(config_path: str | None = None) -> dict:
    """Load MCP config from YAML file."""
    if config_path is None:
        config_path = os.path.join("config", "mcp.yaml")
    path = Path(config_path)
    if not path.exists():
        return {"enabled": False, "servers": []}
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


async def connect_mcp_servers(
    registry: ToolRegistry,
    config_path: str | None = None,
) -> int:
    """Connect to all configured MCP servers and register their tools.

    Returns the number of tools registered.
    """
    config = load_mcp_config(config_path)

    if not config.get("enabled", False):
        logger.debug("MCP server connections disabled")
        return 0

    servers = config.get("servers") or []
    if not servers:
        logger.debug("No MCP servers configured")
        return 0

    total_tools = 0

    for server_cfg in servers:
        name = server_cfg.get("name", "unknown")
        try:
            client = _create_client(server_cfg)
            await client.connect()
            _active_clients.append(client)

            # Discover tools
            tools = await client.list_tools()
            require_approval = server_cfg.get("require_approval", False)

            for tool_def in tools:
                bridge = McpBridgeTool(
                    client=client,
                    tool_def=tool_def,
                    require_approval=require_approval,
                )
                try:
                    registry.register(bridge)
                    total_tools += 1
                    logger.info("Registered MCP tool: %s", bridge.name)
                except ValueError:
                    logger.warning("MCP tool name collision, skipping: %s", bridge.name)

        except McpError as exc:
            logger.error("Failed to connect MCP server '%s': %s", name, exc)
        except Exception as exc:
            logger.error("Unexpected error connecting MCP server '%s': %s", name, exc)

    logger.info("MCP: registered %d tools from %d servers", total_tools, len(servers))
    return total_tools


def _create_client(server_cfg: dict) -> McpClient:
    """Create an McpClient from a server config dict."""
    name = server_cfg.get("name", "unknown")
    transport_type = server_cfg.get("transport", "stdio")
    require_approval = server_cfg.get("require_approval", False)

    if transport_type == "stdio":
        command = server_cfg.get("command", [])
        if not command:
            raise McpError(f"MCP server '{name}': missing 'command' for stdio transport")
        env = _expand_env_dict(server_cfg.get("env", {}))
        cwd = server_cfg.get("cwd")
        transport = McpStdioTransport(command=command, env=env, cwd=cwd)
    elif transport_type == "sse":
        url = server_cfg.get("url", "")
        if not url:
            raise McpError(f"MCP server '{name}': missing 'url' for SSE transport")
        headers = _expand_env_dict(server_cfg.get("headers", {}))
        transport = McpSseTransport(url=url, headers=headers)
    else:
        raise McpError(f"MCP server '{name}': unknown transport '{transport_type}'")

    return McpClient(name=name, transport=transport, require_approval=require_approval)


async def shutdown_mcp_servers() -> None:
    """Disconnect all active MCP clients."""
    for client in _active_clients:
        try:
            await client.disconnect()
        except Exception as exc:
            logger.warning("Error disconnecting MCP client '%s': %s", client.name, exc)
    _active_clients.clear()
    logger.info("All MCP servers disconnected")
