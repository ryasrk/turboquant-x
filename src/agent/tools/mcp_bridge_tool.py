"""Bridge tool — wraps a single MCP server tool as a native TurboQuant-X Tool."""
from __future__ import annotations

from typing import Any

from src.agent.base import Tool
from src.agent.mcp_client import McpClient


class McpBridgeTool(Tool):
    """Wraps a tool from an MCP server as a native agent tool.

    The tool schema is taken directly from the MCP server's tool definition.
    Execution delegates to the MCP client.
    """

    def __init__(
        self,
        client: McpClient,
        tool_def: dict,
        require_approval: bool = False,
    ) -> None:
        self._client = client
        self._tool_def = tool_def
        self._require_approval = require_approval

    @property
    def name(self) -> str:
        # Prefix with server name to avoid collisions with built-in tools
        server = self._client.name
        tool_name = self._tool_def["name"]
        return f"mcp_{server}_{tool_name}"

    @property
    def description(self) -> str:
        desc = self._tool_def.get("description", "")
        server = self._client.name
        return f"[MCP:{server}] {desc}" if desc else f"[MCP:{server}] {self._tool_def['name']}"

    @property
    def parameters(self) -> dict[str, Any]:
        # MCP uses "inputSchema" which is JSON Schema — same as OpenAI format
        schema = self._tool_def.get("inputSchema", {})
        if not schema:
            return {"type": "object", "properties": {}, "required": []}
        return schema

    @property
    def requires_approval(self) -> bool:
        return self._require_approval

    async def execute(self, **kwargs: Any) -> str:
        try:
            return await self._client.call_tool(self._tool_def["name"], kwargs)
        except Exception as exc:
            return f"Error: MCP tool call failed: {exc}"
