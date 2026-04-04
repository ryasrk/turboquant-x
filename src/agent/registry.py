from __future__ import annotations

from src.agent.base import Tool

_MAX_RESULT_LENGTH = 16_000


class ToolRegistry:
    """Registry that stores, resolves, and executes agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        del self._tools[name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[dict]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict) -> str:
        try:
            tool = self._tools.get(name)
            if tool is None:
                return f"Error: KeyError: Tool not found: {name}"
            casted = tool.cast_params(params)
            tool.validate_params(casted)
            result = await tool.execute(**casted)
            result = str(result)
            if len(result) > _MAX_RESULT_LENGTH:
                half = _MAX_RESULT_LENGTH // 2
                result = result[:half] + "\n...[truncated]...\n" + result[-half:]
            return result
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def subset(self, prefix: str) -> "ToolRegistry":
        """Return a new registry containing only tools whose name starts with *prefix*."""
        filtered = ToolRegistry()
        for name, tool in self._tools.items():
            if name.startswith(prefix):
                filtered._tools[name] = tool
        return filtered
