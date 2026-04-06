"""Meta-tools for lazy tool loading — search and invoke tools on demand.

Instead of loading all 70+ tool definitions into the LLM context, the agent
starts with a small core set + these two meta-tools.  When it needs a
specialized capability, it searches the full registry → gets the tool's
parameter schema → invokes it by name.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.base import Tool

logger = logging.getLogger(__name__)

# Set by app.py after the full registry is built — includes ALL tools
_full_registry: Any = None  # ToolRegistry
# Prefixes to exclude from chat dashboard meta-tools (e.g. n8n tools)
_excluded_prefixes: list[str] = []


def set_full_registry(registry: Any, *, excluded_prefixes: list[str] | None = None) -> None:
    """Store a reference to the complete tool registry for meta-tool access.

    Args:
        registry: The full ToolRegistry.
        excluded_prefixes: Tool name prefixes to hide from chat dashboard
            meta-tools (search/detail/invoke). Workspace uses its own
            registry and is unaffected.
    """
    global _full_registry, _excluded_prefixes
    _full_registry = registry
    _excluded_prefixes = excluded_prefixes or []


class SearchToolsTool(Tool):
    """Search available tools by keyword. Returns matching tool names and
    one-line descriptions so the agent can decide which tool to use,
    then call ``get_tool_detail`` or ``invoke_tool`` with the name.
    """

    @property
    def name(self) -> str:
        return "search_tools"

    @property
    def description(self) -> str:
        return (
            "Search for available tools by keyword. Returns matching tool names "
            "and short descriptions. Use this to discover what tools are available "
            "before calling invoke_tool."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search keyword(s) to match against tool names and descriptions. "
                        "Examples: 'git', 'time', 'file', 'n8n', 'memory', 'fetch'"
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(self, *, query: str = "", **kwargs: Any) -> str:
        if _full_registry is None:
            return "Error: Tool registry not initialized"

        query_lower = query.lower()
        # Split query into individual words for flexible matching
        query_words = query_lower.replace("_", " ").split()
        matches = []

        for tool_obj in _full_registry._tools.values():
            name = tool_obj.name
            # Skip tools excluded from chat dashboard (e.g. n8n tools)
            if any(name.startswith(p) for p in _excluded_prefixes):
                continue
            desc = tool_obj.description
            # Normalize underscores to spaces for matching
            name_norm = name.lower().replace("_", " ")
            desc_norm = desc.lower()
            searchable = f"{name_norm} {desc_norm}"
            # Match if ALL query words appear somewhere in name or description
            if all(w in searchable for w in query_words):
                matches.append({"name": name, "description": desc[:120]})

        if not matches:
            return f"No tools found matching '{query}'. Try broader keywords."

        lines = [f"Found {len(matches)} tool(s) matching '{query}':\n"]
        for m in matches:
            lines.append(f"- **{m['name']}**: {m['description']}")
        lines.append(
            "\nTo use a tool, call invoke_tool with the tool name and arguments."
            "\nTo see full parameters, call get_tool_detail with the tool name."
        )
        return "\n".join(lines)


class GetToolDetailTool(Tool):
    """Get full parameter schema for a specific tool by name."""

    @property
    def name(self) -> str:
        return "get_tool_detail"

    @property
    def description(self) -> str:
        return (
            "Get the full parameter schema for a tool by name. "
            "Use after search_tools to understand what arguments a tool expects."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Exact name of the tool (e.g. 'mcp_git_git_status')",
                },
            },
            "required": ["tool_name"],
        }

    async def execute(self, *, tool_name: str = "", **kwargs: Any) -> str:
        if _full_registry is None:
            return "Error: Tool registry not initialized"

        # Block access to excluded tools (e.g. n8n tools in chat dashboard)
        if any(tool_name.startswith(p) for p in _excluded_prefixes):
            return f"Error: Tool '{tool_name}' is not available in this context."

        tool = _full_registry.get(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found. Use search_tools to find available tools."

        schema = tool.to_schema()
        return json.dumps(schema, indent=2)


class InvokeToolTool(Tool):
    """Invoke any registered tool by name with JSON arguments.

    This is the proxy through which the agent calls tools that aren't
    loaded into its active context.
    """

    @property
    def name(self) -> str:
        return "invoke_tool"

    @property
    def description(self) -> str:
        return (
            "Call any available tool by name. First use search_tools to find the "
            "tool, optionally get_tool_detail for its parameters, then call this "
            "with the tool name and arguments as a JSON object."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Exact name of the tool to invoke (e.g. 'mcp_time_get_current_time')",
                },
                "arguments": {
                    "type": "string",
                    "description": 'JSON string of arguments to pass to the tool (e.g. \'{"timezone": "UTC"}\')',
                },
            },
            "required": ["tool_name"],
        }

    async def execute(self, *, tool_name: str = "", arguments: str = "{}", **kwargs: Any) -> str:
        if _full_registry is None:
            return "Error: Tool registry not initialized"

        # Block invocation of excluded tools (e.g. n8n tools in chat dashboard)
        if any(tool_name.startswith(p) for p in _excluded_prefixes):
            return f"Error: Tool '{tool_name}' is not available in this context."

        tool = _full_registry.get(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found. Use search_tools to find available tools."

        try:
            if isinstance(arguments, str):
                # Unescape HTML entities the model sometimes emits
                import html
                arguments = html.unescape(arguments)
                args = json.loads(arguments)
            else:
                args = arguments
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments: {e}"

        if not isinstance(args, dict):
            return "Error: Arguments must be a JSON object (dict)"

        logger.info("invoke_tool: calling %s with %s", tool_name, args)
        return await _full_registry.execute(tool_name, args)
