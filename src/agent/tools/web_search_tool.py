"""Web search tool using DuckDuckGo."""
from __future__ import annotations

import asyncio
from typing import Any

from src.agent.base import Tool


class WebSearchTool(Tool):
    """Search the web via DuckDuckGo and return formatted results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo. Returns titles, URLs, and snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {
                    "type": "integer",
                    "description": "Number of results (1-10)",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        count: int = min(max(kwargs.get("count", 5), 1), 10)

        try:
            from ddgs import DDGS  # noqa: WPS433
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # noqa: WPS433
            except ImportError:
                return "web_search unavailable: install ddgs (pip install ddgs)"

        def _search() -> list[dict[str, str]]:
            return list(DDGS().text(query, max_results=count))

        try:
            results = await asyncio.to_thread(_search)
        except Exception as exc:  # noqa: BLE001
            return f"Error performing search: {exc}"

        if not results:
            return "No results found."

        lines: list[str] = []
        for idx, r in enumerate(results, 1):
            title = r.get("title", "(no title)")
            url = r.get("href", r.get("link", ""))
            snippet = r.get("body", r.get("snippet", ""))
            lines.append(f"{idx}. {title}\n   {url}\n   {snippet}")
        return "\n\n".join(lines)
