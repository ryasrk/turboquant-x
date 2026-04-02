"""Web fetching and HTTP request tools."""
from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp

from src.agent.base import Tool

_PRIVATE_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
)

_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE"})


def _validate_url_scheme(url: str) -> None:
    """Raise ValueError if the URL scheme is not http or https."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Blocked URL scheme '{parsed.scheme}'. Only http:// and https:// are allowed."
        )


def _is_private_host(hostname: str) -> bool:
    """Return True if *hostname* resolves to a private/internal IP."""
    if hostname == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return any(addr in net for net in _PRIVATE_NETWORKS)


def _check_ssrf(url: str) -> None:
    """Raise ValueError when the URL targets a private/internal address."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if _is_private_host(hostname):
        raise ValueError(
            f"Blocked request to private/internal address: {hostname}"
        )


def _strip_html(text: str) -> str:
    """Strip HTML tags via regex (fallback when bs4 is unavailable)."""
    return re.sub(r"<[^>]+>", "", text)


class FetchWebpageTool(Tool):
    """Fetch a webpage and extract its text content."""

    @property
    def name(self) -> str:
        return "fetch_webpage"

    @property
    def description(self) -> str:
        return "Fetch a webpage and extract its text content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return",
                    "default": 8000,
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        max_chars: int = kwargs.get("max_chars", 8000)

        _validate_url_scheme(url)

        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        return f"HTTP error {resp.status} fetching {url}"
                    raw = await resp.text()
        except aiohttp.ClientError as exc:
            return f"Connection error: {exc}"
        except TimeoutError:
            return "Request timed out after 15 seconds."

        try:
            from bs4 import BeautifulSoup  # noqa: WPS433

            soup = BeautifulSoup(raw, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            text = _strip_html(raw)

        return text[:max_chars]


class HttpRequestTool(Tool):
    """Make an HTTP request to an API endpoint."""

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return "Make an HTTP request to an API endpoint. Returns response body."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Request URL"},
                "method": {
                    "type": "string",
                    "description": "HTTP method",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers",
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        method: str = kwargs.get("method", "GET").upper()
        headers: dict[str, str] | None = kwargs.get("headers")
        body: str | None = kwargs.get("body")
        timeout_secs: int = kwargs.get("timeout", 30)

        _validate_url_scheme(url)
        _check_ssrf(url)

        if method not in _ALLOWED_METHODS:
            return f"Unsupported HTTP method: {method}"

        default_headers = {"User-Agent": "turboquant-agent/1.0"}
        if headers:
            default_headers.update(headers)

        request_kwargs: dict[str, Any] = {"headers": default_headers}
        if body and method in ("POST", "PUT"):
            request_kwargs["data"] = body

        timeout = aiohttp.ClientTimeout(total=timeout_secs)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, **request_kwargs) as resp:
                    text = await resp.text()
                    truncated = text[:16000]
                    return f"HTTP {resp.status}\n{truncated}"
        except aiohttp.ClientError as exc:
            return f"Connection error: {exc}"
        except TimeoutError:
            return f"Request timed out after {timeout_secs} seconds."
