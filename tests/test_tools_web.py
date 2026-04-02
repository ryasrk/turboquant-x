"""Tests for FetchWebpageTool and HttpRequestTool."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.tools.web_tools import FetchWebpageTool, HttpRequestTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status: int = 200, body: str = "ok") -> AsyncMock:
    """Create a mock aiohttp response usable as an async context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    return resp


def _mock_session(response: AsyncMock) -> MagicMock:
    """Return a mock ClientSession whose .get / .request yield *response*."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aiter__ = None
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.get = MagicMock(return_value=ctx)
    session.request = MagicMock(return_value=ctx)

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    return session_ctx


# ---------------------------------------------------------------------------
# FetchWebpageTool
# ---------------------------------------------------------------------------

class TestFetchWebpageTool:
    tool = FetchWebpageTool()

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        html = "<html><body><p>Hello World</p></body></html>"
        resp = _mock_response(200, html)
        session_ctx = _mock_session(resp)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(url="https://example.com")

        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_fetch_blocks_file_scheme(self):
        with pytest.raises(ValueError, match="Blocked URL scheme"):
            await self.tool.execute(url="file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_fetch_blocks_ftp_scheme(self):
        with pytest.raises(ValueError, match="Blocked URL scheme"):
            await self.tool.execute(url="ftp://evil.com/data")

    @pytest.mark.asyncio
    async def test_fetch_truncation(self):
        long_text = "a" * 20000
        resp = _mock_response(200, long_text)
        session_ctx = _mock_session(resp)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(url="https://example.com", max_chars=500)

        assert len(result) <= 500

    @pytest.mark.asyncio
    async def test_fetch_http_error_status(self):
        resp = _mock_response(404, "not found")
        session_ctx = _mock_session(resp)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(url="https://example.com/missing")

        assert "HTTP error 404" in result

    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        import aiohttp as _aiohttp

        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(side_effect=TimeoutError)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(url="https://example.com")

        assert "timed out" in result.lower()


# ---------------------------------------------------------------------------
# HttpRequestTool
# ---------------------------------------------------------------------------

class TestHttpRequestTool:
    tool = HttpRequestTool()

    @pytest.mark.asyncio
    async def test_get_success(self):
        resp = _mock_response(200, '{"ok": true}')
        session_ctx = _mock_session(resp)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(url="https://api.example.com/data")

        assert "HTTP 200" in result
        assert '{"ok": true}' in result

    @pytest.mark.asyncio
    async def test_post_with_body(self):
        resp = _mock_response(201, "created")
        session_ctx = _mock_session(resp)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(
                url="https://api.example.com/items",
                method="POST",
                body='{"name": "test"}',
            )

        assert "HTTP 201" in result

    @pytest.mark.asyncio
    async def test_ssrf_blocks_localhost(self):
        with pytest.raises(ValueError, match="private/internal"):
            await self.tool.execute(url="https://localhost/admin")

    @pytest.mark.asyncio
    async def test_ssrf_blocks_127(self):
        with pytest.raises(ValueError, match="private/internal"):
            await self.tool.execute(url="http://127.0.0.1/secret")

    @pytest.mark.asyncio
    async def test_ssrf_blocks_10_network(self):
        with pytest.raises(ValueError, match="private/internal"):
            await self.tool.execute(url="http://10.0.0.1/internal")

    @pytest.mark.asyncio
    async def test_ssrf_blocks_192_168(self):
        with pytest.raises(ValueError, match="private/internal"):
            await self.tool.execute(url="http://192.168.1.1/router")

    @pytest.mark.asyncio
    async def test_ssrf_blocks_172_16(self):
        with pytest.raises(ValueError, match="private/internal"):
            await self.tool.execute(url="http://172.16.0.1/internal")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(side_effect=TimeoutError)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.web_tools.aiohttp.ClientSession", return_value=session_ctx):
            result = await self.tool.execute(url="https://api.example.com/slow", timeout=5)

        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_blocks_file_scheme(self):
        with pytest.raises(ValueError, match="Blocked URL scheme"):
            await self.tool.execute(url="file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_unsupported_method(self):
        result = await self.tool.execute(url="https://example.com", method="PATCH")
        assert "Unsupported HTTP method" in result
