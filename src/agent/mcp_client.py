"""MCP (Model Context Protocol) client — connects to external MCP servers
and exposes their tools as native TurboQuant-X agent tools.

Supports both stdio (subprocess) and SSE (HTTP) transports.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "turboquant-x", "version": "1.0.0"}


class McpError(Exception):
    """Error communicating with an MCP server."""


class McpStdioTransport:
    """MCP transport over stdio (JSON-RPC 2.0 over stdin/stdout)."""

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self._command = command
        self._env = {**os.environ, **(env or {})}
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=self._cwd,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.info("MCP stdio transport started: %s (pid=%d)", self._command, self._process.pid)

    async def stop(self) -> None:
        """Terminate the subprocess."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            logger.info("MCP stdio transport stopped: %s", self._command)
        # Reject all pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(McpError("Transport closed"))
        self._pending.clear()

    async def send_request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if self._process is None or self._process.stdin is None:
            raise McpError("Transport not started")

        async with self._lock:
            req_id = self._next_id
            self._next_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        line = json.dumps(request) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise McpError(f"Request timed out: {method}")

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            raise McpError("Transport not started")

        notification: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notification["params"] = params

        line = json.dumps(notification) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Read JSON-RPC responses from stdout."""
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    future = self._pending.pop(msg_id)
                    if not future.done():
                        if "error" in msg:
                            future.set_exception(
                                McpError(f"MCP error: {msg['error'].get('message', msg['error'])}")
                            )
                        else:
                            future.set_result(msg.get("result"))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("MCP read loop error: %s", exc)


class McpSseTransport:
    """MCP transport over SSE (Server-Sent Events) for HTTP-based MCP servers."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url.rstrip("/")
        self._headers = headers or {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """No persistent connection needed — we use HTTP per request."""
        logger.info("MCP SSE transport configured: %s", self._url)

    async def stop(self) -> None:
        """Nothing to tear down."""
        pass

    async def send_request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request via HTTP POST."""
        try:
            import aiohttp
        except ImportError:
            raise McpError("aiohttp required for SSE transport: pip install aiohttp")

        async with self._lock:
            req_id = self._next_id
            self._next_id += 1

        request = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            request["params"] = params

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._url,
                    json=request,
                    headers={**self._headers, "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        raise McpError(f"MCP SSE request failed: HTTP {resp.status}")
                    result = await resp.json()
                    if "error" in result:
                        raise McpError(f"MCP error: {result['error'].get('message', result['error'])}")
                    return result.get("result")
        except aiohttp.ClientError as exc:
            raise McpError(f"MCP SSE connection error: {exc}")

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a notification (fire-and-forget)."""
        try:
            await self.send_request(method, params)
        except Exception:
            pass  # Notifications don't require responses


class McpClient:
    """High-level MCP client that manages a transport and provides tool access."""

    def __init__(
        self,
        name: str,
        transport: McpStdioTransport | McpSseTransport,
        require_approval: bool = False,
    ) -> None:
        self.name = name
        self.transport = transport
        self.require_approval = require_approval
        self._initialized = False
        self._server_info: dict = {}
        self._tools: list[dict] = []

    async def connect(self) -> None:
        """Start transport + MCP initialize handshake."""
        await self.transport.start()

        # Initialize handshake
        result = await self.transport.send_request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        })
        self._server_info = result or {}
        logger.info(
            "MCP server '%s' initialized: %s",
            self.name,
            self._server_info.get("serverInfo", {}),
        )

        # Send initialized notification
        await self.transport.send_notification("notifications/initialized")
        self._initialized = True

    async def disconnect(self) -> None:
        """Stop the transport."""
        await self.transport.stop()
        self._initialized = False

    async def list_tools(self) -> list[dict]:
        """Discover available tools from the MCP server."""
        if not self._initialized:
            raise McpError("Client not initialized")
        result = await self.transport.send_request("tools/list")
        self._tools = result.get("tools", []) if result else []
        logger.info("MCP server '%s' has %d tools: %s", self.name, len(self._tools),
                     [t["name"] for t in self._tools])
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on the MCP server and return the result as text."""
        if not self._initialized:
            raise McpError("Client not initialized")

        result = await self.transport.send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        if result is None:
            return "No result returned."

        # Extract text from content array
        is_error = result.get("isError", False)
        content_parts = result.get("content", [])
        texts = []
        for part in content_parts:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image":
                texts.append(f"[image: {part.get('mimeType', 'unknown')}]")
            elif part.get("type") == "resource":
                texts.append(f"[resource: {part.get('uri', 'unknown')}]")
            else:
                texts.append(str(part))

        output = "\n".join(texts)
        if is_error:
            output = f"Error: {output}"
        return output

    @property
    def is_connected(self) -> bool:
        return self._initialized
