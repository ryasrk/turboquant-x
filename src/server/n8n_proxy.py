"""Reverse proxy for n8n backend.

All requests to ``/workspace/n8n/*`` are forwarded to the n8n service
with path rewriting, header forwarding, streaming response support,
and WebSocket proxying for the push channel.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, StreamingResponse
from starlette.websockets import WebSocketState

from src.server.n8n_auth import get_n8n_base_url, get_n8n_headers, N8N_API_KEY

logger = logging.getLogger(__name__)

router = APIRouter(tags=["n8n-proxy"])

# Headers that must NOT be forwarded to n8n.
_STRIP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "authorization",  # TQ JWT — not valid for n8n
})

# Timeout for proxied requests (generous — n8n can be slow on first load).
_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


def _forward_headers(request: Request) -> dict[str, str]:
    """Build the header dict to send to n8n.

    * Strips hop-by-hop and auth headers
    * Injects ``X-N8N-API-KEY`` from environment
    * Preserves all other client headers
    """
    out: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() not in _STRIP_HEADERS:
            out[key] = value

    # Inject n8n auth headers (overwrites if client sent them)
    n8n_headers = get_n8n_headers()
    out.update(n8n_headers)

    # When using session auth (no API key), inject the internal cookie
    if not N8N_API_KEY:
        from src.server.n8n_setup import get_session_cookies
        cookie_str = get_session_cookies()
        if cookie_str:
            out["Cookie"] = cookie_str

    return out


# ── Auto-login: set n8n session cookie in browser ────────────────────

@router.get("/workspace/n8n-login")
async def n8n_auto_login(request: Request):
    """Auto-login to n8n and redirect to the editor.

    Sets the n8n session cookie in the browser so the editor
    sees the user as authenticated without requiring manual login.
    Supports ?next=/workspace/n8n/... to redirect to a specific page.
    """
    # Determine redirect target (only allow /workspace/n8n/ paths)
    next_url = request.query_params.get("next", "/workspace/n8n/")
    if not next_url.startswith("/workspace/n8n"):
        next_url = "/workspace/n8n/"

    if N8N_API_KEY:
        # API key mode — no session cookie needed, redirect directly
        return RedirectResponse(url=next_url, status_code=302)

    from src.server.n8n_setup import ensure_n8n_ready, get_session_cookies

    if not await ensure_n8n_ready():
        raise HTTPException(status_code=503, detail="n8n not available")

    cookie_str = get_session_cookies()
    if not cookie_str:
        raise HTTPException(status_code=503, detail="n8n session not established")

    # Build redirect response with n8n session cookie set for the browser
    response = RedirectResponse(url=next_url, status_code=302)
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            name, value = part.split("=", 1)
            response.set_cookie(
                key=name.strip(),
                value=value.strip(),
                path="/workspace/n8n/",
                samesite="lax",
                httponly=False,  # n8n editor JS needs to read this
            )

    return response


@router.api_route(
    "/workspace/n8n/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_to_n8n(request: Request, path: str) -> StreamingResponse:
    """Proxy any request to the n8n backend.

    Path rewriting: ``/workspace/n8n/api/v1/workflows`` →
    ``http://localhost:5678/api/v1/workflows``
    """
    target_url = f"{get_n8n_base_url()}/{path}"

    # Preserve query string
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = _forward_headers(request)
    body = await request.body()

    logger.debug("Proxying %s %s → %s", request.method, request.url.path, target_url)

    client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)
    try:
        n8n_resp = await client.send(
            client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            ),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        logger.error("Cannot connect to n8n at %s", get_n8n_base_url())
        raise HTTPException(status_code=502, detail="n8n service unavailable")
    except httpx.TimeoutException:
        await client.aclose()
        logger.error("Timeout connecting to n8n at %s", get_n8n_base_url())
        raise HTTPException(status_code=504, detail="n8n service timeout")

    # Build response headers — strip hop-by-hop from n8n response too.
    resp_headers: dict[str, str] = {}
    for key, value in n8n_resp.headers.items():
        if key.lower() not in _STRIP_HEADERS:
            resp_headers[key] = value

    async def _stream_body():
        try:
            async for chunk in n8n_resp.aiter_bytes():
                yield chunk
        finally:
            await n8n_resp.aclose()
            await client.aclose()

    return StreamingResponse(
        content=_stream_body(),
        status_code=n8n_resp.status_code,
        headers=resp_headers,
        media_type=resp_headers.get("content-type"),
    )


# ── WebSocket proxy (n8n push channel) ──────────────────────────────

@router.websocket("/workspace/n8n/{path:path}")
async def ws_proxy_to_n8n(ws: WebSocket, path: str):
    """Proxy WebSocket connections to n8n (used for push notifications).

    Cookie resolution order:
    1. Browser's ``n8n-auth`` cookie from the WS upgrade request (set by /workspace/n8n-login)
    2. Server-side session cookie from auto-provisioning
    3. Fresh login attempt via ensure_n8n_ready()
    """
    await ws.accept()

    # Build n8n WebSocket URL
    base = get_n8n_base_url().replace("http://", "ws://").replace("https://", "wss://")
    target = f"{base}/{path}"
    if ws.scope.get("query_string"):
        target = f"{target}?{ws.scope['query_string'].decode()}"

    # ── Resolve n8n auth cookie ─────────────────────────────────────
    cookie_str = ""

    # 1. Try browser's cookie from the WS upgrade request headers
    browser_cookies = ws.headers.get("cookie", "")
    if browser_cookies:
        # Extract n8n-auth cookie from the browser cookie string
        for part in browser_cookies.split(";"):
            part = part.strip()
            if part.startswith("n8n-auth="):
                cookie_str = part
                logger.debug("WS proxy: using browser n8n-auth cookie")
                break

    # 2. Fall back to server-side session cookie
    if not cookie_str and not N8N_API_KEY:
        from src.server.n8n_setup import get_session_cookies
        cookie_str = get_session_cookies() or ""
        if cookie_str:
            logger.debug("WS proxy: using server-side session cookie")

    # 3. Last resort — try to login fresh
    if not cookie_str and not N8N_API_KEY:
        try:
            from src.server.n8n_setup import ensure_n8n_ready, get_session_cookies
            await ensure_n8n_ready()
            cookie_str = get_session_cookies() or ""
            if cookie_str:
                logger.debug("WS proxy: obtained fresh session cookie via ensure_n8n_ready")
        except Exception:
            logger.debug("WS proxy: ensure_n8n_ready failed, proceeding without cookie")

    from websockets.asyncio.client import connect as ws_connect
    extra_headers: dict[str, str] = {}
    if cookie_str:
        extra_headers["Cookie"] = cookie_str
    if N8N_API_KEY:
        extra_headers["X-N8N-API-KEY"] = N8N_API_KEY
    # Don't forward Content-Type/Accept for WebSocket — they're irrelevant
    # and some WS servers reject unexpected headers

    logger.info(
        "WS proxy: connecting to %s (auth: %s)",
        target,
        "cookie" if "Cookie" in extra_headers else ("api-key" if "X-N8N-API-KEY" in extra_headers else "NONE"),
    )

    try:
        async with ws_connect(
            target,
            additional_headers=extra_headers,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=5,
        ) as n8n_ws:
            logger.info("WS proxy: connected to n8n backend for %s", path)

            async def client_to_n8n():
                """Forward messages from browser → n8n."""
                try:
                    while True:
                        data = await ws.receive_text()
                        await n8n_ws.send(data)
                except WebSocketDisconnect:
                    logger.debug("WS proxy: browser disconnected")
                except Exception as e:
                    logger.debug("WS proxy: client→n8n error: %s: %s", type(e).__name__, e)

            async def n8n_to_client():
                """Forward messages from n8n → browser."""
                try:
                    async for msg in n8n_ws:
                        if ws.client_state == WebSocketState.CONNECTED:
                            if isinstance(msg, str):
                                await ws.send_text(msg)
                            elif isinstance(msg, bytes):
                                await ws.send_bytes(msg)
                    # If we get here, n8n closed the connection
                    logger.debug("WS proxy: n8n closed the WebSocket (iterator ended)")
                except Exception as e:
                    logger.debug("WS proxy: n8n→client error: %s: %s", type(e).__name__, e)

            t1 = asyncio.create_task(client_to_n8n(), name="client→n8n")
            t2 = asyncio.create_task(n8n_to_client(), name="n8n→client")

            done, pending = await asyncio.wait(
                [t1, t2],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                exc = task.exception()
                logger.info(
                    "WS proxy: [%s] ended%s",
                    task.get_name(),
                    f" with error: {exc}" if exc else " cleanly",
                )
            for task in pending:
                task.cancel()

    except Exception as exc:
        logger.warning("WS proxy error for %s: %s: %s", path, type(exc).__name__, exc)
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except Exception:
                pass
