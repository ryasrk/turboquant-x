"""n8n authentication bridge and API client.

Bridges TurboQuant-X JWT auth with n8n API access. Uses auto-provisioned
session auth (via n8n_setup) as the primary method — no manual API key
or email/password configuration needed.

Fallback: if N8N_API_KEY is set, uses that instead.
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

from src.server.n8n_setup import (
    ensure_n8n_ready,
    get_session_cookies,
    get_n8n_base_url as _setup_base_url,
    is_n8n_running,
)

logger = logging.getLogger(__name__)

N8N_API_KEY: str = os.getenv("N8N_API_KEY", "")
N8N_BACKEND_URL: str = os.getenv("N8N_BACKEND_URL", "http://localhost:5678")

# Timeout configuration (seconds)
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 120.0  # AI builds can be slow


def get_n8n_headers() -> dict[str, str]:
    """Return headers required for authenticated n8n API calls.

    If N8N_API_KEY is set, uses that. Otherwise returns minimal headers
    (session cookies are attached separately by the httpx client).
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if N8N_API_KEY:
        headers["X-N8N-API-KEY"] = N8N_API_KEY
    return headers


def _get_auth_kwargs() -> dict:
    """Get auth kwargs for httpx requests (cookie header or nothing)."""
    if N8N_API_KEY:
        return {}  # API key is in headers already
    cookie_str = get_session_cookies()
    if cookie_str:
        return {"headers": {"Cookie": cookie_str}}
    return {}


def get_n8n_base_url() -> str:
    """Return the n8n backend base URL (no trailing slash)."""
    return N8N_BACKEND_URL.rstrip("/")


async def verify_n8n_access(user: dict) -> bool:
    """Check whether n8n is available for this user.

    Uses auto-provisioned session auth (preferred) or API key.
    Returns True if n8n is reachable and authenticated.
    """
    # If API key is configured, use it (legacy path)
    if N8N_API_KEY:
        return True

    # Try auto-provisioned session auth
    ready = await ensure_n8n_ready()
    if not ready:
        running = await is_n8n_running()
        if running:
            logger.warning("n8n is running but auto-setup failed for user %s", user.get("user_id"))
        else:
            logger.debug("n8n is not running")
    return ready


async def n8n_create_session() -> dict:
    """Create a new AI builder session in n8n."""
    url = f"{get_n8n_base_url()}/api/v1/ai/session"
    async with httpx.AsyncClient(timeout=httpx.Timeout(_CONNECT_TIMEOUT)) as client:
        resp = await client.post(url, headers={**get_n8n_headers(), **_get_auth_kwargs().get('headers', {})}, json={})
        resp.raise_for_status()
        data: dict = resp.json()
        logger.info("Created n8n AI session: %s", data.get("sessionId") or data.get("id"))
        return data


async def n8n_build_workflow(session_id: str, prompt: str) -> AsyncIterator[dict]:
    """Stream AI build events from n8n for the given session.

    Yields dicts with at least ``{"event": str, ...}`` for each SSE
    event received from n8n.  The caller should iterate with
    ``async for event in n8n_build_workflow(...)``.
    """
    url = f"{get_n8n_base_url()}/api/v1/ai/session/{session_id}/build"
    headers = get_n8n_headers()
    headers["Accept"] = "text/event-stream"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_CONNECT_TIMEOUT, pool=_CONNECT_TIMEOUT),
    ) as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json={"prompt": prompt},
            **_get_auth_kwargs(),
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    parsed = _parse_sse_event(raw_event)
                    if parsed is not None:
                        yield parsed


async def n8n_get_workflow(workflow_id: str) -> dict:
    """Fetch a single workflow by ID from n8n."""
    if N8N_API_KEY:
        url = f"{get_n8n_base_url()}/api/v1/workflows/{workflow_id}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(_CONNECT_TIMEOUT)) as client:
            resp = await client.get(url, headers=get_n8n_headers())
            resp.raise_for_status()
            return resp.json()
    else:
        from src.server.n8n_setup import n8n_api_call
        resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data) if isinstance(data, dict) else data


async def n8n_activate_workflow(workflow_id: str) -> dict:
    """Activate (enable) a workflow in n8n."""
    if N8N_API_KEY:
        url = f"{get_n8n_base_url()}/api/v1/workflows/{workflow_id}/activate"
        async with httpx.AsyncClient(timeout=httpx.Timeout(_CONNECT_TIMEOUT)) as client:
            resp = await client.post(url, headers=get_n8n_headers())
            resp.raise_for_status()
            data: dict = resp.json()
            logger.info("Activated n8n workflow %s", workflow_id)
            return data
    else:
        from src.server.n8n_setup import n8n_api_call
        # REST activate requires versionId
        wf_resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
        wf_resp.raise_for_status()
        wf_data = wf_resp.json()
        wf = wf_data.get("data", wf_data) if isinstance(wf_data, dict) else wf_data
        version_id = wf.get("versionId", "")
        act_resp = await n8n_api_call("POST", f"/rest/workflows/{workflow_id}/activate", json_data={"versionId": version_id})
        act_resp.raise_for_status()
        act_data = act_resp.json()
        result = act_data.get("data", act_data) if isinstance(act_data, dict) else act_data
        logger.info("Activated n8n workflow %s", workflow_id)
        return result


async def n8n_deactivate_workflow(workflow_id: str) -> dict:
    """Deactivate (disable) a workflow in n8n."""
    if N8N_API_KEY:
        url = f"{get_n8n_base_url()}/api/v1/workflows/{workflow_id}/deactivate"
        async with httpx.AsyncClient(timeout=httpx.Timeout(_CONNECT_TIMEOUT)) as client:
            resp = await client.post(url, headers=get_n8n_headers())
            resp.raise_for_status()
            data: dict = resp.json()
            logger.info("Deactivated n8n workflow %s", workflow_id)
            return data
    else:
        from src.server.n8n_setup import n8n_api_call
        wf_resp = await n8n_api_call("GET", f"/rest/workflows/{workflow_id}")
        wf_resp.raise_for_status()
        wf_data = wf_resp.json()
        wf = wf_data.get("data", wf_data) if isinstance(wf_data, dict) else wf_data
        version_id = wf.get("versionId", "")
        deact_resp = await n8n_api_call("POST", f"/rest/workflows/{workflow_id}/deactivate", json_data={"versionId": version_id})
        deact_resp.raise_for_status()
        deact_data = deact_resp.json()
        result = deact_data.get("data", deact_data) if isinstance(deact_data, dict) else deact_data
        logger.info("Deactivated n8n workflow %s", workflow_id)
        return result


# ── Internal helpers ─────────────────────────────────────────────────

def _parse_sse_event(raw: str) -> dict | None:
    """Parse a single SSE event block into a dict.

    Returns None for comment-only or empty blocks.
    """
    event_type = "message"
    data_lines: list[str] = []

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line.startswith(":") or not line:
            continue

    if not data_lines:
        return None

    data_str = "\n".join(data_lines)
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        data = {"text": data_str}

    return {"event": event_type, **data}
