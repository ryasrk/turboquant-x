"""n8n auto-provisioning and internal session management.

Manages the n8n owner account lifecycle automatically so that
TurboQuant-X users never need to interact with n8n auth directly.

Flow:
  1. On TQ startup (or first workspace API call), check if n8n is running
  2. If n8n is fresh (no owner), auto-create an owner account
  3. Login and cache the session cookie for API calls
  4. All n8n API calls from TQ use this internal session

Credentials are stored in a local file (~/.turboquant/n8n_credentials.json)
and never exposed to end users.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

_CREDS_DIR = Path.home() / ".turboquant"
_CREDS_FILE = _CREDS_DIR / "n8n_credentials.json"
_N8N_INTERNAL_EMAIL = "admin@turboquant.local"

# Cached state
_session_cookie_str: str | None = None
_n8n_ready = False


def get_n8n_base_url() -> str:
    return os.getenv("N8N_BACKEND_URL", "http://localhost:5678").rstrip("/")


def _load_credentials() -> dict[str, str]:
    """Load stored n8n internal credentials, or return empty dict."""
    if _CREDS_FILE.exists():
        try:
            data = json.loads(_CREDS_FILE.read_text())
            if data.get("email") and data.get("password"):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_credentials(email: str, password: str) -> None:
    """Save n8n internal credentials to local file."""
    _CREDS_DIR.mkdir(parents=True, exist_ok=True)
    _CREDS_FILE.write_text(json.dumps({
        "email": email,
        "password": password,
    }))
    _CREDS_FILE.chmod(0o600)
    logger.info("Saved n8n internal credentials to %s", _CREDS_FILE)


async def is_n8n_running() -> bool:
    """Check if n8n is reachable."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{get_n8n_base_url()}/healthz")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def _check_n8n_setup_state() -> dict[str, Any]:
    """Check n8n's setup state via /rest/settings."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.get(f"{get_n8n_base_url()}/rest/settings")
        if resp.status_code == 200:
            return resp.json().get("data", resp.json())
        return {}


async def _create_owner(email: str, password: str, first_name: str = "TQ", last_name: str = "Admin") -> bool:
    """Create the n8n owner account (first-time setup)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.post(
            f"{get_n8n_base_url()}/rest/owner/setup",
            json={
                "email": email,
                "password": password,
                "firstName": first_name,
                "lastName": last_name,
            },
        )
        if resp.status_code in (200, 201):
            logger.info("Created n8n owner account: %s", email)
            return True
        logger.warning("Failed to create n8n owner: %s %s", resp.status_code, resp.text[:200])
        return False


async def _login(email: str, password: str) -> str | None:
    """Login to n8n and return cookie header string."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.post(
            f"{get_n8n_base_url()}/rest/login",
            json={"emailOrLdapLoginId": email, "password": password},
        )
        if resp.status_code == 200:
            logger.info("Logged into n8n as %s", email)
            # Extract raw cookie string for cross-client reuse
            return "; ".join(f"{k}={v}" for k, v in resp.cookies.items())
        logger.warning("n8n login failed: %s %s", resp.status_code, resp.text[:200])
        return None


async def ensure_n8n_ready() -> bool:
    """Ensure n8n is running, owner exists, and we have a valid session.

    This is the main entry point — call on startup or before first API call.
    Returns True if n8n is ready for API calls, False otherwise.
    """
    global _session_cookie_str, _n8n_ready

    if _n8n_ready and _session_cookie_str:
        return True

    # Check if n8n is running
    if not await is_n8n_running():
        logger.debug("n8n is not running")
        return False

    # Check if we have stored credentials
    creds = _load_credentials()
    if creds:
        # Try login with stored credentials first
        cookie_str = await _login(creds["email"], creds["password"])
        if cookie_str:
            _session_cookie_str = cookie_str
            _n8n_ready = True
            return True

        # Login failed — stored creds are stale, try re-provisioning
        logger.warning("Stored n8n credentials failed, attempting re-provision...")
        _CREDS_FILE.unlink(missing_ok=True)
        creds = {}

    if not creds:
        # Check if n8n is still in setup state (no owner yet)
        settings = await _check_n8n_setup_state()
        user_mgmt = settings.get("userManagement", {})
        # n8n returns showSetup=true when no owner exists
        needs_setup = user_mgmt.get("showSetup", False) or not settings.get("instanceId")

        password = "TQ1_" + secrets.token_urlsafe(20)
        if needs_setup:
            if await _create_owner(_N8N_INTERNAL_EMAIL, password):
                _save_credentials(_N8N_INTERNAL_EMAIL, password)
                creds = {"email": _N8N_INTERNAL_EMAIL, "password": password}
            else:
                logger.warning("Failed to create n8n owner account")
                return False
        else:
            # Owner exists but we don't have valid credentials
            logger.warning(
                "n8n has an existing owner but no valid stored credentials. "
                "Run: npx n8n user-management:reset && rm ~/.turboquant/n8n_credentials.json "
                "then restart, or set N8N_API_KEY."
            )
            return False

    # Login with fresh credentials
    cookie_str = await _login(creds["email"], creds["password"])
    if cookie_str:
        _session_cookie_str = cookie_str
        _n8n_ready = True
        return True

    return False


def get_session_cookies() -> str | None:
    """Get the cached n8n session cookie header string for API calls."""
    return _session_cookie_str


async def n8n_api_call(
    method: str,
    path: str,
    json_data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """Make an authenticated call to n8n's internal REST API.

    Auto-provisions and logs in if needed.
    Returns the raw httpx.Response for flexible handling.
    """
    if not await ensure_n8n_ready():
        raise RuntimeError("n8n is not available")

    url = f"{get_n8n_base_url()}{path}"
    headers = {"Cookie": _session_cookie_str} if _session_cookie_str else {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.request(
            method,
            url,
            json=json_data,
            params=params,
            headers=headers,
        )

        # If auth expired, try to re-login
        if resp.status_code == 401:
            global _n8n_ready
            _n8n_ready = False
            if await ensure_n8n_ready():
                headers = {"Cookie": _session_cookie_str} if _session_cookie_str else {}
                resp = await client.request(
                    method,
                    url,
                    json=json_data,
                    params=params,
                    headers=headers,
                )

        return resp
