"""Workspace CRUD and AI Design Lifecycle API.

/v1/workspaces              GET   list workspaces
/v1/workspaces              POST  create workspace
/v1/workspaces/{id}         PATCH rename workspace
/v1/workspaces/{id}         DELETE delete workspace + cascade
/v1/workspaces/{id}/design  POST  start AI design (SSE stream)
/v1/workspaces/{id}/approve POST  approve design → activate
/v1/workspaces/{id}/modify  POST  modify with new prompt (SSE stream)
/v1/workspaces/{id}/reject  POST  reject → reset to draft
/v1/workspaces/{id}/status  GET   poll n8n workflow status
/v1/workspaces/{id}/designs GET   list design history
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.server.auth_routes import get_current_user
from src.server.database import get_connection
from src.server.n8n_auth import (
    n8n_activate_workflow,
    n8n_get_workflow,
    verify_n8n_access,
)
from src.server.n8n_setup import ensure_n8n_ready
from src.server.workflow_designer import (
    generate_workflow_via_llm,
    n8n_import_workflow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


# ── Credential helpers ───────────────────────────────────────────────

def _extract_required_credentials(workflow_json: dict) -> list[dict]:
    """Extract credential types required by workflow nodes.

    Returns a list of dicts: [{node_name, node_type, cred_type, cred_name}]
    """
    required: list[dict] = []
    for node in workflow_json.get("nodes", []):
        creds = node.get("credentials")
        if not creds:
            continue
        node_name = node.get("name", "Unknown")
        node_type = node.get("type", "unknown")
        for cred_name, cred_ref in creds.items():
            cred_type = cred_name  # n8n uses the credential key as the type
            cred_id = None
            if isinstance(cred_ref, dict):
                cred_id = cred_ref.get("id")
            required.append({
                "node_name": node_name,
                "node_type": node_type,
                "cred_type": cred_type,
                "cred_display_name": cred_name,
                "linked_id": cred_id,
            })
    return required


async def _check_missing_credentials(workflow_json: dict) -> dict:
    """Check which credentials a workflow needs vs. what n8n has.

    Returns:
        {
            "required": [...],    # All credential refs from nodes
            "available": [...],   # Existing credentials in n8n
            "missing": [...],     # Required but not linked / not existing
            "ok": bool,           # True if all credentials are satisfied
        }
    """
    required = _extract_required_credentials(workflow_json)
    if not required:
        return {"required": [], "available": [], "missing": [], "ok": True}

    from src.server.n8n_setup import n8n_api_call

    available: list[dict] = []
    try:
        resp = await n8n_api_call("GET", "/rest/credentials")
        resp.raise_for_status()
        data = resp.json()
        creds = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(creds, dict):
            creds = creds.get("data", [])
        available = [
            {"id": c.get("id"), "name": c.get("name"), "type": c.get("type")}
            for c in (creds or [])
        ]
    except Exception as exc:
        logger.warning("Could not list n8n credentials: %s", exc)

    # Build lookup: cred_type → list of cred IDs
    available_by_type: dict[str, list[str]] = {}
    for c in available:
        ctype = c.get("type", "")
        available_by_type.setdefault(ctype, []).append(c["id"])

    missing: list[dict] = []
    for req in required:
        ctype = req["cred_type"]
        has_linked = req.get("linked_id") and any(
            c["id"] == req["linked_id"] for c in available
        )
        has_type = ctype in available_by_type
        if not has_linked and not has_type:
            missing.append(req)

    return {
        "required": required,
        "available": available,
        "missing": missing,
        "ok": len(missing) == 0,
    }


# ── Schemas ──────────────────────────────────────────────────────────

class WorkspaceCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class WorkspacePatch(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class DesignRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10_000)
    model: str | None = Field(None, description="Cloud model override, e.g. 'glm-4.5-flash', 'deepseek-chat'")
    provider: str | None = Field(None, description="Cloud provider override, e.g. 'nvidia', 'zhipu', 'openai'. Use 'local' for local model.")


class WorkspaceChatRequest(BaseModel):
    """Chat message for the workspace agent (n8n-scoped tools)."""
    message: str = Field(..., min_length=1, max_length=10_000)
    history: list[dict] = Field(default_factory=list, description="Previous messages [{role, content}]")
    model: str | None = Field(None, description="Cloud model override")
    provider: str | None = Field(None, description="Cloud provider override, or 'local' for local model.")


class WorkflowJsonUpdate(BaseModel):
    """Update the raw workflow JSON stored in the latest design."""
    workflow_json: dict = Field(..., description="Complete workflow JSON object")


# ── Engine creation helper ───────────────────────────────────────────

def _create_engine_for_provider(
    provider: str | None,
    model: str | None,
) -> tuple[Any, bool]:
    """Get or create an engine for the given provider/model.

    Returns (engine, needs_dispose) where engine is a CloudEngine or local
    InferenceEngine.  Returns (None, False) if unavailable.
    """
    from src.server.app import get_engine, get_or_create_cloud_engine

    # "local" → use the local inference engine
    if provider == "local":
        try:
            local = get_engine()
            if local.is_loaded:
                return local, False
        except RuntimeError:
            pass
        return None, False

    # No provider specified → use default cloud (same as before)
    if not provider:
        cloud, is_temp = get_or_create_cloud_engine()
        if cloud and cloud.is_loaded:
            # Model override within the same provider
            if model and model != cloud.model_name:
                try:
                    from dataclasses import replace as dc_replace
                    from src.engine.cloud_engine import CloudEngine
                    cfg = dc_replace(cloud._config, model=model, timeout=300.0)
                    e = CloudEngine(cfg)
                    e.load_model()
                    return e, True
                except Exception:
                    pass
            return cloud, is_temp
        # Fallback to local
        try:
            local = get_engine()
            if local.is_loaded:
                return local, False
        except RuntimeError:
            pass
        return None, False

    # Specific cloud provider requested → build a temporary engine
    import os
    from src.engine.cloud.provider import CloudConfig
    from src.engine.cloud.registry import build_cloud_configs
    from src.engine.cloud_engine import CloudEngine

    # Try to get config from cached YAML
    from src.server.app import _cloud_yaml_config_raw
    existing_cloud, _ = get_or_create_cloud_engine()

    cloud_section = _cloud_yaml_config_raw or {}

    configs = build_cloud_configs({"cloud": cloud_section}) if cloud_section else {}

    cfg: CloudConfig | None = configs.get(provider)
    if cfg is None:
        # Try env var
        env_key = f"TURBOQUANT_CLOUD_{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_key, "")
        if not api_key:
            return None, False
        from src.engine.cloud.openai_compat import _DEFAULT_BASE_URLS, _DEFAULT_MODELS
        cfg = CloudConfig(
            provider=provider,
            api_key=api_key,
            model=model or _DEFAULT_MODELS.get(provider, ""),
            timeout=300.0,
        )
    else:
        if model:
            from dataclasses import replace as dc_replace
            cfg = dc_replace(cfg, model=model, timeout=300.0)
        else:
            from dataclasses import replace as dc_replace
            cfg = dc_replace(cfg, timeout=300.0)

    try:
        engine = CloudEngine(cfg)
        engine.load_model()
        return engine, True
    except Exception as exc:
        logger.warning("Failed to create engine for provider %s: %s", provider, exc)
        return None, False


# ── Design lifecycle states ──────────────────────────────────────────

DESIGN_STATES = {"draft", "designing", "designed", "approved", "rejected", "active", "failed"}


# ── DB helpers (workspace + design tables) ───────────────────────────

def _init_workspace_tables() -> None:
    """Create workspace tables if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title       TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'draft',
                n8n_workflow_id TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_workspaces_user
                ON workspaces(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_designs (
                id              TEXT PRIMARY KEY,
                workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                prompt          TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'designing',
                n8n_session_id  TEXT,
                n8n_workflow_id TEXT,
                result_data     TEXT,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_designs_workspace
                ON workspace_designs(workspace_id, created_at DESC);
        """)
        conn.commit()
    finally:
        conn.close()


# Run on import so tables exist before first request.
_init_workspace_tables()


# ── Workspace CRUD DB functions ──────────────────────────────────────

def create_workspace(user_id: str, title: str) -> dict:
    wid = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO workspaces (id, user_id, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'draft', ?, ?)",
            (wid, user_id, title, now, now),
        )
        conn.commit()
        return {"id": wid, "user_id": user_id, "title": title, "status": "draft",
                "n8n_workflow_id": None, "created_at": now, "updated_at": now}
    finally:
        conn.close()


def list_workspaces(user_id: str, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, title, status, n8n_workflow_id, created_at, updated_at "
            "FROM workspaces WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_workspace(workspace_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, user_id, title, status, n8n_workflow_id, created_at, updated_at "
            "FROM workspaces WHERE id = ?",
            (workspace_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_workspace(workspace_id: str, user_id: str, **kwargs: object) -> bool:
    allowed = {"title", "status", "n8n_workflow_id"}
    fields = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not fields:
        return False
    fields["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [workspace_id, user_id]
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE workspaces SET {set_clause} WHERE id = ? AND user_id = ?",
            values,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_workspace(workspace_id: str, user_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Design record DB functions ───────────────────────────────────────

def create_workspace_design(workspace_id: str, prompt: str, n8n_session_id: str | None = None) -> dict:
    did = uuid.uuid4().hex[:16]
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO workspace_designs "
            "(id, workspace_id, prompt, status, n8n_session_id, created_at, updated_at) "
            "VALUES (?, ?, ?, 'designing', ?, ?, ?)",
            (did, workspace_id, prompt, n8n_session_id, now, now),
        )
        conn.commit()
        return {"id": did, "workspace_id": workspace_id, "prompt": prompt,
                "status": "designing", "n8n_session_id": n8n_session_id,
                "n8n_workflow_id": None, "result_data": None,
                "created_at": now, "updated_at": now}
    finally:
        conn.close()


def list_workspace_designs(workspace_id: str, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, prompt, status, n8n_session_id, n8n_workflow_id, result_data, "
            "created_at, updated_at "
            "FROM workspace_designs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_workspace_design(design_id: str, **kwargs: object) -> bool:
    allowed = {"status", "n8n_session_id", "n8n_workflow_id", "result_data"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return False
    fields["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [design_id]
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE workspace_designs SET {set_clause} WHERE id = ?",
            values,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def _get_latest_design(workspace_id: str) -> dict | None:
    """Return the most recent design for a workspace, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, prompt, status, n8n_session_id, n8n_workflow_id, result_data, "
            "created_at, updated_at "
            "FROM workspace_designs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT 1",
            (workspace_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Auth + ownership helper ──────────────────────────────────────────

def _get_user_workspace(workspace_id: str, user: dict) -> dict:
    """Fetch workspace and verify ownership. Raises 404/403."""
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your workspace")
    return ws


# ── Routes ───────────────────────────────────────────────────────────

@router.get("/models")
async def list_workspace_models_endpoint():
    """List all available models for workspace design & chat.

    Returns local model (if loaded) plus all configured cloud providers
    with their models.
    """
    import os
    from src.server.app import (
        get_engine, get_cloud_engine, get_inference_mode,
        _cloud_yaml_config_raw, InferenceMode,
    )
    from src.engine.cloud.registry import SUPPORTED_PROVIDERS, build_cloud_configs

    groups: list[dict] = []

    # 1. Local model
    try:
        local = get_engine()
        if local.is_loaded:
            mode = get_inference_mode()
            model_name = local.model_config.model_name
            groups.append({
                "provider": "local",
                "display_name": f"Local ({mode.value})",
                "models": [{"id": "local", "name": model_name, "label": model_name}],
            })
    except RuntimeError:
        pass

    # 2. Cloud providers from config
    cloud_section = _cloud_yaml_config_raw or {}
    configs = build_cloud_configs({"cloud": cloud_section}) if cloud_section else {}

    # Also check env vars for providers not in YAML
    for name, display in SUPPORTED_PROVIDERS.items():
        env_key = f"TURBOQUANT_CLOUD_{name.upper()}_API_KEY"
        has_key = name in configs or bool(os.environ.get(env_key, ""))
        if not has_key:
            continue

        # Get configured model from YAML
        provider_cfg = cloud_section.get("providers", {}).get(name, {})
        default_model = provider_cfg.get("model", "")

        active_cloud = get_cloud_engine()
        is_active = active_cloud and active_cloud.provider_name == name

        models_list = []
        if default_model:
            models_list.append({
                "id": default_model,
                "name": default_model,
                "label": f"{default_model}{' (active)' if is_active else ''}",
            })

        groups.append({
            "provider": name,
            "display_name": f"{display}{' ✓' if is_active else ''}",
            "models": models_list,
        })

    return {"groups": groups}


@router.get("")
async def list_user_workspaces(user: dict = Depends(get_current_user)):
    """List all workspaces for the authenticated user."""
    return {"workspaces": list_workspaces(user["user_id"])}


@router.post("", status_code=201)
async def create_user_workspace(body: WorkspaceCreate, user: dict = Depends(get_current_user)):
    """Create a new workspace."""
    ws = create_workspace(user["user_id"], body.title)
    return ws


@router.patch("/{workspace_id}")
async def rename_workspace(workspace_id: str, body: WorkspacePatch, user: dict = Depends(get_current_user)):
    """Rename a workspace."""
    _get_user_workspace(workspace_id, user)
    if not update_workspace(workspace_id, user["user_id"], title=body.title):
        raise HTTPException(status_code=500, detail="Failed to update workspace")
    return {"ok": True}


@router.delete("/{workspace_id}", status_code=204)
async def delete_user_workspace(workspace_id: str, user: dict = Depends(get_current_user)):
    """Delete a workspace and all its designs (cascade)."""
    _get_user_workspace(workspace_id, user)
    if not delete_workspace(workspace_id, user["user_id"]):
        raise HTTPException(status_code=500, detail="Failed to delete workspace")
    return None


@router.post("/{workspace_id}/design")
async def start_design(workspace_id: str, body: DesignRequest, user: dict = Depends(get_current_user)):
    """Start an AI design session — returns an SSE stream of build events.

    Uses the local/cloud LLM to generate an n8n workflow JSON directly,
    then optionally imports it into n8n if available.

    State transitions: draft|designed|rejected|approved → designing → designed|failed
    """
    ws = _get_user_workspace(workspace_id, user)

    if ws["status"] not in ("draft", "designed", "rejected", "approved", "failed", "designing"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start design from state '{ws['status']}'. Must be draft, designed, rejected, approved, failed, or designing.",
        )

    return EventSourceResponse(_design_event_stream(workspace_id, user, body.prompt, body.model, body.provider))


async def _design_event_stream(workspace_id: str, user: dict, prompt: str, model: str | None = None, provider: str | None = None):
    """Async generator that drives the AI design lifecycle via LLM.

    The LLM generates a complete n8n workflow JSON. If n8n is reachable,
    the workflow is imported automatically.

    Yields SSE events: thinking → building → workflow_json → complete | error
    """
    design: dict | None = None
    try:
        # 1. Record design + set workspace to "designing"
        design = create_workspace_design(workspace_id, prompt)
        update_workspace(workspace_id, user["user_id"], status="designing")

        # 2. Stream LLM generation events
        workflow_json: dict | None = None
        async for event in generate_workflow_via_llm(prompt, model=model, provider=provider):
            event_type = event.get("event", "message")

            if event_type == "workflow_json":
                workflow_json = event.get("workflow")
            
            yield {"event": event_type, "data": json.dumps(event)}

        if not workflow_json:
            update_workspace_design(design["id"], status="failed", result_data="LLM did not produce valid workflow JSON")
            update_workspace(workspace_id, user["user_id"], status="failed")
            yield {"event": "error", "data": json.dumps({"message": "LLM failed to generate a valid workflow"})}
            return

        # 3. Store the workflow JSON in the design record
        update_workspace_design(
            design["id"],
            status="designed",
            result_data=json.dumps(workflow_json),
        )

        # 4. Try to import into n8n if available (auto-provisioned auth)
        workflow_id: str | None = None
        n8n_ready = await ensure_n8n_ready()

        if n8n_ready:
            try:
                n8n_base = os.environ.get("N8N_BACKEND_URL", "http://localhost:5678")
                n8n_key = os.environ.get("N8N_API_KEY", "")
                yield {"event": "importing", "data": json.dumps({"message": "Importing workflow into n8n..."})}
                import_result = await n8n_import_workflow(workflow_json, n8n_base, n8n_key)
                workflow_id = import_result.get("id")
                if workflow_id:
                    update_workspace_design(design["id"], n8n_workflow_id=workflow_id)
                    update_workspace(workspace_id, user["user_id"], status="designed", n8n_workflow_id=workflow_id)
                    yield {"event": "imported", "data": json.dumps({
                        "message": "Workflow imported into n8n",
                        "workflow_id": workflow_id,
                    })}
            except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
                logger.warning("Could not import workflow into n8n: %s", exc)
                yield {"event": "import_skipped", "data": json.dumps({
                    "message": "n8n not reachable — workflow saved locally, import when n8n is available",
                })}
        else:
            update_workspace(workspace_id, user["user_id"], status="designed")
            yield {"event": "import_skipped", "data": json.dumps({
                "message": "n8n not running — workflow saved locally",
            })}

        # 5. Complete
        yield {"event": "complete", "data": json.dumps({
            "design_id": design["id"],
            "workflow_id": workflow_id,
            "status": "designed",
        })}

    except Exception:
        logger.exception("Unexpected error during AI design for workspace %s", workspace_id)
        if design:
            update_workspace_design(design["id"], status="failed")
        update_workspace(workspace_id, user["user_id"], status="failed")
        yield {"event": "error", "data": json.dumps({"message": "Internal error during design"})}


@router.post("/{workspace_id}/approve")
async def approve_design(workspace_id: str, user: dict = Depends(get_current_user)):
    """Approve the latest design and optionally activate the n8n workflow.

    If n8n is configured and a workflow ID exists, activates the workflow.
    Otherwise, simply marks the design as approved.

    State transition: designed → approved (→ active if n8n available)
    """
    ws = _get_user_workspace(workspace_id, user)

    if ws["status"] != "designed":
        raise HTTPException(status_code=409, detail=f"Cannot approve from state '{ws['status']}'. Must be 'designed'.")

    design = _get_latest_design(workspace_id)
    raw_wf_id = ws.get("n8n_workflow_id") or (design and design.get("n8n_workflow_id"))
    workflow_id: str | None = str(raw_wf_id) if raw_wf_id else None

    update_workspace(workspace_id, user["user_id"], status="approved")
    if design:
        update_workspace_design(design["id"], status="approved")

    # If n8n is available and we have a workflow ID, try to activate it
    if workflow_id and await verify_n8n_access(user):
        # Pre-activation credential check
        cred_check: dict = {"ok": True, "missing": []}
        if design and design.get("result_data"):
            try:
                wf_json = json.loads(design["result_data"])
                cred_check = await _check_missing_credentials(wf_json)
            except (json.JSONDecodeError, TypeError):
                pass

        if not cred_check["ok"]:
            missing_types = list({m["cred_type"] for m in cred_check["missing"]})
            return {
                "ok": True,
                "status": "approved",
                "workflow_id": workflow_id,
                "missing_credentials": cred_check["missing"],
                "message": (
                    f"Approved but cannot activate — missing credentials: {', '.join(missing_types)}. "
                    "Create the required credentials first, then retry activation."
                ),
            }

        try:
            result = await n8n_activate_workflow(workflow_id)
            update_workspace(workspace_id, user["user_id"], status="active")
            if design:
                update_workspace_design(design["id"], status="active")
            return {"ok": True, "status": "active", "workflow_id": workflow_id, "n8n_result": result}
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            logger.warning("Could not activate workflow in n8n: %s", exc)
            # Still approved, just not activated in n8n
            return {
                "ok": True,
                "status": "approved",
                "workflow_id": workflow_id,
                "message": "Approved but n8n activation failed — can retry later",
            }

    # No n8n or no workflow_id — just approve
    return {
        "ok": True,
        "status": "approved",
        "design_id": design["id"] if design else None,
        "message": "Design approved" + (" — import to n8n when available" if not workflow_id else ""),
    }


@router.post("/{workspace_id}/modify")
async def modify_design(workspace_id: str, body: DesignRequest, user: dict = Depends(get_current_user)):
    """Modify the current design with a new prompt — returns SSE stream.

    State transitions: designed|active|failed → designing → designed|failed
    """
    ws = _get_user_workspace(workspace_id, user)

    if ws["status"] not in ("designed", "active", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot modify from state '{ws['status']}'. Must be designed, active, or failed.",
        )

    return EventSourceResponse(_design_event_stream(workspace_id, user, body.prompt, body.model))


@router.post("/{workspace_id}/reject")
async def reject_design(workspace_id: str, user: dict = Depends(get_current_user)):
    """Reject the current design, resetting workspace to draft.

    State transition: designed → rejected
    """
    ws = _get_user_workspace(workspace_id, user)

    if ws["status"] != "designed":
        raise HTTPException(status_code=409, detail=f"Cannot reject from state '{ws['status']}'. Must be 'designed'.")

    design = _get_latest_design(workspace_id)
    if design:
        update_workspace_design(design["id"], status="rejected")

    update_workspace(workspace_id, user["user_id"], status="rejected")
    return {"ok": True, "status": "rejected"}


@router.delete("/{workspace_id}/workflow")
async def remove_workflow(workspace_id: str, user: dict = Depends(get_current_user)):
    """Remove the linked n8n workflow and/or reset workspace to draft.

    If an n8n workflow is linked, attempts to delete it from n8n.
    Always resets the workspace status to 'draft' so a new design can be started.
    """
    ws = _get_user_workspace(workspace_id, user)

    if ws["status"] == "draft":
        raise HTTPException(status_code=409, detail="Workspace is already in draft state")

    wf_id = ws.get("n8n_workflow_id")

    # Try to delete from n8n if there's a workflow linked and n8n is reachable
    n8n_deleted = False
    if wf_id and await verify_n8n_access(user):
        try:
            from src.server.n8n_setup import n8n_api_call
            resp = await n8n_api_call("DELETE", f"/rest/workflows/{wf_id}")
            n8n_deleted = resp.status_code in (200, 204)
            if n8n_deleted:
                logger.info("Deleted n8n workflow %s", wf_id)
        except Exception:
            logger.warning("Could not delete workflow %s from n8n", wf_id, exc_info=True)

    # Clear the workflow link and reset status to draft
    update_workspace(workspace_id, user["user_id"], status="draft", n8n_workflow_id="")

    return {
        "ok": True,
        "status": "draft",
        "n8n_deleted": n8n_deleted,
        "message": "Workspace reset to draft"
            + (" — n8n workflow deleted" if n8n_deleted else "")
            + (" — n8n cleanup skipped" if wf_id and not n8n_deleted else ""),
    }


@router.put("/{workspace_id}/design/json")
async def update_design_json(
    workspace_id: str,
    body: WorkflowJsonUpdate,
    user: dict = Depends(get_current_user),
):
    """Update the raw workflow JSON in the latest design record.

    Allows direct editing of the workflow JSON before redeploying.
    """
    ws = _get_user_workspace(workspace_id, user)

    design = _get_latest_design(workspace_id)
    if not design:
        # No existing design — create one so the JSON is stored
        design = create_workspace_design(workspace_id, "(manual edit)")
        if ws["status"] == "draft":
            update_workspace(workspace_id, user["user_id"], status="designed")

    workflow_json = body.workflow_json
    if not workflow_json.get("nodes") and not workflow_json.get("connections"):
        raise HTTPException(status_code=422, detail="Workflow JSON must contain nodes or connections.")

    update_workspace_design(
        design["id"],
        result_data=json.dumps(workflow_json),
        status="designed",
    )

    return {
        "ok": True,
        "design_id": design["id"],
        "node_count": len(workflow_json.get("nodes", [])),
        "message": "Design JSON updated. Use Redeploy to push to n8n.",
    }


@router.post("/{workspace_id}/redeploy")
async def redeploy_workflow(workspace_id: str, user: dict = Depends(get_current_user)):
    """Re-deploy the latest design JSON to n8n.

    Useful when the original import produced an empty workflow.  If an
    n8n workflow ID already exists, it is updated in-place.  Otherwise a
    new workflow is created and the workspace is linked to it.
    """
    ws = _get_user_workspace(workspace_id, user)

    # Find the latest design with result data
    design = _get_latest_design(workspace_id)
    if not design or not design.get("result_data"):
        raise HTTPException(status_code=404, detail="No design data found to redeploy.")

    try:
        workflow_json = json.loads(design["result_data"])
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=422, detail="Design data is not valid JSON.")

    if not workflow_json.get("nodes"):
        raise HTTPException(status_code=422, detail="Design has no nodes to deploy.")

    if not await ensure_n8n_ready():
        raise HTTPException(status_code=503, detail="n8n is not available.")

    existing_wf_id = ws.get("n8n_workflow_id")

    from src.server.n8n_setup import n8n_api_call

    # Normalize the payload
    payload = {
        "name": workflow_json.get("name", ws.get("title", "Untitled")),
        "nodes": workflow_json["nodes"],
        "connections": workflow_json.get("connections", {}),
        "active": False,
        "settings": workflow_json.get("settings", {"executionOrder": "v1"}),
    }

    if existing_wf_id:
        # Update existing workflow in-place (fall through to create if 404)
        try:
            resp = await n8n_api_call("PUT", f"/rest/workflows/{existing_wf_id}", json_data=payload)
            resp.raise_for_status()
            result = resp.json()
            wf = result.get("data", result) if isinstance(result, dict) else result
            node_count = len(wf.get("nodes", []))

            # Check credentials for informational warning
            cred_check = await _check_missing_credentials(workflow_json)

            resp_data: dict = {
                "ok": True,
                "action": "updated",
                "workflow_id": existing_wf_id,
                "node_count": node_count,
                "message": f"Re-deployed {node_count} nodes to workflow {existing_wf_id}",
            }
            if not cred_check["ok"]:
                missing_types = list({m["cred_type"] for m in cred_check["missing"]})
                resp_data["missing_credentials"] = cred_check["missing"]
                resp_data["message"] += f" — ⚠ missing credentials: {', '.join(missing_types)}"
            return resp_data
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Workflow %s not found in n8n (stale ID), creating new", existing_wf_id)
                # Fall through to create a new workflow
            else:
                raise

    # Create new workflow (either no existing ID or stale 404)
    from src.server.workflow_designer import n8n_import_workflow
    result = await n8n_import_workflow(workflow_json)
    new_id = result.get("id")
    if new_id:
        update_workspace(workspace_id, user["user_id"], n8n_workflow_id=str(new_id))

    cred_check = await _check_missing_credentials(workflow_json)
    resp_data = {
        "ok": True,
        "action": "created",
        "workflow_id": new_id,
        "node_count": len(result.get("nodes", [])),
        "message": f"Created new workflow {new_id}",
    }
    if not cred_check["ok"]:
        missing_types = list({m["cred_type"] for m in cred_check["missing"]})
        resp_data["missing_credentials"] = cred_check["missing"]
        resp_data["message"] += f" — ⚠ missing credentials: {', '.join(missing_types)}"
    return resp_data


# ── Credential check & create endpoints ─────────────────────────────

@router.get("/{workspace_id}/credentials/check")
async def check_credentials(workspace_id: str, user: dict = Depends(get_current_user)):
    """Check which credentials a workspace workflow needs and which are missing.

    Returns required, available, and missing credentials so the UI can prompt the user.
    """
    ws = _get_user_workspace(workspace_id, user)
    design = _get_latest_design(workspace_id)
    if not design or not design.get("result_data"):
        return {"ok": True, "required": [], "available": [], "missing": [], "message": "No design data"}

    try:
        workflow_json = json.loads(design["result_data"])
    except (json.JSONDecodeError, TypeError):
        return {"ok": True, "required": [], "available": [], "missing": [], "message": "Invalid design JSON"}

    if not await ensure_n8n_ready():
        raise HTTPException(status_code=503, detail="n8n is not available")

    result = await _check_missing_credentials(workflow_json)
    return result


class CredentialCreateRequest(BaseModel):
    """Create a credential in n8n for a workspace workflow."""
    cred_type: str = Field(..., description="Credential type (e.g. 'telegramApi', 'openAiApi')")
    name: str = Field(..., min_length=1, max_length=200, description="Display name")
    data: dict = Field(..., description="Credential data key-value pairs")


@router.post("/{workspace_id}/credentials/create")
async def create_workspace_credential(
    workspace_id: str,
    req: CredentialCreateRequest,
    user: dict = Depends(get_current_user),
):
    """Create a credential in n8n and optionally relink it to the workspace workflow nodes.

    After creation, the credential ID is available for linking to workflow nodes.
    """
    ws = _get_user_workspace(workspace_id, user)

    if not await ensure_n8n_ready():
        raise HTTPException(status_code=503, detail="n8n is not available")

    from src.server.n8n_setup import n8n_api_call

    try:
        resp = await n8n_api_call(
            "POST",
            "/rest/credentials",
            json_data={"type": req.cred_type, "name": req.name, "data": req.data},
        )
        resp.raise_for_status()
        result = resp.json()
        cred = result.get("data", result) if isinstance(result, dict) else result
        cred_id = cred.get("id", "?")

        # Try to auto-link the credential to workflow nodes that need it
        linked_nodes = await _autolink_credential(ws, workspace_id, user, req.cred_type, str(cred_id))

        return {
            "ok": True,
            "credential_id": cred_id,
            "name": req.name,
            "type": req.cred_type,
            "linked_nodes": linked_nodes,
        }
    except httpx.HTTPStatusError as exc:
        detail = f"n8n rejected credential creation: {exc.response.status_code}"
        try:
            detail = exc.response.json().get("message", detail)
        except Exception:
            pass
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc


async def _autolink_credential(
    ws: dict,
    workspace_id: str,
    user: dict,
    cred_type: str,
    cred_id: str,
) -> list[str]:
    """Auto-link a newly created credential to matching workflow nodes.

    Updates both the design record and the live n8n workflow (if it exists).
    Returns a list of node names that were linked.
    """
    design = _get_latest_design(workspace_id)
    if not design or not design.get("result_data"):
        return []

    try:
        wf = json.loads(design["result_data"])
    except (json.JSONDecodeError, TypeError):
        return []

    linked: list[str] = []
    for node in wf.get("nodes", []):
        creds = node.get("credentials")
        if not creds:
            continue
        if cred_type in creds:
            # Update the credential reference with the new ID
            if isinstance(creds[cred_type], dict):
                creds[cred_type]["id"] = cred_id
            else:
                creds[cred_type] = {"id": cred_id, "name": cred_type}
            linked.append(node.get("name", "Unknown"))

    if linked:
        # Save updated design
        update_workspace_design(design["id"], result_data=json.dumps(wf))

        # Push to live n8n workflow if it exists
        n8n_wf_id = ws.get("n8n_workflow_id")
        if n8n_wf_id:
            from src.server.n8n_setup import n8n_api_call
            payload = {
                "name": wf.get("name", ws.get("title", "Untitled")),
                "nodes": wf["nodes"],
                "connections": wf.get("connections", {}),
                "active": False,
                "settings": wf.get("settings", {"executionOrder": "v1"}),
            }
            try:
                await n8n_api_call("PUT", f"/rest/workflows/{n8n_wf_id}", json_data=payload)
                logger.info("Auto-linked credential %s to %d nodes in workflow %s", cred_id, len(linked), n8n_wf_id)
            except Exception as exc:
                logger.warning("Could not push credential update to n8n: %s", exc)

    return linked


@router.get("/{workspace_id}/status")
async def get_workspace_status(workspace_id: str, user: dict = Depends(get_current_user)):
    """Poll the current workspace + n8n workflow status."""
    ws = _get_user_workspace(workspace_id, user)

    result: dict = {
        "workspace_id": ws["id"],
        "status": ws["status"],
        "n8n_workflow_id": ws.get("n8n_workflow_id"),
    }

    # If there's an active workflow, fetch live status from n8n
    if ws.get("n8n_workflow_id") and await verify_n8n_access(user):
        try:
            wf = await n8n_get_workflow(ws["n8n_workflow_id"])
            result["n8n_workflow_active"] = wf.get("active", False)
            result["n8n_workflow_name"] = wf.get("name")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                result["n8n_workflow_active"] = None
                result["n8n_workflow_stale"] = True
                result["n8n_error"] = "Workflow not found in n8n — use Redeploy to recreate"
            else:
                result["n8n_workflow_active"] = None
                result["n8n_error"] = "Could not fetch workflow status"
        except httpx.ConnectError:
            result["n8n_workflow_active"] = None
            result["n8n_error"] = "n8n service unavailable"

    return result


@router.get("/{workspace_id}/designs")
async def list_designs(workspace_id: str, user: dict = Depends(get_current_user)):
    """List all design history records for a workspace."""
    _get_user_workspace(workspace_id, user)
    return {"designs": list_workspace_designs(workspace_id)}


# ── n8n management endpoints ────────────────────────────────────────

@router.get("/{workspace_id}/executions")
async def get_executions(workspace_id: str, user: dict = Depends(get_current_user)):
    """Get execution history for a workspace's n8n workflow."""
    ws = _get_user_workspace(workspace_id, user)
    n8n_wf_id = ws.get("n8n_workflow_id")
    if not n8n_wf_id:
        return {"executions": [], "message": "No n8n workflow linked"}

    from src.server.n8n_manager import get_executions as _get_execs
    try:
        execs = await _get_execs(workflow_id=n8n_wf_id, limit=20)
        return {"executions": execs}
    except (httpx.ConnectError, httpx.HTTPStatusError):
        return {"executions": [], "message": "n8n not reachable"}
    except Exception as exc:
        logger.warning("Failed to list executions: %s", exc)
        return {"executions": [], "message": "Could not fetch executions"}


@router.get("/{workspace_id}/executions/{execution_id}")
async def get_execution_detail_route(
    workspace_id: str, execution_id: str, user: dict = Depends(get_current_user),
):
    """Get detailed execution data including node outputs and errors."""
    _get_user_workspace(workspace_id, user)
    from src.server.n8n_manager import get_execution_summary
    try:
        summary = await get_execution_summary(execution_id)
        return {"summary": summary}
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        raise HTTPException(status_code=502, detail=f"n8n unreachable: {exc}") from exc
    except Exception as exc:
        logger.warning("Failed to get execution detail %s: %s", execution_id, exc)
        raise HTTPException(status_code=502, detail="Could not fetch execution detail from n8n") from exc


class NodeInstallRequest(BaseModel):
    package_name: str = Field(..., min_length=1, max_length=200)


@router.post("/{workspace_id}/nodes/install")
async def install_node(
    workspace_id: str, req: NodeInstallRequest, user: dict = Depends(get_current_user),
):
    """Install a community node package in n8n."""
    _get_user_workspace(workspace_id, user)
    from src.server.n8n_manager import install_community_node
    result = await install_community_node(req.package_name)
    return {"installed": True, "package": req.package_name, "detail": result}


@router.get("/{workspace_id}/nodes")
async def list_available_nodes(
    workspace_id: str, user: dict = Depends(get_current_user),
):
    """List available node types in n8n (cached)."""
    _get_user_workspace(workspace_id, user)
    from src.server.n8n_manager import get_available_nodes
    nodes = await get_available_nodes()
    # Return condensed list (name + displayName only)
    return {
        "total": len(nodes),
        "nodes": [{"name": n.get("name", ""), "displayName": n.get("displayName", "")} for n in nodes[:500]],
    }


# ── Workspace Agent Chat ─────────────────────────────────────────────

def _load_agent_skills() -> str:
    """Load agent skill files from data/skills/ and return as a combined string."""
    skills_dir = Path(__file__).resolve().parents[2] / "data" / "skills"
    if not skills_dir.is_dir():
        return ""
    parts = []
    for md_file in sorted(skills_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            parts.append(content.strip())
        except OSError:
            continue
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


_AGENT_SKILLS_CACHE: str | None = None


def _get_agent_skills() -> str:
    global _AGENT_SKILLS_CACHE
    if _AGENT_SKILLS_CACHE is None:
        _AGENT_SKILLS_CACHE = _load_agent_skills()
    return _AGENT_SKILLS_CACHE


_N8N_SYSTEM_PROMPT = (
    "You are a proactive workspace assistant specialised in n8n workflow automation. "
    "You have tools to inspect, diagnose, update, and manage n8n workflows, "
    "executions, credentials, and community nodes.\n\n"
    "You also have access to a library of ~290 community templates and the official "
    "n8n.io template gallery. When the user asks to build or improve a workflow:\n"
    "1. Search local templates first (n8n_search_templates) for a close match.\n"
    "2. If no good local match, search official templates (n8n_search_official).\n"
    "3. Get the full JSON (n8n_get_template_detail or n8n_fetch_official_template).\n"
    "4. Deploy directly: pass the JSON output as workflow_json to n8n_create_workflow.\n"
    "   - The template tools output raw JSON that pipes directly into create/update tools.\n"
    "   - Override the name with the name parameter if needed.\n"
    "5. Install any missing node types with n8n_install_node.\n\n"
    "TOOL CHAINING:\n"
    "- Template → Deploy: n8n_get_template_detail → n8n_create_workflow(workflow_json=output)\n"
    "- Get → Update: n8n_get_workflow → modify → n8n_update_workflow(workflow_json=modified)\n"
    "- Both n8n_create_workflow and n8n_update_workflow accept workflow_json (full JSON string) "
    "or individual name/nodes/connections params.\n\n"
    "CREDENTIAL SETUP (CRITICAL — do this BEFORE activation):\n"
    "When deploying or activating a workflow:\n"
    "1. Inspect the workflow nodes for credential references (look at 'credentials' field)\n"
    "2. Call n8n_list_credentials to check what's already configured\n"
    "3. For EACH missing credential:\n"
    "   a. Tell the user which service needs a credential (e.g. 'Telegram Bot Token')\n"
    "   b. Ask for the secret value (API key, token, etc.)\n"
    "   c. Call n8n_create_credential with the correct type, name, and data\n"
    "4. After creating credentials, update the workflow nodes to link the credential IDs\n"
    "5. ONLY THEN attempt to activate the workflow\n"
    "Common credential types: telegramApi (accessToken), openAiApi (apiKey), "
    "slackApi (accessToken), httpHeaderAuth (name, value), githubApi (accessToken)\n\n"
    "CRITICAL RULES:\n"
    "1. ALWAYS use your tools FIRST before asking the user for information. "
    "If you have a workflow ID, call n8n_workflow_status immediately.\n"
    "2. When asked about errors, call n8n_list_executions (filter status='error') "
    "then n8n_diagnose_error on the failed execution. DO NOT ask for execution IDs.\n"
    "3. When the user says 'fix', 'help', or 'error' — start by gathering data with tools.\n"
    "4. NEVER respond with 'I need more information' if you have tools that can look it up.\n"
    "5. After tool results, synthesize a clear diagnosis and actionable fix.\n"
    "6. When building workflows, always search for existing templates first before designing from scratch.\n"
    "7. ALWAYS check credentials before trying to activate a workflow.\n\n"
    "Keep answers concise and actionable. Show what you found, what went wrong, and how to fix it."
)


@router.post("/{workspace_id}/chat")
async def workspace_agent_chat(
    workspace_id: str,
    body: WorkspaceChatRequest,
    user: dict = Depends(get_current_user),
):
    """Chat with an n8n-scoped agent for this workspace. Returns SSE stream.

    The agent has access only to n8n tools (workflow status, executions,
    credentials, node install, diagnose, update, suggest improvements).
    """
    ws = _get_user_workspace(workspace_id, user)

    # Inject workspace context into the system prompt
    wf_id = ws.get("n8n_workflow_id") or ""
    context_note = "\n\nWorkspace context:"
    context_note += f"\n- Workspace ID: {workspace_id}"
    context_note += f"\n- Workspace title: {ws.get('title', 'Untitled')}"
    context_note += f"\n- Status: {ws['status']}"
    if wf_id:
        context_note += f"\n- n8n Workflow ID: {wf_id}"
        context_note += "\n\nYou HAVE the workflow ID. Use n8n_workflow_status and n8n_list_executions immediately — do NOT ask the user for it."
    else:
        context_note += "\n- No n8n workflow linked yet."

    system_msg = _N8N_SYSTEM_PROMPT + context_note

    # Inject agent skills
    skills_text = _get_agent_skills()
    if skills_text:
        system_msg += "\n\n## Your Skills Reference\n\n" + skills_text

    # Build message history
    messages: list[dict] = [{"role": "system", "content": system_msg}]
    for m in body.history[-20:]:  # cap history
        role = m.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": m.get("content", "")})
    messages.append({"role": "user", "content": body.message})

    return EventSourceResponse(_workspace_chat_stream(messages, body.model, body.provider))


async def _workspace_chat_stream(messages: list[dict], model: str | None = None, provider: str | None = None):
    """SSE generator for workspace agent chat using only n8n tools.

    If the server is running in local mode (STANDARD/TURBOQUANT/etc.),
    a temporary cloud engine is created on-the-fly from the stored cloud
    config — so workspace chat works regardless of inference mode.
    """
    from src.agent.cloud_loop import CloudAgentLoop
    from src.server.app import get_agent_registry

    registry = get_agent_registry()
    if registry is None:
        yield {"data": json.dumps({"type": "error", "message": "Agent tools not initialized"})}
        yield {"data": "[DONE]"}
        return

    # Filter to n8n tools only
    n8n_registry = registry.subset("n8n_")

    if not n8n_registry.list_tools():
        yield {"data": json.dumps({"type": "error", "message": "No n8n tools available"})}
        yield {"data": "[DONE]"}
        return

    # For chat (agent with tools), we need a cloud engine that supports tool calling.
    # Local engine doesn't support the tools/tool_choice kwargs.
    # If provider is explicitly "local", warn the user.
    if provider == "local":
        yield {"data": json.dumps({
            "type": "error",
            "message": "Local model does not support agent tool calling. Select a cloud provider for workspace chat.",
        })}
        yield {"data": "[DONE]"}
        return

    cloud_engine, is_temporary = _create_engine_for_provider(provider, model)
    if cloud_engine is None:
        yield {"data": json.dumps({
            "type": "error",
            "message": (
                "No inference engine available. "
                "Configure a cloud provider in config/cloud.yaml or set "
                "TURBOQUANT_CLOUD_<PROVIDER>_API_KEY environment variable, "
                "or start the server with a local model."
            ),
        })}
        yield {"data": "[DONE]"}
        return

    engine_to_use = cloud_engine
    dispose_engine = is_temporary

    try:
        loop = CloudAgentLoop(n8n_registry)

        async for event in loop.run(engine_to_use, messages, max_tokens=4096):
            yield {"data": json.dumps(event)}

        yield {"data": "[DONE]"}
    finally:
        if dispose_engine:
            try:
                engine_to_use.unload()
            except Exception:
                pass
