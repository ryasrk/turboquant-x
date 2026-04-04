"""n8n management: node discovery, installation, and execution logs.

Provides capabilities for the TQ agent to understand and manage n8n:
- Fetch available node types (cached)
- Install/uninstall community nodes
- Read execution logs and details
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Node type cache ──────────────────────────────────────────────────

_node_cache: list[dict[str, Any]] = []
_node_names_cache: set[str] = set()
_cache_ts: float = 0
_CACHE_TTL = 300  # 5 minutes


async def get_available_nodes(force_refresh: bool = False) -> list[dict[str, Any]]:
    """Fetch all available node types from n8n (cached for 5 min)."""
    global _node_cache, _node_names_cache, _cache_ts

    if not force_refresh and _node_cache and (time.time() - _cache_ts) < _CACHE_TTL:
        return _node_cache

    try:
        from src.server.n8n_setup import n8n_api_call
        resp = await n8n_api_call("GET", "/types/nodes.json")
        resp.raise_for_status()
        nodes = resp.json()

        _node_cache = nodes
        _node_names_cache = {n.get("name", "") for n in nodes if n.get("name")}
        _cache_ts = time.time()
        logger.info("Cached %d n8n node types", len(_node_cache))
        return _node_cache

    except Exception:
        logger.exception("Failed to fetch n8n node types")
        return _node_cache  # Return stale cache on error


async def get_node_names() -> set[str]:
    """Get set of available node type names (e.g. 'n8n-nodes-base.httpRequest')."""
    if not _node_names_cache:
        await get_available_nodes()
    return _node_names_cache


async def is_node_available(node_type: str) -> bool:
    """Check if a specific node type is available."""
    names = await get_node_names()
    return node_type in names


async def get_core_node_list() -> str:
    """Get a condensed list of available core nodes for LLM prompts.

    Returns a formatted string organized by category.
    """
    nodes = await get_available_nodes()
    if not nodes:
        return "(Could not fetch node types from n8n)"

    categories: dict[str, list[str]] = {
        "triggers": [],
        "logic": [],
        "data": [],
        "communication": [],
        "ai": [],
        "files": [],
        "other_services": [],
    }

    for n in nodes:
        name = n.get("name", "")
        if not name:
            continue

        lower = name.lower()

        if "trigger" in lower or "cron" in lower:
            categories["triggers"].append(name)
        elif any(x in lower for x in [".if", ".switch", ".merge", ".splitinbatches",
                                       ".splitout", ".filter", ".sort", ".limit",
                                       ".aggregate", ".removeduplicates", ".noop",
                                       ".wait", ".executeworkflow", ".code",
                                       ".function", ".set", ".rename"]):
            categories["logic"].append(name)
        elif any(x in lower for x in [".httprequest", ".postgres", ".mysql",
                                       ".mongodb", ".redis", ".graphql",
                                       ".rssfeed", ".html", ".xml", ".json"]):
            categories["data"].append(name)
        elif any(x in lower for x in [".slack", ".discord", ".telegram",
                                       ".gmail", ".email", ".smtp",
                                       ".microsoftteams", ".whatsapp"]):
            categories["communication"].append(name)
        elif "langchain" in lower or "n8n-nodes-langchain" in lower:
            categories["ai"].append(name)
        elif any(x in lower for x in [".ftp", ".ssh", ".compression",
                                       ".converttofile", ".extractfromfile",
                                       ".readbinary", ".writebinary", ".markdown"]):
            categories["files"].append(name)

    lines = ["Available n8n node types (use ONLY these):"]
    for cat, names in categories.items():
        if names:
            lines.append(f"\n### {cat.replace('_', ' ').title()}")
            for n in sorted(set(names))[:30]:  # Cap per category for prompt size
                lines.append(f"  - {n}")
            if len(names) > 30:
                lines.append(f"  ... and {len(names) - 30} more")

    return "\n".join(lines)


# ── Community node management ────────────────────────────────────────

async def install_community_node(package_name: str) -> dict[str, Any]:
    """Install a community node package in n8n.

    Args:
        package_name: npm package name (e.g. 'n8n-nodes-google-drive')

    Returns:
        Installation result from n8n API.
    """
    from src.server.n8n_setup import n8n_api_call

    resp = await n8n_api_call(
        "POST",
        "/rest/community-packages",
        json_data={"name": package_name},
    )
    resp.raise_for_status()
    result = resp.json()
    logger.info("Installed community node: %s → %s", package_name, resp.status_code)

    # Invalidate node cache
    global _cache_ts
    _cache_ts = 0

    return result


async def uninstall_community_node(package_name: str) -> dict[str, Any]:
    """Uninstall a community node package from n8n."""
    from src.server.n8n_setup import n8n_api_call

    resp = await n8n_api_call(
        "DELETE",
        "/rest/community-packages",
        params={"name": package_name},
    )
    resp.raise_for_status()
    result = resp.json()
    logger.info("Uninstalled community node: %s → %s", package_name, resp.status_code)

    global _cache_ts
    _cache_ts = 0

    return result


async def list_community_nodes() -> list[dict[str, Any]]:
    """List installed community node packages."""
    from src.server.n8n_setup import n8n_api_call

    resp = await n8n_api_call("GET", "/rest/community-packages")
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data


# ── Execution logs ───────────────────────────────────────────────────

async def get_executions(
    workflow_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get n8n execution history.

    Args:
        workflow_id: Filter by specific workflow.
        status: Filter by status ('success', 'error', 'waiting').
        limit: Max results.
    """
    from src.server.n8n_setup import n8n_api_call

    params: dict[str, Any] = {"limit": limit}
    if workflow_id:
        params["workflowId"] = workflow_id
    if status:
        params["status"] = status

    resp = await n8n_api_call("GET", "/rest/executions", params=params)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("data", {}).get("results", []) if isinstance(data.get("data"), dict) else data.get("data", [])
    return results


async def get_execution_detail(execution_id: str) -> dict[str, Any]:
    """Get detailed execution data including node outputs and errors."""
    from src.server.n8n_setup import n8n_api_call

    resp = await n8n_api_call("GET", f"/rest/executions/{execution_id}")
    resp.raise_for_status()
    raw = resp.json()
    detail = raw.get("data", raw) if isinstance(raw, dict) else raw

    # n8n may return the inner 'data' (run results) as a JSON string
    if isinstance(detail, dict) and isinstance(detail.get("data"), str):
        import json as _json
        try:
            detail["data"] = _json.loads(detail["data"])
        except (ValueError, TypeError):
            pass

    # n8n v2 uses a compact flattened-array format for execution data.
    # Reconstruct error info from the flat array so downstream consumers
    # get a consistent dict structure.
    if isinstance(detail, dict) and isinstance(detail.get("data"), list):
        detail["_raw_data"] = detail["data"]
        detail["data"] = _reconstruct_execution_data(detail["data"])

    return detail


def _reconstruct_execution_data(flat: list) -> dict[str, Any]:
    """Reconstruct error info from n8n v2's compressed execution array.

    The array format is: [{schema}, {startData}, {resultData}, {execData}, ...]
    where inner values like "5" are references to flat[5].
    We extract error info by searching for known error patterns.
    """
    result: dict[str, Any] = {"resultData": {}}

    # Search for error objects in the flat array
    for item in flat:
        if isinstance(item, dict):
            # Look for error-like objects (have 'name'+'message' or 'error' key)
            if "message" in item and "name" in item and "stack" in item:
                # This looks like a resolved error reference
                pass
            if "error" in item and "runData" in item:
                # This is the resultData schema — resolve references
                error_ref = item.get("error")
                if isinstance(error_ref, str) and error_ref.isdigit():
                    idx = int(error_ref)
                    if idx < len(flat) and isinstance(flat[idx], dict):
                        # Resolve the nested error fields
                        err_obj = {}
                        for ek, ev in flat[idx].items():
                            if isinstance(ev, str) and ev.isdigit():
                                ref_idx = int(ev)
                                if ref_idx < len(flat):
                                    err_obj[ek] = flat[ref_idx]
                                else:
                                    err_obj[ek] = ev
                            else:
                                err_obj[ek] = ev
                        result["resultData"]["error"] = err_obj
                    elif isinstance(error_ref, str):
                        result["resultData"]["error"] = {"message": error_ref}

    # Also look for string items that match common error patterns
    for item in flat:
        if isinstance(item, str) and (
            "Error" in item or "error" in item
        ) and len(item) > 10 and not item.startswith("{"):
            # Could be an error message or stack trace
            if "resultData" not in result:
                result["resultData"] = {}
            if "error" not in result["resultData"]:
                if "\n" in item:  # Stack trace
                    result["resultData"].setdefault("error", {})["stack"] = item
                else:  # Error message
                    result["resultData"].setdefault("error", {})["message"] = item

    return result


async def get_execution_summary(execution_id: str) -> str:
    """Get a human-readable summary of an execution for the agent."""
    detail = await get_execution_detail(execution_id)

    lines = [
        f"Execution #{execution_id}",
        f"Status: {detail.get('status', 'unknown')}",
        f"Workflow: {detail.get('workflowData', {}).get('name', 'unknown')}",
        f"Started: {detail.get('startedAt', 'unknown')}",
        f"Finished: {detail.get('stoppedAt', 'unknown')}",
        "",
    ]

    # Get run data — 'data' is a dict (already parsed by get_execution_detail)
    inner_data = detail.get("data", {})
    if not isinstance(inner_data, dict):
        inner_data = {}
    result_data = inner_data.get("resultData", {})
    if not isinstance(result_data, dict):
        result_data = {}

    # Show top-level execution error if present
    top_error = result_data.get("error")
    if isinstance(top_error, dict):
        lines.append(f"Error: {top_error.get('name', 'Unknown')}: {top_error.get('message', 'no message')}")
        lines.append("")
    elif isinstance(top_error, str):
        lines.append(f"Error: {top_error}")
        lines.append("")

    # Show per-node errors
    run_data = result_data.get("runData", {})
    if not isinstance(run_data, dict):
        run_data = {}

    for node_name, node_runs in run_data.items():
        if not isinstance(node_runs, list):
            node_runs = [node_runs]
        for run in node_runs:
            if not isinstance(run, dict):
                continue
            status = "OK" if not run.get("error") else "ERROR"
            lines.append(f"  [{status}] {node_name}")
            if run.get("error"):
                err = run["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                lines.append(f"    Error: {msg}")

    return "\n".join(lines)


# ── Workflow node validation ─────────────────────────────────────────

async def validate_workflow_nodes(workflow: dict[str, Any]) -> list[str]:
    """Check all nodes in a workflow against available node types.

    Returns list of unavailable node type names.
    """
    names = await get_node_names()
    missing = []
    for node in workflow.get("nodes", []):
        ntype = node.get("type", "")
        if ntype and ntype not in names:
            missing.append(ntype)
    return missing
