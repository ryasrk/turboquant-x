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

    base_url = os.getenv("N8N_BACKEND_URL", "http://127.0.0.1:5678").rstrip("/")

    try:
        from src.server.n8n_setup import get_session_cookies
        cookie = get_session_cookies() or ""
        headers = {"Cookie": cookie} if cookie else {}

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(f"{base_url}/types/nodes.json", headers=headers)
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
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data


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

    # Node execution results
    run_data = detail.get("data", {}).get("resultData", {}).get("runData", {})
    for node_name, node_runs in run_data.items():
        for run in node_runs:
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
