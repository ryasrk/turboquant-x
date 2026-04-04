"""n8n workflow template index and search.

Indexes 291 community templates from awesome-n8n-templates repo and
provides search + retrieval for the workspace AI agent. Also supports
searching official n8n.io templates via their public API.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "data" / "n8n_templates"

# ── In-memory template index ─────────────────────────────────────────

_template_index: list[dict[str, Any]] = []
_index_built = False


def _derive_category(rel_path: str) -> str:
    """Extract category from the relative folder path."""
    parts = Path(rel_path).parts
    if len(parts) > 1:
        return parts[0].replace("_", " ")
    return "Other"


def _extract_node_types(workflow: dict) -> list[str]:
    types = set()
    for node in workflow.get("nodes", []):
        ntype = node.get("type", "")
        if ntype:
            types.add(ntype)
    return sorted(types)


def _extract_node_names(workflow: dict) -> list[str]:
    return [n.get("name", "") for n in workflow.get("nodes", []) if n.get("name")]


def build_template_index() -> int:
    """Scan the awesome-n8n-templates directory and build the in-memory index.

    Returns the number of templates indexed. Safe to call multiple times.
    """
    global _template_index, _index_built

    if _index_built:
        return len(_template_index)

    if not _TEMPLATES_DIR.is_dir():
        logger.warning("Template directory not found: %s", _TEMPLATES_DIR)
        _index_built = True
        return 0

    templates: list[dict[str, Any]] = []
    idx = 0

    for json_file in sorted(_TEMPLATES_DIR.rglob("*.json")):
        rel = json_file.relative_to(_TEMPLATES_DIR)
        # Skip hidden / non-template files
        if any(p.startswith(".") for p in rel.parts):
            continue

        try:
            with open(json_file, encoding="utf-8") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping %s: %s", rel, exc)
            continue

        # Must look like an n8n workflow (has nodes)
        if not isinstance(data, dict) or "nodes" not in data:
            continue

        name = data.get("name") or json_file.stem
        category = _derive_category(str(rel))
        node_types = _extract_node_types(data)
        node_names = _extract_node_names(data)
        node_count = len(data.get("nodes", []))

        templates.append({
            "id": idx,
            "name": name,
            "filename": str(rel),
            "category": category,
            "node_types": node_types,
            "node_names": node_names,
            "node_count": node_count,
            "path": str(json_file),
        })
        idx += 1

    _template_index = templates
    _index_built = True
    logger.info("Indexed %d n8n templates from %s", len(templates), _TEMPLATES_DIR)
    return len(templates)


def get_template_index() -> list[dict[str, Any]]:
    """Return the template index, building it on first call."""
    if not _index_built:
        build_template_index()
    return _template_index


# ── Search ───────────────────────────────────────────────────────────

def search_templates(
    query: str,
    category: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search the local template index by keyword.

    Searches name, category, node types, and node names.
    Returns up to ``limit`` results sorted by relevance score.
    """
    index = get_template_index()
    if not index:
        return []

    query_lower = query.lower()
    terms = query_lower.split()

    scored: list[tuple[float, dict]] = []

    for tpl in index:
        if category and tpl["category"].lower() != category.lower():
            continue

        score = 0.0
        name_lower = tpl["name"].lower()
        cat_lower = tpl["category"].lower()
        types_str = " ".join(tpl["node_types"]).lower()
        names_str = " ".join(tpl["node_names"]).lower()
        searchable = f"{name_lower} {cat_lower} {types_str} {names_str}"

        for term in terms:
            if term in name_lower:
                score += 3.0
            if term in cat_lower:
                score += 1.0
            if term in types_str:
                score += 2.0
            if term in names_str:
                score += 1.5
            # Partial match
            if term in searchable:
                score += 0.5

        if score > 0:
            scored.append((score, tpl))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:limit]]


def get_template_by_id(template_id: int) -> dict[str, Any] | None:
    """Return a single template's full JSON by its index ID."""
    index = get_template_index()
    for tpl in index:
        if tpl["id"] == template_id:
            try:
                with open(tpl["path"], encoding="utf-8") as fp:
                    workflow_json = json.load(fp)
                return {**tpl, "workflow": workflow_json}
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Failed reading template %d: %s", template_id, exc)
                return None
    return None


def get_categories() -> list[str]:
    """Return sorted list of template categories."""
    index = get_template_index()
    return sorted({t["category"] for t in index})


# ── Template adaptation ──────────────────────────────────────────────

def strip_credentials(workflow: dict[str, Any]) -> dict[str, Any]:
    """Remove credential IDs/names from a template so it can be safely imported.

    Credential *types* are kept so n8n auto-prompts for configuration.
    """
    import copy
    wf = copy.deepcopy(workflow)
    for node in wf.get("nodes", []):
        creds = node.get("credentials")
        if isinstance(creds, dict):
            for cred_key in creds:
                if isinstance(creds[cred_key], dict):
                    creds[cred_key] = {"id": "", "name": ""}
    # Strip meta instanceId
    if "meta" in wf:
        wf["meta"] = {}
    return wf


# ── Official n8n.io template search ─────────────────────────────────

_N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/search"


async def search_official_templates(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search the official n8n.io template gallery.

    Uses the public n8n API at api.n8n.io.
    Returns a list of template summaries (id, name, description, nodes, user).
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.get(
                _N8N_TEMPLATES_API,
                params={"q": query, "rows": limit, "page": 1},
            )
            resp.raise_for_status()
            data = resp.json()

        workflows = data.get("workflows", [])
        results = []
        for wf in workflows[:limit]:
            results.append({
                "id": wf.get("id"),
                "name": wf.get("name", ""),
                "description": (wf.get("description") or "")[:300],
                "nodes": [n.get("displayName", n.get("type", "")) for n in wf.get("nodes", [])],
                "created_at": wf.get("createdAt", ""),
                "user": wf.get("user", {}).get("username", ""),
                "url": f"https://n8n.io/workflows/{wf.get('id')}/",
            })
        return results
    except Exception as exc:
        logger.warning("Official template search failed: %s", exc)
        return []


async def fetch_official_template(template_id: int) -> dict[str, Any] | None:
    """Fetch a single official template's full workflow JSON from n8n.io."""
    url = f"https://api.n8n.io/api/templates/workflows/{template_id}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        wf = data.get("workflow", {}).get("workflow", data.get("workflow", {}))
        return wf if isinstance(wf, dict) and "nodes" in wf else None
    except Exception as exc:
        logger.warning("Failed to fetch official template %d: %s", template_id, exc)
        return None
