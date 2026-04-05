"""TOON (Token-Oriented Object Notation) encoder/decoder for n8n workflows.

Converts between n8n workflow JSON and a compact TOON representation that
uses ~50% fewer tokens, allowing smaller LLMs to generate workflows within
their context windows.

TOON format for n8n workflows:
```
name: My Workflow

nodes[2]{name,type,ver}:
  Schedule Trigger,n8n-nodes-base.scheduleTrigger,1.2
  Telegram,n8n-nodes-base.telegram,1.2

params:
  Telegram:
    chatId: 1086032366
    text: Reminder: Drink coffee

creds:
  Telegram:
    telegramApi: 1

connections[1]{src,dest}:
  Schedule Trigger,Telegram
```
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


def workflow_to_toon(wf: dict[str, Any]) -> str:
    """Convert an n8n workflow JSON dict to compact TOON representation."""
    lines: list[str] = []
    name = wf.get("name", "Untitled")
    lines.append(f"name: {name}")

    nodes = wf.get("nodes", [])

    # Nodes header
    lines.append(f"\nnodes[{len(nodes)}]{{name,type,ver}}:")
    for n in nodes:
        nname = n.get("name", "Node")
        ntype = n.get("type", "unknown")
        ver = n.get("typeVersion", 1)
        lines.append(f"  {nname},{ntype},{ver}")

    # Parameters — only for nodes that have non-empty params
    param_sections: list[str] = []
    for n in nodes:
        params = n.get("parameters", {})
        # Filter out empty/default params
        useful_params = {
            k: v for k, v in params.items()
            if v and v != {} and k not in ("additionalFields",)
            or k == "additionalFields" and v
        }
        if useful_params:
            section_lines = [f"  {n['name']}:"]
            for k, v in useful_params.items():
                if isinstance(v, (dict, list)):
                    # Compact JSON for complex values
                    val = json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=str)
                    section_lines.append(f"    {k}: {val}")
                elif isinstance(v, str) and v and (v.isdigit() or v.replace(".", "", 1).isdigit() or v.replace("-", "", 1).isdigit()):
                    # Quote strings that look like numbers to preserve type
                    section_lines.append(f'    {k}: "{v}"')
                else:
                    section_lines.append(f"    {k}: {v}")
            param_sections.append("\n".join(section_lines))

    if param_sections:
        lines.append("\nparams:")
        lines.extend(param_sections)

    # Credentials
    cred_sections: list[str] = []
    for n in nodes:
        creds = n.get("credentials", {})
        if creds:
            section_lines = [f"  {n['name']}:"]
            for ctype, cvalue in creds.items():
                if isinstance(cvalue, dict):
                    cid = cvalue.get("id", "?")
                    section_lines.append(f"    {ctype}: {cid}")
                else:
                    section_lines.append(f"    {ctype}: {cvalue}")
            cred_sections.append("\n".join(section_lines))

    if cred_sections:
        lines.append("\ncreds:")
        lines.extend(cred_sections)

    # Connections
    conn_pairs: list[tuple[str, str]] = []
    for src_name, targets in wf.get("connections", {}).items():
        if isinstance(targets, dict):
            for conn_type, target_list in targets.items():
                if isinstance(target_list, list):
                    for group in target_list:
                        if isinstance(group, list):
                            for t in group:
                                dest = t.get("node", "?")
                                ctype = t.get("type", "main")
                                if ctype == "main":
                                    conn_pairs.append((src_name, dest))
                                else:
                                    conn_pairs.append((src_name, f"{dest}:{ctype}"))

    if conn_pairs:
        lines.append(f"\nconnections[{len(conn_pairs)}]{{src,dest}}:")
        for src, dest in conn_pairs:
            lines.append(f"  {src},{dest}")

    return "\n".join(lines)


def toon_to_workflow(toon_str: str) -> dict[str, Any]:
    """Convert a TOON-formatted n8n workflow string back to JSON dict.

    Parses the compact TOON format and reconstructs a full n8n workflow JSON
    with proper UUIDs, positions, and connection structures.
    """
    lines = toon_str.strip().split("\n")

    name = "Untitled"
    nodes_raw: list[dict[str, str]] = []
    params_map: dict[str, dict[str, Any]] = {}
    creds_map: dict[str, dict[str, str]] = {}
    connections_raw: list[tuple[str, str]] = []

    section = None  # current section: nodes, params, creds, connections
    current_node_name: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key: value
        if stripped.startswith("name:"):
            name = stripped[5:].strip()
            section = None
            continue

        # Section headers
        if re.match(r"nodes\[\d+\]\{.*\}:", stripped):
            section = "nodes"
            continue
        if stripped == "params:":
            section = "params"
            current_node_name = None
            continue
        if stripped == "creds:":
            section = "creds"
            current_node_name = None
            continue
        if re.match(r"connections\[\d+\]\{.*\}:", stripped):
            section = "connections"
            continue

        # Parse based on current section
        if section == "nodes":
            # Format: name,type,ver
            parts = stripped.split(",", 2)
            if len(parts) >= 2:
                nodes_raw.append({
                    "name": parts[0].strip(),
                    "type": parts[1].strip(),
                    "ver": parts[2].strip() if len(parts) > 2 else "1",
                })

        elif section == "params":
            # Node name header: "  NodeName:"
            if stripped.endswith(":") and not stripped.startswith(" ") or (
                line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":")
            ):
                current_node_name = stripped.rstrip(":").strip()
                if current_node_name not in params_map:
                    params_map[current_node_name] = {}
            elif current_node_name and ":" in stripped:
                # Parameter: "    key: value"
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()
                # Try to parse JSON values
                params_map[current_node_name][key] = _parse_value(value)

        elif section == "creds":
            if stripped.endswith(":") and not stripped.startswith(" ") or (
                line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":")
            ):
                current_node_name = stripped.rstrip(":").strip()
                if current_node_name not in creds_map:
                    creds_map[current_node_name] = {}
            elif current_node_name and ":" in stripped:
                key, _, value = stripped.partition(":")
                creds_map[current_node_name][key.strip()] = value.strip()

        elif section == "connections":
            # Format: src,dest or src,dest:type
            parts = stripped.split(",", 1)
            if len(parts) == 2:
                connections_raw.append((parts[0].strip(), parts[1].strip()))

    # Build n8n workflow JSON
    n8n_nodes: list[dict[str, Any]] = []
    x_pos = 250
    for i, nr in enumerate(nodes_raw):
        node: dict[str, Any] = {
            "parameters": params_map.get(nr["name"], {}),
            "id": str(uuid.uuid4()),
            "name": nr["name"],
            "type": nr["type"],
            "typeVersion": _parse_version(nr.get("ver", "1")),
            "position": [x_pos, 300],
        }

        # Add credentials
        if nr["name"] in creds_map:
            node["credentials"] = {}
            for ctype, cid in creds_map[nr["name"]].items():
                node["credentials"][ctype] = {
                    "id": str(cid),
                    "name": f"{ctype} credentials",
                }

        n8n_nodes.append(node)
        x_pos += 250

    # Build connections
    n8n_connections: dict[str, Any] = {}
    for src, dest_raw in connections_raw:
        # Handle typed connections: "dest:ai_languageModel"
        if ":" in dest_raw:
            dest, conn_type = dest_raw.rsplit(":", 1)
        else:
            dest = dest_raw
            conn_type = "main"

        if src not in n8n_connections:
            n8n_connections[src] = {}
        if conn_type not in n8n_connections[src]:
            n8n_connections[src][conn_type] = [[]]

        n8n_connections[src][conn_type][0].append({
            "node": dest,
            "type": conn_type,
            "index": 0,
        })

    return {
        "name": name,
        "nodes": n8n_nodes,
        "connections": n8n_connections,
        "pinData": {},
    }


def _parse_value(val: str) -> Any:
    """Parse a TOON value string into a Python object."""
    if not val:
        return ""
    # Try JSON parse first
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    # Boolean
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    # Number
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        pass
    # String (may contain expressions like ={{ $json.xx }})
    return val


def _parse_version(ver: str) -> float | int:
    """Parse a version string like '1.2' into a number."""
    try:
        v = float(ver)
        return int(v) if v == int(v) else v
    except ValueError:
        return 1


def extract_toon_from_response(text: str) -> str | None:
    """Extract TOON content from LLM response, handling code fences."""
    text = text.strip()

    # Try extracting from toon/text code fence
    for marker in ("```toon", "```text", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end_pos = text.find("```", start)
            if end_pos > start:
                return text[start:end_pos].strip()

    # If it starts with "name:" it's likely raw TOON
    if text.startswith("name:") or text.startswith("# "):
        return text

    # Check if it looks like TOON (has nodes[...]{...}: pattern)
    if re.search(r"nodes\[\d+\]\{", text):
        return text

    return None
