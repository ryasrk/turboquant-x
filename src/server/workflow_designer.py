"""LLM-powered n8n workflow designer.

Uses TurboQuant's local or cloud LLM to generate valid n8n workflow JSON
from natural language prompts.  The generated workflow is then importable
into n8n via ``POST /api/v1/workflows``.

Flow:
  1. User prompt → system prompt + few-shot examples → LLM
  2. LLM returns n8n workflow JSON
  3. validate_workflow() sanity-checks the JSON structure
  4. n8n_import_workflow() pushes it into n8n via REST API
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# ── Few-shot n8n workflow templates ──────────────────────────────────

_EXAMPLE_WEBHOOK_TO_SLACK = {
    "name": "Webhook to Slack Notification",
    "nodes": [
        {
            "parameters": {"httpMethod": "POST", "path": "incoming-data", "options": {}},
            "id": str(uuid.uuid4()),
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "webhookId": str(uuid.uuid4()),
        },
        {
            "parameters": {
                "resource": "message",
                "channel": {"__rl": True, "value": "#general", "mode": "name"},
                "text": "=New data received: {{ $json.body.message }}",
                "otherOptions": {},
            },
            "id": str(uuid.uuid4()),
            "name": "Slack",
            "type": "n8n-nodes-base.slack",
            "typeVersion": 2.2,
            "position": [500, 300],
            "credentials": {"slackApi": {"id": "1", "name": "Slack credentials"}},
        },
    ],
    "connections": {
        "Webhook": {"main": [[{"node": "Slack", "type": "main", "index": 0}]]},
    },
    "pinData": {},
}

_EXAMPLE_SCHEDULE_HTTP_EMAIL = {
    "name": "Scheduled API Check + Email Alert",
    "nodes": [
        {
            "parameters": {"rule": {"interval": [{"field": "hours", "hoursInterval": 1}]}},
            "id": str(uuid.uuid4()),
            "name": "Schedule Trigger",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [250, 300],
        },
        {
            "parameters": {
                "url": "https://api.example.com/status",
                "options": {},
            },
            "id": str(uuid.uuid4()),
            "name": "HTTP Request",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [500, 300],
        },
        {
            "parameters": {
                "conditions": {
                    "options": {"caseSensitive": True, "leftValue": ""},
                    "conditions": [
                        {
                            "id": str(uuid.uuid4()),
                            "leftValue": "={{ $json.status }}",
                            "rightValue": "ok",
                            "operator": {"type": "string", "operation": "notEquals"},
                        }
                    ],
                    "combinator": "and",
                },
                "options": {},
            },
            "id": str(uuid.uuid4()),
            "name": "If",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2.2,
            "position": [750, 300],
        },
        {
            "parameters": {
                "fromEmail": "alerts@example.com",
                "toEmail": "team@example.com",
                "subject": "API Status Alert",
                "text": "=The API returned status: {{ $json.status }}",
                "options": {},
            },
            "id": str(uuid.uuid4()),
            "name": "Send Email",
            "type": "n8n-nodes-base.sendEmail",
            "typeVersion": 2.1,
            "position": [1000, 200],
            "credentials": {"smtp": {"id": "2", "name": "SMTP credentials"}},
        },
    ],
    "connections": {
        "Schedule Trigger": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]},
        "HTTP Request": {"main": [[{"node": "If", "type": "main", "index": 0}]]},
        "If": {"main": [[{"node": "Send Email", "type": "main", "index": 0}], []]},
    },
    "pinData": {},
}

_EXAMPLE_CHAT_AI_AGENT = {
    "name": "AI Chat Agent with Memory",
    "nodes": [
        {
            "parameters": {"options": {}},
            "id": str(uuid.uuid4()),
            "name": "When chat message received",
            "type": "@n8n/n8n-nodes-langchain.chatTrigger",
            "typeVersion": 1.1,
            "position": [340, 20],
            "webhookId": str(uuid.uuid4()),
        },
        {
            "parameters": {"options": {}},
            "id": str(uuid.uuid4()),
            "name": "OpenAI Chat Model",
            "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
            "typeVersion": 1,
            "position": [560, 240],
        },
        {
            "parameters": {},
            "id": str(uuid.uuid4()),
            "name": "Simple Memory",
            "type": "@n8n/n8n-nodes-langchain.memoryBufferWindow",
            "typeVersion": 1.3,
            "position": [740, 240],
        },
        {
            "parameters": {
                "promptType": "define",
                "text": "={{ $json.chatInput }}",
                "options": {
                    "systemMessage": "You are a helpful AI assistant.",
                },
            },
            "id": str(uuid.uuid4()),
            "name": "AI Agent",
            "type": "@n8n/n8n-nodes-langchain.agent",
            "typeVersion": 1.7,
            "position": [580, 20],
        },
    ],
    "connections": {
        "When chat message received": {
            "main": [[{"node": "AI Agent", "type": "main", "index": 0}]],
        },
        "OpenAI Chat Model": {
            "ai_languageModel": [
                [{"node": "AI Agent", "type": "ai_languageModel", "index": 0}]
            ],
        },
        "Simple Memory": {
            "ai_memory": [
                [{"node": "AI Agent", "type": "ai_memory", "index": 0}]
            ],
        },
    },
    "pinData": {},
}

# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert n8n workflow designer. Given a user's natural language description, you generate a complete, valid n8n workflow JSON object.

## n8n Workflow JSON Format

A workflow JSON has these top-level keys:
- "name": (string) Human-readable workflow name
- "nodes": (array) List of node objects
- "connections": (object) Maps source node names to their output connections
- "pinData": (object) Usually empty {}

### Node object structure:
```json
{
  "parameters": { ... },
  "id": "uuid-v4",
  "name": "Unique Node Name",
  "type": "n8n-nodes-base.nodetype",
  "typeVersion": 1,
  "position": [x, y],
  "credentials": { "credName": { "id": "1", "name": "Credential Name" } }
}
```

### Connection object structure:
```json
{
  "Source Node Name": {
    "main": [
      [{ "node": "Target Node Name", "type": "main", "index": 0 }]
    ]
  }
}
```
- "main" outputs are arrays of arrays (one per output port)
- AI sub-nodes use connection types: "ai_languageModel", "ai_memory", "ai_tool", "ai_outputParser"

### Common node types:
- Triggers: n8n-nodes-base.webhook, n8n-nodes-base.scheduleTrigger, n8n-nodes-base.manualWorkflowTrigger, @n8n/n8n-nodes-langchain.chatTrigger
- Logic: n8n-nodes-base.if, n8n-nodes-base.switch, n8n-nodes-base.merge, n8n-nodes-base.splitInBatches
- Data: n8n-nodes-base.httpRequest, n8n-nodes-base.set, n8n-nodes-base.code, n8n-nodes-base.filter
- Services: n8n-nodes-base.slack, n8n-nodes-base.gmail, n8n-nodes-base.postgres, n8n-nodes-base.googleSheets
- AI: @n8n/n8n-nodes-langchain.agent, @n8n/n8n-nodes-langchain.lmChatOpenAi, @n8n/n8n-nodes-langchain.memoryBufferWindow

IMPORTANT: Only use node types that are ACTUALLY INSTALLED. If provided, consult the "Available Nodes" section below.

### Rules:
1. Every workflow MUST have exactly one trigger node as the first node
2. Each node MUST have a unique "name" and a valid UUID "id"
3. Credentials objects should use placeholder IDs ("1", "2", etc.) — the user will configure real credentials later
4. Position nodes left-to-right with ~250px horizontal spacing
5. Use n8n expression syntax: {{ $json.fieldName }} for referencing data
6. Return ONLY the JSON object — no markdown, no explanation
7. NEVER use node types that are not in the available list

## Examples

### Example 1: Webhook → Slack notification
User: "When I receive a webhook, send the message to Slack #general channel"
""" + json.dumps(_EXAMPLE_WEBHOOK_TO_SLACK, indent=2) + """

### Example 2: Scheduled API check with email alert
User: "Every hour, check an API status endpoint and email me if it's not OK"
""" + json.dumps(_EXAMPLE_SCHEDULE_HTTP_EMAIL, indent=2) + """

### Example 3: AI Chat Agent with memory
User: "Create a chat agent with OpenAI and conversation memory"
""" + json.dumps(_EXAMPLE_CHAT_AI_AGENT, indent=2) + """

Now generate a workflow for the user's request. Return ONLY valid JSON."""


# ── Workflow validation ──────────────────────────────────────────────

def validate_workflow(data: dict[str, Any]) -> list[str]:
    """Validate an n8n workflow JSON structure.

    Returns a list of error strings. Empty list means valid.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Workflow must be a JSON object"]

    if "name" not in data:
        errors.append("Missing 'name' field")
    if "nodes" not in data or not isinstance(data.get("nodes"), list):
        errors.append("Missing or invalid 'nodes' array")
    elif len(data["nodes"]) == 0:
        errors.append("Workflow must have at least one node")

    if "connections" not in data or not isinstance(data.get("connections"), dict):
        errors.append("Missing or invalid 'connections' object")

    if errors:
        return errors

    # Check each node
    node_names: set[str] = set()
    has_trigger = False
    for i, node in enumerate(data["nodes"]):
        if not isinstance(node, dict):
            errors.append(f"Node {i} is not an object")
            continue

        name = node.get("name")
        if not name:
            errors.append(f"Node {i} missing 'name'")
        elif name in node_names:
            errors.append(f"Duplicate node name: '{name}'")
        else:
            node_names.add(name)

        if not node.get("type"):
            errors.append(f"Node '{name or i}' missing 'type'")

        ntype = node.get("type", "")
        if "trigger" in ntype.lower() or "Trigger" in node.get("name", ""):
            has_trigger = True

        # Ensure node has an id
        if not node.get("id"):
            node["id"] = str(uuid.uuid4())

        # Ensure node has a position
        if "position" not in node:
            node["position"] = [250 + i * 250, 300]

    if not has_trigger:
        errors.append("Workflow must have at least one trigger node")

    # Validate connections reference existing nodes
    for source_name in data.get("connections", {}):
        if source_name not in node_names:
            errors.append(f"Connection source '{source_name}' not found in nodes")

    return errors


def extract_json_from_llm_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fence
    patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"(\{[\s\S]*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


# ── LLM integration ─────────────────────────────────────────────────

async def generate_workflow_via_llm(
    prompt: str,
    *,
    model: str | None = None,
    use_cloud: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Generate an n8n workflow JSON from a natural language prompt.

    Args:
        prompt: Natural language description of the desired workflow.
        model: Optional cloud model override (e.g. 'deepseek-chat', 'gpt-4o').
        use_cloud: Try cloud engine first if available.

    Yields SSE-compatible dicts with event types:
      - thinking: status updates
      - building: intermediate progress
      - workflow_json: the generated workflow JSON (final result)
      - error: if something goes wrong

    Uses TurboQuant-X's own chat completions endpoint internally.
    """
    from src.server.app import get_engine, get_or_create_cloud_engine

    yield {"event": "thinking", "message": "Analyzing your automation request..."}

    # Fetch available node types to constrain the LLM
    node_list_str = ""
    try:
        from src.server.n8n_manager import get_core_node_list
        node_list_str = await get_core_node_list()
    except Exception:
        logger.warning("Could not fetch n8n node types for prompt")

    # Search for relevant community templates as reference
    template_context = ""
    try:
        from src.server.n8n_templates import search_templates, get_template_by_id, strip_credentials
        matches = search_templates(prompt, limit=3)
        if matches:
            yield {"event": "thinking", "message": f"Found {len(matches)} relevant template(s) for reference..."}
            ref_parts = []
            for m in matches[:2]:  # at most 2 full references
                tpl = get_template_by_id(m["id"])
                if tpl and tpl.get("workflow"):
                    cleaned = strip_credentials(tpl["workflow"])
                    ref_parts.append(
                        f"### Community template: {m['name']} (category: {m['category']})\n"
                        f"```json\n{json.dumps(cleaned, indent=2)}\n```"
                    )
            if ref_parts:
                template_context = (
                    "\n\n## Reference Templates (adapt these to match the user's request)\n\n"
                    + "\n\n".join(ref_parts)
                )
    except Exception:
        logger.debug("Template search for designer context failed")

    # Build system prompt with dynamic node list and template context
    system_content = SYSTEM_PROMPT
    if node_list_str:
        system_content += f"\n\n## {node_list_str}"
    if template_context:
        system_content += template_context

    # Build messages for the LLM
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    model_label = model or "default"
    yield {"event": "building", "message": f"Generating n8n workflow design (model: {model_label})..."}

    try:
        import asyncio

        # Try cloud engine first (active or temporary), fall back to local
        cloud_engine, is_temp_engine = get_or_create_cloud_engine()
        local_engine = None
        try:
            local_engine = get_engine()
        except RuntimeError:
            pass

        response_text = ""

        if cloud_engine and cloud_engine.is_loaded:
            # If a model override is requested, create a dedicated temporary engine
            engine_to_use = cloud_engine
            dispose_after = is_temp_engine
            if model and model != cloud_engine.model_name:
                try:
                    from dataclasses import replace as dc_replace
                    from src.engine.cloud_engine import CloudEngine
                    override_cfg = dc_replace(cloud_engine._config, model=model)
                    engine_to_use = CloudEngine(override_cfg)
                    engine_to_use.load_model()
                    dispose_after = True
                except Exception:
                    logger.debug("Model override to %s failed, using default", model)

            try:
                msg, _stats = await asyncio.to_thread(
                    engine_to_use.chat, messages, max_tokens=4096, temperature=0.3
                )
                response_text = msg.get("content", "")
            finally:
                if dispose_after:
                    try:
                        engine_to_use.unload()
                    except Exception:
                        pass
        elif local_engine and local_engine.is_loaded:
            result = await asyncio.to_thread(
                local_engine.chat, messages, max_tokens=4096, temperature=0.3
            )
            if isinstance(result, tuple):
                msg, _stats = result
                response_text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            elif isinstance(result, dict):
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                response_text = str(result)
        else:
            yield {"event": "error", "message": "No inference engine available. Start in cloud or local mode first."}
            return

        if not response_text:
            yield {"event": "error", "message": "LLM returned empty response"}
            return

        yield {"event": "building", "message": "Parsing workflow JSON..."}

        # Extract and validate JSON
        workflow = extract_json_from_llm_response(response_text)
        if workflow is None:
            logger.error("Failed to extract JSON from LLM response: %s", response_text[:500])
            yield {"event": "error", "message": "LLM did not return valid JSON. Try rephrasing your request."}
            return

        # Ensure pinData exists
        if "pinData" not in workflow:
            workflow["pinData"] = {}

        # Validate
        validation_errors = validate_workflow(workflow)
        if validation_errors:
            logger.warning("Workflow validation errors: %s", validation_errors)
            yield {
                "event": "building",
                "message": f"Fixing {len(validation_errors)} validation issue(s)...",
            }
            # Try to auto-fix common issues
            workflow = _auto_fix_workflow(workflow)
            remaining = validate_workflow(workflow)
            if remaining:
                yield {
                    "event": "error",
                    "message": f"Generated workflow has issues: {'; '.join(remaining)}",
                }
                return

        yield {
            "event": "workflow_json",
            "workflow": workflow,
            "message": f"Generated workflow '{workflow.get('name', 'Untitled')}' with {len(workflow.get('nodes', []))} nodes",
        }

        # Check for unavailable nodes and warn
        try:
            from src.server.n8n_manager import validate_workflow_nodes
            missing_nodes = await validate_workflow_nodes(workflow)
            if missing_nodes:
                yield {
                    "event": "building",
                    "message": f"Warning: {len(missing_nodes)} node type(s) may not be installed: {', '.join(missing_nodes)}. You may need to install them in n8n.",
                }
        except Exception:
            pass

    except Exception:
        logger.exception("Error generating workflow via LLM")
        yield {"event": "error", "message": "Failed to generate workflow. Please try again."}


def _auto_fix_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Attempt to auto-fix common workflow issues and align with n8n format."""
    nodes = workflow.get("nodes", [])

    # Add missing IDs
    for node in nodes:
        if not node.get("id"):
            node["id"] = str(uuid.uuid4())

    # Add missing positions
    for i, node in enumerate(nodes):
        if "position" not in node:
            node["position"] = [250 + i * 250, 300]

    # Ensure position is a list of 2 ints
    for node in nodes:
        pos = node.get("position", [250, 300])
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            node["position"] = [int(pos[0]), int(pos[1])]

    # Add webhookId for webhook/chatTrigger nodes
    for node in nodes:
        ntype = node.get("type", "")
        if ("webhook" in ntype.lower() or "chatTrigger" in ntype.lower()) and "webhookId" not in node:
            node["webhookId"] = str(uuid.uuid4())

    # Add manual trigger if no trigger exists
    has_trigger = any(
        "trigger" in (n.get("type", "")).lower() or "Trigger" in n.get("name", "")
        for n in nodes
    )
    if not has_trigger and nodes:
        trigger = {
            "parameters": {},
            "id": str(uuid.uuid4()),
            "name": "Manual Trigger",
            "type": "n8n-nodes-base.manualWorkflowTrigger",
            "typeVersion": 1,
            "position": [0, 300],
        }
        nodes.insert(0, trigger)
        # Shift other nodes right
        for n in nodes[1:]:
            if "position" in n:
                n["position"][0] += 250
        # Connect trigger to first non-trigger node
        if len(nodes) > 1:
            connections = workflow.get("connections", {})
            connections["Manual Trigger"] = {
                "main": [[{"node": nodes[1]["name"], "type": "main", "index": 0}]]
            }

    # Ensure n8n-standard top-level fields
    if "connections" not in workflow:
        workflow["connections"] = {}
    if "pinData" not in workflow:
        workflow["pinData"] = {}
    if "settings" not in workflow:
        workflow["settings"] = {
            "executionOrder": "v1",
        }
    if "staticData" not in workflow:
        workflow["staticData"] = None
    if "meta" not in workflow:
        workflow["meta"] = {
            "templateCredsSetupCompleted": False,
        }

    return workflow


# ── n8n API integration ──────────────────────────────────────────────

async def n8n_import_workflow(
    workflow_json: dict[str, Any],
    n8n_base_url: str | None = None,
    n8n_api_key: str | None = None,
) -> dict[str, Any]:
    """Import a workflow into n8n via the REST API.

    Auth methods (in priority order):
      1. API key (N8N_API_KEY) → POST /api/v1/workflows
      2. Auto-provisioned session (n8n_setup) → POST /rest/workflows

    Returns the created workflow object from n8n (includes assigned ID).
    """
    import os
    from src.server.n8n_setup import ensure_n8n_ready, get_session_cookies

    base_url = (n8n_base_url or os.getenv("N8N_BACKEND_URL", "http://localhost:5678")).rstrip("/")
    api_key = n8n_api_key or os.getenv("N8N_API_KEY", "")

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        if api_key:
            # Method 1: Public API with explicit API key
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-N8N-API-KEY": api_key,
            }
            resp = await client.post(f"{base_url}/api/v1/workflows", headers=headers, json=workflow_json)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
        else:
            # Method 2: Auto-provisioned session auth (no config needed)
            if not await ensure_n8n_ready():
                raise RuntimeError("n8n is not available for workflow import")
            cookie_str = get_session_cookies()
            headers = {"Cookie": cookie_str} if cookie_str else {}
            resp = await client.post(
                f"{base_url}/rest/workflows",
                json=workflow_json,
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json().get("data", resp.json())

        logger.info("Imported workflow into n8n: id=%s name=%s", result.get("id"), result.get("name"))
        return result
