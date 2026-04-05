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

# Example 4: Cron/Schedule → Telegram reminder (common use case)
_EXAMPLE_CRON_TELEGRAM_REMINDER = {
    "name": "Daily Telegram Reminder",
    "nodes": [
        {
            "parameters": {},
            "id": str(uuid.uuid4()),
            "name": "Every day at 7pm",
            "type": "n8n-nodes-base.cron",
            "typeVersion": 1,
            "position": [250, 300],
        },
        {
            "parameters": {
                "chatId": "1086032366",
                "text": "Reminder: Drink coffee ☕",
                "additionalFields": {},
            },
            "id": str(uuid.uuid4()),
            "name": "Telegram",
            "type": "n8n-nodes-base.telegram",
            "typeVersion": 1.2,
            "position": [500, 300],
            "credentials": {
                "telegramApi": {"id": "1", "name": "Telegram Bot Token"},
            },
        },
    ],
    "connections": {
        "Every day at 7pm": {
            "main": [[{"node": "Telegram", "type": "main", "index": 0}]],
        },
    },
    "pinData": {},
}

# ── System prompt ────────────────────────────────────────────────────

# Generate TOON examples from the JSON examples
def _build_system_prompt() -> str:
    from src.server.workflow_toon import workflow_to_toon

    toon_examples = {
        "Webhook → Slack": (
            "When I receive a webhook, send the message to Slack #general",
            workflow_to_toon(_EXAMPLE_WEBHOOK_TO_SLACK),
        ),
        "Scheduled API + email": (
            "Every hour, check an API status endpoint and email me if it's not OK",
            workflow_to_toon(_EXAMPLE_SCHEDULE_HTTP_EMAIL),
        ),
        "AI Chat Agent": (
            "Create a chat agent with OpenAI and conversation memory",
            workflow_to_toon(_EXAMPLE_CHAT_AI_AGENT),
        ),
        "Telegram reminder": (
            "Create a bot telegram to reminder me everyday at 7pm to drink coffee",
            workflow_to_toon(_EXAMPLE_CRON_TELEGRAM_REMINDER),
        ),
    }

    prompt = """You are an expert n8n workflow designer. Generate workflows in compact TOON format (not JSON).

## TOON Workflow Format

```
name: Workflow Name

nodes[COUNT]{name,type,ver}:
  Node Name,n8n-nodes-base.nodeType,1.2
  Other Node,n8n-nodes-base.otherType,2

params:
  Node Name:
    paramKey: paramValue
    complexParam: {"key":"value"}

creds:
  Node Name:
    credType: 1

connections[COUNT]{src,dest}:
  Source Node,Target Node
  AI Model,Agent Node:ai_languageModel
```

## TOON Rules:
- nodes section: comma-separated rows (name,type,version)
- params section: indented key:value under node name
- creds section: credentialType: realCredentialID under node name (use IDs from "Your n8n Credentials" list)
- connections: src,dest for main type; src,dest:connectionType for AI sub-nodes
- Use n8n expressions: {{ $json.fieldName }}

## Common Node Types & Required Params

TRIGGERS: n8n-nodes-base.cron (daily/weekly), scheduleTrigger (interval: rule.interval), webhook (httpMethod,path), manualTrigger, @n8n/n8n-nodes-langchain.chatTrigger

MESSAGING:
- telegram: chatId (REQUIRED), text (REQUIRED). Creds: telegramApi
- slack: resource,channel,text. Creds: slackApi
- sendEmail: fromEmail,toEmail,subject,text. Creds: smtp

LOGIC: if (conditions), switch, merge, code (jsCode)
DATA: httpRequest (url,method), set (assignments), filter
AI: agent (promptType,text,options.systemMessage), lmChatOpenAi (creds:openAiApi), memoryBufferWindow

## RULES:
1. Every workflow MUST have one trigger node
2. Use REAL credential IDs from the "Your n8n Credentials" list. If no credentials are available, use descriptive placeholder like "TELEGRAM_BOT_CRED"
3. ALWAYS fill required params (chatId, text, url, etc.)
4. Return ONLY TOON — no markdown fences, no explanation
5. CREDENTIALS ARE CRITICAL — every node that talks to external services (Telegram, Slack, Gmail, HTTP with auth, OpenAI, etc.) MUST have a creds entry. Never omit credentials.
6. Match credential types exactly: telegramApi for Telegram, slackApi for Slack, openAiApi for OpenAI, httpHeaderAuth for authenticated HTTP

## Examples
"""

    for title, (user_msg, toon_text) in toon_examples.items():
        prompt += f"\n### {title}\nUser: \"{user_msg}\"\n{toon_text}\n"

    prompt += "\nNow generate a TOON workflow for the user's request. Return ONLY TOON."
    return prompt


# Lazy-initialize to avoid import cycle
_SYSTEM_PROMPT_CACHE: str | None = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is None:
        _SYSTEM_PROMPT_CACHE = _build_system_prompt()
    return _SYSTEM_PROMPT_CACHE


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

    # Validate connections reference existing nodes (both source AND dest)
    for source_name, conn_data in data.get("connections", {}).items():
        if source_name not in node_names:
            errors.append(f"Connection source '{source_name}' not found in nodes")
        # Check destination nodes exist
        if isinstance(conn_data, dict):
            for conn_type, outputs in conn_data.items():
                if isinstance(outputs, list):
                    for output_group in outputs:
                        if isinstance(output_group, list):
                            for link in output_group:
                                if isinstance(link, dict):
                                    dest = link.get("node", "")
                                    if dest and dest not in node_names:
                                        errors.append(f"Connection dest '{dest}' (from '{source_name}') not found in nodes")

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
    provider: str | None = None,
    use_cloud: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Generate an n8n workflow JSON from a natural language prompt.

    Args:
        prompt: Natural language description of the desired workflow.
        model: Optional cloud model override (e.g. 'deepseek-chat', 'gpt-4o').
        provider: Optional cloud provider override (e.g. 'nvidia', 'zhipu'). Use 'local' for local model.
        use_cloud: Try cloud engine first if available.

    Yields SSE-compatible dicts with event types:
      - thinking: status updates
      - building: intermediate progress
      - workflow_json: the generated workflow JSON (final result)
      - error: if something goes wrong

    Uses TurboQuant-X's own chat completions endpoint internally.
    """

    yield {"event": "thinking", "message": "Analyzing your automation request..."}

    # ── Smart search for relevant nodes (TF-IDF) ────────────────
    node_list_str = ""
    try:
        from src.server.smart_search import search_nodes
        relevant_nodes = await search_nodes(prompt, top_k=20)

        if relevant_nodes:
            # Build compact TOON-style node list with credential type hints
            node_entries = []
            seen = set()
            for n in relevant_nodes:
                name = n.get("name", "")
                if name in seen:
                    continue
                seen.add(name)
                cred_types = n.get("credentials", [])
                cred_str = ""
                if cred_types and isinstance(cred_types, list):
                    cred_names = [c.get("name", "") for c in cred_types if isinstance(c, dict) and c.get("name")]
                    if cred_names:
                        cred_str = f" [creds:{','.join(cred_names[:2])}]"
                node_entries.append({"name": name, "cred": cred_str})

            node_list_str = "Available n8n nodes (ranked by relevance):\n"
            node_list_str += "nodes[{}]{{type}}:\n".format(len(node_entries))
            node_list_str += "\n".join(f"  {e['name']}{e['cred']}" for e in node_entries)
    except Exception:
        logger.debug("Smart node search failed, falling back to no node list")

    # ── Fetch real credentials from n8n ─────────────────────────
    creds_context = ""
    try:
        from src.server.n8n_setup import n8n_api_call
        resp = await n8n_api_call("GET", "/rest/credentials")
        resp.raise_for_status()
        data = resp.json()
        creds = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(creds, dict):
            creds = creds.get("data", [])
        if creds:
            lines = ["\n\n## Your n8n Credentials (use these REAL IDs)"]
            lines.append(f"credentials[{len(creds)}]{{id,name,type}}:")
            for c in creds:
                lines.append(f"  {c.get('id', '?')},{c.get('name', '?')},{c.get('type', '?')}")
            lines.append("\nIMPORTANT: Use the REAL credential IDs above in the creds section. Match credential type to the node that needs it.")
            creds_context = "\n".join(lines)
    except Exception:
        logger.debug("Could not fetch n8n credentials for designer context")

    # ── Smart search for relevant templates ───────────────────────
    template_context = ""
    try:
        from src.server.smart_search import search_templates_smart
        from src.server.n8n_templates import get_template_by_id, strip_credentials
        matches = await search_templates_smart(prompt, top_k=2)
        if matches:
            yield {"event": "thinking", "message": f"Found {len(matches)} relevant template(s) for reference..."}
            ref_parts = []
            for m in matches:
                tpl_id = m.get("id")
                if not tpl_id:
                    continue
                tpl = get_template_by_id(tpl_id)
                if tpl and tpl.get("workflow"):
                    cleaned = strip_credentials(tpl["workflow"])
                    try:
                        from src.server.workflow_toon import workflow_to_toon
                        ref_parts.append(
                            f"### Template: {m.get('name', 'Untitled')}\n{workflow_to_toon(cleaned)}"
                        )
                    except Exception:
                        ref_parts.append(
                            f"### Template: {m.get('name', 'Untitled')}\n"
                            f"```json\n{json.dumps(cleaned, separators=(',', ':'))}\n```"
                        )
            if ref_parts:
                template_context = (
                    "\n\n## Reference Templates (adapt to match user request)\n\n"
                    + "\n\n".join(ref_parts)
                )
    except Exception:
        logger.debug("Template search for designer context failed")

    # Build system prompt with filtered node list, credentials, and template context
    system_content = _get_system_prompt()
    if node_list_str:
        system_content += f"\n\n## {node_list_str}"
    if creds_context:
        system_content += creds_context
    if template_context:
        system_content += template_context

    # Build messages for the LLM
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    model_label = model or provider or "default"
    yield {"event": "building", "message": f"Generating n8n workflow design (model: {model_label})..."}

    try:
        import asyncio
        from src.server.workspace_routes import _create_engine_for_provider

        engine_to_use, dispose_after = _create_engine_for_provider(provider, model)

        if engine_to_use is None:
            yield {"event": "error", "message": "No inference engine available. Configure a cloud provider or start with a local model."}
            return

        # Detect context window to avoid exceeding it
        ctx_size = 0
        try:
            if hasattr(engine_to_use, '_config') and hasattr(engine_to_use._config, 'provider'):
                # Cloud engine — assume large context
                ctx_size = 128_000
            elif hasattr(engine_to_use, 'model_config'):
                ctx_size = getattr(engine_to_use.model_config, 'n_ctx', 0)
            if not ctx_size or ctx_size < 0:
                ctx_size = 8192  # conservative default
        except Exception:
            ctx_size = 8192

        # Rough token estimate: ~4 chars per token
        est_tokens = len(system_content) // 4 + len(prompt) // 4 + 4096  # +4096 for response
        if est_tokens > ctx_size * 0.85:
            # Smart truncation priorities: keep creds (critical), trim templates first, then nodes
            yield {"event": "thinking", "message": "Optimizing prompt for model context window..."}
            budget = int(ctx_size * 4 * 0.80) - len(_get_system_prompt()) - len(prompt) - 16000  # chars budget

            system_content = _get_system_prompt()

            # Priority 1: Always keep credentials (they're small and critical)
            if creds_context:
                budget -= len(creds_context)
                system_content += creds_context

            # Priority 2: Node list (truncate if needed)
            if node_list_str and budget > 500:
                if len(node_list_str) > budget:
                    node_list_str = node_list_str[:max(500, budget)] + "\n... (truncated)"
                system_content += f"\n\n## {node_list_str}"
                budget -= len(node_list_str)

            # Priority 3: Templates only if enough space remains
            if template_context and budget > 1000:
                if len(template_context) > budget:
                    template_context = ""  # Drop entirely if too big
                else:
                    system_content += template_context

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

        # Ensure sufficient timeout for workflow generation
        try:
            from dataclasses import replace as dc_replace
            if hasattr(engine_to_use, '_config'):
                original_timeout = engine_to_use._config.timeout
                engine_to_use._config = dc_replace(engine_to_use._config, timeout=max(original_timeout, 300.0))
        except Exception:
            pass

        response_text = ""
        try:
            result = await asyncio.to_thread(
                engine_to_use.chat, messages, max_tokens=4096, temperature=0.3
            )
            if isinstance(result, tuple):
                msg, _stats = result
                response_text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            elif isinstance(result, dict):
                content = result.get("content", "")
                if not content:
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                response_text = content
            else:
                response_text = str(result)
        finally:
            if dispose_after:
                try:
                    engine_to_use.unload()
                except Exception:
                    pass

        if not response_text:
            yield {"event": "error", "message": "LLM returned empty response"}
            return

        yield {"event": "building", "message": "Parsing workflow response..."}

        # ── Parse + validate with one retry on failure ──────────
        async def _parse_response(text: str) -> dict[str, Any] | None:
            """Try TOON first, fall back to JSON."""
            wf = None
            try:
                from src.server.workflow_toon import extract_toon_from_response, toon_to_workflow
                toon_text = extract_toon_from_response(text)
                if toon_text:
                    wf = toon_to_workflow(toon_text)
                    logger.info("Parsed TOON response → %d nodes", len(wf.get("nodes", [])))
            except Exception as exc:
                logger.debug("TOON parsing failed, falling back to JSON: %s", exc)
            if wf is None:
                wf = extract_json_from_llm_response(text)
            return wf

        workflow = await _parse_response(response_text)

        # Retry once on parse failure or validation failure
        retry_reason = None
        if workflow is None:
            retry_reason = "Your response could not be parsed as TOON or JSON. Return ONLY valid TOON format — no markdown, no explanation."
        else:
            if "pinData" not in workflow:
                workflow["pinData"] = {}
            validation_errors = validate_workflow(workflow)
            if validation_errors:
                workflow = _auto_fix_workflow(workflow)
                remaining = validate_workflow(workflow)
                if remaining:
                    retry_reason = f"Workflow has validation errors: {'; '.join(remaining)}. Fix these issues and return corrected TOON."
                    workflow = None

        if retry_reason and not dispose_after:
            # Retry with error feedback
            yield {"event": "building", "message": "Retrying with corrections..."}
            retry_messages = messages + [
                {"role": "assistant", "content": response_text[:2000]},
                {"role": "user", "content": retry_reason},
            ]
            try:
                result2 = await asyncio.to_thread(
                    engine_to_use.chat, retry_messages, max_tokens=4096, temperature=0.2
                )
                if isinstance(result2, tuple):
                    msg2, _ = result2
                    retry_text = msg2.get("content", "") if isinstance(msg2, dict) else str(msg2)
                elif isinstance(result2, dict):
                    retry_text = result2.get("content", "") or result2.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    retry_text = str(result2)

                if retry_text:
                    workflow = await _parse_response(retry_text)
            except Exception as e:
                logger.debug("Retry failed: %s", e)

        if workflow is None:
            logger.error("Failed to extract workflow from LLM response: %s", response_text[:500])
            yield {"event": "error", "message": "LLM did not return valid workflow. Try rephrasing your request."}
            return

        # Ensure pinData exists
        if "pinData" not in workflow:
            workflow["pinData"] = {}

        # Final validation
        validation_errors = validate_workflow(workflow)
        if validation_errors:
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

        # Attempt test execution for early error detection
        wf_id = workflow.get("id")
        if wf_id:
            try:
                from src.server.n8n_manager import test_execute_workflow
                yield {"event": "building", "message": "Running test execution..."}
                test_result = await test_execute_workflow(str(wf_id), timeout=20.0)
                if test_result["success"]:
                    yield {"event": "building", "message": "Test execution passed."}
                elif test_result["error"] or test_result["node_errors"]:
                    err_parts = []
                    if test_result["error"]:
                        err_parts.append(f"Error: {test_result['error']}")
                    for ne in test_result["node_errors"][:3]:
                        err_parts.append(f"Node '{ne['node']}': {ne['message']}")
                    yield {
                        "event": "building",
                        "message": f"Test execution had issues: {'; '.join(err_parts)}. The workflow was imported but may need manual adjustment.",
                    }
            except Exception:
                logger.debug("Test execution skipped (workflow not yet imported or n8n unavailable)")

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
            "type": "n8n-nodes-base.manualTrigger",
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

    # Normalize the payload: only send fields n8n expects, and ensure
    # required defaults are present.  Raw LLM output may contain extra
    # keys (pinData, meta, etc.) that the internal REST API ignores or
    # that cause it to return an empty workflow.
    payload: dict[str, Any] = {
        "name": workflow_json.get("name", "Untitled Workflow"),
        "nodes": workflow_json.get("nodes", []),
        "connections": workflow_json.get("connections", {}),
        "active": workflow_json.get("active", False),
        "settings": workflow_json.get("settings", {"executionOrder": "v1"}),
    }
    if "staticData" in workflow_json:
        payload["staticData"] = workflow_json["staticData"]

    logger.info(
        "Importing workflow to n8n: name=%s nodes=%d",
        payload["name"],
        len(payload["nodes"]),
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        if api_key:
            # Method 1: Public API with explicit API key
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-N8N-API-KEY": api_key,
            }
            resp = await client.post(f"{base_url}/api/v1/workflows", headers=headers, json=payload)
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
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json().get("data", resp.json())

        created_nodes = len(result.get("nodes", []))
        logger.info(
            "Imported workflow into n8n: id=%s name=%s nodes=%d",
            result.get("id"), result.get("name"), created_nodes,
        )
        if created_nodes == 0 and len(payload["nodes"]) > 0:
            logger.warning(
                "n8n returned 0 nodes but we sent %d — workflow may not have imported correctly",
                len(payload["nodes"]),
            )
        return result
