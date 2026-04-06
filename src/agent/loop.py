"""Agent execution loop — iterative LLM ↔ tool calling cycle."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from src.agent.approval import get_approval_store
from src.agent.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Dedicated logger for chain-of-thought output — written to file only,
# never streamed to the client.  Keeps debug reasoning available without
# burning extra tokens in the response.
_thought_logger = logging.getLogger("agent.thoughts")
_thought_logger.propagate = False  # don't bubble up to root/console

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Text-format thought/action scaffolding produced by the system prompt
_THOUGHT_TEXT_RE = re.compile(
    r"^Thought:\s*.*?(?=\nAction:|\n\n|$)", re.MULTILINE | re.DOTALL
)
_ACTION_TEXT_RE = re.compile(
    r"^Action:\s*(?:Final Answer:\s*)?", re.MULTILINE
)


def init_thought_log(path: str | None = None) -> None:
    """Attach a file handler to the thought logger.

    Called once at startup.  If *path* is ``None`` the default is
    ``logs/agent-thoughts.log`` relative to the working directory.
    """
    if _thought_logger.handlers:
        return  # already initialised
    path = path or os.path.join("logs", "agent-thoughts.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    )
    _thought_logger.addHandler(fh)
    _thought_logger.setLevel(logging.DEBUG)


def _strip_think_blocks(text: str, *, tag: str = "") -> str:
    """Remove thought scaffolding from model output, logging it.

    Handles both XML ``<think>…</think>`` blocks and text-format
    ``Thought: …`` / ``Action: Final Answer: …`` lines produced by
    the agent system prompt.

    *tag* is an optional label written to the log for traceability.
    """
    prefix = f"[{tag}] " if tag else ""

    # 1. XML think blocks
    for m in _THINK_RE.finditer(text):
        thought = m.group(1).strip()
        if thought:
            _thought_logger.debug("%s%s", prefix, thought)
    text = _THINK_RE.sub("", text)

    # 1b. Orphaned opening tag (truncated by max_tokens before </think>)
    if "<think>" in text and "</think>" not in text:
        text = text[:text.index("<think>")]
    # 1c. Orphaned closing tag (opening was in a previous chunk)
    if "</think>" in text and "<think>" not in text:
        text = text[text.index("</think>") + len("</think>"):]

    # 2. Text-format "Thought: …" lines
    for m in _THOUGHT_TEXT_RE.finditer(text):
        thought = m.group(0).strip()
        if thought:
            _thought_logger.debug("%s%s", prefix, thought)
    text = _THOUGHT_TEXT_RE.sub("", text)

    # 3. "Action: Final Answer:" prefix (keep the answer after it)
    text = _ACTION_TEXT_RE.sub("", text)

    return text.strip()

MAX_ITERATIONS = 20
MAX_TOOL_RESULT_CHARS = 16_000
# Reserve tokens for output (max_tokens) + tool schema overhead
_CTX_RESERVE_TOKENS = 1500


def _estimate_message_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _trim_messages_to_fit(
    messages: list[dict[str, Any]],
    max_ctx: int,
    reserve: int,
    tools_schema: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Drop oldest non-system messages until total fits within context budget.

    Keeps: system prompt (first message if role=system) + the most recent messages.
    Drops from the second message forward (oldest first).
    """
    budget = max_ctx - reserve  # tokens available for prompt
    if budget <= 0:
        return messages

    # Estimate tool schema overhead
    schema_tokens = 0
    if tools_schema:
        schema_tokens = _estimate_message_tokens(json.dumps(tools_schema))

    def total_tokens(msgs: list[dict]) -> int:
        t = schema_tokens
        for m in msgs:
            content = m.get("content", "") or ""
            t += _estimate_message_tokens(content) + 4  # role/separator overhead
            # tool_calls in assistant messages add tokens
            if m.get("tool_calls"):
                t += _estimate_message_tokens(json.dumps(m["tool_calls"], default=str))
        return t

    if total_tokens(messages) <= budget:
        return messages

    # Split: system (index 0 if system) + rest
    system_msgs = []
    rest = list(messages)
    if rest and rest[0].get("role") == "system":
        system_msgs = [rest.pop(0)]

    # Keep at least the last 2 messages (current user turn + latest context)
    while len(rest) > 2 and total_tokens(system_msgs + rest) > budget:
        dropped = rest.pop(0)
        logger.info("Context trim: dropped %s message (len=%d)",
                     dropped.get("role", "?"), len(dropped.get("content", "") or ""))

    trimmed = system_msgs + rest
    logger.info("Context trimmed: %d messages, ~%d tokens (budget=%d)",
                len(trimmed), total_tokens(trimmed), budget)
    return trimmed

_AGENT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant with access to the following tools:\n{tool_list}\n\n"
    "### Tool Discovery Workflow\n"
    "You have meta-tools that let you discover and use many specialized tools:\n"
    "1. **search_tools** — searches for TOOLS by capability keyword (NOT a web search). "
    "Use short keywords like 'web', 'git', 'time', 'file', 'memory'.\n"
    "2. **get_tool_detail** — shows full parameter schema for a discovered tool.\n"
    "3. **invoke_tool** — calls any discovered tool with arguments.\n\n"
    "**Example workflow** to search the web:\n"
    '  Step 1: search_tools({{"query": "web"}}) → finds "web_search" tool\n'
    '  Step 2: invoke_tool({{"tool_name": "web_search", "arguments": "{{\\"query\\": \\"your topic\\"}}" }}) → returns search results\n\n'
    "### Instructions\n"
    "- **ALWAYS use tools** for: current time/date, real-time data, live information, file operations, calculations, or web searches.\n"
    "- For general knowledge, conversation, or creative tasks, answer directly without tools.\n"
    "- Do NOT guess or hallucinate real-time information — use the appropriate tool instead.\n"
    "- After receiving tool results, synthesize a clear and complete answer.\n"
    "- If a tool returns an error, fix the arguments and try again.\n"
    "- Format responses with markdown (headings, lists, bold, code blocks) for readability.\n\n"
    "### Tool Call Format\n"
    "When you need to call a tool, output EXACTLY this format with ALL required arguments:\n"
    '<tool_call>{{"name": "TOOL_NAME", "arguments": {{"param1": "value1", "param2": "value2"}}}}</tool_call>\n\n'
    "Examples:\n"
    '<tool_call>{{"name": "current_time", "arguments": {{}}}}</tool_call>\n'
    '<tool_call>{{"name": "save_note", "arguments": {{"key": "user_name", "value": "Alex"}}}}</tool_call>\n'
)
# Regex to extract tool calls from content (various model formats)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?:</?function_call>)?\s*(\{.*?\})\s*(?:</?function_call>)?\s*</tool_call>",
    re.DOTALL,
)
# Sub-tag format: <tool_call> <tool_name>name</tool_name> <tool_args>{...}</tool_args> </tool_call>
_TOOL_CALL_SUBTAG_RE = re.compile(
    r"<tool_call>\s*<tool_name>\s*(\w+)\s*</tool_name>\s*<tool_args>\s*(.*?)\s*</tool_args>\s*</tool_call>",
    re.DOTALL,
)
# <tool_code> name(args) </tool_code>  format
_TOOL_CODE_RE = re.compile(
    r"<tool_code>\s*(\w+)\(([^)]*)\)\s*</tool_code>",
    re.DOTALL,
)
# Marker regex — find <tool_call> positions for raw_decode extraction
_TOOL_CALL_MARKER_RE = re.compile(r"<tool_call>\s*", re.DOTALL)
_FUNC_CALL_RE = re.compile(
    r"<function_call>\s*(\{.*?\})\s*</function_call>",
    re.DOTALL,
)
# chatml-function-calling format: "functions.tool_name:\n{...}" or just "functions.tool_name:"
_FUNCTIONS_RE = re.compile(
    r"functions\.(\w+):\s*(\{.*\})?\s*$",
    re.DOTALL,
)
# Text "Action: tool_name()" or "Action: tool_name(arg1, arg2)"  (system prompt format)
_ACTION_CALL_RE = re.compile(
    r"Action:\s*(\w+)\(([^)]*)\)",
    re.MULTILINE,
)
# JSON tool calls inside markdown code blocks: ```json\n{"name":"x","arguments":{...}}\n```
# Also matches ```python blocks since small LLMs sometimes use that tag.
_MD_JSON_BLOCK_RE = re.compile(
    r"```(?:json|python)?\s*\n?\s*(\{[^`]*\})\s*\n?\s*```",
    re.DOTALL,
)

_json_decoder = json.JSONDecoder()


def _extract_tool_calls_from_markers(content: str) -> list[dict] | None:
    """Extract tool calls using <tool_call> markers + json.raw_decode.

    Handles nested JSON objects (e.g. arguments with nested dicts) and
    tolerates models that use ``<tool_call>`` instead of ``</tool_call>``
    as the closing tag.
    """
    tool_calls: list[dict] = []
    seen_names: set[str] = set()
    for marker in _TOOL_CALL_MARKER_RE.finditer(content):
        remaining = content[marker.end():]
        # Skip if remaining doesn't start with '{'
        stripped = remaining.lstrip()
        if not stripped.startswith("{"):
            continue
        offset = len(remaining) - len(stripped)
        try:
            parsed, _ = _json_decoder.raw_decode(remaining, offset)
        except (json.JSONDecodeError, ValueError):
            continue
        name = parsed.get("name") or parsed.get("tool") or ""
        if not name or name in seen_names:
            continue  # skip hallucinated duplicate results
        seen_names.add(name)
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        tool_calls.append({
            "id": f"call_{len(tool_calls)}_{name}",
            "function": {"name": name, "arguments": arguments},
        })
    return tool_calls if tool_calls else None


def _parse_tool_calls_from_content(content: str) -> list[dict] | None:
    """Extract tool calls from text content (multiple model formats).

    Handles:
    - <tool_call>{"name": "x", "arguments": {}}</tool_call>  (Qwen3 native)
    - <tool_call>{"tool": "x", "arguments": {}}<tool_call>  (loose format)
    - <tool_call><tool_name>x</tool_name><tool_args>{}</tool_args></tool_call>  (sub-tag format)
    - <tool_code>name(args)</tool_code>  (code-style format)
    - <function_call>{"name": "x", "arguments": {}}</function_call>
    - functions.tool_name:\n{"key": "val"}  (chatml-function-calling)
    - functions.tool_name:  (no args — default to {})
    - Action: tool_name()  (system prompt text format)

    Returns a list of tool_call dicts or None if no tool calls found.
    """
    # Try sub-tag format: <tool_call><tool_name>x</tool_name><tool_args>{}</tool_args></tool_call>
    subtag_matches = _TOOL_CALL_SUBTAG_RE.findall(content)
    if subtag_matches:
        tool_calls = []
        for i, (name, raw_args) in enumerate(subtag_matches):
            args_str = raw_args.strip() or "{}"
            try:
                json.loads(args_str)  # validate
            except json.JSONDecodeError:
                args_str = "{}"
            tool_calls.append({
                "id": f"call_{i}_{name}",
                "function": {"name": name, "arguments": args_str},
            })
        return tool_calls if tool_calls else None

    # Try <tool_code>name(args)</tool_code> format
    code_matches = _TOOL_CODE_RE.findall(content)
    if code_matches:
        tool_calls = []
        for i, (name, raw_args) in enumerate(code_matches):
            args_dict: dict[str, Any] = {}
            raw_args = raw_args.strip()
            if raw_args:
                try:
                    args_dict = json.loads(raw_args)
                except json.JSONDecodeError:
                    for pair in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))', raw_args):
                        key = pair.group(1)
                        val = pair.group(2) if pair.group(2) is not None else pair.group(3)
                        args_dict[key] = val
            tool_calls.append({
                "id": f"call_{i}_{name}",
                "function": {"name": name, "arguments": json.dumps(args_dict)},
            })
        return tool_calls if tool_calls else None

    # Try strict XML formats
    matches = _TOOL_CALL_RE.findall(content) or _FUNC_CALL_RE.findall(content)
    if matches:
        tool_calls = []
        for i, raw_json in enumerate(matches):
            try:
                parsed = json.loads(raw_json)
                name = parsed.get("name") or parsed.get("tool") or ""
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                elif not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                if name:
                    tool_calls.append({
                        "id": f"call_{i}_{name}",
                        "function": {"name": name, "arguments": arguments},
                    })
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to parse tool call from content: %s", exc)
        if tool_calls:
            return tool_calls

    # Try <tool_call> markers with raw_decode (handles nested JSON, loose closing)
    marker_result = _extract_tool_calls_from_markers(content)
    if marker_result:
        return marker_result

    # Try chatml-function-calling format
    func_match = _FUNCTIONS_RE.search(content)
    if func_match:
        name = func_match.group(1)
        args_str = func_match.group(2) or "{}"
        try:
            json.loads(args_str)  # validate
        except json.JSONDecodeError:
            args_str = "{}"
        return [{
            "id": f"call_0_{name}",
            "function": {"name": name, "arguments": args_str},
        }]

    # Try text "Action: tool_name(...)" format (from system prompt)
    action_match = _ACTION_CALL_RE.search(content)
    if action_match:
        name = action_match.group(1)
        raw_args = action_match.group(2).strip()
        # Parse simple key=value args or JSON
        args_dict: dict[str, Any] = {}
        if raw_args:
            try:
                args_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                # Try key=value pairs: path="/foo", n=10
                for pair in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))', raw_args):
                    key = pair.group(1)
                    val = pair.group(2) if pair.group(2) is not None else pair.group(3)
                    args_dict[key] = val
        return [{
            "id": f"call_0_{name}",
            "function": {"name": name, "arguments": json.dumps(args_dict)},
        }]

    # Try markdown code blocks containing JSON tool call objects.
    # Must have BOTH "name"/"tool" AND "arguments" keys to distinguish
    # from workflow JSON payloads that also have a "name" field.
    md_matches = _MD_JSON_BLOCK_RE.findall(content)
    if md_matches:
        tool_calls = []
        for i, raw_json in enumerate(md_matches):
            try:
                parsed = json.loads(raw_json)
                if not isinstance(parsed, dict):
                    continue
                name = parsed.get("name") or parsed.get("tool") or ""
                # Require "arguments" key to confirm this is a tool call,
                # not a random JSON payload (e.g. workflow_json)
                if not name or "arguments" not in parsed:
                    continue
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                elif not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                tool_calls.append({
                    "id": f"call_{i}_{name}",
                    "function": {"name": name, "arguments": arguments},
                })
            except (json.JSONDecodeError, KeyError):
                pass
        if tool_calls:
            return tool_calls

    return None


def _strip_tool_call_tags(content: str) -> str:
    """Remove tool call XML/text tags from content for display."""
    content = _TOOL_CALL_SUBTAG_RE.sub("", content)
    content = _TOOL_CODE_RE.sub("", content)
    content = _TOOL_CALL_RE.sub("", content)
    content = _FUNC_CALL_RE.sub("", content)
    content = _FUNCTIONS_RE.sub("", content)
    # Also strip loose <tool_call>...JSON...<tool_call> (model using open tag as close)
    content = re.sub(r"<tool_call>.*?(?=<tool_call>|$)", "", content, flags=re.DOTALL)
    # Strip markdown JSON code blocks that look like tool calls (have "name" key)
    def _strip_md_tool_block(m: re.Match) -> str:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict) and ("name" in parsed or "tool" in parsed):
                return ""
        except (json.JSONDecodeError, KeyError):
            pass
        return m.group(0)
    content = _MD_JSON_BLOCK_RE.sub(_strip_md_tool_block, content)
    return content.strip()


def _build_system_prompt(registry: ToolRegistry) -> str:
    """Build the agent system prompt with an enumerated tool list."""
    tool_lines = []
    for defn in registry.get_definitions():
        fn = defn["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        if params:
            param_parts = []
            for pname, pinfo in params.items():
                req = "*" if pname in required else "?"
                desc = pinfo.get("description", "")
                param_parts.append(f"{pname}{req}: {desc}" if desc else f"{pname}{req}")
            params_str = "; ".join(param_parts)
        else:
            params_str = ""
        line = f"- **{fn['name']}**"
        if params_str:
            line += f" — params: {params_str}"
        line += f"\n  {fn.get('description', '')}"
        tool_lines.append(line)
    return _AGENT_SYSTEM_PROMPT_TEMPLATE.format(tool_list="\n".join(tool_lines))


class AgentLoop:
    """Runs iterative LLM ↔ tool execution cycle.

    Uses a two-phase approach:
    1. Phase 1 (FC handler): Call model with tools to get tool_calls
    2. Phase 2 (plain handler): Convert tool results to plain messages
       and call model again for the final text answer.

    This works around the chatml-function-calling handler not properly
    rendering tool results back into the prompt.

    Yields SSE-compatible event dicts during execution:
    - {"type": "tool_call", "id": str, "name": str, "arguments": dict}
    - {"type": "tool_result", "id": str, "name": str, "content": str}
    - {"type": "content", "delta": str}
    - {"type": "done", "finish_reason": str, "iterations": int, "tools_used": list}
    - {"type": "error", "message": str}
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
        max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
        content_only: bool = False,
    ):
        self.registry = registry
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.content_only = content_only

    async def run(
        self,
        engine: Any,  # InferenceEngine
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the agent loop. Yields event dicts."""
        tools_used: list[str] = []
        seen_calls: set[str] = set()
        tool_results_log: list[dict[str, str]] = []  # Plaintext log for final synthesis

        try:
            tools_schema = self.registry.get_definitions()

            if not tools_schema:
                result = await asyncio.to_thread(
                    self._call_model_plain, engine, messages,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                )
                content = result["choices"][0]["message"].get("content", "")
                content = _strip_think_blocks(content, tag="no-tools")
                model_fr = result["choices"][0].get("finish_reason", "stop")
                yield {"type": "content", "delta": content}
                yield {"type": "done", "finish_reason": "length" if model_fr == "length" else "completed", "iterations": 1, "tools_used": []}
                return

            # Inject agent system prompt if none exists
            if not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": _build_system_prompt(self.registry)}] + messages

            for iteration in range(self.max_iterations):
                logger.debug("Agent iteration %d/%d (content_only=%s)", iteration + 1, self.max_iterations, self.content_only)

                # Trim conversation history to fit context window
                max_ctx = getattr(getattr(engine, '_model_config', None), 'n_ctx', 8192)
                messages = _trim_messages_to_fit(
                    messages, max_ctx, _CTX_RESERVE_TOKENS + max_tokens,
                    tools_schema=tools_schema if not self.content_only else None,
                )

                tool_calls = None
                content = ""
                model_fr = "stop"

                if self.content_only:
                    # ── Content-only mode (local models) ──────────────────
                    # Use plain handler exclusively; parse <tool_call> tags
                    # from text.  Avoids chatml-function-calling handler
                    # which repeats the same call on every iteration for
                    # small models.
                    result = await asyncio.to_thread(
                        self._call_model_plain, engine, messages,
                        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    )
                    message = result["choices"][0]["message"]
                    content = message.get("content", "") or ""
                    model_fr = result["choices"][0].get("finish_reason", "stop")
                    logger.info("Iter %d — content_only, content_len: %d", iteration + 1, len(content))

                    parsed_tc = _parse_tool_calls_from_content(content)
                    if parsed_tc:
                        # Filter to known tools only
                        parsed_tc = [
                            tc for tc in parsed_tc
                            if self.registry.get(tc['function']['name']) is not None
                        ]
                    logger.info("Iter %d — parsed tool_calls: %s",
                                iteration + 1,
                                [(tc['function']['name']) for tc in parsed_tc] if parsed_tc else None)
                    if parsed_tc:  # non-empty after registry validation
                        tool_calls = parsed_tc
                        # Strip tool call tags and thoughts from content for display
                        remaining = _strip_tool_call_tags(content)
                        remaining = _strip_think_blocks(remaining, tag=f"iter-{iteration + 1}-co")
                        if remaining.strip():
                            yield {"type": "content", "delta": remaining}
                        # Build a pseudo message for history (plain text, no FC)
                        message = {"role": "assistant", "content": content}
                else:
                    # ── FC mode (default) ─────────────────────────────────
                    result = await asyncio.to_thread(
                        self._call_model_fc, engine, messages, tools_schema,
                        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    )

                    message = result["choices"][0]["message"]
                    tool_calls = message.get("tool_calls")
                    model_fr = result["choices"][0].get("finish_reason", "stop")
                    logger.info("Iter %d — FC tool_calls: %s, content_len: %d",
                                iteration + 1,
                                [(tc['function']['name']) for tc in tool_calls] if tool_calls else None,
                                len(message.get("content", "") or ""))

                    # If FC handler returned structured tool_calls, also strip any tool
                    # call tags from the content to prevent the model from seeing them
                    # in conversation history and re-issuing the same calls.
                    # Save raw content BEFORE cleaning for fallback parsing.
                    raw_content_before_clean = (message.get("content", "") or "")
                    if tool_calls:
                        if raw_content_before_clean:
                            cleaned = _strip_tool_call_tags(raw_content_before_clean)
                            cleaned = _strip_think_blocks(cleaned, tag=f"iter-{iteration + 1}-fc-cleanup")
                            message["content"] = cleaned.strip() or None

                        # Deduplicate tool_calls by (name, arguments)
                        seen_tc_keys: set[str] = set()
                        deduped: list[dict] = []
                        for tc in tool_calls:
                            key = f"{tc['function']['name']}:{tc['function']['arguments']}"
                            if key not in seen_tc_keys:
                                seen_tc_keys.add(key)
                                deduped.append(tc)
                            else:
                                logger.debug("Deduped tool_call: %s", key)
                        tool_calls = deduped
                        if not tool_calls:
                            tool_calls = None

                        # If ALL FC tool_calls are repeats (already in seen_calls),
                        # also check the RAW content for additional tool calls that the
                        # FC handler missed (small models often put extra calls in text)
                        if tool_calls:
                            all_fc_repeated = all(
                                f"{tc['function']['name']}:{tc['function']['arguments']}" in seen_calls
                                for tc in tool_calls
                            )
                            if all_fc_repeated and raw_content_before_clean:
                                extra_tc = _parse_tool_calls_from_content(raw_content_before_clean)
                                if extra_tc:
                                    # Filter out already-seen calls
                                    new_tc = [
                                        tc for tc in extra_tc
                                        if f"{tc['function']['name']}:{tc['function']['arguments']}" not in seen_calls
                                    ]
                                    if new_tc:
                                        tool_calls = new_tc
                                        logger.info("Recovered %d tool calls from content after FC repeats", len(new_tc))

                    # Qwen3 outputs tool calls as text tags — parse from content
                    if not tool_calls:
                        content = message.get("content", "") or ""
                        logger.debug("FC content (no tool_calls): %s", content[:500])
                        parsed_tc = _parse_tool_calls_from_content(content)
                        logger.info("Parsed tool_calls from content (%d chars): %s",
                                    len(content),
                                    [(tc['function']['name'], tc['function']['arguments'][:60]) for tc in parsed_tc] if parsed_tc else None)
                        if parsed_tc:
                            tool_calls = parsed_tc
                            # Strip tool call tags and thoughts from content
                            remaining = _strip_tool_call_tags(content)
                            remaining = _strip_think_blocks(remaining, tag=f"iter-{iteration + 1}-tc")
                            if remaining:
                                yield {"type": "content", "delta": remaining}

                if not tool_calls:
                    # Model decided to answer directly (or no more tool calls)
                    content = message.get("content", "") or ""
                    model_fr = result["choices"][0].get("finish_reason", "stop")
                    if not content.strip():
                        if tool_results_log:
                            # Empty content after tools — do Phase 2 synthesis
                            content, model_fr = await self._synthesize_answer(
                                engine, messages, tool_results_log,
                                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                            )
                        else:
                            # FC handler returned empty content with no tools — fallback to plain handler
                            logger.debug("FC handler empty content, falling back to plain handler")
                            plain_result = await asyncio.to_thread(
                                self._call_model_plain, engine, messages,
                                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                            )
                            content = plain_result["choices"][0]["message"].get("content", "")
                            model_fr = plain_result["choices"][0].get("finish_reason", "stop")

                            # The plain handler may produce "Action: tool()" text — try parsing
                            fallback_tc = _parse_tool_calls_from_content(content)
                            if fallback_tc:
                                tool_calls = fallback_tc
                                logger.debug("Parsed tool call from plain fallback: %s", fallback_tc)

                if not tool_calls:
                    content = _strip_think_blocks(content, tag=f"iter-{iteration + 1}-answer")

                    # Content-only nudge: if the model narrates intent to call
                    # a tool but didn't produce a <tool_call> tag, inject a
                    # nudge and continue the loop instead of returning.
                    if (
                        self.content_only
                        and tool_results_log
                        and iteration < self.max_iterations - 1
                        and any(kw in content.lower() for kw in (
                            "let's call", "let me call", "i will call", "call this tool",
                            "next tool", "let's use", "let me use",
                            "n8n_get_node_params", "n8n_create_workflow",
                            "n8n_search", "get_node_params",
                        ))
                    ):
                        logger.info("Iter %d — nudging model to produce <tool_call>", iteration + 1)
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                "You said you would call a tool but didn't output the <tool_call> tag. "
                                "Output ONLY the tool call now, no explanation:\n"
                                '<tool_call>{"name": "TOOL_NAME", "arguments": {...}}</tool_call>'
                            ),
                        })
                        continue

                    done_reason = "length" if model_fr == "length" else "completed"
                    yield {"type": "content", "delta": content}
                    yield {"type": "done", "finish_reason": done_reason, "iterations": iteration + 1, "tools_used": tools_used}
                    return

                # Execute tool calls
                has_repeat = False
                iteration_results: list[dict[str, str]] = []
                all_repeated = True  # Track if ALL calls in this iteration are repeats

                for tc in tool_calls:
                    tc_id = tc["id"]
                    fn = tc["function"]
                    name = fn["name"]
                    raw_args = fn["arguments"]

                    # Skip tool calls with unknown names (hallucinated tool names)
                    if self.registry.get(name) is None:
                        logger.warning("Skipping unknown tool: %s", name)
                        continue

                    arguments = json.loads(raw_args)

                    call_key = f"{name}:{raw_args}"
                    if call_key in seen_calls:
                        logger.warning("Skipping repeated tool call: %s", name)
                        has_repeat = True
                        continue  # skip this call, try others
                    seen_calls.add(call_key)
                    all_repeated = False

                    yield {"type": "tool_call", "id": tc_id, "name": name, "arguments": arguments}

                    # Check if tool requires user approval before execution
                    tool_obj = self.registry.get(name)
                    if tool_obj is not None and getattr(tool_obj, "requires_approval", False):
                        approval_id = f"approve_{uuid.uuid4().hex[:12]}"
                        # Run validation pipeline if available (TerminalTool)
                        validation = {}
                        if hasattr(tool_obj, "validate"):
                            cmd = arguments.get("command", "")
                            validation = tool_obj.validate(cmd)
                            # If hard-blocked by validation, skip approval flow
                            if validation.get("blocked"):
                                result_str = validation["blocked"]
                                yield {"type": "tool_result", "id": tc_id, "name": name, "content": result_str}
                                iteration_results.append({"name": name, "args": raw_args, "result": result_str})
                                if name not in tools_used:
                                    tools_used.append(name)
                                continue
                        yield {
                            "type": "tool_approval_request",
                            "id": tc_id,
                            "approval_id": approval_id,
                            "name": name,
                            "arguments": arguments,
                            "risk_level": validation.get("risk_level", "medium"),
                            "intent": validation.get("intent", "unknown"),
                            "warnings": validation.get("warnings", []),
                        }
                        # Wait for user to allow or deny
                        store = get_approval_store()
                        store.create(approval_id)
                        approved = await store.wait_for_approval(approval_id)
                        if not approved:
                            result_str = "Command denied by user."
                            yield {
                                "type": "tool_approval_result",
                                "id": tc_id,
                                "approval_id": approval_id,
                                "approved": False,
                            }
                            yield {"type": "tool_result", "id": tc_id, "name": name, "content": result_str}
                            iteration_results.append({"name": name, "args": raw_args, "result": result_str})
                            if name not in tools_used:
                                tools_used.append(name)
                            continue
                        yield {
                            "type": "tool_approval_result",
                            "id": tc_id,
                            "approval_id": approval_id,
                            "approved": True,
                        }

                    result_str = await self.registry.execute(name, arguments)
                    if len(result_str) > self.max_tool_result_chars:
                        result_str = result_str[:self.max_tool_result_chars] + "\n...[truncated]"

                    yield {"type": "tool_result", "id": tc_id, "name": name, "content": result_str}

                    iteration_results.append({"name": name, "args": raw_args, "result": result_str})
                    if name not in tools_used:
                        tools_used.append(name)

                tool_results_log.extend(iteration_results)

                # Track consecutive all-repeat iterations for stuck detection
                if all_repeated:
                    consecutive_repeats = getattr(self, '_consecutive_repeats', 0) + 1
                    self._consecutive_repeats = consecutive_repeats
                else:
                    self._consecutive_repeats = 0
                    consecutive_repeats = 0

                stuck = (consecutive_repeats >= 2) or (all_repeated and not iteration_results)
                if stuck or iteration == self.max_iterations - 1:
                    # Force final answer via Phase 2
                    content, synth_fr = await self._synthesize_answer(
                        engine, messages, tool_results_log,
                        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    )
                    # Check if synthesis output contains parseable tool calls
                    # (small models often produce explained plans with JSON blocks)
                    synth_tc = _parse_tool_calls_from_content(content)
                    if synth_tc and iteration < self.max_iterations - 1:
                        new_synth_tc = [
                            tc for tc in synth_tc
                            if (
                                self.registry.get(tc['function']['name']) is not None
                                and f"{tc['function']['name']}:{tc['function']['arguments']}" not in seen_calls
                            )
                        ]
                        if new_synth_tc:
                            logger.info("Extracted %d tool calls from synthesis, continuing loop", len(new_synth_tc))
                            tool_calls = new_synth_tc
                            # Re-enter execution by NOT returning; fall through to execution code
                            # We need to execute these tool calls, so continue the loop
                            self._consecutive_repeats = 0
                            # Execute the recovered tool calls inline
                            for tc in new_synth_tc:
                                tc_id = tc["id"]
                                fn = tc["function"]
                                name = fn["name"]
                                raw_args = fn["arguments"]
                                # Double-check tool exists in registry
                                if self.registry.get(name) is None:
                                    logger.warning("Skipping unknown tool from synthesis: %s", name)
                                    continue
                                arguments = json.loads(raw_args)
                                call_key = f"{name}:{raw_args}"
                                if call_key in seen_calls:
                                    continue
                                seen_calls.add(call_key)
                                yield {"type": "tool_call", "id": tc_id, "name": name, "arguments": arguments}
                                result_str = await self.registry.execute(name, arguments)
                                if len(result_str) > self.max_tool_result_chars:
                                    result_str = result_str[:self.max_tool_result_chars] + "\n...[truncated]"
                                yield {"type": "tool_result", "id": tc_id, "name": name, "content": result_str}
                                tool_results_log.append({"name": name, "args": raw_args, "result": result_str})
                                if name not in tools_used:
                                    tools_used.append(name)
                            # After executing recovered calls, do another synthesis
                            content, synth_fr = await self._synthesize_answer(
                                engine, messages, tool_results_log,
                                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                            )
                    content = _strip_think_blocks(content, tag="synthesis-forced")
                    done_reason = "length" if synth_fr == "length" else "completed"
                    yield {"type": "content", "delta": content}
                    yield {"type": "done", "finish_reason": done_reason, "iterations": iteration + 1, "tools_used": tools_used}
                    return

                # Build history for next iteration
                if self.content_only:
                    # Content-only mode: inject tool results as plain text
                    # so the model sees them in a format it can understand.
                    messages.append({"role": "assistant", "content": _strip_tool_call_tags(content)})
                    if iteration_results:
                        # Present tool results as a user message so the model
                        # clearly sees the boundary and can act on results.
                        parts = ["Tool results:\n"]
                        for tr in iteration_results:
                            parts.append(f"**{tr['name']}** returned:\n```\n{tr['result']}\n```\n")
                        parts.append(
                            "IMPORTANT: Output your next action using EXACTLY this format — "
                            "do NOT describe or explain what you will do:\n"
                            '<tool_call>{"name": "TOOL_NAME", "arguments": {"param": "value"}}</tool_call>\n'
                            "If you have all needed info, call n8n_create_workflow now."
                        )
                        messages.append({"role": "user", "content": "\n".join(parts)})
                else:
                    # FC mode: use tool_call_id messages
                    messages.append(message)
                    for tr in iteration_results:
                        tc_match = next(
                            (tc for tc in tool_calls if tc["function"]["name"] == tr["name"]),
                            None,
                        )
                        if tc_match:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_match["id"],
                                "name": tr["name"],
                                "content": tr["result"],
                            })

            # Exhausted max iterations
            yield {"type": "done", "finish_reason": "max_iterations", "iterations": self.max_iterations, "tools_used": tools_used}

        except Exception as e:
            logger.exception("Agent loop error")
            yield {"type": "error", "message": str(e)}

    async def _synthesize_answer(
        self,
        engine: Any,
        original_messages: list[dict[str, Any]],
        tool_results: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Phase 2: Build plain messages with tool results and get final answer.

        Converts tool call/result history into regular user/assistant messages
        so the standard chatml handler (which doesn't understand tool messages)
        can properly format them for the model.
        """
        # Extract original system + user messages (skip tool-related ones)
        plain_msgs = []
        for m in original_messages:
            if m.get("role") in ("system", "user"):
                plain_msgs.append(m)
            elif m.get("role") == "assistant" and not m.get("tool_calls"):
                plain_msgs.append(m)

        # Build tool results summary as an assistant message
        summary_parts = ["I used the following tools to gather information:\n"]
        for tr in tool_results:
            args_str = tr["args"]
            try:
                args_obj = json.loads(args_str)
                args_str = ", ".join(f"{k}={v!r}" for k, v in args_obj.items()) if args_obj else ""
            except (json.JSONDecodeError, AttributeError):
                pass
            summary_parts.append(f"**{tr['name']}**({args_str}):\n```\n{tr['result']}\n```\n")

        plain_msgs.append({"role": "assistant", "content": "\n".join(summary_parts)})
        plain_msgs.append({"role": "user", "content": "Based on the tool results above, please provide a complete answer to my original question."})

        # Trim to fit context window
        max_ctx = getattr(getattr(engine, '_model_config', None), 'n_ctx', 8192)
        plain_msgs = _trim_messages_to_fit(plain_msgs, max_ctx, _CTX_RESERVE_TOKENS + max_tokens)

        result = await asyncio.to_thread(
            self._call_model_plain, engine, plain_msgs,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        content = result["choices"][0]["message"].get("content", "")
        content = _strip_think_blocks(content, tag="synthesis")
        fr = result["choices"][0].get("finish_reason", "stop")
        return content, fr

    @staticmethod
    def _call_model_fc(
        engine: Any,
        messages: list[dict[str, Any]],
        tools_schema: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        """Call model with function-calling handler for tool selection.

        Uses chatml-function-calling handler which properly formats the
        prompt with tool schemas and parses tool_calls from the response.
        """
        from llama_cpp.llama_chat_format import get_chat_completion_handler

        engine._ensure_loaded()
        with engine._lock:
            original_handler = engine._model.chat_handler
            engine._model.chat_handler = get_chat_completion_handler("chatml-function-calling")
            try:
                result = engine._model.create_chat_completion(
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                logger.debug("FC response: %s", json.dumps(result, default=str)[:2000])
                return result
            except ValueError as exc:
                if "exceed context window" in str(exc):
                    logger.warning("Context overflow in FC call: %s — retrying with trimmed messages", exc)
                    # Aggressive trim: keep system + last 2 messages
                    trimmed = [m for m in messages if m.get("role") == "system"]
                    non_system = [m for m in messages if m.get("role") != "system"]
                    trimmed.extend(non_system[-2:] if len(non_system) > 2 else non_system)
                    result = engine._model.create_chat_completion(
                        messages=trimmed,
                        tools=tools_schema,
                        tool_choice="auto",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return result
                raise
            finally:
                engine._model.chat_handler = original_handler

    @staticmethod
    def _call_model_plain(
        engine: Any,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        """Call model with standard handler for text generation."""
        engine._ensure_loaded()
        with engine._lock:
            try:
                return engine._model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except ValueError as exc:
                if "exceed context window" in str(exc):
                    logger.warning("Context overflow in plain call: %s — retrying trimmed", exc)
                    trimmed = [m for m in messages if m.get("role") == "system"]
                    non_system = [m for m in messages if m.get("role") != "system"]
                    trimmed.extend(non_system[-2:] if len(non_system) > 2 else non_system)
                    return engine._model.create_chat_completion(
                        messages=trimmed,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                raise
