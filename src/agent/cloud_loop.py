"""Cloud agent loop — tool calling via cloud LLM native function calling.

Unlike the local agent loop (which parses <tool_call> tags from text),
cloud providers support native tool/function calling via their APIs.
This loop uses the provider's native format for tool definitions and
responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from src.agent.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
MAX_TOOL_RESULT_CHARS = 16_000

_AGENT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "Use tools when you need real-time data, file operations, calculations, "
    "web searches, or any information you don't have. "
    "After receiving tool results, synthesize a clear answer. "
    "Format responses with markdown for readability."
)


def _build_openai_tools_schema(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Convert tool definitions to OpenAI function calling format."""
    return registry.get_definitions()


class CloudAgentLoop:
    """Agent loop using cloud LLM native function/tool calling.

    Works with OpenAI, Anthropic, and any provider that supports
    the OpenAI tool calling format via kwargs passthrough.

    Yields SSE-compatible event dicts:
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
    ):
        self.registry = registry
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars

    async def run(
        self,
        cloud_engine: Any,  # CloudEngine
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the cloud agent loop with native tool calling."""
        tools_used: list[str] = []
        seen_calls: set[str] = set()

        try:
            tools_schema = _build_openai_tools_schema(self.registry)

            if not tools_schema:
                # No tools — just chat
                msg, stats = await asyncio.to_thread(
                    cloud_engine.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                yield {"type": "content", "delta": msg["content"]}
                yield {
                    "type": "done",
                    "finish_reason": "completed",
                    "iterations": 1,
                    "tools_used": [],
                }
                return

            # Add system prompt if missing
            if not any(m.get("role") == "system" for m in messages):
                messages = [
                    {"role": "system", "content": _AGENT_SYSTEM_PROMPT}
                ] + messages

            for iteration in range(self.max_iterations):
                logger.debug(
                    "Cloud agent iteration %d/%d",
                    iteration + 1,
                    self.max_iterations,
                )

                # Call cloud LLM with tools — pass tools definition via kwargs
                # The provider's chat() passes extra kwargs through to the API
                msg, stats = await asyncio.to_thread(
                    cloud_engine.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools_schema,
                    tool_choice="auto",
                )

                content = msg.get("content", "")

                # Check if the response contains tool calls
                # Cloud providers return tool_calls in the message
                tool_calls = msg.get("tool_calls")

                # If no tool calls, try to parse them from text content
                # (some providers embed tool calls in content)
                if not tool_calls and content:
                    tool_calls = self._parse_tool_calls_from_text(content)

                if not tool_calls:
                    # Model answered directly
                    if content:
                        yield {"type": "content", "delta": content}
                    yield {
                        "type": "done",
                        "finish_reason": "completed",
                        "iterations": iteration + 1,
                        "tools_used": tools_used,
                    }
                    return

                # Emit any content before tool calls
                if content:
                    yield {"type": "content", "delta": content}

                # Execute tool calls
                has_repeat = False
                iteration_tool_messages: list[dict[str, Any]] = []

                for tc in tool_calls:
                    tc_id = tc.get("id", f"call_{iteration}_{tc['function']['name']}")
                    fn = tc["function"]
                    name = fn["name"]
                    raw_args = fn.get("arguments", "{}")

                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = raw_args

                    call_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
                    if call_key in seen_calls:
                        logger.warning("Repeated cloud tool call: %s", name)
                        has_repeat = True
                        break
                    seen_calls.add(call_key)

                    yield {
                        "type": "tool_call",
                        "id": tc_id,
                        "name": name,
                        "arguments": arguments,
                    }

                    result_str = await self.registry.execute(name, arguments)
                    if len(result_str) > self.max_tool_result_chars:
                        result_str = (
                            result_str[: self.max_tool_result_chars]
                            + "\n...[truncated]"
                        )

                    yield {
                        "type": "tool_result",
                        "id": tc_id,
                        "name": name,
                        "content": result_str,
                    }

                    if name not in tools_used:
                        tools_used.append(name)

                    # Build tool result message for next iteration
                    iteration_tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": name,
                        "content": result_str,
                    })

                if has_repeat:
                    # Force final answer
                    final_msg, _ = await asyncio.to_thread(
                        cloud_engine.chat,
                        messages=messages + iteration_tool_messages + [
                            {
                                "role": "user",
                                "content": "Based on the tool results, provide your final answer.",
                            }
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    yield {"type": "content", "delta": final_msg.get("content", "")}
                    yield {
                        "type": "done",
                        "finish_reason": "completed",
                        "iterations": iteration + 1,
                        "tools_used": tools_used,
                    }
                    return

                # Add assistant message with tool_calls + tool results to history
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content or "",
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
                messages.extend(iteration_tool_messages)

            # Exhausted max iterations
            yield {
                "type": "done",
                "finish_reason": "max_iterations",
                "iterations": self.max_iterations,
                "tools_used": tools_used,
            }

        except Exception as e:
            logger.exception("Cloud agent loop error")
            yield {"type": "error", "message": str(e)}

    @staticmethod
    def _parse_tool_calls_from_text(content: str) -> list[dict] | None:
        """Try to extract tool calls from text content.

        Some cloud models may embed tool calls in text format.
        """
        import re

        # Try <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        pattern = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
        )
        matches = pattern.findall(content)
        if not matches:
            return None

        tool_calls = []
        for i, raw_json in enumerate(matches):
            try:
                parsed = json.loads(raw_json)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                if name:
                    tool_calls.append({
                        "id": f"call_{i}_{name}",
                        "function": {
                            "name": name,
                            "arguments": (
                                json.dumps(arguments)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        },
                    })
            except json.JSONDecodeError:
                continue

        return tool_calls if tool_calls else None
