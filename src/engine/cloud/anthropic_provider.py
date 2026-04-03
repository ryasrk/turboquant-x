"""Anthropic Claude provider.

Claude uses a different API format than OpenAI — the Messages API.
This provider handles the translation between the common interface
and Anthropic's native format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.engine.cloud.provider import CloudConfig, CloudProvider, CloudResponse

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_API_VERSION = "2023-06-01"


class AnthropicProvider(CloudProvider):
    """Provider for Anthropic's Claude Messages API."""

    def __init__(self, config: CloudConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or _DEFAULT_BASE_URL
        self._model = config.model or _DEFAULT_MODEL
        self._headers = {
            "x-api-key": config.api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
            **config.extra_headers,
        }

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert OpenAI-style messages to Anthropic format.

        Anthropic requires the system prompt as a separate top-level field,
        not as a message with role=system.

        Returns:
            Tuple of (system_prompt, messages_without_system).
        """
        system_prompt = None
        converted = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                converted.append({"role": msg["role"], "content": msg["content"]})
        return system_prompt, converted

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_prompt, converted = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens or self._config.max_tokens,
            "stream": stream,
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Anthropic temperature must be 0.0–1.0
        temp = temperature if temperature is not None else self._config.temperature
        payload["temperature"] = min(max(temp, 0.0), 1.0)

        tp = top_p if top_p is not None else self._config.top_p
        payload["top_p"] = tp

        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        return payload

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> CloudResponse:
        payload = self._build_payload(
            messages, max_tokens, temperature, top_p, stream=False, **kwargs
        )

        with httpx.Client(timeout=self._config.timeout) as client:
            resp = client.post(
                f"{self._base_url}/v1/messages",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        text_parts = [
            block["text"]
            for block in content_blocks
            if block.get("type") == "text"
        ]
        content = "".join(text_parts)

        usage = data.get("usage", {})

        return CloudResponse(
            content=content,
            model=data.get("model", self._model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(
            messages, max_tokens, temperature, top_p, stream=True, **kwargs
        )

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/messages",
                headers=self._headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text

    def list_models(self) -> list[str]:
        return [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
