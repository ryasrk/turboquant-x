"""OpenAI-compatible provider — base for most cloud LLM APIs.

Works with OpenAI, Moonshot (Kimi), Zhipu (GLM), DeepSeek, Groq,
Together AI, and any endpoint that speaks the OpenAI chat completions
protocol.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.engine.cloud.provider import CloudConfig, CloudProvider, CloudResponse

logger = logging.getLogger(__name__)

# Default base URLs per provider
_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
}

# Default models per provider
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "moonshot": "moonshot-v1-8k",
    "zhipu": "glm-4.5",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3-70b-chat-hf",
    "openrouter": "anthropic/claude-sonnet-4-20250514",
    "siliconflow": "Qwen/Qwen2.5-7B-Instruct",
}


class OpenAICompatibleProvider(CloudProvider):
    """Provider for any OpenAI-compatible chat completions API.

    Handles: OpenAI, Moonshot/Kimi, Zhipu/GLM, DeepSeek, Groq,
    Together AI, OpenRouter, SiliconFlow, and custom endpoints.
    """

    def __init__(self, config: CloudConfig) -> None:
        super().__init__(config)
        self._base_url = (
            config.base_url
            or _DEFAULT_BASE_URLS.get(config.provider, "")
        )
        if not self._base_url:
            raise ValueError(
                f"No base_url configured for provider '{config.provider}'. "
                "Set base_url in cloud provider config."
            )
        self._model = config.model or _DEFAULT_MODELS.get(config.provider, "")
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.extra_headers,
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "top_p": top_p if top_p is not None else self._config.top_p,
            "stream": stream,
        }
        # Pass through any extra kwargs (e.g. tools, response_format)
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
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        # Content: some models (GLM-4.5, DeepSeek-R1) put reasoning in
        # a separate field and may leave content empty when max_tokens
        # is exhausted by reasoning.  Fall back to reasoning_content.
        content = message.get("content", "") or ""
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"]

        # Extract native tool calls if present
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", "{}"),
                    },
                }
                for i, tc in enumerate(raw_tool_calls)
                if "function" in tc
            ]

        return CloudResponse(
            content=content,
            model=data.get("model", self._model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
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
                f"{self._base_url}/chat/completions",
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
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    # Some reasoning models (GLM-4.5, DeepSeek-R1) stream
                    # reasoning in a separate field.  Fall back to it when
                    # content is empty so the user sees *something*.
                    if not content:
                        content = delta.get("reasoning_content", "")
                    if content:
                        yield content

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as exc:
            logger.warning("Failed to list models for %s: %s", self._config.provider, exc)
            return []
