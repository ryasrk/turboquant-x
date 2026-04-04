"""Cloud inference engine — wraps cloud LLM providers with the same
interface pattern as InferenceEngine for seamless integration.

This engine does NOT load a local model. Instead, it forwards requests
to cloud LLM APIs (OpenAI, Anthropic, Moonshot/Kimi, Zhipu/GLM, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Generator
from typing import Any

from src.engine.cloud.provider import CloudConfig, CloudProvider, CloudResponse
from src.engine.cloud.registry import ProviderRegistry, create_provider
from src.engine.inference import GenerationStats

logger = logging.getLogger(__name__)

# Known context window sizes for cloud models (in tokens).
# Used to drive the context meter in the UI.
_CLOUD_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000, "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000, "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385, "o1": 200_000,
    # NVIDIA NIM
    "openai/gpt-oss-120b": 131_072, "nvidia/llama-3.1-nemotron-ultra-253b-v1": 131_072,
    "meta/llama-3.3-70b-instruct": 131_072, "deepseek-ai/deepseek-r1": 131_072,
    "moonshotai/kimi-k2.5": 131_072, "nvidia/nemotron-3-super-120b-a12b": 131_072,
    "z-ai/glm5": 131_072, "minimaxai/minimax-m2.5": 131_072,
    # Anthropic
    "claude-sonnet-4-20250514": 200_000, "claude-3-opus": 200_000,
    "claude-3-haiku": 200_000,
    # Zhipu / GLM
    "glm-4": 128_000, "glm-4-plus": 128_000,
    "glm-4.5": 128_000, "glm-4.5-flash": 128_000,
    "glm-4-flash": 128_000, "glm-4-long": 1_000_000,
    # DeepSeek
    "deepseek-chat": 128_000, "deepseek-reasoner": 128_000,
    # Moonshot / Kimi
    "moonshot-v1-8k": 8_000, "moonshot-v1-32k": 32_000,
    "moonshot-v1-128k": 128_000,
    # Groq
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
    # Fallback for unknown models
}

# Cloud models that include built-in chain-of-thought reasoning.
_CLOUD_REASONING_MODELS = {
    "glm-4.5", "glm-4.5-flash", "glm-4-plus",
    "deepseek-reasoner", "deepseek-chat",
    "o1", "o1-mini", "o1-preview",
    # NVIDIA NIM reasoning models
    "moonshotai/kimi-k2.5", "nvidia/nemotron-3-super-120b-a12b", "z-ai/glm5",
}


class CloudEngine:
    """Cloud LLM inference engine.

    Provides the same chat/generate interface as InferenceEngine but
    routes to cloud APIs instead of local GGUF models.

    Usage:
        engine = CloudEngine(cloud_config)
        engine.load_model()  # validates connection

        response = engine.chat(messages=[{"role": "user", "content": "Hello"}])

        for chunk in engine.chat_stream(messages=[...]):
            print(chunk, end="")

        engine.unload()
    """

    def __init__(self, config: CloudConfig) -> None:
        self._config = config
        self._provider: CloudProvider | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def provider_name(self) -> str:
        return self._config.provider

    @property
    def model_name(self) -> str:
        return self._config.model or "cloud-model"

    @property
    def supports_vision(self) -> bool:
        """Check if the current cloud model supports vision/image inputs."""
        _VISION_MODELS = {
            "gpt-4o", "gpt-4-turbo",
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
            "claude-3.5-sonnet", "claude-4-opus",
            "glm-4v",
        }
        model = self.model_name.lower()
        return any(v in model for v in _VISION_MODELS)

    @property
    def model_config(self) -> _CloudModelConfig:
        """Return a duck-typed model config for compatibility with routes."""
        return _CloudModelConfig(
            model_name=self.model_name,
            provider=self._config.provider,
            n_ctx=0,  # Not applicable for cloud
        )

    def load_model(self) -> None:
        """Initialize the cloud provider and validate the API key.

        Raises:
            RuntimeError: If already loaded.
            ValueError: If config is invalid.
        """
        if self._loaded:
            raise RuntimeError("Cloud engine is already loaded. Call unload() first.")

        self._provider = create_provider(self._config)
        ProviderRegistry.register(self._config.provider, self._provider)
        self._loaded = True

        logger.info(
            "Cloud engine ready: provider=%s, model=%s, base_url=%s",
            self._config.provider,
            self._config.model,
            self._config.base_url or "(default)",
        )

    def unload(self) -> None:
        """Release the cloud provider."""
        self._provider = None
        self._loaded = False
        logger.info("Cloud engine unloaded")

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._provider is None:
            raise RuntimeError("Cloud engine not loaded. Call load_model() first.")

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], GenerationStats]:
        """Chat completion via cloud provider.

        Returns:
            Tuple of (response_message, stats).
        """
        self._ensure_loaded()
        assert self._provider is not None

        start = time.monotonic()

        response = self._provider.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        elapsed = time.monotonic() - start

        stats = GenerationStats(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            generation_time_s=elapsed,
            tokens_per_second=response.completion_tokens / max(elapsed, 0.001),
            finish_reason=response.finish_reason,
        )

        msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            msg["tool_calls"] = response.tool_calls
        return msg, stats

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
        **kwargs: Any,
    ) -> Generator[str, None, GenerationStats | None]:
        """Streaming chat via cloud provider.

        Yields content chunks as strings. Uses asyncio internally
        to bridge the async stream from cloud providers.
        """
        self._ensure_loaded()
        assert self._provider is not None

        start = time.monotonic()
        completion_tokens = 0
        finish_reason = "stop"

        # Bridge async streaming to sync generator
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already inside an async context — use a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    chunks = pool.submit(
                        self._collect_stream_sync,
                        messages, max_tokens, temperature, top_p, kwargs,
                    ).result()
                    for chunk in chunks:
                        completion_tokens += 1
                        yield chunk
            else:
                async_gen = self._provider.chat_stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        completion_tokens += 1
                        yield chunk
                    except StopAsyncIteration:
                        break
        except RuntimeError:
            # No event loop — create one
            chunks = asyncio.run(
                self._collect_stream(
                    messages, max_tokens, temperature, top_p, kwargs
                )
            )
            for chunk in chunks:
                completion_tokens += 1
                yield chunk

        elapsed = time.monotonic() - start

        return GenerationStats(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            generation_time_s=elapsed,
            tokens_per_second=completion_tokens / max(elapsed, 0.001),
            finish_reason=finish_reason,
        )

    def _collect_stream_sync(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        kwargs: dict,
    ) -> list[str]:
        """Collect stream chunks in a new event loop (thread-safe)."""
        return asyncio.run(
            self._collect_stream(messages, max_tokens, temperature, top_p, kwargs)
        )

    async def _collect_stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        kwargs: dict,
    ) -> list[str]:
        """Collect all stream chunks into a list."""
        assert self._provider is not None
        chunks: list[str] = []
        async for chunk in self._provider.chat_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        ):
            chunks.append(chunk)
        return chunks

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Text generation from a raw prompt (non-streaming)."""
        messages = [{"role": "user", "content": prompt}]
        response_msg, stats = self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response_msg["content"], stats

    @property
    def context_window(self) -> int:
        """Best-effort context window size for the current cloud model."""
        model = self.model_name.lower()
        # Exact match first
        if model in _CLOUD_CONTEXT_WINDOWS:
            return _CLOUD_CONTEXT_WINDOWS[model]
        # Prefix match (e.g. 'gpt-4o-2024-08-06' → gpt-4o)
        for key, val in _CLOUD_CONTEXT_WINDOWS.items():
            if model.startswith(key):
                return val
        return 128_000  # safe default for modern models

    @property
    def supports_reasoning(self) -> bool:
        """Whether this cloud model has native chain-of-thought reasoning."""
        model = self.model_name.lower()
        return any(model.startswith(r) for r in _CLOUD_REASONING_MODELS)

    def get_stats(self) -> dict[str, Any]:
        """Get current engine status."""
        ctx = self.context_window
        return {
            "model_name": self.model_name,
            "provider": self._config.provider,
            "is_loaded": self._loaded,
            "mode": "cloud",
            "base_url": self._config.base_url or "(default)",
            "n_ctx": ctx,
            "kv_cache_k": "n/a",
            "kv_cache_v": "n/a",
            "flash_attention": False,
            "context_max": ctx,
            "context_used": 0,
            "supports_reasoning": self.supports_reasoning,
        }

    def list_models(self) -> list[str]:
        """List available models from the cloud provider."""
        if self._provider is None:
            return []
        return self._provider.list_models()

    def switch_model(self, model: str) -> None:
        """Switch to a different cloud model at runtime."""
        from dataclasses import replace
        self._config = replace(self._config, model=model)
        if self._provider is not None:
            # Recreate provider with new config
            self._provider = create_provider(self._config)
            logger.info("Switched cloud model to: %s", model)


class _CloudModelConfig:
    """Duck-typed model config for compatibility with routes.py."""

    def __init__(self, model_name: str, provider: str, n_ctx: int) -> None:
        self.model_name = model_name
        self.model_path = f"cloud://{provider}/{model_name}"
        self.n_ctx = n_ctx
