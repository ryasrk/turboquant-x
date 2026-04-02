"""LLM inference engine with TurboQuant KV cache support.

Wraps llama-cpp-python to provide model loading, text generation,
and chat completion with optional TurboQuant KV cache compression.

llama-cpp-python is an OPTIONAL dependency — the engine gracefully
handles its absence for testing and environments without GPU.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from src.engine.kv_cache import KVCacheConfig, to_llama_params
from src.engine.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationStats:
    """Statistics from a single generation call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time_s: float
    tokens_per_second: float


class InferenceEngine:
    """Thread-safe LLM inference engine with TurboQuant KV cache.

    Usage:
        engine = InferenceEngine(model_config, kv_config)
        engine.load_model()

        # Non-streaming
        response = engine.chat(messages=[{"role": "user", "content": "Hello"}])

        # Streaming
        for chunk in engine.chat_stream(messages=[{"role": "user", "content": "Hello"}]):
            print(chunk, end="", flush=True)

        engine.unload()

    Thread safety:
        All model access is serialized via a threading.Lock.
        Multiple threads can call chat/generate, but they will be queued.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        kv_config: KVCacheConfig | None = None,
    ) -> None:
        self._model_config = model_config
        self._kv_config = kv_config or KVCacheConfig()
        self._model: Any = None  # llama_cpp.Llama instance
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @property
    def kv_config(self) -> KVCacheConfig:
        return self._kv_config

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the LLM model into memory.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model is already loaded.
        """
        if self._loaded:
            raise RuntimeError("Model is already loaded. Call unload() first.")

        try:
            from llama_cpp import Llama  # noqa: F811
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for inference. "
                "Install with: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        # Build constructor kwargs
        kv_params = to_llama_params(self._kv_config)

        # Resolve n_threads: -1 → half of logical CPU cores (avoids HT overhead)
        import os as _os
        n_threads = self._model_config.n_threads
        if n_threads == -1:
            n_threads = max(1, (_os.cpu_count() or 4) // 2)

        # Resolve n_threads_batch: -1 → all logical cores (faster prompt eval)
        n_threads_batch = self._model_config.n_threads_batch
        if n_threads_batch == -1:
            n_threads_batch = _os.cpu_count() or n_threads

        logger.info(
            "Loading model %s (ctx=%d, gpu_layers=%d, threads=%d/%d, batch=%d, KV: K=%s V=%s)",
            self._model_config.model_name,
            self._model_config.n_ctx,
            self._model_config.n_gpu_layers,
            n_threads,
            n_threads_batch,
            self._model_config.n_batch,
            self._kv_config.cache_type_k.value,
            self._kv_config.cache_type_v.value,
        )

        start = time.monotonic()

        with self._lock:
            self._model = Llama(
                model_path=self._model_config.model_path,
                n_ctx=self._model_config.n_ctx,
                n_gpu_layers=self._model_config.n_gpu_layers,
                chat_format=self._model_config.chat_format,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                n_batch=self._model_config.n_batch,
                use_mlock=self._model_config.use_mlock,
                verbose=False,
                **kv_params,
            )
            self._loaded = True

        elapsed = time.monotonic() - start
        logger.info("Model loaded in %.1fs", elapsed)

    def unload(self) -> None:
        """Release model from memory."""
        with self._lock:
            self._model = None
            self._loaded = False
        logger.info("Model unloaded")

    def _ensure_loaded(self) -> None:
        """Raise if model not loaded."""
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Generate text from a raw prompt (non-streaming).

        Args:
            prompt: Raw text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            Tuple of (generated_text, stats).
        """
        self._ensure_loaded()

        start = time.monotonic()

        with self._lock:
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
            )

        elapsed = time.monotonic() - start

        text = result["choices"][0]["text"]
        usage = result.get("usage", {})

        stats = GenerationStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            generation_time_s=elapsed,
            tokens_per_second=usage.get("completion_tokens", 0) / max(elapsed, 0.001),
        )

        return text, stats

    def _apply_thinking(
        self, messages: list[dict[str, str]], thinking: bool
    ) -> list[dict[str, str]]:
        """Return messages with Qwen3 thinking control applied.

        When thinking is disabled, appends a partial assistant message with
        an empty <think></think> block. This prefills the model's response
        start, causing it to skip the chain-of-thought reasoning phase.

        Does not mutate the input list.
        """
        if thinking:
            return messages
        return [*messages, {"role": "assistant", "content": "<think>\n\n</think>\n\n"}]

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
    ) -> tuple[dict[str, str], GenerationStats]:
        """Chat completion (non-streaming).

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            thinking: Enable Qwen3 chain-of-thought thinking block.
                      When False, injects an empty <think></think> prefix
                      so the model skips the reasoning phase.

        Returns:
            Tuple of (response_message, stats).
            response_message: {"role": "assistant", "content": "..."}
        """
        self._ensure_loaded()

        effective_messages = self._apply_thinking(messages, thinking)
        start = time.monotonic()

        with self._lock:
            result = self._model.create_chat_completion(
                messages=effective_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        elapsed = time.monotonic() - start

        choice = result["choices"][0]
        message = choice["message"]
        usage = result.get("usage", {})

        stats = GenerationStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            generation_time_s=elapsed,
            tokens_per_second=usage.get("completion_tokens", 0) / max(elapsed, 0.001),
        )

        return {"role": message["role"], "content": message["content"]}, stats

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
    ) -> Generator[str, None, GenerationStats | None]:
        """Chat completion with streaming.

        Yields content chunks as strings.
        The generator's return value (accessible via StopIteration.value)
        contains the GenerationStats.

        Args:
            thinking: Enable Qwen3 chain-of-thought thinking block.

        Usage:
            gen = engine.chat_stream(messages)
            for chunk in gen:
                print(chunk, end="")
        """
        self._ensure_loaded()

        effective_messages = self._apply_thinking(messages, thinking)
        start = time.monotonic()
        completion_tokens = 0

        with self._lock:
            stream = self._model.create_chat_completion(
                messages=effective_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    completion_tokens += 1  # approximate
                    yield content

        elapsed = time.monotonic() - start

        return GenerationStats(
            prompt_tokens=0,  # not available in streaming
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            generation_time_s=elapsed,
            tokens_per_second=completion_tokens / max(elapsed, 0.001),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current engine status and stats.

        Returns dict with:
        - model_name, n_ctx, is_loaded
        - kv_config (K type, V type)
        """
        return {
            "model_name": self._model_config.model_name,
            "n_ctx": self._model_config.n_ctx,
            "is_loaded": self._loaded,
            "kv_cache_k": self._kv_config.cache_type_k.value,
            "kv_cache_v": self._kv_config.cache_type_v.value,
            "flash_attention": self._kv_config.flash_attention,
        }
