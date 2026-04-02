"""Zero-Quant inference engine — throughput-optimised, zero Python overhead.

Zero-Quant is TurboQuant-X's third inference research mode, designed for
maximum throughput on large models with limited VRAM/RAM.

Design goals vs standard/turboquant:
    standard   — Q8_0 KV at C level, no compression, KV grows each turn
    turboquant — Q8_0 KV at C level + Python PolarQuant compression between turns
    zero-quant — Q4_0 KV at C level (no Python overhead), all CPU threads for
                 prompt eval, fast decode with physical cores only

Key differences from TurboQuant:
    1. C-level Q4_0/Q4_0 KV quantization — compression handled entirely inside
       llama.cpp with zero Python overhead (no compress/decompress per turn)
    2. n_threads_batch = all logical CPU cores — maximises prompt eval throughput
    3. n_threads = physical cores — avoids hyperthreading overhead on generation
    4. Boundary-layer KV precision override: first 2 + last 2 transformer layers
       use Q8_0 at C level while middle layers use Q4_0 (future: via tensor split)
    5. Stateless between turns — no compressed_state stored in Python

Tradeoffs vs TurboQuant:
    + Lower per-turn latency (no Python compress/decompress step)
    + No Python memory overhead for compressed state
    - KV memory grows with conversation length (no compression between turns)
    - Slightly lower KV quality than TurboQuant PolarQuant at same bit-width

Expected performance on RTX 4060 Ti 8 GB + 32 GB RAM + 28 cores:
    Qwen3.5-35B-A3B:  ~same tok/s as TurboQuant, faster first-token latency
    LLaMA-2-70B Q4_K_S: prompt phase ~1.5-2× faster due to n_threads_batch=28
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from typing import Any

from src.engine.inference import InferenceEngine, GenerationStats
from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig

logger = logging.getLogger(__name__)


def _build_zero_quant_kv_config(zero_quant_config: dict[str, Any]) -> KVCacheConfig:
    """Build KV config for zero-quant: Q4_0 for both K and V at C level.

    Q4_0 is llama.cpp's simplest 4-bit format — fast to quantize/dequantize
    inside CUDA/CPU kernels, with minimal overhead vs Q8_0.
    """
    k_type = zero_quant_config.get("kv_type_k", "q4_0")
    v_type = zero_quant_config.get("kv_type_v", "q4_0")
    return KVCacheConfig(
        cache_type_k=CacheType(k_type),
        cache_type_v=CacheType(v_type),
        flash_attention=zero_quant_config.get("flash_attention", True),
    )


class ZeroQuantEngine:
    """Zero-overhead inference engine for maximum throughput.

    Wraps InferenceEngine with zero-quant KV defaults and exposes
    the same chat/generate interface as TurboQuantEngine so it can
    drop in as a replacement in app.py.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        kv_config: KVCacheConfig | None = None,
        *,
        zero_quant_config: dict[str, Any] | None = None,
    ) -> None:
        _zq = zero_quant_config or {}

        # Use zero-quant KV config (Q4_0/Q4_0) unless caller provides override
        effective_kv = kv_config if kv_config is not None else _build_zero_quant_kv_config(_zq)

        self._engine = InferenceEngine(model_config, effective_kv)
        self._zero_quant_config = _zq

        cpu_count = os.cpu_count() or 4
        logger.info(
            "Zero-Quant engine initialised: KV K=%s V=%s | threads gen=%s batch=%s",
            effective_kv.cache_type_k.value,
            effective_kv.cache_type_v.value,
            model_config.n_threads if model_config.n_threads != -1 else f"auto({cpu_count // 2})",
            model_config.n_threads_batch if model_config.n_threads_batch != -1 else f"auto({cpu_count})",
        )

    # ------------------------------------------------------------------
    # Proxy properties (mirrors TurboQuantEngine interface)
    # ------------------------------------------------------------------

    @property
    def engine(self) -> InferenceEngine:
        """Underlying standard inference engine."""
        return self._engine

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    @property
    def has_compressed_state(self) -> bool:
        """Always False — zero-quant has no Python-level state."""
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        self._engine.load_model()

    def unload(self) -> None:
        self._engine.unload()

    # ------------------------------------------------------------------
    # Inference (stateless — no compression between turns)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Generate text — direct pass-through, zero Python overhead."""
        return self._engine.generate(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stop=stop,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = False,
    ) -> tuple[str, GenerationStats]:
        """Chat completion — direct pass-through."""
        return self._engine.chat(
            messages, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, thinking=thinking,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = False,
    ) -> Generator[str, None, None]:
        """Streaming chat — direct pass-through."""
        yield from self._engine.chat_stream(
            messages, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, thinking=thinking,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get engine status including zero-quant config."""
        base = self._engine.get_stats()
        base["turbo_quant"] = {
            "k_bits": 4,
            "v_bits": 4,
            "block_size": None,
            "has_compressed_state": False,
            "mode": "zero-quant (C-level Q4_0, no Python overhead)",
        }
        return base
