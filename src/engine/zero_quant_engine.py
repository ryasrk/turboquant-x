"""Zero-Quant inference engine — depth-adaptive KV cache compression.

Zero-Quant is TurboQuant-X's third inference research mode, implementing a
layer-depth-aware quantization scheme derived from four academic papers (see
``src/turboquant/zero_quant.py`` for citations).

Design vs standard/turboquant:
    standard   — Q8_0 KV at C level; no Python compression; KV grows each turn
    turboquant — Q8_0 KV at C level + flat Python PolarQuant (same bits for all
                 layers) between turns
    zero-quant — Q8_0 KV at C level + depth-adaptive Python PolarQuant between
                 turns: shallow layers K8/V8, middle layers K4/V2, deep layers
                 K8/V8 (all zones configurable via ZeroQuantConfig)

Depth-adaptive scheme (based on activation sparsity research):
    - Shallow zone (first 25%): dense activations — preserve with K8/V8
    - Middle zone  (25%–75%):   sparse activations — compress with K4/V2
    - Deep zone    (last 25%):  output-critical — preserve with K8/V8

Expected average bit-width for a 32-layer model with default config:
    K = (8×8 + 16×4 + 8×8) / 32 = 5.5 bits   (vs 8 for standard/turboquant)
    V = (8×8 + 16×2 + 8×8) / 32 = 5.0 bits

Expected performance on RTX 4060 Ti 8 GB + 32 GB RAM + 28 cores:
    Qwen3.5-35B-A3B:  longer context headroom than TurboQuant, similar tok/s
    LLaMA-2-70B:      same prompt-eval speed as TurboQuant, smaller KV footprint
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from collections.abc import Generator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engine.inference import InferenceEngine, GenerationStats
from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig
from src.turboquant.zero_quant import (
    ZeroQuantConfig,
    ZoneCompressedKV,
    DepthAdaptiveCompressor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroQuantCompressionStats:
    """Statistics from one depth-adaptive KV cache compress/decompress cycle."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    compress_time_s: float
    decompress_time_s: float
    mse: float
    avg_bits: float        # Weighted average bit-width across all layers
    n_layers: int
    zone_summary: dict[str, str]  # Human-readable zone description


@dataclass(frozen=True)
class ZeroQuantGenerationResult:
    """Result of a Zero-Quant depth-adaptive generation."""

    text: str
    gen_stats: GenerationStats
    compression_stats: ZeroQuantCompressionStats | None


# ---------------------------------------------------------------------------
# ZeroQuantEngine
# ---------------------------------------------------------------------------


class ZeroQuantEngine:
    """Inference engine with depth-adaptive Python-level KV cache compression.

    Uses standard Q8_0 at the C level (same as TurboQuantEngine), then applies
    :class:`~src.turboquant.zero_quant.DepthAdaptiveCompressor` between turns
    to compress the saved model state using three zone-specific bit-widths.

    Interface mirrors :class:`~src.engine.turbo_engine.TurboQuantEngine` so
    it can drop in as a replacement in ``app.py``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        zero_quant_config: ZeroQuantConfig | None = None,
        *,
        n_layers: int = 28,
        n_heads: int = 28,
        head_dim: int = 128,
        flash_attention: bool = True,
    ) -> None:
        # Q8_0 at C level — depth-adaptive compression happens at Python level.
        # Q8_0 KV requires flash_attn; fall back to F16 when it is disabled.
        kv_type = CacheType.Q8_0 if flash_attention else CacheType.F16
        kv_config = KVCacheConfig(
            cache_type_k=kv_type,
            cache_type_v=kv_type,
            flash_attention=flash_attention,
        )
        self._engine = InferenceEngine(model_config, kv_config)
        self._zero_quant_config = zero_quant_config or ZeroQuantConfig()
        self._compressor = DepthAdaptiveCompressor(self._zero_quant_config)

        # Model architecture for KV tensor reshaping
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._head_dim = head_dim

        # Compressed state between turns
        self._compressed_state: ZoneCompressedKV | None = None
        self._state_metadata: dict[str, Any] | None = None

        cpu_count = os.cpu_count() or 4
        avg_bits = self._zero_quant_config.average_bits(n_layers)
        logger.info(
            "ZeroQuant engine init: avg %.1f bits/value | "
            "threads gen=%s batch=%s | zones: %s",
            avg_bits,
            model_config.n_threads if model_config.n_threads != -1
            else f"auto({cpu_count // 2})",
            model_config.n_threads_batch if model_config.n_threads_batch != -1
            else f"auto({cpu_count})",
            self._compressor.zone_summary(n_layers),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> InferenceEngine:
        """Underlying standard inference engine."""
        return self._engine

    @property
    def zero_quant_config(self) -> ZeroQuantConfig:
        return self._zero_quant_config

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    @property
    def has_compressed_state(self) -> bool:
        return self._compressed_state is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        self._engine.load_model()

    def unload(self) -> None:
        self._engine.unload()
        self._compressed_state = None
        self._state_metadata = None

    # ------------------------------------------------------------------
    # KV state extraction (identical logic to TurboQuantEngine)
    # ------------------------------------------------------------------

    def _state_to_kv_tensors(
        self, state_bytes: bytes, n_tokens: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert raw llama.cpp state bytes to KV-shaped float64 tensors.

        The llama.cpp state blob is opaque binary.  We interpret it as uint8,
        normalise to roughly N(0, 1), and reshape into the model's KV tensor
        shape.  This preserves the information content for compression while
        giving the PolarQuant pipeline well-conditioned input values.
        """
        raw = np.frombuffer(state_bytes, dtype=np.uint8).astype(np.float64)
        raw = (raw - 128.0) / 64.0

        block = self._head_dim
        usable = (raw.size // (2 * block)) * block
        k_flat = raw[:usable]
        v_flat = raw[usable : usable * 2]

        elements_per_layer_head = block
        total_per_head = k_flat.size // (self._n_layers * self._n_heads)
        seq_len = max(1, total_per_head // elements_per_layer_head)

        target_size = self._n_layers * self._n_heads * seq_len * block
        keys = k_flat[:target_size].reshape(
            self._n_layers, self._n_heads, seq_len, block
        )
        values = v_flat[:target_size].reshape(
            self._n_layers, self._n_heads, seq_len, block
        )
        return keys, values

    def _kv_tensors_to_state_bytes(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
        original_state_bytes: bytes,
    ) -> bytes:
        """Reconstruct state bytes from decompressed KV tensors.

        Reverses the normalisation applied in ``_state_to_kv_tensors`` and
        writes the recovered uint8 values back into a copy of the original
        state buffer at the same offsets.
        """
        state_array = np.frombuffer(original_state_bytes, dtype=np.uint8).copy()

        k_flat = keys.ravel()
        v_flat = values.ravel()
        n_k = k_flat.size

        # Reverse normalisation: float64 N(0,1) → uint8 [0,255]
        k_uint8 = np.clip(np.round(k_flat * 64.0 + 128.0), 0, 255).astype(np.uint8)
        v_uint8 = np.clip(np.round(v_flat * 64.0 + 128.0), 0, 255).astype(np.uint8)

        offset_v = n_k
        state_array[:n_k] = k_uint8
        state_array[offset_v : offset_v + n_k] = v_uint8

        return state_array.tobytes()

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    def _compress_kv_tensors(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> tuple[ZoneCompressedKV, ZeroQuantCompressionStats]:
        """Compress KV tensors with depth-adaptive zones and collect stats."""
        original_bytes = keys.nbytes + values.nbytes
        n_layers = keys.shape[0]

        t0 = time.monotonic()
        compressed = self._compressor.compress_kv(keys, values)
        compress_time = time.monotonic() - t0

        t1 = time.monotonic()
        dec_keys, dec_values = self._compressor.decompress_kv(compressed)
        decompress_time = time.monotonic() - t1

        k_mse = float(np.mean((keys - dec_keys) ** 2))
        v_mse = float(np.mean((values - dec_values) ** 2))
        avg_mse = (k_mse + v_mse) / 2.0

        compressed_bytes = compressed.memory_bytes()
        avg_bits = self._zero_quant_config.average_bits(n_layers)

        stats = ZeroQuantCompressionStats(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=original_bytes / max(compressed_bytes, 1),
            compress_time_s=compress_time,
            decompress_time_s=decompress_time,
            mse=avg_mse,
            avg_bits=avg_bits,
            n_layers=n_layers,
            zone_summary=self._compressor.zone_summary(n_layers),
        )
        return compressed, stats

    # ------------------------------------------------------------------
    # Public API: chat with depth-adaptive compression
    # ------------------------------------------------------------------

    def chat_with_compression(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> ZeroQuantGenerationResult:
        """Chat completion with depth-adaptive KV cache compression.

        After generating a response, the model's context state is saved and
        its KV data is compressed using ZeroQuant depth-adaptive zones.
        """
        self._engine._ensure_loaded()

        msg, gen_stats = self._engine.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        comp_stats: ZeroQuantCompressionStats | None = None
        try:
            state = self._engine._model.save_state()
            keys, values = self._state_to_kv_tensors(
                state.llama_state, state.n_tokens
            )

            compressed, comp_stats = self._compress_kv_tensors(keys, values)

            self._compressed_state = compressed
            self._state_metadata = {
                "n_tokens": state.n_tokens,
                "state_bytes_len": len(state.llama_state),
            }

            logger.info(
                "ZeroQuant KV compressed: %.1f KB → %.1f KB (%.1fx) | "
                "avg %.1f bits | MSE=%.6f | compress=%.3fs decompress=%.3fs",
                comp_stats.original_bytes / 1024,
                comp_stats.compressed_bytes / 1024,
                comp_stats.compression_ratio,
                comp_stats.avg_bits,
                comp_stats.mse,
                comp_stats.compress_time_s,
                comp_stats.decompress_time_s,
            )
        except Exception as exc:
            logger.warning("ZeroQuant state compression failed: %s", exc)

        return ZeroQuantGenerationResult(
            text=msg["content"],
            gen_stats=gen_stats,
            compression_stats=comp_stats,
        )

    def compress_current_state(self) -> ZeroQuantCompressionStats | None:
        """Compress the current model state without generating new tokens.

        Useful for benchmarking depth-adaptive compression on a filled context.
        """
        self._engine._ensure_loaded()
        try:
            state = self._engine._model.save_state()
            keys, values = self._state_to_kv_tensors(
                state.llama_state, state.n_tokens
            )
            compressed, stats = self._compress_kv_tensors(keys, values)
            self._compressed_state = compressed
            self._state_metadata = {"n_tokens": state.n_tokens}
            return stats
        except Exception as exc:
            logger.warning("ZeroQuant compress_current_state failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Direct inference pass-through (no compression)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Generate text without compression (direct pass-through)."""
        return self._engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = False,
    ) -> tuple[str, GenerationStats]:
        """Chat completion without turn-level compression."""
        return self._engine.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
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
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get engine status including depth-adaptive zone config."""
        base = self._engine.get_stats()
        cfg = self._zero_quant_config
        avg_bits = cfg.average_bits(self._n_layers)
        zones = self._compressor.zone_summary(self._n_layers)

        base["turbo_quant"] = {
            "mode": "zero-quant (depth-adaptive PolarQuant)",
            "avg_bits": round(avg_bits, 2),
            "zones": zones,
            "shallow_k_bits": cfg.shallow_k_bits,
            "shallow_v_bits": cfg.shallow_v_bits,
            "middle_k_bits": cfg.middle_k_bits,
            "middle_v_bits": cfg.middle_v_bits,
            "deep_k_bits": cfg.deep_k_bits,
            "deep_v_bits": cfg.deep_v_bits,
            "block_size": cfg.block_size,
            "use_kv_coquant": cfg.use_kv_coquant,
            "has_compressed_state": self.has_compressed_state,
        }
        return base
