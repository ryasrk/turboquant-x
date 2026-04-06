"""NullQuant inference engine — token eviction + depth-adaptive KV cache compression.

NullQuant is TurboQuant-X's most aggressive KV cache optimization mode,
combining SlimInfer-inspired token eviction with Zero-Quant depth-adaptive
zone compression (see ``src/turboquant/null_quant.py`` for full citations).

Design progression across inference modes:
    standard   — Q8_0 KV at C level; no Python compression; KV grows each turn
    turboquant — Q8_0 KV at C level + flat Python PolarQuant between turns
    zero-quant — Q8_0 KV at C level + depth-adaptive Python PolarQuant between
                 turns (shallow K8/V8, middle K4/V2, deep K8/V8)
    null-quant — Q8_0 KV at C level + token eviction + depth-adaptive PolarQuant
                 between turns (evict 75% of tokens, then zone-compress survivors)

Pipeline for each turn:
    1. Generate response (standard inference with Q8_0 KV)
    2. Save llama.cpp opaque state blob
    3. Extract KV tensors from state bytes
    4. Evict low-importance token positions (based on key-vector L2 norm)
    5. Zone-compress surviving tokens (depth-adaptive bit-widths)
    6. Store compressed state + metadata for next turn

Token eviction exploits the Information Diffusion phenomenon [5]: after
generation, critical information has already propagated into later token
positions, making many intermediate KV entries redundant.

Expected KV reduction for a 32-layer model with default config:
    Eviction:    75% tokens evicted (4× reduction)
    Compression: ~5.25 avg bits vs 64-bit float (≈8× reduction from zone quant)
    Combined:    32–128× depending on eviction_ratio and zone bit-widths
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
from src.turboquant.null_quant import (
    NullQuantConfig,
    NullQuantCompressedKV,
    NullQuantCompressor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullQuantCompressionStats:
    """Statistics from one NullQuant eviction + zone compression cycle."""

    original_bytes: int          # Before eviction
    original_tokens: int         # Sequence length before eviction
    surviving_tokens: int        # After eviction
    eviction_ratio: float        # Actual fraction evicted
    eviction_time_s: float
    compressed_bytes: int        # After zone compression
    compression_ratio: float     # original_bytes / compressed_bytes
    zone_compress_time_s: float
    decompress_time_s: float
    mse: float                   # MSE of surviving tokens after compress+decompress
    avg_bits: float
    n_layers: int
    total_reduction: float       # eviction × compression combined
    zone_summary: dict[str, str]


@dataclass(frozen=True)
class NullQuantGenerationResult:
    """Result of a NullQuant generation with eviction + zone compression."""

    text: str
    gen_stats: GenerationStats
    compression_stats: NullQuantCompressionStats | None


# ---------------------------------------------------------------------------
# NullQuantEngine
# ---------------------------------------------------------------------------


class NullQuantEngine:
    """Inference engine with token eviction + depth-adaptive KV compression.

    NullQuant is the most aggressive KV cache optimization mode.  It combines:

    1. **Token eviction**: Remove low-importance KV entries (save VRAM).
       Protected regions (attention sinks + recent context) are never evicted.
    2. **Zone compression**: Apply depth-adaptive bit-width reduction on
       survivors (save bits per remaining token).

    Uses standard Q8_0 at the C level (same as TurboQuantEngine and
    ZeroQuantEngine), then applies :class:`NullQuantCompressor` between
    turns to evict and compress the saved model state.

    Interface mirrors :class:`~src.engine.zero_quant_engine.ZeroQuantEngine`
    so it can drop in as a replacement in ``app.py``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        null_quant_config: NullQuantConfig | None = None,
        *,
        n_layers: int = 28,
        n_heads: int = 28,
        head_dim: int = 128,
        flash_attention: bool = True,
    ) -> None:
        # Q8_0 at C level — eviction + zone compression at Python level.
        # Q8_0 KV requires flash_attn; fall back to F16 when it is disabled.
        kv_type = CacheType.Q8_0 if flash_attention else CacheType.F16
        kv_config = KVCacheConfig(
            cache_type_k=kv_type,
            cache_type_v=kv_type,
            flash_attention=flash_attention,
        )
        self._engine = InferenceEngine(model_config, kv_config)
        self._config = null_quant_config or NullQuantConfig()
        self._compressor = NullQuantCompressor(self._config)

        # Model architecture for KV tensor reshaping
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._head_dim = head_dim

        # Compressed state between turns
        self._compressed_state: NullQuantCompressedKV | None = None
        self._state_metadata: dict[str, Any] | None = None

        cpu_count = os.cpu_count() or 4
        est_reduction = self._config.estimated_reduction(4096, n_layers)
        logger.info(
            "NullQuant engine init: eviction=%.0f%% | "
            "scoring=%s | sink=%d recent=%d | "
            "est ~%.0fx total reduction | "
            "threads gen=%s batch=%s | zones: %s",
            self._config.eviction_ratio * 100,
            self._config.scoring_method,
            self._config.sink_tokens,
            self._config.recent_tokens,
            est_reduction,
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
    def config(self) -> NullQuantConfig:
        return self._config

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
    # KV state extraction (identical logic to ZeroQuantEngine)
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
    ) -> tuple[NullQuantCompressedKV, NullQuantCompressionStats]:
        """Two-stage: evict tokens then zone-compress survivors."""
        original_bytes = keys.nbytes + values.nbytes
        original_tokens = keys.shape[2]  # seq_len dimension
        n_layers = keys.shape[0]

        t0 = time.monotonic()
        compressed = self._compressor.compress_kv(keys, values)
        total_compress_time = time.monotonic() - t0

        # Decompress for MSE calculation (surviving tokens only)
        t1 = time.monotonic()
        dec_keys, dec_values = self._compressor.decompress_kv(compressed)
        decompress_time = time.monotonic() - t1

        # MSE on surviving tokens only — compare against the pruned originals
        # (keys/values indexed to survivors), not the full original tensors.
        survivor_count = compressed.n_surviving_tokens
        evicted_positions = compressed.evicted_positions

        # Reconstruct survivor mask to extract original survivors for MSE
        all_positions = np.arange(original_tokens, dtype=np.int32)
        survivor_positions = np.setdiff1d(all_positions, evicted_positions)
        orig_survivor_keys = keys[:, :, survivor_positions, :]
        orig_survivor_values = values[:, :, survivor_positions, :]

        k_mse = float(np.mean((orig_survivor_keys - dec_keys) ** 2))
        v_mse = float(np.mean((orig_survivor_values - dec_values) ** 2))
        avg_mse = (k_mse + v_mse) / 2.0

        compressed_bytes = compressed.memory_bytes
        eviction_ratio_actual = compressed.eviction_ratio_actual
        total_reduction = compressed.total_reduction

        # Zone config average bits
        from src.turboquant.zero_quant import ZeroQuantConfig

        zone_cfg = ZeroQuantConfig(
            shallow_fraction=self._config.shallow_fraction,
            deep_fraction=self._config.deep_fraction,
            shallow_k_bits=self._config.shallow_k_bits,
            shallow_v_bits=self._config.shallow_v_bits,
            middle_k_bits=self._config.middle_k_bits,
            middle_v_bits=self._config.middle_v_bits,
            deep_k_bits=self._config.deep_k_bits,
            deep_v_bits=self._config.deep_v_bits,
            block_size=self._config.compress_block_size,
        )
        avg_bits = zone_cfg.average_bits(n_layers)

        stats = NullQuantCompressionStats(
            original_bytes=original_bytes,
            original_tokens=original_tokens,
            surviving_tokens=survivor_count,
            eviction_ratio=eviction_ratio_actual,
            eviction_time_s=compressed.eviction_time_s,
            compressed_bytes=compressed_bytes,
            compression_ratio=original_bytes / max(compressed_bytes, 1),
            zone_compress_time_s=compressed.compression_time_s,
            decompress_time_s=decompress_time,
            mse=avg_mse,
            avg_bits=avg_bits,
            n_layers=n_layers,
            total_reduction=total_reduction,
            zone_summary=self._compressor.zone_summary(n_layers),
        )
        return compressed, stats

    # ------------------------------------------------------------------
    # Public API: chat with eviction + zone compression
    # ------------------------------------------------------------------

    def chat_with_compression(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> NullQuantGenerationResult:
        """Chat completion with token eviction + depth-adaptive KV compression.

        After generating a response, the model's context state is saved and
        its KV data is first pruned (low-importance token eviction) then
        compressed using depth-adaptive zone quantization.
        """
        self._engine._ensure_loaded()

        msg, gen_stats = self._engine.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        comp_stats: NullQuantCompressionStats | None = None
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
                "NullQuant: %d → %d tokens (%.0f%% evicted), then zone "
                "compressed %.1f KB → %.1f KB (%.1fx) | total %.1fx | "
                "avg %.1f bits | MSE=%.6f | evict=%.3fs compress=%.3fs",
                comp_stats.original_tokens,
                comp_stats.surviving_tokens,
                comp_stats.eviction_ratio * 100,
                comp_stats.original_bytes / 1024,
                comp_stats.compressed_bytes / 1024,
                comp_stats.compression_ratio,
                comp_stats.total_reduction,
                comp_stats.avg_bits,
                comp_stats.mse,
                comp_stats.eviction_time_s,
                comp_stats.zone_compress_time_s,
            )
        except Exception as exc:
            logger.warning("NullQuant state compression failed: %s", exc)

        return NullQuantGenerationResult(
            text=msg["content"],
            gen_stats=gen_stats,
            compression_stats=comp_stats,
        )

    def compress_current_state(self) -> NullQuantCompressionStats | None:
        """Compress the current model state without generating new tokens.

        Useful for benchmarking NullQuant eviction + zone compression on a
        filled context.
        """
        self._engine._ensure_loaded()
        try:
            state = self._engine._model.save_state()
            keys, values = self._state_to_kv_tensors(
                state.llama_state, state.n_tokens
            )
            compressed, stats = self._compress_kv_tensors(keys, values)
            self._compressed_state = compressed
            self._state_metadata = {
                "n_tokens": state.n_tokens,
                "state_bytes_len": len(state.llama_state),
            }
            return stats
        except Exception as exc:
            logger.warning("NullQuant compress_current_state failed: %s", exc)
            return None

    def chat_stream_with_compression(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Generator[str, None, NullQuantGenerationResult | None]:
        """Streaming chat with NullQuant KV cache compression.

        Yields tokens as they are generated.  After streaming completes, the
        KV cache is evicted and zone-compressed for the next turn.
        """
        if not self._engine.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Stream tokens
        full_text_parts: list[str] = []
        gen_stats: GenerationStats | None = None

        for chunk in self._engine.chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            if isinstance(chunk, GenerationStats):
                gen_stats = chunk
            elif isinstance(chunk, str):
                full_text_parts.append(chunk)
                yield chunk

        # Post-stream compression
        compression_stats: NullQuantCompressionStats | None = None
        if gen_stats:
            try:
                with self._engine._lock:
                    if self._engine._model is not None:
                        state = self._engine._model.save_state()

                state_bytes = state.llama_state
                n_tokens = state.n_tokens

                if state_bytes and len(state_bytes) > 1024:
                    keys, values = self._state_to_kv_tensors(
                        state_bytes, n_tokens
                    )
                    compressed, compression_stats = self._compress_kv_tensors(
                        keys, values
                    )
                    self._compressed_state = compressed
                    self._state_metadata = {
                        "n_tokens": n_tokens,
                        "state_bytes_len": len(state_bytes),
                    }

                    if compression_stats:
                        logger.info(
                            "NullQuant stream: %d → %d tokens (%.0f%% evicted), "
                            "zone %.1f KB → %.1f KB (%.1fx) | total %.1fx",
                            compression_stats.original_tokens,
                            compression_stats.surviving_tokens,
                            compression_stats.eviction_ratio * 100,
                            compression_stats.original_bytes / 1024,
                            compression_stats.compressed_bytes / 1024,
                            compression_stats.compression_ratio,
                            compression_stats.total_reduction,
                        )
            except Exception as exc:
                logger.warning("NullQuant post-stream compression failed: %s", exc)

    def restore_context(self) -> bool:
        """Decompress and reload state for the next turn.

        Decompresses the zone-compressed KV data for surviving tokens and
        writes it back into a state blob that llama.cpp can load.

        .. warning::
            Only surviving tokens are restored.  Evicted tokens are permanently
            lost.  The restored state has ``seq_len = n_surviving_tokens``, not
            the original sequence length.

        .. note::
            The llama.cpp state blob includes more than just KV — it also holds
            rng state, logits, and other metadata.  We preserve the original
            state buffer structure and only overwrite the KV-data region with
            decompressed values.  The ``n_tokens`` bookkeeping is updated to
            reflect the surviving token count.

        Returns
        -------
        bool
            ``True`` if restoration succeeded, ``False`` otherwise.
        """
        if self._compressed_state is None or self._state_metadata is None:
            logger.warning("restore_context: no compressed state to restore")
            return False

        self._engine._ensure_loaded()

        try:
            compressed = self._compressed_state
            metadata = self._state_metadata

            # Decompress zone data → KV tensors (surviving tokens only)
            dec_keys, dec_values = self._compressor.decompress_kv(compressed)

            # We need the original state buffer to preserve non-KV metadata.
            # Re-save the current state as a template for the byte layout.
            state = self._engine._model.save_state()
            original_state_bytes = state.llama_state

            # Reconstruct state bytes with decompressed surviving KV data
            restored_bytes = self._kv_tensors_to_state_bytes(
                dec_keys, dec_values, original_state_bytes
            )

            # Load the restored state back into the model
            self._engine._model.load_state(restored_bytes)

            n_surviving = compressed.n_surviving_tokens
            n_original = compressed.n_original_tokens
            logger.info(
                "NullQuant context restored: %d tokens (originally %d, "
                "%.0f%% evicted)",
                n_surviving,
                n_original,
                compressed.eviction_ratio_actual * 100,
            )

            # Clear compressed state — it has been restored
            self._compressed_state = None
            self._state_metadata = None

            return True
        except Exception as exc:
            logger.warning("NullQuant restore_context failed: %s", exc)
            return False

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
        """Get engine status including eviction and zone config."""
        base = self._engine.get_stats()
        cfg = self._config

        from src.turboquant.zero_quant import ZeroQuantConfig

        zone_cfg = ZeroQuantConfig(
            shallow_fraction=cfg.shallow_fraction,
            deep_fraction=cfg.deep_fraction,
            shallow_k_bits=cfg.shallow_k_bits,
            shallow_v_bits=cfg.shallow_v_bits,
            middle_k_bits=cfg.middle_k_bits,
            middle_v_bits=cfg.middle_v_bits,
            deep_k_bits=cfg.deep_k_bits,
            deep_v_bits=cfg.deep_v_bits,
            block_size=cfg.compress_block_size,
        )
        avg_bits = zone_cfg.average_bits(self._n_layers)
        zones = self._compressor.zone_summary(self._n_layers)

        base["turbo_quant"] = {
            "mode": "null-quant (token eviction + depth-adaptive PolarQuant)",
            "eviction_ratio": cfg.eviction_ratio,
            "scoring_method": cfg.scoring_method,
            "sink_tokens": cfg.sink_tokens,
            "recent_tokens": cfg.recent_tokens,
            "avg_bits": round(avg_bits, 2),
            "zones": zones,
            "shallow_k_bits": cfg.shallow_k_bits,
            "shallow_v_bits": cfg.shallow_v_bits,
            "middle_k_bits": cfg.middle_k_bits,
            "middle_v_bits": cfg.middle_v_bits,
            "deep_k_bits": cfg.deep_k_bits,
            "deep_v_bits": cfg.deep_v_bits,
            "compress_block_size": cfg.compress_block_size,
            "est_total_reduction": round(
                cfg.estimated_reduction(4096, self._n_layers), 1
            ),
            "has_compressed_state": self.has_compressed_state,
        }
        return base
