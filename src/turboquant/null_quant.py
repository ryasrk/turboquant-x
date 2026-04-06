"""NullQuant — two-stage KV cache compression: Token Eviction → Zone Compression.

Implements the NullQuant compression pipeline that combines aggressive token
pruning (inspired by SlimInfer) with depth-adaptive zone compression
(from Zero-Quant):

    Stage 1 — Token Eviction:
        Score each token position by its KV importance (L2 norm of key
        vectors across layers).  Evict low-importance positions while
        always preserving "attention sink" tokens (first N) and recent
        context (last N).  This exploits the "Information Diffusion"
        phenomenon: critical token information propagates through the
        sequence, making many intermediate positions redundant after
        generation completes.

    Stage 2 — Zone Compression:
        Apply Zero-Quant depth-adaptive quantization (shallow/middle/deep
        zones at different K/V bit-widths) to the surviving token subset.

Combined effect example:
    eviction_ratio=0.75 (keep 25%) × zone compression ≈ 8×
    → Total KV reduction = 4× × 8× = 32×

References
----------
[1] "Exploring Activation Patterns..." (Peking Univ, 2024, arXiv:2405.17799):
    Shallow layers activate densely; deep layers sparsely.
[2] "TurboSparse" (SJTU, 2024, arXiv:2406.05955): dReLU achieves 97% FFN
    sparsity in MoE models, 90% in dense models.
[3] "LUT-LLM" (UCLA/Microsoft, 2025, arXiv:2511.06174): Activation-weight
    co-quantization captures cross-tensor correlation.
[4] "LLM Inference Hardware Challenges" (Google, 2026, arXiv:2601.05047):
    Memory bandwidth is the primary bottleneck; KV cache size reduction
    directly improves throughput.
[5] "SlimInfer: Accelerating LLM Inference via Token Pruning" (AAAI 2026,
    arXiv:2508.06447): Token importance decays after generation due to
    Information Diffusion — aggressively pruning the KV cache between turns
    preserves quality while dramatically reducing memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.turboquant.compressor import CompressedKV, QuantConfig, TurboQuantCompressor
from src.turboquant.zero_quant import (
    ZeroQuantConfig,
    ZoneCompressedKV,
    DepthAdaptiveCompressor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_BITS: frozenset[int] = frozenset({2, 3, 4, 8})
_VALID_SCORING_METHODS: frozenset[str] = frozenset(
    {"l2_norm", "random", "uniform_stride"}
)


# ---------------------------------------------------------------------------
# NullQuantConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullQuantConfig:
    """Configuration for NullQuant two-stage KV cache compression.

    Stage 1 — Token Eviction
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Controls which token positions are evicted from the KV cache based on
    importance scoring.  The ``eviction_ratio`` determines the fraction of
    tokens to remove; ``sink_tokens`` and ``recent_tokens`` define protected
    regions that are never evicted (attention sinks [5] and recent context).

    Stage 2 — Zone Compression
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    After eviction, surviving tokens are compressed with depth-adaptive
    zone quantization (same scheme as :class:`ZeroQuantConfig`).

    Attributes
    ----------
    eviction_ratio:
        Fraction of tokens to evict.  0.75 means keep 25% of tokens.
    sink_tokens:
        Always keep the first N token positions (attention sinks).
    recent_tokens:
        Always keep the last N token positions (recent context window).
    scoring_method:
        Token importance scoring strategy:
        - ``"l2_norm"``: average L2 norm of key vectors across layers [5].
        - ``"random"``: random baseline for ablation studies.
        - ``"uniform_stride"``: keep every Nth token (deterministic baseline).
    block_size:
        Score and evict in blocks of this size (matches SlimInfer [5]).
    shallow_fraction:
        Fraction of layers in the shallow zone (high-precision).
    deep_fraction:
        Fraction of layers in the deep zone (high-precision).
    shallow_k_bits / shallow_v_bits:
        Key / value bit-width for the shallow zone.
    middle_k_bits / middle_v_bits:
        Key / value bit-width for the middle (sparse) zone.
    deep_k_bits / deep_v_bits:
        Key / value bit-width for the deep zone.
    compress_block_size:
        PolarQuant block size for zone compression.
    """

    # -- Token eviction settings -------------------------------------------
    eviction_ratio: float = 0.75
    sink_tokens: int = 256
    recent_tokens: int = 256
    scoring_method: str = "l2_norm"
    block_size: int = 64

    # -- Zone compression settings -----------------------------------------
    shallow_fraction: float = 0.25
    deep_fraction: float = 0.25
    shallow_k_bits: int = 8
    shallow_v_bits: int = 8
    middle_k_bits: int = 4
    middle_v_bits: int = 2
    deep_k_bits: int = 8
    deep_v_bits: int = 8
    compress_block_size: int = 128

    def __post_init__(self) -> None:
        if not (0.0 < self.eviction_ratio < 1.0):
            raise ValueError(
                f"eviction_ratio must be in (0, 1), got {self.eviction_ratio}"
            )
        if self.sink_tokens < 0:
            raise ValueError(
                f"sink_tokens must be >= 0, got {self.sink_tokens}"
            )
        if self.recent_tokens < 0:
            raise ValueError(
                f"recent_tokens must be >= 0, got {self.recent_tokens}"
            )
        if self.scoring_method not in _VALID_SCORING_METHODS:
            raise ValueError(
                f"scoring_method must be one of {sorted(_VALID_SCORING_METHODS)}, "
                f"got {self.scoring_method!r}"
            )
        for name, bits in [
            ("shallow_k_bits", self.shallow_k_bits),
            ("shallow_v_bits", self.shallow_v_bits),
            ("middle_k_bits", self.middle_k_bits),
            ("middle_v_bits", self.middle_v_bits),
            ("deep_k_bits", self.deep_k_bits),
            ("deep_v_bits", self.deep_v_bits),
        ]:
            if bits not in _VALID_BITS:
                raise ValueError(
                    f"{name} must be one of {sorted(_VALID_BITS)}, got {bits}"
                )
        if self.compress_block_size < 2 or (
            self.compress_block_size & (self.compress_block_size - 1)
        ) != 0:
            raise ValueError(
                f"compress_block_size must be a power of 2 >= 2, "
                f"got {self.compress_block_size}"
            )

    def estimated_reduction(self, n_tokens: int, n_layers: int) -> float:
        """Estimate total KV cache reduction factor from both stages.

        Parameters
        ----------
        n_tokens:
            Number of tokens in the original KV cache.
        n_layers:
            Number of transformer layers in the model.

        Returns
        -------
        float
            Combined reduction factor.  For example, if eviction keeps 25%
            of tokens and zone compression averages 4 bits (vs 16-bit fp16),
            the reduction is ``4.0 × 4.0 = 16.0×``.
        """
        # Stage 1: eviction reduction
        keep_ratio = 1.0 - self.eviction_ratio
        eviction_factor = 1.0 / keep_ratio if keep_ratio > 0 else float("inf")

        # Stage 2: zone compression reduction (bits relative to fp16=16)
        zone_cfg = ZeroQuantConfig(
            shallow_fraction=self.shallow_fraction,
            deep_fraction=self.deep_fraction,
            shallow_k_bits=self.shallow_k_bits,
            shallow_v_bits=self.shallow_v_bits,
            middle_k_bits=self.middle_k_bits,
            middle_v_bits=self.middle_v_bits,
            deep_k_bits=self.deep_k_bits,
            deep_v_bits=self.deep_v_bits,
            block_size=self.compress_block_size,
        )
        avg_bits = zone_cfg.average_bits(n_layers)
        compression_factor = 16.0 / avg_bits if avg_bits > 0 else float("inf")

        return eviction_factor * compression_factor


# ---------------------------------------------------------------------------
# NullQuantCompressedKV
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullQuantCompressedKV:
    """Result of the NullQuant two-stage compression pipeline.

    Holds the compressed KV data for surviving tokens, along with metadata
    about the eviction step (original count, evicted positions, timing).

    Attributes
    ----------
    config:
        The :class:`NullQuantConfig` that produced this object.
    n_original_tokens:
        Number of tokens before eviction.
    n_surviving_tokens:
        Number of tokens after eviction (input to zone compression).
    evicted_positions:
        Sorted array of token positions that were evicted.
    zone_compressed:
        The :class:`ZoneCompressedKV` holding zone-compressed survivors.
    eviction_time_s:
        Wall-clock time spent on the eviction stage (seconds).
    compression_time_s:
        Wall-clock time spent on the zone compression stage (seconds).
    """

    config: NullQuantConfig
    n_original_tokens: int
    n_surviving_tokens: int
    evicted_positions: NDArray[np.int32]
    zone_compressed: ZoneCompressedKV
    eviction_time_s: float
    compression_time_s: float

    @property
    def memory_bytes(self) -> int:
        """Estimated memory footprint of all compressed data in bytes."""
        # Zone-compressed KV data
        zone_bytes = self.zone_compressed.memory_bytes()
        # Evicted positions index
        index_bytes = self.evicted_positions.nbytes
        return zone_bytes + index_bytes

    @property
    def eviction_ratio_actual(self) -> float:
        """Actual fraction of tokens that were evicted."""
        if self.n_original_tokens == 0:
            return 0.0
        return 1.0 - (self.n_surviving_tokens / self.n_original_tokens)

    @property
    def total_reduction(self) -> float:
        """Approximate total KV reduction factor vs uncompressed fp16.

        Combines the eviction reduction with the zone compression reduction.
        """
        if self.n_surviving_tokens == 0:
            return float("inf")
        eviction_factor = self.n_original_tokens / self.n_surviving_tokens
        # Zone compression: average bits vs 16-bit fp16
        avg_bits = self.config.estimated_reduction(
            self.n_surviving_tokens,
            self.zone_compressed.n_layers,
        )
        # estimated_reduction already returns the full combined factor,
        # but it uses the config eviction_ratio rather than actual.
        # Recompute with actuals for accuracy.
        zone_cfg = ZeroQuantConfig(
            shallow_fraction=self.config.shallow_fraction,
            deep_fraction=self.config.deep_fraction,
            shallow_k_bits=self.config.shallow_k_bits,
            shallow_v_bits=self.config.shallow_v_bits,
            middle_k_bits=self.config.middle_k_bits,
            middle_v_bits=self.config.middle_v_bits,
            deep_k_bits=self.config.deep_k_bits,
            deep_v_bits=self.config.deep_v_bits,
            block_size=self.config.compress_block_size,
        )
        avg_bits = zone_cfg.average_bits(self.zone_compressed.n_layers)
        compression_factor = 16.0 / avg_bits if avg_bits > 0 else float("inf")
        return eviction_factor * compression_factor


# ---------------------------------------------------------------------------
# Token scoring functions
# ---------------------------------------------------------------------------


def score_tokens_l2_norm(
    keys: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Score each token position by average L2 norm of its key vectors.

    High-norm keys contribute more to attention scores (dot product), so
    they represent more "important" token positions.  This heuristic is
    derived from the SlimInfer observation [5] that token importance
    correlates with key vector magnitude.

    Parameters
    ----------
    keys:
        Shape ``(n_layers, n_heads, seq_len, head_dim)``.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(seq_len,)`` with importance scores per position.
    """
    # L2 norm per (layer, head, position): shape (n_layers, n_heads, seq_len)
    norms = np.linalg.norm(keys, axis=-1)
    # Average across layers and heads → shape (seq_len,)
    return norms.mean(axis=(0, 1))


def score_tokens_random(
    seq_len: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Random scoring baseline for ablation studies.

    Parameters
    ----------
    seq_len:
        Number of token positions to score.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(seq_len,)`` with random importance scores.
    """
    rng = np.random.default_rng(seed)
    return rng.random(seq_len)


def score_tokens_uniform_stride(
    seq_len: int,
    keep_ratio: float,
) -> NDArray[np.float64]:
    """Uniform stride scoring — keep every Nth token.

    Assigns high scores to positions that would be kept under a uniform
    stride policy, and low scores to the rest.  Deterministic baseline.

    Parameters
    ----------
    seq_len:
        Number of token positions to score.
    keep_ratio:
        Fraction of tokens to keep (e.g. 0.25 for every 4th token).

    Returns
    -------
    NDArray[np.float64]
        Shape ``(seq_len,)`` with binary-ish importance scores.
    """
    scores = np.zeros(seq_len, dtype=np.float64)
    if keep_ratio <= 0.0 or seq_len == 0:
        return scores
    stride = max(1, int(round(1.0 / keep_ratio)))
    scores[::stride] = 1.0
    return scores


# ---------------------------------------------------------------------------
# select_survivors
# ---------------------------------------------------------------------------


def select_survivors(
    scores: NDArray[np.float64],
    eviction_ratio: float,
    sink_tokens: int,
    recent_tokens: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Select which tokens survive eviction.

    Always keeps the first ``sink_tokens`` positions (attention sinks) and
    the last ``recent_tokens`` positions (recent context).  Among the
    remaining "evictable" positions, keeps the highest-scored tokens to
    reach the target keep ratio.

    Parameters
    ----------
    scores:
        Shape ``(seq_len,)`` with importance score per position.
    eviction_ratio:
        Fraction of tokens to evict (e.g. 0.75 = evict 75%).
    sink_tokens:
        Number of leading positions always kept.
    recent_tokens:
        Number of trailing positions always kept.

    Returns
    -------
    survivor_indices:
        Sorted ascending indices of surviving positions.
    evicted_indices:
        Sorted ascending indices of evicted positions.
    """
    seq_len = len(scores)
    if seq_len == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    # Clamp protected region sizes to available tokens
    sink_n = min(sink_tokens, seq_len)
    recent_n = min(recent_tokens, max(0, seq_len - sink_n))

    # Protected indices (always survive)
    protected = set(range(sink_n))
    if recent_n > 0:
        protected.update(range(seq_len - recent_n, seq_len))

    # Evictable positions
    evictable = np.array(
        [i for i in range(seq_len) if i not in protected], dtype=np.int32
    )

    if len(evictable) == 0:
        # All tokens are protected
        return (
            np.arange(seq_len, dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    # Target total survivors
    keep_ratio = 1.0 - eviction_ratio
    total_keep = max(len(protected), int(round(seq_len * keep_ratio)))

    # How many evictable tokens to keep
    evictable_keep = max(0, total_keep - len(protected))
    evictable_keep = min(evictable_keep, len(evictable))

    # Select top-scoring evictable positions
    evictable_scores = scores[evictable]
    # argsort descending — keep the highest scoring
    top_indices = np.argsort(evictable_scores)[::-1][:evictable_keep]
    surviving_evictable = evictable[top_indices]

    # Combine protected + surviving evictable
    survivor_set = np.union1d(
        np.array(sorted(protected), dtype=np.int32),
        surviving_evictable,
    )
    survivor_indices = np.sort(survivor_set).astype(np.int32)

    # Evicted = everything not in survivors
    all_positions = np.arange(seq_len, dtype=np.int32)
    evicted_indices = np.setdiff1d(all_positions, survivor_indices).astype(np.int32)

    return survivor_indices, evicted_indices


# ---------------------------------------------------------------------------
# TokenEvictor
# ---------------------------------------------------------------------------


class TokenEvictor:
    """Scores token importance and selects positions for eviction.

    Wraps scoring functions and the ``select_survivors`` logic into a
    single reusable component.  All configuration is immutable after
    ``__init__``, making instances thread-safe.

    Thread-safety
    -------------
    All state is created in ``__init__`` and never mutated; instances are safe
    to reuse across threads.
    """

    __slots__ = ("_config", "_scoring_fn")

    def __init__(self, config: NullQuantConfig) -> None:
        self._config = config
        self._scoring_fn = {
            "l2_norm": score_tokens_l2_norm,
            "random": score_tokens_random,
            "uniform_stride": score_tokens_uniform_stride,
        }[config.scoring_method]

    @property
    def config(self) -> NullQuantConfig:
        return self._config

    def evict(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], float]:
        """Evict low-importance tokens from KV tensors.

        Parameters
        ----------
        keys:
            Shape ``(n_layers, n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_layers, n_heads, seq_len, head_dim)``.

        Returns
        -------
        pruned_keys:
            Shape ``(n_layers, n_heads, n_surviving, head_dim)``.
        pruned_values:
            Shape ``(n_layers, n_heads, n_surviving, head_dim)``.
        evicted_indices:
            Sorted indices of evicted positions.
        eviction_time_s:
            Wall-clock time for eviction in seconds.
        """
        t0 = time.monotonic()

        seq_len = keys.shape[2]
        cfg = self._config

        # Validate protected regions fit within sequence
        if cfg.sink_tokens + cfg.recent_tokens >= seq_len:
            logger.warning(
                "sink_tokens(%d) + recent_tokens(%d) >= seq_len(%d); "
                "skipping eviction",
                cfg.sink_tokens,
                cfg.recent_tokens,
                seq_len,
            )
            eviction_time = time.monotonic() - t0
            return (
                keys,
                values,
                np.array([], dtype=np.int32),
                eviction_time,
            )

        # Score tokens
        if self._config.scoring_method == "l2_norm":
            scores = self._scoring_fn(keys)
        elif self._config.scoring_method == "random":
            scores = self._scoring_fn(seq_len)
        else:  # uniform_stride
            keep_ratio = 1.0 - cfg.eviction_ratio
            scores = self._scoring_fn(seq_len, keep_ratio)

        # Select survivors
        survivor_indices, evicted_indices = select_survivors(
            scores,
            cfg.eviction_ratio,
            cfg.sink_tokens,
            cfg.recent_tokens,
        )

        logger.debug(
            "TokenEvictor: %d → %d tokens (evicted %d, %.1f%%)",
            seq_len,
            len(survivor_indices),
            len(evicted_indices),
            len(evicted_indices) / seq_len * 100 if seq_len > 0 else 0,
        )

        # Prune KV tensors — index along the seq_len dimension (axis=2)
        pruned_keys = keys[:, :, survivor_indices, :]
        pruned_values = values[:, :, survivor_indices, :]

        eviction_time = time.monotonic() - t0
        return pruned_keys, pruned_values, evicted_indices, eviction_time


# ---------------------------------------------------------------------------
# NullQuantCompressor
# ---------------------------------------------------------------------------


class NullQuantCompressor:
    """Two-stage compressor: Token Eviction → Zone Compression.

    Stage 1 (Eviction): Score token importance, remove low-importance
    positions from the KV cache using :class:`TokenEvictor`.

    Stage 2 (Compression): Apply Zero-Quant depth-adaptive compression on
    survivors using :class:`DepthAdaptiveCompressor`.

    Combined effect::

        If eviction_ratio=0.75 (keep 25%) and zone compression ≈ 8×:
        Total KV reduction = 4× (from eviction) × 8× (from compression) = 32×

        With aggressive settings (keep 6.25%, K4/V2):
        Total KV reduction = 16× × 8× = 128×

    Thread-safety
    -------------
    All state is created in ``__init__`` and never mutated; instances are safe
    to reuse across threads.
    """

    __slots__ = ("_config", "_evictor", "_zone_config", "_zone_compressor")

    def __init__(self, config: NullQuantConfig | None = None) -> None:
        self._config = config or NullQuantConfig()
        cfg = self._config

        self._evictor = TokenEvictor(cfg)
        self._zone_config = ZeroQuantConfig(
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
        self._zone_compressor = DepthAdaptiveCompressor(self._zone_config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> NullQuantConfig:
        return self._config

    # ------------------------------------------------------------------
    # Compress / Decompress
    # ------------------------------------------------------------------

    def compress_kv(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> NullQuantCompressedKV:
        """Two-stage compression: evict tokens then zone-compress survivors.

        Parameters
        ----------
        keys:
            Shape ``(n_layers, n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_layers, n_heads, seq_len, head_dim)``.

        Returns
        -------
        NullQuantCompressedKV
            Compressed KV data with eviction and zone compression metadata.
        """
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "keys/values must be 4D: (n_layers, n_heads, seq_len, head_dim)"
            )
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have the same shape; "
                f"got {keys.shape} vs {values.shape}"
            )

        n_original_tokens = keys.shape[2]

        # Stage 1: Token eviction
        pruned_keys, pruned_values, evicted_indices, eviction_time = (
            self._evictor.evict(keys, values)
        )
        n_surviving = pruned_keys.shape[2]

        logger.debug(
            "NullQuant stage 1: %d → %d tokens (%.3fs)",
            n_original_tokens,
            n_surviving,
            eviction_time,
        )

        # Stage 2: Zone compression on survivors
        t0 = time.monotonic()
        zone_compressed = self._zone_compressor.compress_kv(
            pruned_keys, pruned_values
        )
        compression_time = time.monotonic() - t0

        logger.debug(
            "NullQuant stage 2: zone compression on %d tokens (%.3fs)",
            n_surviving,
            compression_time,
        )

        return NullQuantCompressedKV(
            config=self._config,
            n_original_tokens=n_original_tokens,
            n_surviving_tokens=n_surviving,
            evicted_positions=evicted_indices,
            zone_compressed=zone_compressed,
            eviction_time_s=eviction_time,
            compression_time_s=compression_time,
        )

    def decompress_kv(
        self,
        compressed: NullQuantCompressedKV,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Decompress zone data back to KV tensors (surviving tokens only).

        .. warning::
            Evicted tokens are permanently lost and cannot be reconstructed.
            The returned tensors have shape
            ``(n_layers, n_heads, n_surviving, head_dim)``.

        Parameters
        ----------
        compressed:
            Output of :meth:`compress_kv`.

        Returns
        -------
        keys, values:
            Both of shape ``(n_layers, n_heads, n_surviving_tokens, head_dim)``.
        """
        return self._zone_compressor.decompress_kv(compressed.zone_compressed)

    def zone_summary(self, n_layers: int) -> dict[str, str]:
        """Human-readable zone description including eviction stats.

        Parameters
        ----------
        n_layers:
            Number of transformer layers in the model.

        Returns
        -------
        dict[str, str]
            Zone descriptions with eviction info prepended.
        """
        cfg = self._config
        summary = self._zone_compressor.zone_summary(n_layers)
        summary["eviction"] = (
            f"ratio={cfg.eviction_ratio:.0%} "
            f"(sink={cfg.sink_tokens}, recent={cfg.recent_tokens}, "
            f"scoring={cfg.scoring_method})"
        )
        return summary


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "NullQuantConfig",
    "NullQuantCompressedKV",
    "NullQuantCompressor",
    "TokenEvictor",
    "score_tokens_l2_norm",
    "score_tokens_random",
    "score_tokens_uniform_stride",
    "select_survivors",
]
