"""Depth-adaptive KV cache compression for Zero-Quant mode.

Implements a three-zone quantization scheme derived from layer activation
research papers:

    - Shallow zone (first ~25% of layers): Dense activations → protect with
      high-precision KV cache (K8/V8 by default).
    - Middle zone  (~25%–75%):  Sparse activations → compress aggressively
      (K4/V2 by default).
    - Deep zone    (last ~25%): Output-critical layers → protect with high
      precision (K8/V8 by default).

This scheme reduces average KV bit-width from 8 (Q8_0 standard) to ~4–5 bits
while preserving attention quality in the zones that matter most.

References
----------
[1] "Exploring Activation Patterns..." (Peking Univ, 2024, arXiv:2405.17799):
    Shallow transformer layers activate densely; deep layers activate sparsely.
    Implication: shallow layers need high-fidelity KV cache; middle layers can
    tolerate aggressive quantization.
[2] "TurboSparse" (SJTU, 2024, arXiv:2406.05955): dReLU achieves 97% FFN
    sparsity in MoE models, 90% in dense models.  Confirms middle layers are
    the primary candidates for aggressive compression.
[3] "LUT-LLM" (UCLA/Microsoft, 2025, arXiv:2511.06174): Activation-weight
    *vector* co-quantization outperforms separate K/V quantization.  Joint KV
    codebooks capture cross-tensor correlation for better quality per bit.
[4] "LLM Inference Hardware Challenges" (Google, 2026, arXiv:2601.05047):
    Memory bandwidth is the primary bottleneck.  KV cache size reduction
    directly improves throughput.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.turboquant.compressor import CompressedKV, QuantConfig, TurboQuantCompressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_BITS: frozenset[int] = frozenset({2, 3, 4, 8})


# ---------------------------------------------------------------------------
# ZeroQuantConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroQuantConfig:
    """Depth-adaptive KV quantization config for Zero-Quant mode.

    Divides transformer layers into three zones and applies independent K/V
    bit-widths per zone, based on observed activation sparsity patterns [1][2].

    Attributes
    ----------
    shallow_fraction:
        Fraction of total layers classified as "shallow" (first N layers).
        These have dense activations and must use high-precision KV cache.
    deep_fraction:
        Fraction of total layers classified as "deep" (last N layers).
        These are output-critical and must use high-precision KV cache.
    shallow_k_bits / shallow_v_bits:
        Key / value bit-width for the shallow zone.
    middle_k_bits / middle_v_bits:
        Key / value bit-width for the middle (sparse) zone.
    deep_k_bits / deep_v_bits:
        Key / value bit-width for the deep zone.
    block_size:
        PolarQuant block size.  Must be a power of 2 ≥ 2.
    use_kv_coquant:
        Enable K-V co-quantization (paper [3]).  When True, K and V for each
        zone are concatenated along the head-dimension axis and compressed
        jointly with the same codebook, capturing cross-tensor correlation.
    """

    shallow_fraction: float = 0.25
    deep_fraction: float = 0.25

    shallow_k_bits: int = 8
    shallow_v_bits: int = 8
    middle_k_bits: int = 4
    middle_v_bits: int = 2
    deep_k_bits: int = 8
    deep_v_bits: int = 8

    block_size: int = 128
    use_kv_coquant: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.shallow_fraction < 1.0):
            raise ValueError(
                f"shallow_fraction must be in (0, 1), got {self.shallow_fraction}"
            )
        if not (0.0 < self.deep_fraction < 1.0):
            raise ValueError(
                f"deep_fraction must be in (0, 1), got {self.deep_fraction}"
            )
        if self.shallow_fraction + self.deep_fraction >= 1.0:
            raise ValueError(
                "shallow_fraction + deep_fraction must be < 1.0 (need room for "
                f"middle zone), got sum={self.shallow_fraction + self.deep_fraction}"
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
        if self.block_size < 2 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be a power of 2 >= 2, got {self.block_size}"
            )

    def average_bits(self, n_layers: int) -> float:
        """Compute the weighted average bit-width across all layers."""
        compressor = DepthAdaptiveCompressor(self)
        shallow_end, deep_start = compressor._zone_boundaries(n_layers)
        n_shallow = shallow_end
        n_middle = deep_start - shallow_end
        n_deep = n_layers - deep_start
        total_bits = (
            n_shallow * (self.shallow_k_bits + self.shallow_v_bits)
            + n_middle * (self.middle_k_bits + self.middle_v_bits)
            + n_deep * (self.deep_k_bits + self.deep_v_bits)
        )
        return total_bits / (n_layers * 2)


# ---------------------------------------------------------------------------
# ZoneCompressedKV
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneCompressedKV:
    """Depth-adaptive compressed KV cache with per-zone precision.

    Stores three independent :class:`~src.turboquant.compressor.CompressedKV`
    objects — one for each depth zone — so that each zone's K and V tensors
    are compressed at the appropriate bit-width.

    Attributes
    ----------
    config:
        The :class:`ZeroQuantConfig` that produced this object.
    n_layers:
        Total number of transformer layers in the source model.
    shallow_end:
        Exclusive upper bound for the shallow zone.
        Shallow zone = layers ``[0, shallow_end)``.
    deep_start:
        Inclusive lower bound for the deep zone.
        Deep zone = layers ``[deep_start, n_layers)``.
        Middle zone = layers ``[shallow_end, deep_start)``.
    shallow_kv / middle_kv / deep_kv:
        Compressed KV data for each zone.
    coquant_head_dim:
        Original ``head_dim`` when ``use_kv_coquant=True``; 0 otherwise.
        Used by the decompressor to split the joint K||V tensor on decode.
    """

    config: ZeroQuantConfig
    n_layers: int
    shallow_end: int
    deep_start: int
    shallow_kv: CompressedKV
    middle_kv: CompressedKV
    deep_kv: CompressedKV
    coquant_head_dim: int = 0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_shallow(self) -> int:
        return self.shallow_end

    @property
    def n_middle(self) -> int:
        return self.deep_start - self.shallow_end

    @property
    def n_deep(self) -> int:
        return self.n_layers - self.deep_start

    def memory_bytes(self) -> int:
        """Estimate memory footprint of all compressed tensors in bytes."""

        def _kv_bytes(ckv: CompressedKV) -> int:
            total = 0
            for ct in ckv.keys:
                for b in ct.blocks:
                    total += b.indices.nbytes + 8  # indices + norm + metadata
            for ct in ckv.values:
                for b in ct.blocks:
                    total += b.indices.nbytes + 8
            return total

        return (
            _kv_bytes(self.shallow_kv)
            + _kv_bytes(self.middle_kv)
            + _kv_bytes(self.deep_kv)
        )


# ---------------------------------------------------------------------------
# DepthAdaptiveCompressor
# ---------------------------------------------------------------------------


class DepthAdaptiveCompressor:
    """KV cache compressor with depth-adaptive precision per layer zone.

    Wraps :class:`~src.turboquant.compressor.TurboQuantCompressor` to apply
    different quantization configs per depth zone.  Three
    ``TurboQuantCompressor`` instances are created at init time — one per
    zone — so compression/decompression is fully stateless after construction.

    When ``use_kv_coquant=True`` (paper [3]), K and V for each zone are
    concatenated along the head-dim axis before compression so that a single
    PolarQuant codebook captures cross-tensor K-V correlation.

    Thread-safety
    -------------
    All state is created in ``__init__`` and never mutated; instances are safe
    to reuse across threads.
    """

    __slots__ = (
        "_config",
        "_shallow_compressor",
        "_middle_compressor",
        "_deep_compressor",
    )

    def __init__(self, config: ZeroQuantConfig | None = None) -> None:
        self._config = config or ZeroQuantConfig()
        cfg = self._config

        self._shallow_compressor = TurboQuantCompressor(
            QuantConfig(
                k_bits=cfg.shallow_k_bits,
                v_bits=cfg.shallow_v_bits,
                block_size=cfg.block_size,
            )
        )
        self._middle_compressor = TurboQuantCompressor(
            QuantConfig(
                k_bits=cfg.middle_k_bits,
                v_bits=cfg.middle_v_bits,
                block_size=cfg.block_size,
            )
        )
        self._deep_compressor = TurboQuantCompressor(
            QuantConfig(
                k_bits=cfg.deep_k_bits,
                v_bits=cfg.deep_v_bits,
                block_size=cfg.block_size,
            )
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ZeroQuantConfig:
        return self._config

    # ------------------------------------------------------------------
    # Zone boundaries
    # ------------------------------------------------------------------

    def _zone_boundaries(self, n_layers: int) -> tuple[int, int]:
        """Compute ``(shallow_end, deep_start)`` for a model with ``n_layers``.

        Ensures:
        - Each zone has at least 1 layer.
        - ``shallow_end < deep_start``.

        Returns
        -------
        shallow_end:
            Index of the first layer *not* in the shallow zone.
        deep_start:
            Index of the first layer *in* the deep zone.
        """
        cfg = self._config
        n_shallow = max(1, int(round(n_layers * cfg.shallow_fraction)))
        n_deep = max(1, int(round(n_layers * cfg.deep_fraction)))

        # Safety: guarantee at least 1 middle layer
        if n_shallow + n_deep >= n_layers:
            n_shallow = max(1, n_layers // 4)
            n_deep = max(1, n_layers // 4)
            n_shallow = min(n_shallow, n_layers - 2)
            n_deep = min(n_deep, n_layers - n_shallow - 1)

        return n_shallow, n_layers - n_deep

    def zone_summary(self, n_layers: int) -> dict[str, str]:
        """Human-readable zone descriptions for logging / health endpoint."""
        cfg = self._config
        shallow_end, deep_start = self._zone_boundaries(n_layers)
        return {
            "shallow": (
                f"layers 0-{shallow_end - 1} "
                f"(K{cfg.shallow_k_bits}/V{cfg.shallow_v_bits})"
            ),
            "middle": (
                f"layers {shallow_end}-{deep_start - 1} "
                f"(K{cfg.middle_k_bits}/V{cfg.middle_v_bits})"
            ),
            "deep": (
                f"layers {deep_start}-{n_layers - 1} "
                f"(K{cfg.deep_k_bits}/V{cfg.deep_v_bits})"
            ),
        }

    # ------------------------------------------------------------------
    # Compress / Decompress
    # ------------------------------------------------------------------

    def compress_kv(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> ZoneCompressedKV:
        """Compress full KV cache with depth-adaptive zone precision.

        Parameters
        ----------
        keys:
            Shape ``(n_layers, n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_layers, n_heads, seq_len, head_dim)``.

        Returns
        -------
        ZoneCompressedKV
            Separate compressed state for each depth zone.
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

        n_layers = keys.shape[0]
        head_dim = keys.shape[-1]
        shallow_end, deep_start = self._zone_boundaries(n_layers)

        logger.debug(
            "ZeroQuant compress: %d layers → shallow[0:%d] middle[%d:%d] deep[%d:%d]",
            n_layers,
            shallow_end,
            shallow_end,
            deep_start,
            deep_start,
            n_layers,
        )

        # Slice zones
        sk, sv = keys[:shallow_end], values[:shallow_end]
        mk, mv = keys[shallow_end:deep_start], values[shallow_end:deep_start]
        dk, dv = keys[deep_start:], values[deep_start:]

        if self._config.use_kv_coquant:
            shallow_kv = self._compress_coquant(sk, sv, self._shallow_compressor)
            middle_kv = self._compress_coquant(mk, mv, self._middle_compressor)
            deep_kv = self._compress_coquant(dk, dv, self._deep_compressor)
            coquant_head_dim = head_dim
        else:
            shallow_kv = self._shallow_compressor.compress_kv(sk, sv)
            middle_kv = self._middle_compressor.compress_kv(mk, mv)
            deep_kv = self._deep_compressor.compress_kv(dk, dv)
            coquant_head_dim = 0

        return ZoneCompressedKV(
            config=self._config,
            n_layers=n_layers,
            shallow_end=shallow_end,
            deep_start=deep_start,
            shallow_kv=shallow_kv,
            middle_kv=middle_kv,
            deep_kv=deep_kv,
            coquant_head_dim=coquant_head_dim,
        )

    def decompress_kv(
        self,
        compressed: ZoneCompressedKV,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Decompress a :class:`ZoneCompressedKV` back to full K and V arrays.

        Returns
        -------
        keys, values:
            Both of shape ``(n_layers, n_heads, seq_len, head_dim)``.
        """
        coquant = compressed.coquant_head_dim > 0

        if coquant:
            sk, sv = self._decompress_coquant(
                compressed.shallow_kv,
                compressed.coquant_head_dim,
                self._shallow_compressor,
            )
            mk, mv = self._decompress_coquant(
                compressed.middle_kv,
                compressed.coquant_head_dim,
                self._middle_compressor,
            )
            dk, dv = self._decompress_coquant(
                compressed.deep_kv,
                compressed.coquant_head_dim,
                self._deep_compressor,
            )
        else:
            sk, sv = self._shallow_compressor.decompress_kv(compressed.shallow_kv)
            mk, mv = self._middle_compressor.decompress_kv(compressed.middle_kv)
            dk, dv = self._deep_compressor.decompress_kv(compressed.deep_kv)

        keys = np.concatenate([sk, mk, dk], axis=0)
        values = np.concatenate([sv, mv, dv], axis=0)
        return keys, values

    # ------------------------------------------------------------------
    # K-V co-quantization helpers (paper [3])
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_coquant(
        zone_k: NDArray[np.float64],
        zone_v: NDArray[np.float64],
        compressor: TurboQuantCompressor,
    ) -> CompressedKV:
        """Compress K and V jointly (co-quantization, paper [3]).

        Concatenates ``zone_k`` and ``zone_v`` along the ``head_dim`` axis to
        form a single ``kv_joint`` tensor of shape
        ``(n_layers, n_heads, seq_len, 2 * head_dim)``.  This joint tensor is
        compressed as "keys" so that the PolarQuant codebook captures the
        correlation between K and V vectors.

        The "values" slot of the returned :class:`CompressedKV` holds
        ``zone_v`` compressed independently at the same precision, providing
        a cross-check reference that does not affect memory significantly.

        On decompression, only the joint "keys" tensor is used — it is split
        at ``head_dim`` to recover the co-quantized K and V.
        """
        # Joint K||V tensor: (..., 2*head_dim)
        kv_joint = np.concatenate([zone_k, zone_v], axis=-1)
        # Compress the joint tensor as "keys"; zone_v as independent "values"
        return compressor.compress_kv(kv_joint, zone_v)

    @staticmethod
    def _decompress_coquant(
        zone_compressed: CompressedKV,
        head_dim: int,
        compressor: TurboQuantCompressor,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Decompress a co-quantized zone.

        The "keys" in ``zone_compressed`` hold the joint K||V tensor with
        shape ``(..., 2 * head_dim)``.  Decompress and split at ``head_dim``
        to recover K and V.

        Parameters
        ----------
        zone_compressed:
            Output of :meth:`_compress_coquant`.
        head_dim:
            Original per-head dimension (half the compressed "key" last-dim).
        compressor:
            The same compressor used during compression.
        """
        kv_joint, _v_independent = compressor.decompress_kv(zone_compressed)
        zone_k = kv_joint[..., :head_dim]
        zone_v = kv_joint[..., head_dim:]
        return zone_k, zone_v
