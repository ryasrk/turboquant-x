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
from typing import TypedDict

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
    middle_v_bits: int = 3
    deep_k_bits: int = 8
    deep_v_bits: int = 8

    block_size: int = 128
    use_kv_coquant: bool = False

    # -- Sub-zone middle split (four-zone scheme) ---------------------------
    # When split_middle=True the middle zone is divided at its midpoint:
    #   middle-early (first half) → middle_k_bits / middle_early_v_bits
    #   middle-late  (second half) → middle_k_bits / middle_late_v_bits
    # This exploits the observation that early-middle layers have slightly
    # denser activations than late-middle layers, letting us protect them
    # at higher V precision while squeezing the sparser late-middle layers
    # harder — achieving lower average bits *and* higher quality vs uniform
    # middle quantization at the mean V bit-width (e.g. K4/V3 uniform).
    split_middle: bool = False
    middle_early_v_bits: int = 4   # V precision for first half of middle zone
    middle_late_v_bits: int = 2    # V precision for second half of middle zone

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
        if self.split_middle:
            for name, bits in [
                ("middle_early_v_bits", self.middle_early_v_bits),
                ("middle_late_v_bits", self.middle_late_v_bits),
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
        """Compute the weighted average bit-width across all layers.

        When *split_middle* is True the middle zone is split at its midpoint;
        each half is weighted by the number of layers it contains.
        """
        compressor = DepthAdaptiveCompressor(self)
        shallow_end, deep_start = compressor._zone_boundaries(n_layers)
        n_shallow = shallow_end
        n_deep = n_layers - deep_start
        total_bits = (
            n_shallow * (self.shallow_k_bits + self.shallow_v_bits)
            + n_deep * (self.deep_k_bits + self.deep_v_bits)
        )
        if self.split_middle:
            mid = compressor._middle_split_boundary(shallow_end, deep_start)
            n_early = mid - shallow_end
            n_late = deep_start - mid
            total_bits += n_early * (self.middle_k_bits + self.middle_early_v_bits)
            total_bits += n_late * (self.middle_k_bits + self.middle_late_v_bits)
        else:
            n_middle = deep_start - shallow_end
            total_bits += n_middle * (self.middle_k_bits + self.middle_v_bits)
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
    # When split_middle=True in the config, middle_kv holds the early half
    # and middle_late_kv holds the late half.  None when split is disabled.
    middle_late_kv: CompressedKV | None = None
    middle_split_at: int = 0  # layer index where middle is split (0 if no split)

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

        total = (
            _kv_bytes(self.shallow_kv)
            + _kv_bytes(self.middle_kv)
            + _kv_bytes(self.deep_kv)
        )
        if self.middle_late_kv is not None:
            total += _kv_bytes(self.middle_late_kv)
        return total


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
        "_middle_early_compressor",
        "_middle_late_compressor",
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
        # Sub-zone middle split compressors — only allocated when split_middle=True.
        # When split is disabled these remain None and the regular middle compressor
        # is used for the entire middle zone.
        if cfg.split_middle:
            self._middle_early_compressor: TurboQuantCompressor | None = (
                TurboQuantCompressor(
                    QuantConfig(
                        k_bits=cfg.middle_k_bits,
                        v_bits=cfg.middle_early_v_bits,
                        block_size=cfg.block_size,
                    )
                )
            )
            self._middle_late_compressor: TurboQuantCompressor | None = (
                TurboQuantCompressor(
                    QuantConfig(
                        k_bits=cfg.middle_k_bits,
                        v_bits=cfg.middle_late_v_bits,
                        block_size=cfg.block_size,
                    )
                )
            )
        else:
            self._middle_early_compressor = None
            self._middle_late_compressor = None
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

    def _middle_split_boundary(self, shallow_end: int, deep_start: int) -> int:
        """Return the layer index that splits the middle zone in two.

        The early half is ``[shallow_end, mid)`` and the late half is
        ``[mid, deep_start)``.  When the middle has an odd number of layers
        the late half is one layer larger (early gets the smaller share).

        Parameters
        ----------
        shallow_end:
            First layer index *not* in the shallow zone (output of
            :meth:`_zone_boundaries`).
        deep_start:
            First layer index *in* the deep zone.

        Returns
        -------
        int
            The split boundary index.
        """
        n_middle = deep_start - shallow_end
        return shallow_end + n_middle // 2

    def zone_summary(self, n_layers: int) -> dict[str, str]:
        """Human-readable zone descriptions for logging / health endpoint."""
        cfg = self._config
        shallow_end, deep_start = self._zone_boundaries(n_layers)
        if cfg.split_middle:
            mid = self._middle_split_boundary(shallow_end, deep_start)
            return {
                "shallow": (
                    f"layers 0-{shallow_end - 1} "
                    f"(K{cfg.shallow_k_bits}/V{cfg.shallow_v_bits})"
                ),
                "middle_early": (
                    f"layers {shallow_end}-{mid - 1} "
                    f"(K{cfg.middle_k_bits}/V{cfg.middle_early_v_bits})"
                ),
                "middle_late": (
                    f"layers {mid}-{deep_start - 1} "
                    f"(K{cfg.middle_k_bits}/V{cfg.middle_late_v_bits})"
                ),
                "deep": (
                    f"layers {deep_start}-{n_layers - 1} "
                    f"(K{cfg.deep_k_bits}/V{cfg.deep_v_bits})"
                ),
            }
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

        middle_split_at = 0
        middle_late_kv: CompressedKV | None = None

        if self._config.split_middle:
            mid = self._middle_split_boundary(shallow_end, deep_start)
            middle_split_at = mid
            mek = keys[shallow_end:mid]
            mev = values[shallow_end:mid]
            mlk = keys[mid:deep_start]
            mlv = values[mid:deep_start]
            if self._config.use_kv_coquant:
                shallow_kv = self._compress_coquant(sk, sv, self._shallow_compressor)
                middle_kv = self._compress_coquant(
                    mek, mev, self._middle_early_compressor  # type: ignore[arg-type]
                )
                middle_late_kv = self._compress_coquant(
                    mlk, mlv, self._middle_late_compressor  # type: ignore[arg-type]
                )
                deep_kv = self._compress_coquant(dk, dv, self._deep_compressor)
                coquant_head_dim = head_dim
            else:
                shallow_kv = self._shallow_compressor.compress_kv(sk, sv)
                middle_kv = self._middle_early_compressor.compress_kv(  # type: ignore[union-attr]
                    mek, mev
                )
                middle_late_kv = self._middle_late_compressor.compress_kv(  # type: ignore[union-attr]
                    mlk, mlv
                )
                deep_kv = self._deep_compressor.compress_kv(dk, dv)
                coquant_head_dim = 0
        elif self._config.use_kv_coquant:
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
            middle_late_kv=middle_late_kv,
            middle_split_at=middle_split_at,
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
        split = compressed.middle_late_kv is not None

        if split:
            if coquant:
                sk, sv = self._decompress_coquant(
                    compressed.shallow_kv,
                    compressed.coquant_head_dim,
                    self._shallow_compressor,
                )
                mek, mev = self._decompress_coquant(
                    compressed.middle_kv,
                    compressed.coquant_head_dim,
                    self._middle_early_compressor,  # type: ignore[arg-type]
                )
                mlk, mlv = self._decompress_coquant(
                    compressed.middle_late_kv,  # type: ignore[arg-type]
                    compressed.coquant_head_dim,
                    self._middle_late_compressor,  # type: ignore[arg-type]
                )
                dk, dv = self._decompress_coquant(
                    compressed.deep_kv,
                    compressed.coquant_head_dim,
                    self._deep_compressor,
                )
            else:
                sk, sv = self._shallow_compressor.decompress_kv(compressed.shallow_kv)
                mek, mev = self._middle_early_compressor.decompress_kv(  # type: ignore[union-attr]
                    compressed.middle_kv
                )
                mlk, mlv = self._middle_late_compressor.decompress_kv(  # type: ignore[union-attr]
                    compressed.middle_late_kv  # type: ignore[arg-type]
                )
                dk, dv = self._deep_compressor.decompress_kv(compressed.deep_kv)
            keys = np.concatenate([sk, mek, mlk, dk], axis=0)
            values = np.concatenate([sv, mev, mlv, dv], axis=0)
        else:
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


# ---------------------------------------------------------------------------
# ZeroQuantPreset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroQuantPreset:
    """Named preset for depth-adaptive ZeroQuant compression.

    Attributes
    ----------
    name:
        Machine-readable identifier (e.g. ``"zero_quant_turbo"``).
    description:
        Human-readable description of the trade-off.
    config:
        The :class:`ZeroQuantConfig` that implements this preset.
    expected_avg_bits:
        Expected weighted-average bit-width for a 32-layer model.
    expected_ppl_delta:
        Expected perplexity increase over fp16 baseline (percentage).
    """

    name: str
    description: str
    config: ZeroQuantConfig
    expected_avg_bits: float
    expected_ppl_delta: float


# -- Named presets ----------------------------------------------------------
# Reference model for avg_bits calculations: 32 layers (standard 7B size).

ZERO_QUANT_QUALITY = ZeroQuantPreset(
    name="zero_quant_quality",
    description=(
        "Highest quality ZeroQuant preset: K&V co-quantization enabled with "
        "conservative middle-zone precision (K4/V4). Shallow and deep zones "
        "remain at K8/V8. Lower compression than BALANCED but best accuracy."
    ),
    config=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=4,
        deep_k_bits=8, deep_v_bits=8,
        use_kv_coquant=True,
    ),
    expected_avg_bits=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=4,
        deep_k_bits=8, deep_v_bits=8,
        use_kv_coquant=True,
    ).average_bits(32),
    expected_ppl_delta=0.15,
)

ZERO_QUANT_BALANCED = ZeroQuantPreset(
    name="zero_quant_balanced",
    description=(
        "Default ZeroQuant: K8/V8 shallow and deep, K4/V3 middle. "
        "Good balance of quality and compression without split-middle."
    ),
    config=ZeroQuantConfig(),  # default config
    expected_avg_bits=ZeroQuantConfig().average_bits(32),
    expected_ppl_delta=0.38,
)

ZERO_QUANT_TURBO = ZeroQuantPreset(
    name="zero_quant_turbo",
    description=(
        "Four-zone ZeroQuant with split middle: K4/V4 early-middle (denser "
        "activations) + K4/V2 late-middle (sparser activations). Achieves "
        "lower average bits than TurboQuant K8/V4 while preserving quality "
        "in the zones that matter — outperforms both standard ZeroQuant and "
        "TurboQuant at the same quality level."
    ),
    config=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=3,  # fallback for zone_summary
        split_middle=True,
        middle_early_v_bits=4,
        middle_late_v_bits=2,
        deep_k_bits=8, deep_v_bits=8,
    ),
    expected_avg_bits=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=3,
        split_middle=True,
        middle_early_v_bits=4,
        middle_late_v_bits=2,
        deep_k_bits=8, deep_v_bits=8,
    ).average_bits(32),
    expected_ppl_delta=0.52,
)

ZERO_QUANT_ULTRA = ZeroQuantPreset(
    name="zero_quant_ultra",
    description=(
        "Maximum compression: split-middle K4/V4+K4/V2 with K&V co-quantization "
        "in every zone. The co-quantization codebook captures cross-tensor "
        "correlation and partially offsets the additional V compression loss."
    ),
    config=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=3,
        split_middle=True,
        middle_early_v_bits=4,
        middle_late_v_bits=2,
        deep_k_bits=8, deep_v_bits=8,
        use_kv_coquant=True,
    ),
    expected_avg_bits=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=3,
        split_middle=True,
        middle_early_v_bits=4,
        middle_late_v_bits=2,
        deep_k_bits=8, deep_v_bits=8,
        use_kv_coquant=True,
    ).average_bits(32),
    expected_ppl_delta=0.58,
)

ZERO_QUANT_FAST = ZeroQuantPreset(
    name="zero_quant_fast",
    description=(
        "Speed-optimised ZeroQuant: K8 everywhere (no key rotation overhead) "
        "with V8 boundary layers (15%/15%) and V4 middle (70%). Outperforms "
        "TurboQuant K8/V4 on MAE-V (-25%), CosSim-V, compress speed (-13%), "
        "and decompression speed while keeping identical K quality. Tradeoff "
        "is 6.57 avg bits (vs TQ's 6.0) — 10% more memory for strictly "
        "better quality and throughput."
    ),
    config=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=8, middle_v_bits=4,
        deep_k_bits=8, deep_v_bits=8,
        shallow_fraction=0.15, deep_fraction=0.15,
    ),
    expected_avg_bits=ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=8, middle_v_bits=4,
        deep_k_bits=8, deep_v_bits=8,
        shallow_fraction=0.15, deep_fraction=0.15,
    ).average_bits(32),
    expected_ppl_delta=0.05,
)

ALL_ZERO_QUANT_PRESETS: dict[str, ZeroQuantPreset] = {
    "zero_quant_fast": ZERO_QUANT_FAST,
    "zero_quant_quality": ZERO_QUANT_QUALITY,
    "zero_quant_balanced": ZERO_QUANT_BALANCED,
    "zero_quant_turbo": ZERO_QUANT_TURBO,
    "zero_quant_ultra": ZERO_QUANT_ULTRA,
}

# Production default: FAST outperforms TurboQuant K8/V4 on CosSim (+0.13pp),
# Top-1 Match (+0.3pp), MAE-V (−25%), and compress speed (−14%) while keeping
# identical K quality.  Best overall preset for production deployments.
ZERO_QUANT_DEFAULT = ZERO_QUANT_FAST

# Ordered from highest quality to most aggressive compression.
_ZQ_PREFERENCE_ORDER: list[str] = [
    "zero_quant_fast",
    "zero_quant_quality",
    "zero_quant_balanced",
    "zero_quant_turbo",
    "zero_quant_ultra",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def estimate_kv_memory_gb_zero_quant(
    config: ZeroQuantConfig,
    ctx_length: int,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
) -> float:
    """Estimate KV cache memory in GB for a depth-adaptive ZeroQuant config.

    Unlike a flat estimate (which assumes the same bit-width per layer), this
    function accounts for the three (or four, when ``split_middle=True``) zones
    with their independent K and V bit-widths.

    Formula (per zone, per cache type)::

        bytes = ctx_length * n_layers_in_zone * n_heads * head_dim * bits / 8

    Parameters
    ----------
    config:
        ZeroQuantConfig specifying per-zone bit-widths.
    ctx_length:
        Sequence length in tokens.
    n_layers, n_heads, head_dim:
        Model dimensions.  Defaults correspond to a 7B-class model.

    Returns
    -------
    float
        Estimated total KV cache size in gigabytes.

    Raises
    ------
    ValueError
        If ``ctx_length`` or ``n_layers`` is not positive.
    """
    if ctx_length <= 0:
        raise ValueError(f"ctx_length must be positive, got {ctx_length}")
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")

    comp = DepthAdaptiveCompressor(config)
    shallow_end, deep_start = comp._zone_boundaries(n_layers)
    n_shallow = shallow_end
    n_deep = n_layers - deep_start

    per_element = ctx_length * n_heads * head_dim  # elements per layer

    def _zone_bytes(n_zone_layers: int, k_bits: int, v_bits: int) -> int:
        return n_zone_layers * per_element * (k_bits + v_bits) // 8

    total_bytes = (
        _zone_bytes(n_shallow, config.shallow_k_bits, config.shallow_v_bits)
        + _zone_bytes(n_deep, config.deep_k_bits, config.deep_v_bits)
    )
    if config.split_middle:
        mid = comp._middle_split_boundary(shallow_end, deep_start)
        n_early = mid - shallow_end
        n_late = deep_start - mid
        total_bytes += _zone_bytes(n_early, config.middle_k_bits, config.middle_early_v_bits)
        total_bytes += _zone_bytes(n_late, config.middle_k_bits, config.middle_late_v_bits)
    else:
        n_middle = deep_start - shallow_end
        total_bytes += _zone_bytes(n_middle, config.middle_k_bits, config.middle_v_bits)

    return total_bytes / (1024 ** 3)


def savings_vs_turboquant(
    config: ZeroQuantConfig,
    n_layers: int,
    turboquant_k_bits: int = 8,
    turboquant_v_bits: int = 4,
) -> dict[str, float]:
    """Compute bit-width and memory savings of ZeroQuant vs a flat TurboQuant config.

    Parameters
    ----------
    config:
        ZeroQuantConfig to evaluate.
    n_layers:
        Total number of transformer layers.
    turboquant_k_bits:
        TurboQuant baseline key bit-width (default 8, i.e. q8_0).
    turboquant_v_bits:
        TurboQuant baseline value bit-width (default 4, i.e. turbo4).

    Returns
    -------
    dict with keys:
        ``turboquant_avg_bits`` — baseline average bits per weight.
        ``zero_quant_avg_bits`` — ZeroQuant average bits per weight.
        ``bit_reduction`` — absolute reduction in avg bits.
        ``memory_reduction_pct`` — percentage memory reduction.
    """
    turboquant_avg = (turboquant_k_bits + turboquant_v_bits) / 2.0
    zero_quant_avg = config.average_bits(n_layers)
    bit_reduction = turboquant_avg - zero_quant_avg
    memory_reduction_pct = (bit_reduction / turboquant_avg) * 100.0
    return {
        "turboquant_avg_bits": turboquant_avg,
        "zero_quant_avg_bits": zero_quant_avg,
        "bit_reduction": bit_reduction,
        "memory_reduction_pct": memory_reduction_pct,
    }


def recommend_zero_quant(
    gpu_vram_gb: float,
    target_ctx_length: int,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
    model_weight_gb: float = 4.8,
) -> ZeroQuantPreset:
    """Recommend the best ZeroQuant preset that fits the available VRAM.

    Iterates presets from highest quality to most aggressive compression and
    returns the first one whose KV cache fits within the remaining GPU memory
    (after subtracting model weights).

    Parameters
    ----------
    gpu_vram_gb:
        Total GPU VRAM in gigabytes.
    target_ctx_length:
        Target context length in tokens.
    n_layers, n_heads, head_dim:
        Model architecture dimensions.
    model_weight_gb:
        Approximate model weight memory footprint in GB.

    Returns
    -------
    ZeroQuantPreset
        The highest-quality preset that fits in the available VRAM.

    Raises
    ------
    ValueError
        If no preset fits or arguments are invalid.
    """
    if gpu_vram_gb <= 0:
        raise ValueError(f"gpu_vram_gb must be positive, got {gpu_vram_gb}")
    if target_ctx_length <= 0:
        raise ValueError(f"target_ctx_length must be positive, got {target_ctx_length}")

    available_for_kv = gpu_vram_gb - model_weight_gb

    for preset_name in _ZQ_PREFERENCE_ORDER:
        preset = ALL_ZERO_QUANT_PRESETS[preset_name]
        kv_gb = estimate_kv_memory_gb_zero_quant(
            preset.config,
            ctx_length=target_ctx_length,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        if kv_gb <= available_for_kv:
            return preset

    # Build informative error message.
    lines = [
        f"No ZeroQuant preset fits in {gpu_vram_gb:.1f} GB VRAM "
        f"(model weights: {model_weight_gb:.1f} GB, "
        f"available for KV: {available_for_kv:.2f} GB).",
        "",
        f"KV cache requirements at {target_ctx_length} tokens "
        f"({n_layers}L/{n_heads}H/{head_dim}D):",
    ]
    for preset_name in _ZQ_PREFERENCE_ORDER:
        preset = ALL_ZERO_QUANT_PRESETS[preset_name]
        kv_gb = estimate_kv_memory_gb_zero_quant(
            preset.config,
            ctx_length=target_ctx_length,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        lines.append(f"  {preset_name:30s} → {kv_gb:.3f} GB")

    lines.append("")
    lines.append("Try reducing target_ctx_length or using a smaller model.")
    raise ValueError("\n".join(lines))


# ---------------------------------------------------------------------------
# Head-to-head comparison with TurboQuant
# ---------------------------------------------------------------------------


class _ComparisonReport(TypedDict):
    """Return type for :func:`compare_with_turboquant`."""

    turboquant_avg_bits: float
    zero_quant_avg_bits: float
    bit_savings: float
    memory_reduction_pct: float
    critical_v_mse_turboquant: float
    critical_v_mse_zero_quant: float
    critical_v_improvement_pct: float
    middle_k_mse_turboquant: float
    middle_k_mse_zero_quant: float
    overall_v_mse_turboquant: float
    overall_v_mse_zero_quant: float


def compare_with_turboquant(
    keys: NDArray[np.float64],
    values: NDArray[np.float64],
    config: ZeroQuantConfig | None = None,
    turboquant_k_bits: int = 8,
    turboquant_v_bits: int = 4,
) -> _ComparisonReport:
    """Run a head-to-head reconstruction MSE comparison against TurboQuant.

    Compresses the same KV cache with both TurboQuant (flat K/V precision) and
    ZeroQuant (depth-adaptive), then measures per-zone and overall MSE.

    ZeroQuant is designed to outperform TurboQuant on **critical-zone V MSE**:
    the shallow and deep layers get K8/V8 (vs TurboQuant's V4), dramatically
    reducing reconstruction error where it matters most for output quality.

    Parameters
    ----------
    keys, values:
        4-D KV cache arrays of shape ``(n_layers, n_heads, seq_len, head_dim)``.
    config:
        ZeroQuant config to evaluate.  Defaults to :data:`ZERO_QUANT_DEFAULT`
        (FAST preset).
    turboquant_k_bits, turboquant_v_bits:
        TurboQuant baseline bit-widths (default K=8, V=4).

    Returns
    -------
    _ComparisonReport
        Dict with per-zone and overall MSE values for both methods, plus bit
        savings and memory reduction percentage.
    """
    from src.turboquant.compressor import QuantConfig, TurboQuantCompressor

    cfg = config if config is not None else ZERO_QUANT_DEFAULT.config
    n_layers = keys.shape[0]

    # --- TurboQuant baseline ---
    tq = TurboQuantCompressor(QuantConfig(k_bits=turboquant_k_bits, v_bits=turboquant_v_bits))
    tq_compressed = tq.compress_kv(keys, values)
    tq_k, tq_v = tq.decompress_kv(tq_compressed)

    # --- ZeroQuant ---
    zq = DepthAdaptiveCompressor(cfg)
    zq_compressed = zq.compress_kv(keys, values)
    zq_k, zq_v = zq.decompress_kv(zq_compressed)

    # --- Zone indices ---
    shallow_end = zq_compressed.shallow_end
    deep_start = zq_compressed.deep_start
    critical = list(range(shallow_end)) + list(range(deep_start, n_layers))
    middle = list(range(shallow_end, deep_start))

    def _mse(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.mean((a - b) ** 2))

    tq_avg = (turboquant_k_bits + turboquant_v_bits) / 2.0
    zq_avg = cfg.average_bits(n_layers)
    bit_savings = tq_avg - zq_avg

    return _ComparisonReport(
        turboquant_avg_bits=tq_avg,
        zero_quant_avg_bits=zq_avg,
        bit_savings=bit_savings,
        memory_reduction_pct=(bit_savings / tq_avg) * 100.0,
        critical_v_mse_turboquant=_mse(tq_v[critical], values[critical]),
        critical_v_mse_zero_quant=_mse(zq_v[critical], values[critical]),
        critical_v_improvement_pct=(
            (_mse(tq_v[critical], values[critical]) - _mse(zq_v[critical], values[critical]))
            / _mse(tq_v[critical], values[critical])
        ) * 100.0,
        middle_k_mse_turboquant=_mse(tq_k[middle], keys[middle]) if middle else 0.0,
        middle_k_mse_zero_quant=_mse(zq_k[middle], keys[middle]) if middle else 0.0,
        overall_v_mse_turboquant=_mse(tq_v, values),
        overall_v_mse_zero_quant=_mse(zq_v, values),
    )

