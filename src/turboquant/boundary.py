"""Boundary layer protection for TurboQuant KV cache compression.

Community finding: First 2 + last 2 transformer layers carry disproportionate
importance for output quality. Keeping them at q8_0 (8-bit) recovers 37–91%
of the quality gap from aggressive quantization, at minimal memory cost.

First layers: encode input representation / positional information
Last layers: generate output token distribution
Middle layers: more redundant, tolerate aggressive compression
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.turboquant.compressor import CompressedKV, QuantConfig, TurboQuantCompressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_BITS: frozenset[int] = frozenset({2, 3, 4, 8})

_BITS_NAME: dict[int, str] = {
    8: "q8_0",
    4: "turbo4",
    3: "turbo3",
    2: "turbo2",
}


def _format_bits(bits: int) -> str:
    """Human-readable name for a bit-width."""
    return _BITS_NAME.get(bits, f"q{bits}")


# ---------------------------------------------------------------------------
# BoundaryConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryConfig:
    """Configuration for boundary layer protection.

    Boundary layers (first N and last N) are kept at higher precision to
    preserve output quality, regardless of the global quantization config.

    Attributes:
        first_n: Number of initial layers to protect.  Must be >= 0.
        last_n: Number of final layers to protect.  Must be >= 0.
        protected_k_bits: Bit-width for protected layer keys.
        protected_v_bits: Bit-width for protected layer values.
    """

    first_n: int = 2
    last_n: int = 2
    protected_k_bits: int = 8
    protected_v_bits: int = 8

    def __post_init__(self) -> None:
        if self.first_n < 0:
            raise ValueError(f"first_n must be >= 0, got {self.first_n}")
        if self.last_n < 0:
            raise ValueError(f"last_n must be >= 0, got {self.last_n}")
        if self.protected_k_bits not in VALID_BITS:
            raise ValueError(
                f"protected_k_bits must be one of {sorted(VALID_BITS)}, "
                f"got {self.protected_k_bits}"
            )
        if self.protected_v_bits not in VALID_BITS:
            raise ValueError(
                f"protected_v_bits must be one of {sorted(VALID_BITS)}, "
                f"got {self.protected_v_bits}"
            )


# ---------------------------------------------------------------------------
# BoundaryAwareCompressor
# ---------------------------------------------------------------------------


class BoundaryAwareCompressor:
    """KV cache compressor with boundary layer protection.

    Wraps :class:`TurboQuantCompressor` to apply different quantization
    configs to boundary layers (first/last) vs middle layers.

    Example for 28-layer model with default boundary config::

        Layers 0, 1       (first 2):   K=q8_0, V=q8_0
        Layers 2–25       (middle 24): K=q8_0, V=turbo4  (from global config)
        Layers 26, 27     (last 2):    K=q8_0, V=q8_0

    Thread-safety
    -------------
    Compressor instances are created once in ``__init__`` and reused.
    All compression/decompression calls are stateless — no mutable state is
    shared or accumulated between calls.
    """

    __slots__ = (
        "_global_config",
        "_boundary_config",
        "_protected_config",
        "_global_compressor",
        "_protected_compressor",
    )

    def __init__(
        self,
        global_config: QuantConfig,
        boundary_config: BoundaryConfig | None = None,
    ) -> None:
        """
        Args:
            global_config: Default quantization config for middle layers.
            boundary_config: Override for boundary layers.  Defaults to
                first 2 + last 2 at q8_0/q8_0.
        """
        self._global_config = global_config
        self._boundary_config = boundary_config or BoundaryConfig()

        # Build the protected config, inheriting block_size and seed from
        # the global config so that block alignment and rotation seeds stay
        # consistent across the full model.
        self._protected_config = QuantConfig(
            k_bits=self._boundary_config.protected_k_bits,
            v_bits=self._boundary_config.protected_v_bits,
            block_size=global_config.block_size,
            seed=global_config.seed,
        )

        # Pre-create one compressor per unique config (at most two).
        self._global_compressor = TurboQuantCompressor(self._global_config)
        if self._protected_config == self._global_config:
            self._protected_compressor = self._global_compressor
        else:
            self._protected_compressor = TurboQuantCompressor(
                self._protected_config,
            )

    # -- properties --------------------------------------------------------

    @property
    def global_config(self) -> QuantConfig:
        """Quantization config applied to middle (non-boundary) layers."""
        return self._global_config

    @property
    def boundary_config(self) -> BoundaryConfig:
        """Boundary protection settings."""
        return self._boundary_config

    # -- layer classification ---------------------------------------------

    def _validate_layer_args(self, layer_idx: int, n_layers: int) -> None:
        """Raise :class:`ValueError` for invalid layer arguments."""
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {n_layers})"
            )

    def is_boundary_layer(self, layer_idx: int, n_layers: int) -> bool:
        """Check if a layer is a boundary layer.

        A layer is *boundary* when it falls within the first ``first_n``
        layers **or** the last ``last_n`` layers.  When
        ``first_n + last_n >= n_layers``, **all** layers are boundary
        (overlap is handled correctly via logical OR).

        Raises:
            ValueError: If *layer_idx* is out of ``[0, n_layers)``.
        """
        self._validate_layer_args(layer_idx, n_layers)
        bc = self._boundary_config
        return layer_idx < bc.first_n or layer_idx >= n_layers - bc.last_n

    def get_layer_config(self, layer_idx: int, n_layers: int) -> QuantConfig:
        """Return the quantization config for a specific layer.

        - If ``layer_idx < first_n``: return protected config
        - If ``layer_idx >= n_layers - last_n``: return protected config
        - Otherwise: return global config

        Raises:
            ValueError: If *layer_idx* is out of ``[0, n_layers)``.
        """
        self._validate_layer_args(layer_idx, n_layers)
        if self.is_boundary_layer(layer_idx, n_layers):
            return self._protected_config
        return self._global_config

    # -- internal helpers -------------------------------------------------

    def _get_compressor(self, config: QuantConfig) -> TurboQuantCompressor:
        """Return the pre-created compressor for *config*."""
        if config == self._protected_config:
            return self._protected_compressor
        return self._global_compressor

    @staticmethod
    def _layer_seed(base_seed: int, layer_idx: int) -> tuple[int, int]:
        """Compute (k_seed, v_seed) for a layer.

        Follows the same seed schedule as TurboQuantCompressor:
            K seed = base_seed + layer_idx * 1000
            V seed = base_seed + layer_idx * 1000 + 500
        """
        k_seed = base_seed + layer_idx * 1000
        v_seed = base_seed + layer_idx * 1000 + 500
        return k_seed, v_seed

    # -- compression / decompression --------------------------------------

    def compress_kv(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> CompressedKV:
        """Compress KV cache with boundary layer protection.

        Iterates over layers, applying the protected config to boundary
        layers and the global config to middle layers.  Uses
        :meth:`TurboQuantCompressor.compress_layer` per layer.

        Args:
            keys: Shape ``(n_layers, n_heads, seq_len, head_dim)``.
            values: Shape ``(n_layers, n_heads, seq_len, head_dim)``.

        Returns:
            :class:`CompressedKV` with per-layer compressed tensors using
            mixed quantization configs.

        Raises:
            ValueError: If shapes are mismatched or not 4-D.
        """
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have the same shape, "
                f"got {keys.shape} vs {values.shape}"
            )
        if keys.ndim != 4:
            raise ValueError(
                f"Expected 4D arrays (n_layers, n_heads, seq_len, head_dim), "
                f"got {keys.ndim}D"
            )

        n_layers = keys.shape[0]
        compressed_keys = []
        compressed_values = []

        for layer_idx in range(n_layers):
            config = self.get_layer_config(layer_idx, n_layers)
            compressor = self._get_compressor(config)
            k_seed, v_seed = self._layer_seed(config.seed, layer_idx)

            compressed_keys.append(
                compressor.compress_layer(keys[layer_idx], config.k_bits, k_seed)
            )
            compressed_values.append(
                compressor.compress_layer(values[layer_idx], config.v_bits, v_seed)
            )

        return CompressedKV(
            keys=tuple(compressed_keys),
            values=tuple(compressed_values),
            config=self._global_config,
            n_layers=n_layers,
            original_k_shape=tuple(keys.shape),
            original_v_shape=tuple(values.shape),
        )

    def decompress_kv(
        self,
        compressed: CompressedKV,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Decompress full KV cache.

        Each layer is decompressed with the correct bit-width derived
        from the ``n_bits`` metadata stored in each :class:`CompressedTensor`.

        Returns:
            ``(keys, values)`` arrays of shape
            ``(n_layers, n_heads, seq_len, head_dim)``.
        """
        n_layers = compressed.n_layers

        if n_layers == 0:
            k_shape = compressed.original_k_shape
            v_shape = compressed.original_v_shape
            return (
                np.empty(k_shape, dtype=np.float64),
                np.empty(v_shape, dtype=np.float64),
            )

        keys_list: list[NDArray[np.float64]] = []
        values_list: list[NDArray[np.float64]] = []

        for layer_idx in range(n_layers):
            config = self.get_layer_config(layer_idx, n_layers)
            compressor = self._get_compressor(config)

            # Use n_bits from the compressed tensor for correct dispatch.
            k_tensor = compressed.keys[layer_idx]
            v_tensor = compressed.values[layer_idx]

            keys_list.append(
                compressor.decompress_layer(k_tensor, k_tensor.n_bits)
            )
            values_list.append(
                compressor.decompress_layer(v_tensor, v_tensor.n_bits)
            )

        return np.stack(keys_list, axis=0), np.stack(values_list, axis=0)

    # -- memory estimation ------------------------------------------------

    def estimate_memory(
        self,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> dict[str, object]:
        """Estimate memory with boundary layer overhead.

        Uses the same bits-per-element model as
        :meth:`TurboQuantCompressor.estimate_memory`:

        - 8-bit: 8 bits/element (simple uniform, no block overhead)
        - 2/3/4-bit: ``n_bits + 16 / block_size`` bits/element

        Returns:
            Dictionary with keys:

            - ``total_bytes`` — total compressed size (float)
            - ``boundary_bytes`` — size of boundary layers (float)
            - ``middle_bytes`` — size of middle layers (float)
            - ``compression_ratio`` — vs f16 baseline (float)
            - ``n_boundary_layers`` — count of protected layers (int)
            - ``n_middle_layers`` — count of standard layers (int)
        """
        if n_layers <= 0:
            return {
                "total_bytes": 0.0,
                "boundary_bytes": 0.0,
                "middle_bytes": 0.0,
                "compression_ratio": float("inf"),
                "n_boundary_layers": 0,
                "n_middle_layers": 0,
            }

        elements_per_layer = n_heads * seq_len * head_dim
        block_size = self._global_config.block_size

        def _bits_per_element(n_bits: int) -> float:
            """Effective bits per element including overhead."""
            if n_bits == 8:
                return 8.0
            return n_bits + 16.0 / block_size

        def _layer_bytes(config: QuantConfig) -> float:
            """Compressed byte count for one layer (keys + values)."""
            k_bpe = _bits_per_element(config.k_bits)
            v_bpe = _bits_per_element(config.v_bits)
            return elements_per_layer * (k_bpe + v_bpe) / 8.0

        n_boundary = 0
        n_middle = 0
        boundary_bytes = 0.0
        middle_bytes = 0.0

        for layer_idx in range(n_layers):
            if self.is_boundary_layer(layer_idx, n_layers):
                n_boundary += 1
                boundary_bytes += _layer_bytes(self._protected_config)
            else:
                n_middle += 1
                middle_bytes += _layer_bytes(self._global_config)

        total_bytes = boundary_bytes + middle_bytes

        # f16 baseline: 2 bytes per element for both K and V
        f16_bytes = float(2 * n_layers * elements_per_layer * 2)
        compression_ratio = (
            f16_bytes / total_bytes if total_bytes > 0 else float("inf")
        )

        return {
            "total_bytes": total_bytes,
            "boundary_bytes": boundary_bytes,
            "middle_bytes": middle_bytes,
            "compression_ratio": compression_ratio,
            "n_boundary_layers": n_boundary,
            "n_middle_layers": n_middle,
        }

    # -- human-readable description ---------------------------------------

    def describe(self, n_layers: int) -> str:
        """Human-readable description of layer allocation.

        Example output::

            28 layers: [0,1] q8_0/q8_0 | [2-25] q8_0/turbo4 | [26,27] q8_0/q8_0
        """
        if n_layers <= 0:
            return "0 layers: (empty)"

        bc = self._boundary_config
        prot_label = (
            f"{_format_bits(self._protected_config.k_bits)}/"
            f"{_format_bits(self._protected_config.v_bits)}"
        )
        glob_label = (
            f"{_format_bits(self._global_config.k_bits)}/"
            f"{_format_bits(self._global_config.v_bits)}"
        )

        # Classify layers into three disjoint segments:
        #   first_boundary:  indices in [0, first_n)
        #   last_boundary:   indices in [max(first_n, n_layers-last_n), n_layers)
        #   middle:          everything else (non-boundary)
        first_boundary = [i for i in range(n_layers) if i < bc.first_n]
        last_boundary = [
            i
            for i in range(n_layers)
            if i >= n_layers - bc.last_n and i >= bc.first_n
        ]
        middle = [
            i
            for i in range(n_layers)
            if not self.is_boundary_layer(i, n_layers)
        ]

        parts: list[str] = []

        if first_boundary:
            parts.append(
                f"{_format_layer_indices(first_boundary)} {prot_label}"
            )
        if middle:
            parts.append(
                f"{_format_layer_indices(middle)} {glob_label}"
            )
        if last_boundary:
            parts.append(
                f"{_format_layer_indices(last_boundary)} {prot_label}"
            )

        return f"{n_layers} layers: {' | '.join(parts)}"


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _format_layer_indices(indices: list[int]) -> str:
    """Format a contiguous list of layer indices for :meth:`describe`.

    Short lists (≤ 3 elements) use comma notation ``[0,1,2]``.
    Longer contiguous ranges use dash notation ``[4-25]``.
    """
    if not indices:
        return "[]"
    if len(indices) == 1:
        return f"[{indices[0]}]"
    is_contiguous = indices == list(range(indices[0], indices[-1] + 1))
    if is_contiguous and len(indices) > 3:
        return f"[{indices[0]}-{indices[-1]}]"
    return f"[{','.join(str(i) for i in indices)}]"
