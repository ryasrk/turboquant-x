"""TurboQuant KV cache compressor — MSE-only, no QJL.

Community research (6+ teams) proved QJL variance is amplified by softmax.
MSE-only reconstruction wins across all tested models and sequence lengths.

Design:
- Asymmetric K/V precision via QuantConfig (default K=8, V=4)
- 8-bit mode uses simple min-max uniform quantization (no PolarQuant overhead)
- 2/3/4-bit modes use the full PolarQuant pipeline (norm + rotation + codebook)
- Per-layer unique seeds ensure rotation diversity across the model
- All output dataclasses are frozen (immutable)
- Compressor is stateless — holds only config
"""

from __future__ import annotations

import logging
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numpy.typing import NDArray

from src.turboquant.polar_quant import (
    CompressedTensor,
    CompressedBlock,
    polar_quantize as _py_polar_quantize,
    polar_dequantize as _py_polar_dequantize,
    compression_ratio,
)

# Try C++ backend — fall back to pure Python if not built
try:
    from src.turboquant_cpp import (
        CPP_AVAILABLE,
        polar_quantize as _cpp_polar_quantize,
        polar_dequantize as _cpp_polar_dequantize,
        polar_quantize_f32 as _cpp_polar_quantize_f32,
        polar_dequantize_f32 as _cpp_polar_dequantize_f32,
    )
except ImportError:
    CPP_AVAILABLE = False

_logger = logging.getLogger(__name__)
if CPP_AVAILABLE:
    _logger.info("Using C++ backend for PolarQuant compression")
else:
    _logger.info("C++ backend not available, using pure-Python PolarQuant")


# ------------------------------------------------------------------
# C++ ↔ Python dataclass adapters
# ------------------------------------------------------------------

def _cpp_compress_to_dataclass(
    tensor: NDArray[np.float64],
    n_bits: int,
    seed: int,
    block_size: int,
) -> CompressedTensor:
    """Compress via C++ zero-copy float32 path and convert dict to dataclasses.

    numpy astype(float32) is SIMD-accelerated (~10 GB/s), much faster than
    per-element conversion in the C++ bindings.
    """
    tensor_f32 = tensor.astype(np.float32, copy=False)
    result = _cpp_polar_quantize_f32(tensor_f32, n_bits=n_bits, seed=seed, block_size=block_size)
    blocks = tuple(
        CompressedBlock(
            indices=np.asarray(b["indices"], dtype=np.uint8),
            norm=np.float32(b["norm"]),
            n_bits=b["n_bits"],
            block_size=b["block_size"],
            seed=b["seed"],
        )
        for b in result["blocks"]
    )
    return CompressedTensor(
        blocks=blocks,
        original_shape=tuple(result["original_shape"]),
        n_bits=result["n_bits"],
        block_size=result["block_size"],
    )


def _cpp_decompress_from_dataclass(compressed: CompressedTensor) -> NDArray[np.float64]:
    """Decompress via C++ zero-copy float32 path, convert result to float64."""
    ct_dict = {
        "blocks": [
            {
                "indices": np.asarray(b.indices, dtype=np.uint8),
                "norm": float(b.norm),
                "n_bits": b.n_bits,
                "block_size": b.block_size,
                "seed": b.seed,
            }
            for b in compressed.blocks
        ],
        "original_shape": tuple(compressed.original_shape),
        "n_bits": compressed.n_bits,
        "block_size": compressed.block_size,
    }
    result_f32 = _cpp_polar_dequantize_f32(ct_dict)
    return result_f32.astype(np.float64)


@dataclass(frozen=True)
class QuantConfig:
    """Configuration for KV cache quantization."""

    k_bits: int = 8  # Key cache bit-width (2, 3, 4, or 8)
    v_bits: int = 4  # Value cache bit-width (2, 3, 4, or 8)
    block_size: int = 128  # Block size for PolarQuant
    seed: int = 42  # Base seed for rotations

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_bits = {2, 3, 4, 8}
        if self.k_bits not in valid_bits:
            raise ValueError(
                f"k_bits must be one of {valid_bits}, got {self.k_bits}"
            )
        if self.v_bits not in valid_bits:
            raise ValueError(
                f"v_bits must be one of {valid_bits}, got {self.v_bits}"
            )
        if self.block_size < 2 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be a power of 2 >= 2, got {self.block_size}"
            )


@dataclass(frozen=True)
class CompressedKV:
    """Compressed key-value cache for all layers."""

    keys: tuple[CompressedTensor, ...]  # One CompressedTensor per layer
    values: tuple[CompressedTensor, ...]  # One CompressedTensor per layer
    config: QuantConfig
    n_layers: int
    original_k_shape: tuple[int, ...]
    original_v_shape: tuple[int, ...]


class TurboQuantCompressor:
    """Stateless KV cache compressor using TurboQuant PolarQuant pipeline.

    Key design decisions:
    - MSE-only reconstruction (NO QJL — proven to hurt under softmax)
    - Asymmetric K/V precision supported via QuantConfig
    - 8-bit means no PolarQuant compression (simple min-max uniform quantization)
    - Per-layer compression with unique seeds
    """

    def __init__(self, config: QuantConfig | None = None) -> None:
        self._config = config or QuantConfig()

    @property
    def config(self) -> QuantConfig:
        return self._config

    # ------------------------------------------------------------------
    # Full KV cache operations
    # ------------------------------------------------------------------

    def compress_kv(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> CompressedKV:
        """Compress full KV cache.

        Args:
            keys: shape (n_layers, n_heads, seq_len, head_dim)
            values: shape (n_layers, n_heads, seq_len, head_dim)

        Returns:
            CompressedKV with per-layer compressed tensors.

        Seed schedule per layer:
            K seed = base_seed + layer_idx * 1000
            V seed = base_seed + layer_idx * 1000 + 500
        """
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "keys and values must be 4D: (n_layers, n_heads, seq_len, head_dim)"
            )

        n_layers = keys.shape[0]
        cfg = self._config

        # Parallel layer compression — only for large data where thread
        # pool overhead is amortized.  C++ already uses OpenMP internally
        # for block-level parallelism, so threading mainly helps with
        # Python glue overhead between layers.
        total_bytes = keys.nbytes + values.nbytes
        use_parallel = n_layers > 4 and total_bytes > 50_000_000  # 50 MB
        max_workers = min(os.cpu_count() or 4, n_layers, 8)

        def _compress_layer_pair(layer_idx: int) -> tuple[CompressedTensor, CompressedTensor]:
            k_seed = cfg.seed + layer_idx * 1000
            v_seed = cfg.seed + layer_idx * 1000 + 500
            k_comp = self.compress_layer(keys[layer_idx], cfg.k_bits, k_seed)
            v_comp = self.compress_layer(values[layer_idx], cfg.v_bits, v_seed)
            return k_comp, v_comp

        if use_parallel and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(_compress_layer_pair, range(n_layers)))
            compressed_keys = [r[0] for r in results]
            compressed_values = [r[1] for r in results]
        else:
            compressed_keys = []
            compressed_values = []
            for layer_idx in range(n_layers):
                k, v = _compress_layer_pair(layer_idx)
                compressed_keys.append(k)
                compressed_values.append(v)

        return CompressedKV(
            keys=tuple(compressed_keys),
            values=tuple(compressed_values),
            config=cfg,
            n_layers=n_layers,
            original_k_shape=tuple(keys.shape),
            original_v_shape=tuple(values.shape),
        )

    def decompress_kv(
        self, compressed: CompressedKV
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Decompress full KV cache back to original arrays."""
        n_layers = compressed.n_layers
        # Estimate size from original shapes for threshold
        k_elements = 1
        for s in compressed.original_k_shape:
            k_elements *= s
        total_bytes = k_elements * 8 * 2  # float64, K+V
        use_parallel = n_layers > 4 and total_bytes > 50_000_000
        max_workers = min(os.cpu_count() or 4, n_layers, 8)

        def _decompress_layer_pair(layer_idx: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            k = self.decompress_layer(compressed.keys[layer_idx], compressed.config.k_bits)
            v = self.decompress_layer(compressed.values[layer_idx], compressed.config.v_bits)
            return k, v

        if use_parallel and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(_decompress_layer_pair, range(n_layers)))
            keys = [r[0] for r in results]
            values = [r[1] for r in results]
        else:
            keys = []
            values = []
            for layer_idx in range(n_layers):
                k, v = _decompress_layer_pair(layer_idx)
                keys.append(k)
                values.append(v)

        return np.stack(keys, axis=0), np.stack(values, axis=0)

    # ------------------------------------------------------------------
    # Per-layer operations
    # ------------------------------------------------------------------

    def compress_layer(
        self,
        tensor: NDArray[np.float64],
        n_bits: int,
        layer_seed: int,
    ) -> CompressedTensor:
        """Compress a single layer's K or V tensor.

        If n_bits == 8, use simple 8-bit uniform quantization (no PolarQuant).
        If n_bits in {2, 3, 4}, use full PolarQuant pipeline.
        Uses C++ backend when available for 10-50x speedup.
        """
        if n_bits == 8:
            return self._quantize_8bit(tensor)

        if CPP_AVAILABLE:
            return _cpp_compress_to_dataclass(
                tensor, n_bits, layer_seed, self._config.block_size
            )
        return _py_polar_quantize(
            tensor, n_bits=n_bits, seed=layer_seed, block_size=self._config.block_size
        )

    def decompress_layer(
        self,
        compressed: CompressedTensor,
        n_bits: int,
    ) -> NDArray[np.float64]:
        """Decompress a single layer."""
        if n_bits == 8:
            return self._dequantize_8bit(compressed)

        if CPP_AVAILABLE:
            return _cpp_decompress_from_dataclass(compressed)
        return _py_polar_dequantize(compressed)

    # ------------------------------------------------------------------
    # 8-bit uniform quantization (q8_0 pass-through)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_8bit(x: NDArray[np.float64]) -> CompressedTensor:
        """Simple min-max 8-bit uniform quantization (for q8_0 mode).

        scale = (max - min) / 255
        indices = round((x - min) / scale)   [uint8 in 0..255]
        dequant = indices * scale + min

        Stored as two CompressedBlocks:
          block[0] — data indices, norm = scale
          block[1] — metadata (empty indices), norm = offset (x_min)
        """
        original_shape = x.shape
        flat = x.ravel().astype(np.float64)

        x_min = float(np.min(flat))
        x_max = float(np.max(flat))

        if x_max - x_min < 1e-30:
            # Constant tensor — avoid division by zero
            indices = np.zeros(len(flat), dtype=np.uint8)
            scale = 1.0
            x_min = float(np.mean(flat))
        else:
            scale = (x_max - x_min) / 255.0
            indices = np.clip(
                np.round((flat - x_min) / scale), 0, 255
            ).astype(np.uint8)

        data_block = CompressedBlock(
            indices=indices,
            norm=np.float64(scale),
            n_bits=8,
            block_size=len(flat),
            seed=0,
        )
        meta_block = CompressedBlock(
            indices=np.array([], dtype=np.uint8),
            norm=np.float64(x_min),
            n_bits=8,
            block_size=0,
            seed=0,
        )

        return CompressedTensor(
            blocks=(data_block, meta_block),
            original_shape=original_shape,
            n_bits=8,
            block_size=0,
        )

    @staticmethod
    def _dequantize_8bit(compressed: CompressedTensor) -> NDArray[np.float64]:
        """Dequantize 8-bit uniform."""
        data_block = compressed.blocks[0]
        meta_block = compressed.blocks[1]

        scale = float(data_block.norm)
        offset = float(meta_block.norm)

        flat = data_block.indices.astype(np.float64) * scale + offset
        return flat.reshape(compressed.original_shape)

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    def estimate_memory(
        self,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> dict[str, float]:
        """Estimate memory usage in bytes for compressed KV cache.

        Returns dict with:
        - 'k_bytes': key cache size
        - 'v_bytes': value cache size
        - 'total_bytes': total
        - 'compression_ratio': vs float16
        - 'original_bytes': float16 baseline
        """
        cfg = self._config
        elements_per_layer = n_heads * seq_len * head_dim

        # Float16 baseline: K + V each at 2 bytes per element
        original_bytes = float(2 * n_layers * elements_per_layer * 2)

        # Bits per element — 8-bit has no block overhead
        if cfg.k_bits == 8:
            k_bpe = 8.0
        else:
            k_bpe = cfg.k_bits + 16.0 / cfg.block_size

        if cfg.v_bits == 8:
            v_bpe = 8.0
        else:
            v_bpe = cfg.v_bits + 16.0 / cfg.block_size

        k_bytes = float(n_layers * elements_per_layer * k_bpe / 8.0)
        v_bytes = float(n_layers * elements_per_layer * v_bpe / 8.0)
        total_bytes = k_bytes + v_bytes

        ratio = original_bytes / total_bytes if total_bytes > 0 else float("inf")

        return {
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "total_bytes": total_bytes,
            "compression_ratio": ratio,
            "original_bytes": original_bytes,
        }
