"""PolarQuant compression pipeline for TurboQuant KV cache.

Implements the full PolarQuant pipeline:
  norm extraction → rotation → scalar quantization → packing

Polar decomposition separates magnitude (norm) from direction (unit vector).
The norm is stored at full float32 precision, while the direction is rotated
via Walsh-Hadamard transform and then quantized with a Lloyd-Max codebook
optimised for Gaussian data.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from src.turboquant.rotation import rotate, inverse_rotate
from src.turboquant.codebook import get_codebook, quantize_scalar, dequantize_scalar

_SUPPORTED_BITS = frozenset({2, 3, 4})


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= *n*."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class CompressedBlock:
    """Single compressed block of a KV cache vector.

    Attributes:
        indices:    Quantized centroid indices (uint8).
                    Length equals the padded (power-of-2) block dimension.
        norm:       L2 norm of the original block, stored as float32.
        n_bits:     Quantization bit-width (2, 3, or 4).
        block_size: Number of elements in the *original* (unpadded) block.
        seed:       Rotation seed used for this block.
    """

    indices: NDArray[np.uint8]
    norm: np.float32
    n_bits: int
    block_size: int
    seed: int


@dataclass(frozen=True)
class CompressedTensor:
    """Collection of compressed blocks representing a full tensor.

    Attributes:
        blocks:         Immutable tuple of :class:`CompressedBlock` instances.
        original_shape: Shape of the tensor before compression.
        n_bits:         Quantization bit-width.
        block_size:     Nominal block size used during compression.
    """

    blocks: tuple[CompressedBlock, ...]
    original_shape: tuple[int, ...]
    n_bits: int
    block_size: int


# ---------------------------------------------------------------------------
# Block-level compress / decompress
# ---------------------------------------------------------------------------


def polar_quantize_block(
    x: NDArray[np.float64],
    n_bits: int,
    seed: int,
) -> CompressedBlock:
    """Compress a single block using PolarQuant.

    Pipeline:
      1. Extract norm: γ = ||x||₂
      2. Normalize:    x̂ = x / γ   (unit vector — "polar" decomposition)
      3. Pad to next power of 2 for WHT (if needed)
      4. Rotate:       ŷ = rotate(x̂, seed)  — entries ≈ N(0, 1/d)
      5. Scale:        z = ŷ · √d            — entries ≈ N(0, 1)
      6. Quantize:     indices = quantize_scalar(z, codebook)

    Zero vectors are handled gracefully: norm=0 and indices map to the
    centroid nearest 0.

    Parameters
    ----------
    x : 1-D float array
        Block data to compress.
    n_bits : int
        Quantization bit-width (2, 3, or 4).
    seed : int
        RNG seed for the Walsh-Hadamard rotation.

    Returns
    -------
    CompressedBlock
    """
    if n_bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"n_bits must be one of {sorted(_SUPPORTED_BITS)}, got {n_bits}"
        )

    x = np.asarray(x, dtype=np.float64).ravel()
    original_len = len(x)
    if original_len == 0:
        raise ValueError("Cannot compress an empty block")

    codebook = get_codebook(n_bits)
    padded_len = _next_power_of_two(original_len)

    # 1. Extract L2 norm
    norm_val = np.float32(np.linalg.norm(x))

    # 2. Zero-vector fast-path
    if norm_val == 0.0:
        indices = quantize_scalar(np.zeros(padded_len), codebook)
        return CompressedBlock(
            indices=indices,
            norm=norm_val,
            n_bits=n_bits,
            block_size=original_len,
            seed=seed,
        )

    # 3. Normalize to unit vector
    x_hat = x / np.float64(norm_val)

    # 4. Pad to next power of 2 for WHT
    if padded_len != original_len:
        x_hat = np.pad(x_hat, (0, padded_len - original_len))

    # 5. Rotate (Walsh-Hadamard + random sign flip)
    y = rotate(x_hat, seed)

    # 6. Scale: rotated unit vector entries ≈ N(0, 1/d), multiply by √d → N(0, 1)
    z = y * np.sqrt(padded_len)

    # 7. Quantize with Lloyd-Max codebook
    indices = quantize_scalar(z, codebook)

    return CompressedBlock(
        indices=indices,
        norm=norm_val,
        n_bits=n_bits,
        block_size=original_len,
        seed=seed,
    )


def polar_dequantize_block(block: CompressedBlock) -> NDArray[np.float64]:
    """Decompress a single :class:`CompressedBlock`.

    Pipeline:
      1. Dequantize:      z  = dequantize_scalar(indices, codebook)
      2. Unscale:          ŷ  = z / √d
      3. Inverse rotate:   x̂  = inverse_rotate(ŷ, seed)
      4. Trim padding:     x̂  = x̂[:block_size]
      5. Rescale:          x  = x̂ · norm

    Returns
    -------
    NDArray[np.float64]
        Reconstructed block of shape ``(block_size,)``.
    """
    # Zero-vector fast-path
    if block.norm == 0.0:
        return np.zeros(block.block_size, dtype=np.float64)

    codebook = get_codebook(block.n_bits)
    d = len(block.indices)

    # 1. Dequantize
    z = dequantize_scalar(block.indices, codebook)

    # 2. Unscale
    y_hat = z / np.sqrt(d)

    # 3. Inverse rotate
    x_hat = inverse_rotate(y_hat, block.seed)

    # 4. Trim padding
    x_hat = x_hat[: block.block_size]

    # 5. Rescale by norm
    return x_hat * np.float64(block.norm)


# ---------------------------------------------------------------------------
# Tensor-level compress / decompress
# ---------------------------------------------------------------------------


def polar_quantize(
    x: NDArray[np.float64],
    n_bits: int,
    seed: int = 42,
    block_size: int = 128,
) -> CompressedTensor:
    """Compress a full tensor using PolarQuant with blocking.

    The tensor is flattened and split into blocks of ``block_size`` elements.
    Each block is compressed independently with a unique seed
    (``seed_i = seed + block_index``).  The last block may be smaller than
    ``block_size`` and will be padded internally to the next power of 2 for
    the Walsh-Hadamard transform.

    Parameters
    ----------
    x : array
        Tensor of any shape.
    n_bits : int
        Quantization bit-width (2, 3, or 4).
    seed : int
        Base RNG seed.  Block *i* uses ``seed + i``.
    block_size : int
        Number of elements per block (default 128).

    Returns
    -------
    CompressedTensor
    """
    if n_bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"n_bits must be one of {sorted(_SUPPORTED_BITS)}, got {n_bits}"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    flat = x.ravel()
    n = len(flat)

    n_blocks = (n + block_size - 1) // block_size
    blocks: list[CompressedBlock] = []

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        blocks.append(polar_quantize_block(flat[start:end], n_bits, seed + i))

    return CompressedTensor(
        blocks=tuple(blocks),
        original_shape=original_shape,
        n_bits=n_bits,
        block_size=block_size,
    )


def polar_dequantize(compressed: CompressedTensor) -> NDArray[np.float64]:
    """Decompress a :class:`CompressedTensor` back to its original shape.

    Returns
    -------
    NDArray[np.float64]
        Tensor with the same shape as the original input.
    """
    parts = [polar_dequantize_block(b) for b in compressed.blocks]
    flat = np.concatenate(parts)
    return flat.reshape(compressed.original_shape)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compression_ratio(n_bits: int, block_size: int) -> float:
    """Calculate the compression ratio versus float16 storage.

    Original cost:    16 bits per value (float16).
    Compressed cost:  *n_bits* per value + 32-bit float32 norm per block.

    .. math::

        \\text{ratio} = \\frac{16}{n\\_bits + 32 / \\text{block\\_size}}

    Parameters
    ----------
    n_bits : int
        Quantization bit-width.
    block_size : int
        Number of elements per block.

    Returns
    -------
    float
        Compression ratio (e.g. ≈3.76× for turbo4 with block_size=128).
    """
    return 16.0 / (n_bits + 32.0 / block_size)
