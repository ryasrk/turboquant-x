"""Lloyd-Max optimal scalar quantization for TurboQuant.

After Walsh-Hadamard rotation, KV cache values follow a Gaussian N(0, σ²)
distribution. Lloyd-Max codebooks are provably optimal for this case,
minimizing mean squared error for a given number of quantization levels.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class Codebook:
    """Immutable quantization codebook with centroids and decision boundaries.

    Attributes:
        n_bits: Number of bits per quantized value.
        centroids: Sorted centroid values, shape (2^n_bits,).
        boundaries: Decision boundaries between adjacent centroids,
                     shape (2^n_bits - 1,).
    """

    n_bits: int
    centroids: NDArray[np.float64]
    boundaries: NDArray[np.float64]

    def __post_init__(self) -> None:
        n_levels = 1 << self.n_bits
        if self.centroids.shape != (n_levels,):
            raise ValueError(
                f"Expected {n_levels} centroids for {self.n_bits}-bit, "
                f"got shape {self.centroids.shape}"
            )
        if self.boundaries.shape != (n_levels - 1,):
            raise ValueError(
                f"Expected {n_levels - 1} boundaries for {self.n_bits}-bit, "
                f"got shape {self.boundaries.shape}"
            )


def lloyd_max_codebook(
    n_bits: int, n_iterations: int = 50, tol: float = 1e-8
) -> Codebook:
    """Compute Lloyd-Max optimal codebook for N(0,1) distribution.

    Uses the iterative Lloyd-Max algorithm:
      1. Initialize centroids uniformly across [-3, 3].
      2. Update boundaries: midpoints between adjacent centroids.
      3. Update centroids: conditional expectation E[X | b_i < X < b_{i+1}]
         using φ(b_i) - φ(b_{i+1}) / (Φ(b_{i+1}) - Φ(b_i)).
      4. Repeat until convergence (max centroid change < tol) or max iterations.

    Args:
        n_bits: Quantization bit-width (produces 2^n_bits levels).
        n_iterations: Maximum number of Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        A frozen Codebook with optimal centroids and boundaries.
    """
    if n_bits < 1 or n_bits > 8:
        raise ValueError(f"n_bits must be in [1, 8], got {n_bits}")

    n_levels = 1 << n_bits
    centroids = np.linspace(-3.0, 3.0, n_levels)

    for _ in range(n_iterations):
        # Boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Extended boundaries with ±∞ sentinels
        b = np.concatenate([[-np.inf], boundaries, [np.inf]])

        new_centroids = np.empty(n_levels, dtype=np.float64)
        for i in range(n_levels):
            lo, hi = b[i], b[i + 1]
            cdf_diff = norm.cdf(hi) - norm.cdf(lo)
            if cdf_diff < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0
            else:
                # Conditional expectation: E[X | lo < X < hi] for N(0,1)
                pdf_diff = norm.pdf(lo) - norm.pdf(hi)
                new_centroids[i] = pdf_diff / cdf_diff

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return Codebook(
        n_bits=n_bits,
        centroids=centroids,
        boundaries=boundaries,
    )


def quantize_scalar(
    x: NDArray[np.float64], codebook: Codebook
) -> NDArray[np.uint8]:
    """Quantize continuous values to nearest centroid index.

    Uses ``np.searchsorted`` on decision boundaries for O(n log k) performance.

    Args:
        x: Array of values to quantize.
        codebook: Codebook with decision boundaries and centroids.

    Returns:
        Array of uint8 centroid indices in [0, 2^n_bits).
    """
    indices = np.searchsorted(codebook.boundaries, x).astype(np.uint8)
    return indices


def dequantize_scalar(
    indices: NDArray[np.uint8], codebook: Codebook
) -> NDArray[np.float64]:
    """Reconstruct continuous values from centroid indices.

    Args:
        indices: uint8 array of centroid indices.
        codebook: Codebook containing centroid values.

    Returns:
        Array of reconstructed values (centroid look-up).
    """
    return codebook.centroids[indices]


# ---------------------------------------------------------------------------
# Precomputed Lloyd-Max codebooks for N(0,1) — TurboQuant paper Table 1
# Values computed via lloyd_max_codebook() with 200 iterations / tol=1e-12.
# ---------------------------------------------------------------------------

TURBO2_CODEBOOK = Codebook(
    n_bits=2,
    centroids=np.array(
        [-1.5104176085002023, -0.4527800346370284,
         0.4527800346370285,  1.5104176085002028],
        dtype=np.float64,
    ),
    boundaries=np.array(
        [-0.9815988215686153, 0.0, 0.9815988215686157],
        dtype=np.float64,
    ),
)

TURBO3_CODEBOOK = Codebook(
    n_bits=3,
    centroids=np.array(
        [-2.1519457045434560, -1.3439092785115518,
         -0.7560052812106792, -0.2450941789459803,
          0.2450941789459804,  0.7560052812106796,
          1.3439092785115505,  2.1519457045434582],
        dtype=np.float64,
    ),
    boundaries=np.array(
        [-1.7479274915275038, -1.0499572798611156,
         -0.5005497300783297,  0.0,
          0.5005497300783299,  1.0499572798611150,
          1.7479274915275043],
        dtype=np.float64,
    ),
)

TURBO4_CODEBOOK = Codebook(
    n_bits=4,
    centroids=np.array(
        [-2.7329730276314383, -2.0694504218818746,
         -1.6184883483594745, -1.2566476802931554,
         -0.9427007640381189, -0.6570370151078325,
         -0.3882235722082744, -0.1284549100577299,
          0.1284549100577301,  0.3882235722082740,
          0.6570370151078336,  0.9427007640381218,
          1.2566476802931565,  1.6184883483594787,
          2.0694504218818760,  2.7329730276314304],
        dtype=np.float64,
    ),
    boundaries=np.array(
        [-2.4012117247566565, -1.8439693851206744,
         -1.4375680143263150, -1.0996742221656373,
         -0.7998688895729758, -0.5226302936580535,
         -0.2583392411330021,  0.0,
          0.2583392411330021,  0.5226302936580538,
          0.7998688895729778,  1.0996742221656390,
          1.4375680143263176,  1.8439693851206773,
          2.4012117247566529],
        dtype=np.float64,
    ),
)

_PRECOMPUTED: dict[int, Codebook] = {
    2: TURBO2_CODEBOOK,
    3: TURBO3_CODEBOOK,
    4: TURBO4_CODEBOOK,
}


def get_codebook(n_bits: int) -> Codebook:
    """Return a precomputed Lloyd-Max codebook for the given bit-width.

    Args:
        n_bits: Quantization bit-width. Must be 2, 3, or 4.

    Returns:
        The corresponding precomputed Codebook.

    Raises:
        ValueError: If n_bits is not one of {2, 3, 4}.
    """
    try:
        return _PRECOMPUTED[n_bits]
    except KeyError:
        raise ValueError(
            f"No precomputed codebook for {n_bits}-bit. "
            f"Supported: {sorted(_PRECOMPUTED)}"
        ) from None
