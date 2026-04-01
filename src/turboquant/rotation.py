"""Walsh-Hadamard Transform with random sign flips for TurboQuant rotation.

The WHT-based random rotation spreads outlier magnitudes uniformly across all
coordinates, making every entry approximately N(0, ||x||²/d).  This is the
key pre-processing step that enables near-optimal scalar quantization in the
PolarQuant pipeline.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray


def _validate_power_of_two(d: int) -> None:
    """Raise ValueError if *d* is not a positive power of 2."""
    if not isinstance(d, (int, np.integer)) or d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(
            f"d must be a positive power of 2, got {d}"
        )


@functools.lru_cache(maxsize=32)
def hadamard_matrix(d: int) -> NDArray[np.float64]:
    """Construct a normalized Hadamard matrix of size d×d.

    *d* must be a power of 2.  Uses :func:`scipy.linalg.hadamard` when
    available, otherwise falls back to the recursive Sylvester construction.

    The result is normalized by ``1 / sqrt(d)`` so that ``H @ H.T == I``.
    """
    _validate_power_of_two(d)

    try:
        from scipy.linalg import hadamard as _scipy_hadamard
        H = _scipy_hadamard(d).astype(np.float64)
    except ImportError:  # pragma: no cover – scipy optional
        H = _sylvester_hadamard(d)

    H /= np.sqrt(d)
    return H


def _sylvester_hadamard(d: int) -> NDArray[np.float64]:
    """Recursive Sylvester construction (un-normalized)."""
    if d == 1:
        return np.array([[1.0]])
    half = _sylvester_hadamard(d // 2)
    return np.block([[half, half], [half, -half]])


def random_sign_diagonal(d: int, seed: int) -> NDArray[np.float64]:
    """Generate a 1-D array of *d* random ±1 values.

    Uses a :class:`numpy.random.Generator` seeded with *seed* for full
    determinism.  Apply via element-wise multiplication rather than
    constructing a full diagonal matrix.
    """
    _validate_power_of_two(d)
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=d)


def rotate(x: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """Apply the random rotation  ``y = H @ diag(signs) @ x``  (normalized).

    After rotation, coordinates of any fixed vector become approximately
    ``N(0, ||x||² / d)`` — the key property that enables optimal scalar
    quantization in the TurboQuant paper.

    Parameters
    ----------
    x : array, shape ``(..., d)``
        Input tensor.  The last dimension *d* must be a power of 2.
    seed : int
        RNG seed for the sign-flip diagonal.

    Returns
    -------
    y : array, same shape as *x*
    """
    d = x.shape[-1]
    _validate_power_of_two(d)

    signs = random_sign_diagonal(d, seed)
    H = hadamard_matrix(d)

    # sign-flip then WHT: y = (x * signs) @ H.T  ≡  H @ diag(signs) @ x  per-row
    return (x * signs) @ H.T


def inverse_rotate(y: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """Inverse rotation  ``x = diag(signs) @ H^T @ y``.

    Since *H* is orthogonal and normalized, ``H^T = H^{-1}``.
    Since ``diag(signs)² = I``, the inverse of ``diag(signs)`` is itself.

    Parameters
    ----------
    y : array, shape ``(..., d)``
        Rotated tensor.
    seed : int
        Same seed used in :func:`rotate`.

    Returns
    -------
    x : array, same shape as *y*
    """
    d = y.shape[-1]
    _validate_power_of_two(d)

    signs = random_sign_diagonal(d, seed)
    H = hadamard_matrix(d)

    # inverse: x = (y @ H) * signs  ≡  diag(signs) @ H^T @ y  per-row
    return (y @ H) * signs
