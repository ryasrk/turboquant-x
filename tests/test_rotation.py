"""Tests for Walsh-Hadamard Transform rotation module."""

import numpy as np
import pytest
from scipy import stats
from unittest.mock import patch

from src.turboquant.rotation import (
    hadamard_matrix,
    inverse_rotate,
    random_sign_diagonal,
    rotate,
)


# ---------------------------------------------------------------------------
# Hadamard matrix
# ---------------------------------------------------------------------------
class TestHadamardMatrix:
    @pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64, 128])
    def test_valid_sizes(self, d: int) -> None:
        H = hadamard_matrix(d)
        assert H.shape == (d, d)
        assert H.dtype == np.float64

    @pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64, 128])
    def test_orthonormality(self, d: int) -> None:
        """H @ H.T must equal the identity (normalized Hadamard)."""
        H = hadamard_matrix(d)
        np.testing.assert_allclose(H @ H.T, np.eye(d), atol=1e-10)

    @pytest.mark.parametrize("d", [2, 4, 8, 16, 64])
    def test_symmetry(self, d: int) -> None:
        """Sylvester-type Hadamard matrices are symmetric."""
        H = hadamard_matrix(d)
        np.testing.assert_allclose(H, H.T, atol=1e-10)

    @pytest.mark.parametrize("d", [3, 5, 7, 0, -1])
    def test_invalid_size_raises(self, d: int) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_matrix(d)


# ---------------------------------------------------------------------------
# Random sign diagonal
# ---------------------------------------------------------------------------
class TestRandomSignDiagonal:
    def test_all_values_pm1(self) -> None:
        signs = random_sign_diagonal(64, seed=0)
        assert set(np.unique(signs)).issubset({-1.0, 1.0})

    def test_deterministic_same_seed(self) -> None:
        a = random_sign_diagonal(32, seed=42)
        b = random_sign_diagonal(32, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = random_sign_diagonal(64, seed=0)
        b = random_sign_diagonal(64, seed=1)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Rotation round-trip and properties
# ---------------------------------------------------------------------------
class TestRotation:
    def test_invertibility(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.standard_normal(128)
        recovered = inverse_rotate(rotate(x, seed=7), seed=7)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_norm_preservation(self) -> None:
        """Orthogonal transforms preserve the L2 norm."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal(64)
        y = rotate(x, seed=3)
        np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x), atol=1e-10)

    def test_distribution_after_rotation(self) -> None:
        """Rotated coordinates should have mean ≈ 0 and var ≈ ||x||²/d.

        For a dense input vector, the randomized Hadamard transform
        produces entries that are approximately N(0, ||x||²/d).
        We use the Kolmogorov-Smirnov test at α = 0.001 on a dense input
        (the Gaussian CLT approximation requires many non-zero entries).
        """
        d = 1024
        rng = np.random.default_rng(123)
        x = rng.standard_normal(d) * 5.0  # dense vector with varied magnitudes

        y = rotate(x, seed=55)
        expected_std = np.linalg.norm(x) / np.sqrt(d)

        # Verify mean ≈ 0 and std ≈ expected_std
        assert abs(np.mean(y)) < 3 * expected_std / np.sqrt(d)
        np.testing.assert_allclose(np.std(y), expected_std, rtol=0.15)

        # KS test against normal distribution
        _, p_value = stats.kstest(y, "norm", args=(0, expected_std))
        assert p_value > 0.001, f"KS test failed with p={p_value:.4f}"

    def test_determinism(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        y1 = rotate(x, seed=10)
        y2 = rotate(x, seed=10)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_results(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        y1 = rotate(x, seed=10)
        y2 = rotate(x, seed=11)
        assert not np.array_equal(y1, y2)

    def test_batched(self) -> None:
        """Shape (batch, d) must work and be invertible per-row."""
        rng = np.random.default_rng(77)
        batch, d = 8, 64
        x = rng.standard_normal((batch, d))
        y = rotate(x, seed=5)
        assert y.shape == (batch, d)

        recovered = inverse_rotate(y, seed=5)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_batched_3d(self) -> None:
        """Arbitrary leading dimensions (..., d) should also work."""
        rng = np.random.default_rng(88)
        x = rng.standard_normal((2, 4, 32))
        y = rotate(x, seed=1)
        assert y.shape == x.shape

        recovered = inverse_rotate(y, seed=1)
        np.testing.assert_allclose(recovered, x, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_d2_smallest_valid(self) -> None:
        x = np.array([3.0, 4.0])
        y = rotate(x, seed=0)
        recovered = inverse_rotate(y, seed=0)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_zero_vector(self) -> None:
        x = np.zeros(16)
        y = rotate(x, seed=0)
        np.testing.assert_allclose(y, np.zeros(16), atol=1e-15)

    def test_large_d256(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        y = rotate(x, seed=99)
        np.testing.assert_allclose(
            np.linalg.norm(y), np.linalg.norm(x), atol=1e-10
        )
        recovered = inverse_rotate(y, seed=99)
        np.testing.assert_allclose(recovered, x, atol=1e-10)


class TestSylvesterFallback:
    """Test hadamard_matrix Sylvester fallback (lines 48-51)."""

    def test_sylvester_hadamard_without_scipy(self) -> None:
        """When scipy.linalg is unavailable, Sylvester construction is used."""
        import importlib
        import src.turboquant.rotation as rot_module

        # Make 'from scipy.linalg import hadamard' raise ImportError
        with patch.dict("sys.modules", {"scipy.linalg": None}):
            H = rot_module.hadamard_matrix(8)
            assert H.shape == (8, 8)
            # Verify orthogonality: H @ H.T ≈ I
            np.testing.assert_allclose(H @ H.T, np.eye(8), atol=1e-12)

    def test_sylvester_hadamard_d1(self) -> None:
        """Test _sylvester_hadamard base case: d=1 returns [[1.0]]."""
        from src.turboquant.rotation import _sylvester_hadamard

        H = _sylvester_hadamard(1)
        assert H.shape == (1, 1)
        assert H[0, 0] == 1.0

    def test_sylvester_hadamard_d4(self) -> None:
        """Test _sylvester_hadamard recursive case: d=4."""
        from src.turboquant.rotation import _sylvester_hadamard

        H = _sylvester_hadamard(4)
        assert H.shape == (4, 4)
        # Un-normalized Hadamard: H @ H.T = d * I
        np.testing.assert_allclose(H @ H.T, 4 * np.eye(4), atol=1e-12)
