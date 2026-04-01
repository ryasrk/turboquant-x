"""Tests for Lloyd-Max codebook and scalar quantization."""

from __future__ import annotations

import numpy as np
import pytest

from src.turboquant.codebook import (
    Codebook,
    lloyd_max_codebook,
    quantize_scalar,
    dequantize_scalar,
    get_codebook,
    TURBO2_CODEBOOK,
    TURBO3_CODEBOOK,
    TURBO4_CODEBOOK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip_mse(codebook: Codebook, n_samples: int = 10_000, seed: int = 42) -> float:
    """Quantize-dequantize N(0,1) samples and return MSE."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(np.float64)
    indices = quantize_scalar(x, codebook)
    recon = dequantize_scalar(indices, codebook)
    return float(np.mean((x - recon) ** 2))


# ---------------------------------------------------------------------------
# TestLloydMaxCodebook
# ---------------------------------------------------------------------------

class TestLloydMaxCodebook:
    """Verify the Lloyd-Max algorithm converges to optimal codebooks."""

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_convergence(self, n_bits: int) -> None:
        """Algorithm should converge without hitting max iterations."""
        cb = lloyd_max_codebook(n_bits, n_iterations=200, tol=1e-10)
        assert cb.n_bits == n_bits
        assert cb.centroids.shape == (1 << n_bits,)

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_centroids_sorted(self, n_bits: int) -> None:
        cb = lloyd_max_codebook(n_bits)
        assert np.all(np.diff(cb.centroids) > 0), "Centroids must be strictly ascending"

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_centroids_symmetric(self, n_bits: int) -> None:
        """Codebook for symmetric Gaussian must be symmetric around 0."""
        cb = lloyd_max_codebook(n_bits)
        n = len(cb.centroids)
        for i in range(n // 2):
            assert cb.centroids[i] == pytest.approx(-cb.centroids[n - 1 - i], abs=1e-6)

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_correct_num_centroids(self, n_bits: int) -> None:
        cb = lloyd_max_codebook(n_bits)
        assert len(cb.centroids) == 1 << n_bits

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_correct_num_boundaries(self, n_bits: int) -> None:
        cb = lloyd_max_codebook(n_bits)
        assert len(cb.boundaries) == (1 << n_bits) - 1

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_boundaries_between_centroids(self, n_bits: int) -> None:
        """Each boundary must lie strictly between its adjacent centroids."""
        cb = lloyd_max_codebook(n_bits)
        for i in range(len(cb.boundaries)):
            assert cb.centroids[i] < cb.boundaries[i] < cb.centroids[i + 1]


# ---------------------------------------------------------------------------
# TestQuantizeDequantize
# ---------------------------------------------------------------------------

class TestQuantizeDequantize:
    """Verify quantize → dequantize round-trip quality."""

    def test_roundtrip_mse_turbo4(self) -> None:
        mse = _roundtrip_mse(TURBO4_CODEBOOK)
        assert mse == pytest.approx(0.009497, rel=0.15)

    def test_roundtrip_mse_turbo3(self) -> None:
        mse = _roundtrip_mse(TURBO3_CODEBOOK)
        assert mse == pytest.approx(0.03454, rel=0.15)

    def test_roundtrip_mse_turbo2(self) -> None:
        mse = _roundtrip_mse(TURBO2_CODEBOOK)
        assert mse == pytest.approx(0.1175, rel=0.15)

    @pytest.mark.parametrize("codebook", [TURBO2_CODEBOOK, TURBO3_CODEBOOK, TURBO4_CODEBOOK])
    def test_indices_in_valid_range(self, codebook: Codebook) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000).astype(np.float64)
        indices = quantize_scalar(x, codebook)
        assert indices.dtype == np.uint8
        assert np.all(indices < (1 << codebook.n_bits))

    @pytest.mark.parametrize("codebook", [TURBO2_CODEBOOK, TURBO3_CODEBOOK, TURBO4_CODEBOOK])
    def test_dequantized_values_are_centroids(self, codebook: Codebook) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(500).astype(np.float64)
        indices = quantize_scalar(x, codebook)
        recon = dequantize_scalar(indices, codebook)
        for val in recon:
            assert val in codebook.centroids

    @pytest.mark.parametrize("codebook", [TURBO2_CODEBOOK, TURBO3_CODEBOOK, TURBO4_CODEBOOK])
    def test_monotonicity(self, codebook: Codebook) -> None:
        """Larger input values should produce larger or equal indices."""
        x = np.linspace(-4.0, 4.0, 200)
        indices = quantize_scalar(x, codebook)
        assert np.all(np.diff(indices.astype(np.int16)) >= 0)


# ---------------------------------------------------------------------------
# TestPrecomputedCodebooks
# ---------------------------------------------------------------------------

class TestPrecomputedCodebooks:
    """Verify the precomputed codebook constants."""

    def test_turbo2_num_centroids(self) -> None:
        assert len(TURBO2_CODEBOOK.centroids) == 4

    def test_turbo3_num_centroids(self) -> None:
        assert len(TURBO3_CODEBOOK.centroids) == 8

    def test_turbo4_num_centroids(self) -> None:
        assert len(TURBO4_CODEBOOK.centroids) == 16

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_get_codebook_returns_correct(self, n_bits: int) -> None:
        expected = {2: TURBO2_CODEBOOK, 3: TURBO3_CODEBOOK, 4: TURBO4_CODEBOOK}
        assert get_codebook(n_bits) is expected[n_bits]

    @pytest.mark.parametrize("bad_bits", [0, 1, 5, 8, -1])
    def test_get_codebook_invalid_raises(self, bad_bits: int) -> None:
        with pytest.raises(ValueError, match="No precomputed codebook"):
            get_codebook(bad_bits)

    @pytest.mark.parametrize("n_bits", [2, 3, 4])
    def test_precomputed_matches_lloyd_max(self, n_bits: int) -> None:
        """Precomputed values should match fresh Lloyd-Max output."""
        fresh = lloyd_max_codebook(n_bits, n_iterations=200, tol=1e-12)
        precomputed = get_codebook(n_bits)
        np.testing.assert_allclose(
            precomputed.centroids, fresh.centroids, atol=1e-4
        )
        np.testing.assert_allclose(
            precomputed.boundaries, fresh.boundaries, atol=1e-4
        )


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case inputs for quantize / dequantize."""

    def test_single_value(self) -> None:
        x = np.array([0.5], dtype=np.float64)
        idx = quantize_scalar(x, TURBO2_CODEBOOK)
        recon = dequantize_scalar(idx, TURBO2_CODEBOOK)
        assert idx.shape == (1,)
        assert recon.shape == (1,)

    def test_all_same_values(self) -> None:
        x = np.full(100, 0.0, dtype=np.float64)
        idx = quantize_scalar(x, TURBO3_CODEBOOK)
        recon = dequantize_scalar(idx, TURBO3_CODEBOOK)
        # All inputs identical → all indices and reconstructions identical
        assert np.all(idx == idx[0])
        assert np.all(recon == recon[0])

    def test_extreme_positive_maps_to_last_centroid(self) -> None:
        x = np.array([10.0], dtype=np.float64)
        idx = quantize_scalar(x, TURBO4_CODEBOOK)
        assert idx[0] == len(TURBO4_CODEBOOK.centroids) - 1

    def test_extreme_negative_maps_to_first_centroid(self) -> None:
        x = np.array([-10.0], dtype=np.float64)
        idx = quantize_scalar(x, TURBO4_CODEBOOK)
        assert idx[0] == 0

    def test_extreme_values_both(self) -> None:
        x = np.array([-10.0, 10.0], dtype=np.float64)
        idx = quantize_scalar(x, TURBO2_CODEBOOK)
        assert idx[0] == 0
        assert idx[1] == len(TURBO2_CODEBOOK.centroids) - 1


# ---------------------------------------------------------------------------
# Missing coverage tests
# ---------------------------------------------------------------------------


class TestCodebookValidation:
    """Test Codebook __post_init__ validation (lines 34, 39)."""

    def test_wrong_centroids_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="centroids"):
            Codebook(
                n_bits=2,
                centroids=np.array([1.0, 2.0, 3.0]),  # need 4 for 2-bit
                boundaries=np.array([1.5, 2.5, 3.5]),
            )

    def test_wrong_boundaries_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="boundaries"):
            Codebook(
                n_bits=2,
                centroids=np.array([1.0, 2.0, 3.0, 4.0]),
                boundaries=np.array([1.5, 2.5]),  # need 3 for 2-bit
            )


class TestLloydMaxEdgeCases:
    """Test lloyd_max_codebook edge cases (lines 66, 83)."""

    def test_n_bits_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be in"):
            lloyd_max_codebook(n_bits=0)

    def test_n_bits_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be in"):
            lloyd_max_codebook(n_bits=-1)

    def test_n_bits_9_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bits must be in"):
            lloyd_max_codebook(n_bits=9)

    def test_n_bits_1_valid(self) -> None:
        cb = lloyd_max_codebook(n_bits=1)
        assert cb.n_bits == 1
        assert len(cb.centroids) == 2
        assert len(cb.boundaries) == 1

    def test_n_bits_8_valid(self) -> None:
        cb = lloyd_max_codebook(n_bits=8)
        assert cb.n_bits == 8
        assert len(cb.centroids) == 256
        assert len(cb.boundaries) == 255
