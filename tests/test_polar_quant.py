"""Tests for PolarQuant compression pipeline.

Covers block-level and tensor-level round-trips, compression quality
relative to Lloyd-Max codebook theory, compression-ratio arithmetic,
and edge cases (zero vectors, partial blocks, invalid inputs).
"""

import numpy as np
import pytest

from src.turboquant.polar_quant import (
    CompressedBlock,
    CompressedTensor,
    compression_ratio,
    polar_dequantize,
    polar_dequantize_block,
    polar_quantize,
    polar_quantize_block,
)


# ---------------------------------------------------------------------------
# Block-level compression
# ---------------------------------------------------------------------------


class TestBlockCompression:
    """Test single-block compress / decompress round trips."""

    @pytest.mark.parametrize(
        "n_bits, max_mse",
        [
            (4, 0.02),    # turbo4: codebook MSE ≈ 0.0094
            (3, 0.06),    # turbo3: codebook MSE ≈ 0.0344
            (2, 0.18),    # turbo2: codebook MSE ≈ 0.1175
        ],
    )
    def test_round_trip_block_mse(self, n_bits: int, max_mse: float) -> None:
        """Round-trip block compression has MSE below threshold."""
        rng = np.random.default_rng(123)
        x = rng.standard_normal(128)
        block = polar_quantize_block(x, n_bits, seed=7)
        x_recon = polar_dequantize_block(block)
        mse = float(np.mean((x - x_recon) ** 2))
        assert mse < max_mse, f"{n_bits}-bit block MSE {mse:.6f} exceeds {max_mse}"

    def test_norm_preserved(self) -> None:
        """CompressedBlock stores the correct L2 norm."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        block = polar_quantize_block(x, n_bits=4, seed=0)
        expected_norm = np.float32(np.linalg.norm(x))
        assert block.norm == expected_norm

    def test_indices_shape_and_dtype(self) -> None:
        """Indices have uint8 dtype and length equal to padded block size."""
        x = np.ones(128)
        block = polar_quantize_block(x, n_bits=4, seed=0)
        assert block.indices.shape == (128,)
        assert block.indices.dtype == np.uint8

    def test_zero_vector(self) -> None:
        """Zero vector produces norm=0 and recovers as zeros."""
        x = np.zeros(128)
        block = polar_quantize_block(x, n_bits=4, seed=0)
        assert block.norm == 0.0
        x_recon = polar_dequantize_block(block)
        np.testing.assert_array_equal(x_recon, np.zeros(128))

    def test_determinism(self) -> None:
        """Identical input + seed produces identical output."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal(128)
        b1 = polar_quantize_block(x, n_bits=3, seed=42)
        b2 = polar_quantize_block(x, n_bits=3, seed=42)
        np.testing.assert_array_equal(b1.indices, b2.indices)
        assert b1.norm == b2.norm

    def test_different_seed_different_output(self) -> None:
        """Different seeds produce different indices (with high probability)."""
        rng = np.random.default_rng(77)
        x = rng.standard_normal(128)
        b1 = polar_quantize_block(x, n_bits=4, seed=0)
        b2 = polar_quantize_block(x, n_bits=4, seed=999)
        assert not np.array_equal(b1.indices, b2.indices)


# ---------------------------------------------------------------------------
# Tensor-level compression
# ---------------------------------------------------------------------------


class TestTensorCompression:
    """Test full-tensor compress / decompress."""

    def test_single_block(self) -> None:
        """Tensor of exactly one block (128 elements)."""
        rng = np.random.default_rng(10)
        x = rng.standard_normal(128)
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=128)
        assert len(ct.blocks) == 1
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == x.shape

    def test_multiple_blocks(self) -> None:
        """Tensor splits into four blocks (512 elements)."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal(512)
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=128)
        assert len(ct.blocks) == 4
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == x.shape
        mse = float(np.mean((x - x_recon) ** 2))
        assert mse < 0.02

    def test_partial_last_block(self) -> None:
        """Non-divisible size pads the last block for WHT."""
        rng = np.random.default_rng(12)
        x = rng.standard_normal(100)
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=128)
        # 100 elements < 128 → single partial block
        assert len(ct.blocks) == 1
        assert ct.blocks[0].block_size == 100
        # indices padded to 128 (next power of 2 >= 100)
        assert len(ct.blocks[0].indices) == 128
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == (100,)

    def test_multi_block_partial_last(self) -> None:
        """300 elements → 128 + 128 + 44, last block padded to 64."""
        rng = np.random.default_rng(15)
        x = rng.standard_normal(300)
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=128)
        assert len(ct.blocks) == 3
        assert ct.blocks[0].block_size == 128
        assert ct.blocks[1].block_size == 128
        assert ct.blocks[2].block_size == 44
        # Padded to next power of 2: 44 → 64
        assert len(ct.blocks[2].indices) == 64
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == (300,)

    def test_original_shape_preserved(self) -> None:
        """CompressedTensor records and restores original shape."""
        rng = np.random.default_rng(13)
        x = rng.standard_normal((4, 32))
        ct = polar_quantize(x, n_bits=3, seed=0, block_size=128)
        assert ct.original_shape == (4, 32)
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == (4, 32)

    def test_2d_tensor(self) -> None:
        """2-D tensor (32, 128) round-trips correctly."""
        rng = np.random.default_rng(14)
        x = rng.standard_normal((32, 128))
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=128)
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == (32, 128)
        mse = float(np.mean((x - x_recon) ** 2))
        assert mse < 0.02

    def test_n_bits_stored(self) -> None:
        """n_bits value stored correctly in CompressedTensor."""
        x = np.ones(128)
        for n_bits in (2, 3, 4):
            ct = polar_quantize(x, n_bits=n_bits, seed=0)
            assert ct.n_bits == n_bits


# ---------------------------------------------------------------------------
# Compression quality
# ---------------------------------------------------------------------------


class TestCompressionQuality:
    """Verify pipeline MSE matches Lloyd-Max codebook theory."""

    @pytest.fixture()
    def gaussian_data(self) -> np.ndarray:
        """Large Gaussian dataset for stable MSE measurement."""
        return np.random.default_rng(0).standard_normal(4096)

    @pytest.mark.parametrize(
        "n_bits, expected_mse",
        [
            (4, 0.0094),   # turbo4 theoretical
            (3, 0.0344),   # turbo3 theoretical
            (2, 0.1175),   # turbo2 theoretical
        ],
    )
    def test_mse_matches_codebook(
        self,
        gaussian_data: np.ndarray,
        n_bits: int,
        expected_mse: float,
    ) -> None:
        """Pipeline MSE on Gaussian data within 20% of codebook MSE."""
        ct = polar_quantize(gaussian_data, n_bits=n_bits, seed=42, block_size=128)
        x_recon = polar_dequantize(ct)
        mse = float(np.mean((gaussian_data - x_recon) ** 2))
        assert mse == pytest.approx(expected_mse, rel=0.20), (
            f"{n_bits}-bit MSE {mse:.6f} not within 20% of {expected_mse}"
        )

    def test_higher_bits_lower_mse(self, gaussian_data: np.ndarray) -> None:
        """More bits → lower MSE: MSE(4) < MSE(3) < MSE(2)."""
        mse_by_bits = {}
        for n_bits in (2, 3, 4):
            ct = polar_quantize(gaussian_data, n_bits=n_bits, seed=42, block_size=128)
            x_recon = polar_dequantize(ct)
            mse_by_bits[n_bits] = float(np.mean((gaussian_data - x_recon) ** 2))
        assert mse_by_bits[4] < mse_by_bits[3] < mse_by_bits[2]

    def test_larger_blocks_better_ratio(self) -> None:
        """Larger blocks reduce norm overhead → higher compression ratio."""
        r128 = compression_ratio(4, 128)
        r256 = compression_ratio(4, 256)
        r512 = compression_ratio(4, 512)
        assert r128 < r256 < r512


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    """Test compression-ratio formula: 16 / (n_bits + 32 / block_size)."""

    def test_turbo4_block128(self) -> None:
        # 16 / (4 + 32/128) = 16 / 4.25 ≈ 3.765
        assert compression_ratio(4, 128) == pytest.approx(16.0 / 4.25, rel=1e-9)

    def test_turbo3_block128(self) -> None:
        # 16 / (3 + 32/128) = 16 / 3.25 ≈ 4.923
        assert compression_ratio(3, 128) == pytest.approx(16.0 / 3.25, rel=1e-9)

    def test_turbo2_block128(self) -> None:
        # 16 / (2 + 32/128) = 16 / 2.25 ≈ 7.111
        assert compression_ratio(2, 128) == pytest.approx(16.0 / 2.25, rel=1e-9)

    def test_smaller_block_reduces_ratio(self) -> None:
        """Smaller blocks → more norm overhead → lower compression ratio."""
        r32 = compression_ratio(4, 32)
        r128 = compression_ratio(4, 128)
        assert r32 < r128

    def test_all_ratios_positive(self) -> None:
        for n_bits in (2, 3, 4):
            for bs in (2, 4, 8, 16, 32, 64, 128, 256):
                assert compression_ratio(n_bits, bs) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test boundary conditions and invalid input handling."""

    def test_block_size_2(self) -> None:
        """Minimum non-trivial WHT size."""
        rng = np.random.default_rng(50)
        x = rng.standard_normal(2)
        block = polar_quantize_block(x, n_bits=4, seed=0)
        x_recon = polar_dequantize_block(block)
        assert x_recon.shape == (2,)

    def test_single_element(self) -> None:
        """Single-element block is handled (WHT d=1 is identity)."""
        x = np.array([3.14])
        block = polar_quantize_block(x, n_bits=4, seed=0)
        x_recon = polar_dequantize_block(block)
        assert x_recon.shape == (1,)
        assert abs(x_recon[0]) > 0

    def test_large_tensor(self) -> None:
        """4096-element tensor compresses into 32 blocks."""
        rng = np.random.default_rng(60)
        x = rng.standard_normal(4096)
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=128)
        assert len(ct.blocks) == 32
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == (4096,)

    @pytest.mark.parametrize("bad_bits", [0, 1, 5, -1, 8])
    def test_invalid_n_bits_block(self, bad_bits: int) -> None:
        """polar_quantize_block rejects unsupported bit-widths."""
        x = np.ones(128)
        with pytest.raises(ValueError, match="n_bits"):
            polar_quantize_block(x, n_bits=bad_bits, seed=0)

    @pytest.mark.parametrize("bad_bits", [0, 1, 5, -1, 8])
    def test_invalid_n_bits_tensor(self, bad_bits: int) -> None:
        """polar_quantize rejects unsupported bit-widths."""
        x = np.ones(128)
        with pytest.raises(ValueError, match="n_bits"):
            polar_quantize(x, n_bits=bad_bits, seed=0)

    def test_empty_block_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            polar_quantize_block(np.array([]), n_bits=4, seed=0)

    def test_invalid_block_size(self) -> None:
        """block_size < 1 raises ValueError."""
        x = np.ones(128)
        with pytest.raises(ValueError, match="block_size"):
            polar_quantize(x, n_bits=4, seed=0, block_size=0)

    def test_block_size_not_power_of_two(self) -> None:
        """Non-power-of-2 block_size is handled via internal padding."""
        rng = np.random.default_rng(70)
        x = rng.standard_normal(50)
        # block_size=50 → each block of 50 elements padded to 64 for WHT
        ct = polar_quantize(x, n_bits=4, seed=0, block_size=50)
        x_recon = polar_dequantize(ct)
        assert x_recon.shape == (50,)

    def test_frozen_dataclasses(self) -> None:
        """CompressedBlock and CompressedTensor are immutable."""
        block = polar_quantize_block(np.ones(128), n_bits=4, seed=0)
        with pytest.raises(AttributeError):
            block.norm = 999.0  # type: ignore[misc]

        ct = polar_quantize(np.ones(128), n_bits=4, seed=0)
        with pytest.raises(AttributeError):
            ct.n_bits = 99  # type: ignore[misc]


class TestNextPowerOfTwo:
    """Test _next_power_of_two edge cases (line 27)."""

    def test_zero_raises(self) -> None:
        from src.turboquant.polar_quant import _next_power_of_two

        with pytest.raises(ValueError, match="positive"):
            _next_power_of_two(0)

    def test_negative_raises(self) -> None:
        from src.turboquant.polar_quant import _next_power_of_two

        with pytest.raises(ValueError, match="positive"):
            _next_power_of_two(-5)

    def test_one_returns_one(self) -> None:
        from src.turboquant.polar_quant import _next_power_of_two

        assert _next_power_of_two(1) == 1
