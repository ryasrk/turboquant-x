"""Tests for TurboQuant KV cache compressor."""

import numpy as np
import pytest

from src.turboquant.compressor import (
    QuantConfig,
    CompressedKV,
    TurboQuantCompressor,
)


# ======================================================================
# QuantConfig validation
# ======================================================================


class TestQuantConfig:
    """Tests for QuantConfig construction and validation."""

    def test_default_config(self) -> None:
        cfg = QuantConfig()
        assert cfg.k_bits == 8
        assert cfg.v_bits == 4
        assert cfg.block_size == 128
        assert cfg.seed == 42

    @pytest.mark.parametrize(
        "k,v",
        [
            (2, 2), (2, 3), (2, 4), (2, 8),
            (3, 2), (3, 3), (3, 4), (3, 8),
            (4, 2), (4, 3), (4, 4), (4, 8),
            (8, 2), (8, 3), (8, 4), (8, 8),
        ],
    )
    def test_valid_bit_combinations(self, k: int, v: int) -> None:
        cfg = QuantConfig(k_bits=k, v_bits=v)
        assert cfg.k_bits == k
        assert cfg.v_bits == v

    @pytest.mark.parametrize("bad_bits", [1, 5, 6, 7, 16])
    def test_invalid_k_bits_raises(self, bad_bits: int) -> None:
        with pytest.raises(ValueError, match="k_bits"):
            QuantConfig(k_bits=bad_bits)

    @pytest.mark.parametrize("bad_bits", [1, 5, 6, 7, 16])
    def test_invalid_v_bits_raises(self, bad_bits: int) -> None:
        with pytest.raises(ValueError, match="v_bits"):
            QuantConfig(v_bits=bad_bits)

    @pytest.mark.parametrize("bad_bs", [0, 3, 7])
    def test_invalid_block_size_raises(self, bad_bs: int) -> None:
        with pytest.raises(ValueError, match="block_size"):
            QuantConfig(block_size=bad_bs)

    def test_immutability(self) -> None:
        cfg = QuantConfig()
        with pytest.raises(AttributeError):
            cfg.k_bits = 4  # type: ignore[misc]


# ======================================================================
# Compress / decompress round-trip
# ======================================================================


class TestCompressDecompress:
    """Tests for full KV cache compress→decompress round-trip."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(12345)

    @pytest.fixture
    def small_kv(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Small KV cache: 2 layers, 4 heads, 8 tokens, dim 64."""
        keys = rng.standard_normal((2, 4, 8, 64))
        values = rng.standard_normal((2, 4, 8, 64))
        return keys, values

    def test_roundtrip_small_kv(
        self, small_kv: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = small_kv
        comp = TurboQuantCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_roundtrip_default_config(
        self, small_kv: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Default config K=8, V=4 produces acceptable MSE."""
        keys, values = small_kv
        comp = TurboQuantCompressor()  # k=8, v=4
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        # K at 8-bit — very low relative MSE
        k_rel_mse = float(np.mean((keys - k_out) ** 2) / np.var(keys))
        assert k_rel_mse < 0.01, f"K relative MSE too high: {k_rel_mse}"

        # V at 4-bit — lossy but bounded
        v_rel_mse = float(np.mean((values - v_out) ** 2) / np.var(values))
        assert v_rel_mse < 0.5, f"V relative MSE too high: {v_rel_mse}"

    def test_k_8bit_very_low_mse(
        self, small_kv: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """8-bit keys should have near-zero reconstruction error."""
        keys, values = small_kv
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=8))
        compressed = comp.compress_kv(keys, values)
        k_out, _ = comp.decompress_kv(compressed)

        relative_mse = float(np.mean((keys - k_out) ** 2) / np.var(keys))
        assert relative_mse < 0.01

    def test_v_turbo4_mse_bounded(
        self, small_kv: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """4-bit values produce nonzero but reasonable MSE."""
        keys, values = small_kv
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=4))
        compressed = comp.compress_kv(keys, values)
        _, v_out = comp.decompress_kv(compressed)

        mse = float(np.mean((values - v_out) ** 2))
        assert mse > 0, "MSE should be nonzero for lossy 4-bit"
        assert mse < float(np.var(values)), "MSE should be less than signal variance"

    def test_shape_preserved(
        self, small_kv: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = small_kv
        comp = TurboQuantCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_n_layers_stored(
        self, small_kv: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = small_kv
        comp = TurboQuantCompressor()
        compressed = comp.compress_kv(keys, values)
        assert compressed.n_layers == 2
        assert len(compressed.keys) == 2
        assert len(compressed.values) == 2


# ======================================================================
# Asymmetric K/V precision
# ======================================================================


class TestAsymmetric:
    """Tests for asymmetric K/V bit-width configurations."""

    @pytest.fixture
    def kv_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(999)
        keys = rng.standard_normal((2, 4, 16, 64))
        values = rng.standard_normal((2, 4, 16, 64))
        return keys, values

    def test_different_precision_stored(
        self, kv_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """K and V compressed tensors record their respective bit-widths."""
        keys, values = kv_data
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=4))
        compressed = comp.compress_kv(keys, values)
        assert compressed.keys[0].n_bits == 8
        assert compressed.values[0].n_bits == 4

    def test_k8_v4_default(
        self, kv_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Recommended default K=8  V=4: K has much lower MSE than V."""
        keys, values = kv_data
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=4))
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        k_mse = float(np.mean((keys - k_out) ** 2))
        v_mse = float(np.mean((values - v_out) ** 2))
        assert k_mse < v_mse, "8-bit K should have lower MSE than 4-bit V"

    def test_k8_v2_max_compression(
        self, kv_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Maximum compression K=8  V=2 round-trips with correct shapes."""
        keys, values = kv_data
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=2))
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape
        # V at 2-bit is lossy but should still reconstruct something
        v_mse = float(np.mean((values - v_out) ** 2))
        assert v_mse > 0

    def test_k4_v4_symmetric(
        self, kv_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Symmetric turbo4 for both K and V."""
        keys, values = kv_data
        comp = TurboQuantCompressor(QuantConfig(k_bits=4, v_bits=4))
        compressed = comp.compress_kv(keys, values)

        assert compressed.keys[0].n_bits == 4
        assert compressed.values[0].n_bits == 4

        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape


# ======================================================================
# Memory estimation
# ======================================================================


class TestMemoryEstimation:
    """Tests for estimate_memory calculations."""

    def test_returns_all_expected_keys(self) -> None:
        comp = TurboQuantCompressor()
        mem = comp.estimate_memory(
            n_layers=32, n_heads=32, seq_len=2048, head_dim=128
        )
        expected = {
            "k_bytes", "v_bytes", "total_bytes",
            "compression_ratio", "original_bytes",
        }
        assert set(mem.keys()) == expected

    def test_compression_ratio_greater_than_1(self) -> None:
        """Any sub-8-bit config should compress more than 1x."""
        comp = TurboQuantCompressor(QuantConfig(k_bits=4, v_bits=4))
        mem = comp.estimate_memory(
            n_layers=32, n_heads=32, seq_len=2048, head_dim=128
        )
        assert mem["compression_ratio"] > 1.0

    def test_k8_v4_compression_ratio_approx(self) -> None:
        """K=8 V=4 should yield roughly 2.5–3× compression."""
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=4))
        mem = comp.estimate_memory(
            n_layers=32, n_heads=32, seq_len=2048, head_dim=128
        )
        ratio = mem["compression_ratio"]
        assert 2.0 < ratio < 4.0, f"Expected ~2.6×, got {ratio:.2f}×"

    def test_memory_units_in_bytes(self) -> None:
        comp = TurboQuantCompressor()
        mem = comp.estimate_memory(
            n_layers=1, n_heads=1, seq_len=1, head_dim=128
        )
        # original: K+V × 1 layer × 1 head × 1 token × 128 dim × 2 bytes fp16
        # = 2 × 1 × 128 × 2 = 512 bytes
        assert mem["original_bytes"] == 512.0
        assert mem["total_bytes"] < mem["original_bytes"]
        assert mem["total_bytes"] == mem["k_bytes"] + mem["v_bytes"]


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    """Tests for boundary and degenerate inputs."""

    def test_single_layer_single_head(self) -> None:
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((1, 1, 16, 64))
        values = rng.standard_normal((1, 1, 16, 64))

        comp = TurboQuantCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        assert compressed.n_layers == 1
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_seq_len_1(self) -> None:
        """Single token KV cache."""
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 4, 1, 64))
        values = rng.standard_normal((2, 4, 1, 64))

        comp = TurboQuantCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_large_head_dim_256(self) -> None:
        """head_dim=256 — larger than default block_size."""
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 4, 8, 256))
        values = rng.standard_normal((2, 4, 8, 256))

        comp = TurboQuantCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

        # 8-bit K should still have low MSE with large dim
        k_rel_mse = float(np.mean((keys - k_out) ** 2) / np.var(keys))
        assert k_rel_mse < 0.01, f"8-bit K relative MSE too high: {k_rel_mse}"


# ======================================================================
# Missing coverage tests
# ======================================================================


class TestCompressKvValidation:
    """Test compress_kv dimension validation (line 83)."""

    def test_2d_input_raises(self) -> None:
        comp = TurboQuantCompressor()
        with pytest.raises(ValueError, match="4D"):
            comp.compress_kv(np.zeros((2, 64)), np.zeros((2, 64)))

    def test_3d_input_raises(self) -> None:
        comp = TurboQuantCompressor()
        with pytest.raises(ValueError, match="4D"):
            comp.compress_kv(np.zeros((2, 4, 64)), np.zeros((2, 4, 64)))


class TestQuantize8bitConstant:
    """Test _quantize_8bit with constant tensor (lines 214-216)."""

    def test_constant_tensor_roundtrip(self) -> None:
        comp = TurboQuantCompressor(QuantConfig(k_bits=8, v_bits=8))
        keys = np.full((1, 1, 1, 128), 5.0)
        values = np.full((1, 1, 1, 128), 3.0)
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        np.testing.assert_allclose(k_out, 5.0, atol=0.1)
        np.testing.assert_allclose(v_out, 3.0, atol=0.1)


class TestEstimateMemory:
    """Test estimate_memory return values (line 290)."""

    def test_returns_all_keys(self) -> None:
        comp = TurboQuantCompressor()
        mem = comp.estimate_memory(n_layers=2, n_heads=4, seq_len=8, head_dim=128)
        assert "k_bytes" in mem
        assert "v_bytes" in mem
        assert "total_bytes" in mem
        assert "compression_ratio" in mem
        assert "original_bytes" in mem

    def test_values_are_floats(self) -> None:
        comp = TurboQuantCompressor()
        mem = comp.estimate_memory(n_layers=2, n_heads=4, seq_len=8, head_dim=128)
        for key in ("k_bytes", "v_bytes", "total_bytes", "compression_ratio", "original_bytes"):
            assert isinstance(mem[key], float)

    def test_v_bits_8_path(self) -> None:
        """Line 290: v_bpe = 8.0 when v_bits == 8."""
        comp = TurboQuantCompressor(QuantConfig(k_bits=4, v_bits=8))
        mem = comp.estimate_memory(n_layers=2, n_heads=4, seq_len=8, head_dim=128)
        assert mem["v_bytes"] > 0

    def test_config_property(self) -> None:
        """Line 83: config property."""
        cfg = QuantConfig(k_bits=4, v_bits=3)
        comp = TurboQuantCompressor(cfg)
        assert comp.config is cfg
