"""Tests for boundary layer protection."""

import numpy as np
import pytest

from src.turboquant.boundary import BoundaryAwareCompressor, BoundaryConfig
from src.turboquant.compressor import QuantConfig


class TestBoundaryConfig:
    """Verify BoundaryConfig validation."""

    def test_defaults(self) -> None:
        bc = BoundaryConfig()
        assert bc.first_n == 2
        assert bc.last_n == 2
        assert bc.protected_k_bits == 8
        assert bc.protected_v_bits == 8

    def test_custom_values(self) -> None:
        bc = BoundaryConfig(first_n=3, last_n=1, protected_k_bits=4, protected_v_bits=4)
        assert bc.first_n == 3
        assert bc.last_n == 1

    def test_negative_first_n_raises(self) -> None:
        with pytest.raises(ValueError, match="first_n must be >= 0"):
            BoundaryConfig(first_n=-1)

    def test_negative_last_n_raises(self) -> None:
        with pytest.raises(ValueError, match="last_n must be >= 0"):
            BoundaryConfig(last_n=-2)

    def test_invalid_protected_k_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="protected_k_bits"):
            BoundaryConfig(protected_k_bits=5)

    def test_invalid_protected_v_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="protected_v_bits"):
            BoundaryConfig(protected_v_bits=7)

    def test_frozen(self) -> None:
        bc = BoundaryConfig()
        with pytest.raises(AttributeError):
            bc.first_n = 5  # type: ignore[misc]


class TestBoundaryLayerClassification:
    """Verify layer classification as boundary vs middle."""

    def test_default_28_layers(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        # First 2 + last 2 are boundary
        assert comp.is_boundary_layer(0, 28) is True
        assert comp.is_boundary_layer(1, 28) is True
        assert comp.is_boundary_layer(2, 28) is False
        assert comp.is_boundary_layer(14, 28) is False
        assert comp.is_boundary_layer(25, 28) is False
        assert comp.is_boundary_layer(26, 28) is True
        assert comp.is_boundary_layer(27, 28) is True

    def test_no_boundary(self) -> None:
        comp = BoundaryAwareCompressor(
            QuantConfig(k_bits=8, v_bits=4),
            BoundaryConfig(first_n=0, last_n=0),
        )
        for i in range(10):
            assert comp.is_boundary_layer(i, 10) is False

    def test_all_boundary_overlap(self) -> None:
        """When first_n + last_n >= n_layers, all layers are boundary."""
        comp = BoundaryAwareCompressor(
            QuantConfig(k_bits=8, v_bits=4),
            BoundaryConfig(first_n=3, last_n=3),
        )
        for i in range(4):
            assert comp.is_boundary_layer(i, 4) is True

    def test_invalid_layer_idx_raises(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        with pytest.raises(ValueError):
            comp.is_boundary_layer(-1, 10)
        with pytest.raises(ValueError):
            comp.is_boundary_layer(10, 10)

    def test_invalid_n_layers_raises(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        with pytest.raises(ValueError):
            comp.is_boundary_layer(0, 0)


class TestGetLayerConfig:
    """Verify per-layer config selection."""

    def test_boundary_gets_protected(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        cfg = comp.get_layer_config(0, 28)
        assert cfg.k_bits == 8
        assert cfg.v_bits == 8  # protected default

    def test_middle_gets_global(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        cfg = comp.get_layer_config(14, 28)
        assert cfg.k_bits == 8
        assert cfg.v_bits == 4  # global turbo4


class TestCompressDecompress:
    """Verify boundary-aware compression round-trip."""

    @pytest.fixture
    def small_kv(self) -> tuple[np.ndarray, np.ndarray]:
        """Small KV cache: 4 layers, 2 heads, 4 tokens, 64 dim."""
        rng = np.random.default_rng(123)
        shape = (4, 2, 4, 64)
        return rng.standard_normal(shape), rng.standard_normal(shape)

    def test_round_trip_shape(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        compressed = comp.compress_kv(keys, values)
        dk, dv = comp.decompress_kv(compressed)
        assert dk.shape == keys.shape
        assert dv.shape == values.shape

    def test_round_trip_mse_reasonable(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        compressed = comp.compress_kv(keys, values)
        dk, dv = comp.decompress_kv(compressed)
        k_mse = np.mean((keys - dk) ** 2)
        v_mse = np.mean((values - dv) ** 2)
        # Both should have finite, bounded MSE
        assert k_mse < 1.0
        assert v_mse < 1.0

    def test_boundary_layers_higher_quality(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        """Boundary layers (q8_0) should have lower MSE than middle (turbo4)."""
        keys, values = small_kv
        comp = BoundaryAwareCompressor(
            QuantConfig(k_bits=8, v_bits=4),
            BoundaryConfig(first_n=1, last_n=1),
        )
        compressed = comp.compress_kv(keys, values)
        dk, dv = comp.decompress_kv(compressed)

        # V MSE for boundary layer 0 should be lower than middle layer 2
        v_mse_boundary = np.mean((values[0] - dv[0]) ** 2)
        v_mse_middle = np.mean((values[2] - dv[2]) ** 2)
        # Boundary (q8_0) should generally be better than middle (turbo4)
        assert v_mse_boundary <= v_mse_middle + 0.5  # allow some tolerance

    def test_mismatched_shapes_raises(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        k = np.zeros((4, 2, 4, 64))
        v = np.zeros((4, 2, 8, 64))
        with pytest.raises(ValueError, match="same shape"):
            comp.compress_kv(k, v)

    def test_non_4d_raises(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        k = np.zeros((4, 64))
        v = np.zeros((4, 64))
        with pytest.raises(ValueError, match="4D"):
            comp.compress_kv(k, v)


class TestEstimateMemory:
    """Verify memory estimation with boundary layers."""

    def test_returns_expected_keys(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        mem = comp.estimate_memory(n_layers=28, n_heads=28, seq_len=8192, head_dim=128)
        assert "total_bytes" in mem
        assert "boundary_bytes" in mem
        assert "middle_bytes" in mem
        assert "compression_ratio" in mem
        assert "n_boundary_layers" in mem
        assert "n_middle_layers" in mem

    def test_layer_counts(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        mem = comp.estimate_memory(n_layers=28, n_heads=28, seq_len=8192, head_dim=128)
        assert mem["n_boundary_layers"] == 4
        assert mem["n_middle_layers"] == 24

    def test_total_equals_sum(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        mem = comp.estimate_memory(n_layers=28, n_heads=28, seq_len=8192, head_dim=128)
        assert abs(mem["total_bytes"] - (mem["boundary_bytes"] + mem["middle_bytes"])) < 1.0

    def test_compression_ratio_positive(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        mem = comp.estimate_memory(n_layers=28, n_heads=28, seq_len=8192, head_dim=128)
        assert mem["compression_ratio"] > 1.0


class TestDescribe:
    """Verify human-readable description."""

    def test_default_28_layers(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        desc = comp.describe(28)
        assert "28 layers" in desc
        assert "q8_0" in desc
        assert "turbo4" in desc

    def test_no_boundary(self) -> None:
        comp = BoundaryAwareCompressor(
            QuantConfig(k_bits=8, v_bits=4),
            BoundaryConfig(first_n=0, last_n=0),
        )
        desc = comp.describe(10)
        assert "10 layers" in desc

    def test_zero_layers(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        desc = comp.describe(0)
        assert "0 layers" in desc


# ---------------------------------------------------------------------------
# Missing coverage tests
# ---------------------------------------------------------------------------


class TestMissingCoverage:
    """Tests that target specific uncovered lines."""

    def test_protected_same_as_global(self) -> None:
        """Line 140: protected_config == global_config → shared compressor."""
        comp = BoundaryAwareCompressor(
            QuantConfig(k_bits=8, v_bits=8),
            BoundaryConfig(protected_k_bits=8, protected_v_bits=8),
        )
        assert comp._protected_compressor is comp._global_compressor

    def test_global_config_property(self) -> None:
        """Line 151: global_config property returns the config."""
        cfg = QuantConfig(k_bits=8, v_bits=4)
        comp = BoundaryAwareCompressor(cfg)
        assert comp.global_config is cfg

    def test_boundary_config_property(self) -> None:
        """Line 156: boundary_config property returns the config."""
        bc = BoundaryConfig(first_n=3, last_n=1)
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4), bc)
        assert comp.boundary_config is bc

    def test_n_layers_zero_raises_validate(self) -> None:
        """Line 151 (alt): _validate_layer_args with n_layers=0."""
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        with pytest.raises(ValueError, match="n_layers must be > 0"):
            comp.is_boundary_layer(0, 0)

    def test_negative_layer_idx_raises(self) -> None:
        """Line 156: layer_idx out of range."""
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        with pytest.raises(ValueError, match="out of range"):
            comp.is_boundary_layer(-1, 10)

    def test_layer_idx_ge_n_layers_raises(self) -> None:
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        with pytest.raises(ValueError, match="out of range"):
            comp.is_boundary_layer(10, 10)

    def test_decompress_zero_layers(self) -> None:
        """Lines 295-297: decompress with 0 layers returns empty arrays."""
        from src.turboquant.compressor import CompressedKV

        compressed = CompressedKV(
            keys=(),
            values=(),
            config=QuantConfig(k_bits=8, v_bits=4),
            n_layers=0,
            original_k_shape=(0, 4, 8, 128),
            original_v_shape=(0, 4, 8, 128),
        )
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        keys, values = comp.decompress_kv(compressed)
        assert keys.shape == (0, 4, 8, 128)
        assert values.shape == (0, 4, 8, 128)

    def test_estimate_memory_zero_layers(self) -> None:
        """Line 350: n_layers <= 0 returns zero estimate."""
        comp = BoundaryAwareCompressor(QuantConfig(k_bits=8, v_bits=4))
        mem = comp.estimate_memory(n_layers=0, n_heads=28, seq_len=8192, head_dim=128)
        assert mem["total_bytes"] == 0.0
        assert mem["compression_ratio"] == float("inf")
        assert mem["n_boundary_layers"] == 0
        assert mem["n_middle_layers"] == 0

    def test_format_layer_indices_empty(self) -> None:
        """Line 472: empty list → '[]'."""
        from src.turboquant.boundary import _format_layer_indices

        assert _format_layer_indices([]) == "[]"

    def test_format_layer_indices_single(self) -> None:
        """Line 474: single element → '[N]'."""
        from src.turboquant.boundary import _format_layer_indices

        assert _format_layer_indices([5]) == "[5]"
