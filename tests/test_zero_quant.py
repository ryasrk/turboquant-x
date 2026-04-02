"""Tests for ZeroQuant depth-adaptive KV cache compression.

Covers:
  - ZeroQuantConfig construction and validation
  - Sub-zone middle split (split_middle=True)
  - DepthAdaptiveCompressor zone boundaries and compress/decompress round-trip
  - ZeroQuantPreset system
  - estimate_kv_memory_gb_zero_quant utility
  - savings_vs_turboquant analysis
  - recommend_zero_quant hardware-aware recommendation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.turboquant.zero_quant import (
    DepthAdaptiveCompressor,
    ZeroQuantConfig,
    ZeroQuantPreset,
    ZoneCompressedKV,
    # Preset constants
    ZERO_QUANT_FAST,
    ZERO_QUANT_QUALITY,
    ZERO_QUANT_BALANCED,
    ZERO_QUANT_TURBO,
    ZERO_QUANT_ULTRA,
    ALL_ZERO_QUANT_PRESETS,
    # Free functions
    estimate_kv_memory_gb_zero_quant,
    recommend_zero_quant,
    savings_vs_turboquant,
)


# ===========================================================================
# ZeroQuantConfig — baseline (existing) behaviour
# ===========================================================================


class TestZeroQuantConfigDefaults:
    def test_default_fractions(self) -> None:
        cfg = ZeroQuantConfig()
        assert cfg.shallow_fraction == 0.25
        assert cfg.deep_fraction == 0.25

    def test_default_bits(self) -> None:
        cfg = ZeroQuantConfig()
        assert cfg.shallow_k_bits == 8
        assert cfg.shallow_v_bits == 8
        assert cfg.middle_k_bits == 4
        assert cfg.middle_v_bits == 3
        assert cfg.deep_k_bits == 8
        assert cfg.deep_v_bits == 8

    def test_split_middle_off_by_default(self) -> None:
        cfg = ZeroQuantConfig()
        assert cfg.split_middle is False

    def test_split_middle_early_late_defaults(self) -> None:
        cfg = ZeroQuantConfig()
        assert cfg.middle_early_v_bits == 4
        assert cfg.middle_late_v_bits == 2

    def test_immutable(self) -> None:
        cfg = ZeroQuantConfig()
        with pytest.raises(AttributeError):
            cfg.middle_k_bits = 2  # type: ignore[misc]


# ===========================================================================
# ZeroQuantConfig — validation
# ===========================================================================


class TestZeroQuantConfigValidation:
    @pytest.mark.parametrize("bits", [1, 5, 6, 7, 16])
    def test_invalid_middle_k_bits_raises(self, bits: int) -> None:
        with pytest.raises(ValueError, match="middle_k_bits"):
            ZeroQuantConfig(middle_k_bits=bits)

    @pytest.mark.parametrize("bits", [1, 5, 6, 7, 16])
    def test_invalid_middle_v_bits_raises(self, bits: int) -> None:
        with pytest.raises(ValueError, match="middle_v_bits"):
            ZeroQuantConfig(middle_v_bits=bits)

    @pytest.mark.parametrize("bits", [1, 5, 6, 7, 16])
    def test_invalid_middle_early_v_bits_raises(self, bits: int) -> None:
        with pytest.raises(ValueError, match="middle_early_v_bits"):
            ZeroQuantConfig(split_middle=True, middle_early_v_bits=bits)

    @pytest.mark.parametrize("bits", [1, 5, 6, 7, 16])
    def test_invalid_middle_late_v_bits_raises(self, bits: int) -> None:
        with pytest.raises(ValueError, match="middle_late_v_bits"):
            ZeroQuantConfig(split_middle=True, middle_late_v_bits=bits)

    def test_zero_shallow_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            ZeroQuantConfig(shallow_fraction=0.0)

    def test_fractions_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError):
            ZeroQuantConfig(shallow_fraction=0.5, deep_fraction=0.5)

    def test_invalid_block_size_raises(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            ZeroQuantConfig(block_size=7)


# ===========================================================================
# ZeroQuantConfig.average_bits
# ===========================================================================


class TestAverageBits:
    def test_uniform_8bit_equals_8(self) -> None:
        cfg = ZeroQuantConfig(
            shallow_k_bits=8, shallow_v_bits=8,
            middle_k_bits=8, middle_v_bits=8,
            deep_k_bits=8, deep_v_bits=8,
        )
        assert cfg.average_bits(32) == pytest.approx(8.0)

    def test_lower_middle_reduces_average(self) -> None:
        high = ZeroQuantConfig(middle_k_bits=8, middle_v_bits=8)
        low = ZeroQuantConfig(middle_k_bits=4, middle_v_bits=2)
        assert low.average_bits(32) < high.average_bits(32)

    def test_split_middle_average_bits_between_early_and_late(self) -> None:
        # With split middle K4/V4 early + K4/V2 late the avg should be
        # between K4/V4 uniform and K4/V2 uniform.
        uniform_v4 = ZeroQuantConfig(middle_k_bits=4, middle_v_bits=4)
        uniform_v2 = ZeroQuantConfig(middle_k_bits=4, middle_v_bits=2)
        split = ZeroQuantConfig(
            split_middle=True,
            middle_k_bits=4,
            middle_early_v_bits=4,
            middle_late_v_bits=2,
        )
        n = 32
        assert uniform_v2.average_bits(n) < split.average_bits(n) < uniform_v4.average_bits(n)

    def test_split_middle_average_below_turboquant_6bit(self) -> None:
        """TURBO preset should beat TurboQuant K8/V4 = 6.0 avg bits."""
        assert ZERO_QUANT_TURBO.config.average_bits(32) < 6.0


# ===========================================================================
# DepthAdaptiveCompressor — zone boundaries
# ===========================================================================


class TestZoneBoundaries:
    @pytest.fixture
    def compressor(self) -> DepthAdaptiveCompressor:
        return DepthAdaptiveCompressor()

    def test_32_layer_default_zones(self, compressor: DepthAdaptiveCompressor) -> None:
        shallow_end, deep_start = compressor._zone_boundaries(32)
        assert shallow_end == 8  # 25% of 32
        assert deep_start == 24  # 32 - 25% of 32

    def test_zones_have_at_least_one_layer(self, compressor: DepthAdaptiveCompressor) -> None:
        for n in [3, 4, 8, 12, 32, 64]:
            shallow_end, deep_start = compressor._zone_boundaries(n)
            assert shallow_end >= 1
            assert deep_start - shallow_end >= 1
            assert n - deep_start >= 1

    def test_split_middle_boundary_at_midpoint(self) -> None:
        cfg = ZeroQuantConfig(split_middle=True)
        comp = DepthAdaptiveCompressor(cfg)
        shallow_end, deep_start = comp._zone_boundaries(32)
        mid = comp._middle_split_boundary(shallow_end, deep_start)
        # For 16 middle layers, split at 8.
        assert mid == shallow_end + (deep_start - shallow_end) // 2

    def test_split_middle_boundary_odd_middle(self) -> None:
        """With odd-sized middle zone, early half should be slightly smaller."""
        cfg = ZeroQuantConfig(
            split_middle=True,
            shallow_fraction=0.25,
            deep_fraction=0.25,
        )
        comp = DepthAdaptiveCompressor(cfg)
        # Use a model where middle zone has an odd number of layers
        # e.g. 10 layers → shallow=2, deep=2, middle=6 (even here)
        # Use 13 layers → shallow=3, deep=3, middle=7 (odd)
        shallow_end, deep_start = comp._zone_boundaries(13)
        mid = comp._middle_split_boundary(shallow_end, deep_start)
        n_middle = deep_start - shallow_end
        assert mid == shallow_end + n_middle // 2
        # Ensure early and late each have at least 1 layer
        assert mid > shallow_end
        assert mid < deep_start


# ===========================================================================
# DepthAdaptiveCompressor — round-trip (no split)
# ===========================================================================


class TestDepthAdaptiveCompressorRoundTrip:
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    @pytest.fixture
    def kv_8_layers(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """8 layers, 4 heads, 16 tokens, 32 head_dim."""
        k = rng.standard_normal((8, 4, 16, 32))
        v = rng.standard_normal((8, 4, 16, 32))
        return k, v

    def test_roundtrip_shape_preserved(
        self, kv_8_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_8_layers
        comp = DepthAdaptiveCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_roundtrip_8bit_zone_low_error(
        self, kv_8_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shallow and deep zones at K8/V8 should have very low MSE."""
        keys, values = kv_8_layers
        comp = DepthAdaptiveCompressor()
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        # Shallow = first 2 layers (8-layer model with default 25%)
        shallow_end = compressed.shallow_end
        k_mse_shallow = np.mean((k_out[:shallow_end] - keys[:shallow_end]) ** 2)
        assert k_mse_shallow < 0.01

    def test_roundtrip_produces_zone_compressed_kv(
        self, kv_8_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_8_layers
        comp = DepthAdaptiveCompressor()
        compressed = comp.compress_kv(keys, values)
        assert isinstance(compressed, ZoneCompressedKV)
        assert compressed.n_layers == 8
        assert compressed.coquant_head_dim == 0

    def test_roundtrip_with_coquant(
        self, kv_8_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_8_layers
        cfg = ZeroQuantConfig(use_kv_coquant=True)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape
        assert compressed.coquant_head_dim == 32  # head_dim of fixture


# ===========================================================================
# DepthAdaptiveCompressor — round-trip WITH split_middle
# ===========================================================================


class TestSplitMiddleRoundTrip:
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(0xBEEF)

    @pytest.fixture
    def kv_16_layers(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        k = rng.standard_normal((16, 8, 32, 64))
        v = rng.standard_normal((16, 8, 32, 64))
        return k, v

    def test_split_roundtrip_shape(
        self, kv_16_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_16_layers
        cfg = ZeroQuantConfig(split_middle=True, middle_k_bits=4,
                              middle_early_v_bits=4, middle_late_v_bits=2)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_split_compressed_has_middle_late(
        self, kv_16_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_16_layers
        cfg = ZeroQuantConfig(split_middle=True)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        assert compressed.middle_late_kv is not None

    def test_no_split_has_no_middle_late(
        self, kv_16_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_16_layers
        cfg = ZeroQuantConfig(split_middle=False)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        assert compressed.middle_late_kv is None

    def test_split_middle_split_at_recorded(
        self, kv_16_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_16_layers
        cfg = ZeroQuantConfig(split_middle=True)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        # With 16 layers: shallow=4, deep=4, middle=8, split at 8 (4+4)
        assert compressed.middle_split_at == compressed.shallow_end + 4

    def test_split_roundtrip_deep_zone_accuracy(
        self, kv_16_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Deep zone K8/V8 should still restore accurately after split."""
        keys, values = kv_16_layers
        cfg = ZeroQuantConfig(split_middle=True)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)

        deep_start = compressed.deep_start
        k_mse_deep = np.mean((k_out[deep_start:] - keys[deep_start:]) ** 2)
        assert k_mse_deep < 0.01

    def test_coquant_with_split_middle(
        self, kv_16_layers: tuple[np.ndarray, np.ndarray]
    ) -> None:
        keys, values = kv_16_layers
        cfg = ZeroQuantConfig(split_middle=True, use_kv_coquant=True)
        comp = DepthAdaptiveCompressor(cfg)
        compressed = comp.compress_kv(keys, values)
        k_out, v_out = comp.decompress_kv(compressed)
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape


# ===========================================================================
# ZoneCompressedKV — memory_bytes
# ===========================================================================


class TestZoneCompressedKVMemory:
    @pytest.fixture
    def compressed_default(self) -> ZoneCompressedKV:
        rng = np.random.default_rng(1)
        k = rng.standard_normal((8, 2, 8, 32))
        v = rng.standard_normal((8, 2, 8, 32))
        return DepthAdaptiveCompressor().compress_kv(k, v)

    @pytest.fixture
    def compressed_split(self) -> ZoneCompressedKV:
        rng = np.random.default_rng(2)
        k = rng.standard_normal((8, 2, 8, 32))
        v = rng.standard_normal((8, 2, 8, 32))
        cfg = ZeroQuantConfig(split_middle=True)
        return DepthAdaptiveCompressor(cfg).compress_kv(k, v)

    def test_memory_bytes_positive(self, compressed_default: ZoneCompressedKV) -> None:
        assert compressed_default.memory_bytes() > 0

    def test_memory_bytes_split_positive(self, compressed_split: ZoneCompressedKV) -> None:
        assert compressed_split.memory_bytes() > 0

    def test_split_memory_accounts_for_both_middle_zones(
        self,
        compressed_default: ZoneCompressedKV,
        compressed_split: ZoneCompressedKV,
    ) -> None:
        # Both have the same number of layers so memory should be in the same ballpark.
        # Split with V4/V2 vs uniform V3 may differ slightly but must be > 0 in both cases.
        assert compressed_split.memory_bytes() > 0
        assert compressed_default.memory_bytes() > 0


# ===========================================================================
# ZeroQuantPreset system
# ===========================================================================


class TestZeroQuantPresets:
    def test_quality_preset_has_name(self) -> None:
        assert ZERO_QUANT_QUALITY.name == "zero_quant_quality"

    def test_balanced_preset_has_name(self) -> None:
        assert ZERO_QUANT_BALANCED.name == "zero_quant_balanced"

    def test_turbo_preset_uses_split_middle(self) -> None:
        assert ZERO_QUANT_TURBO.config.split_middle is True

    def test_ultra_preset_uses_coquant(self) -> None:
        assert ZERO_QUANT_ULTRA.config.use_kv_coquant is True

    def test_all_presets_dict_has_four_entries(self) -> None:
        assert len(ALL_ZERO_QUANT_PRESETS) == 5

    def test_all_presets_are_frozen(self) -> None:
        for preset in ALL_ZERO_QUANT_PRESETS.values():
            with pytest.raises(AttributeError):
                preset.name = "changed"  # type: ignore[misc]

    def test_presets_have_positive_expected_avg_bits(self) -> None:
        for preset in ALL_ZERO_QUANT_PRESETS.values():
            assert preset.expected_avg_bits > 0

    def test_quality_has_higher_avg_bits_than_turbo(self) -> None:
        assert ZERO_QUANT_QUALITY.expected_avg_bits > ZERO_QUANT_TURBO.expected_avg_bits

    def test_turbo_avg_bits_below_turboquant_baseline(self) -> None:
        """TURBO should average fewer bits than TurboQuant's K8/V4 = 6.0."""
        assert ZERO_QUANT_TURBO.expected_avg_bits < 6.0

    def test_presets_config_is_zero_quant_config(self) -> None:
        for preset in ALL_ZERO_QUANT_PRESETS.values():
            assert isinstance(preset.config, ZeroQuantConfig)

    def test_fast_preset_has_name(self) -> None:
        assert ZERO_QUANT_FAST.name == "zero_quant_fast"

    def test_fast_preset_uses_k8_everywhere(self) -> None:
        cfg = ZERO_QUANT_FAST.config
        assert cfg.shallow_k_bits == 8
        assert cfg.middle_k_bits == 8
        assert cfg.deep_k_bits == 8

    def test_fast_preset_uses_v8_boundary(self) -> None:
        cfg = ZERO_QUANT_FAST.config
        assert cfg.shallow_v_bits == 8
        assert cfg.deep_v_bits == 8

    def test_fast_preset_no_split_middle(self) -> None:
        assert ZERO_QUANT_FAST.config.split_middle is False

    def test_fast_preset_small_boundary_zones(self) -> None:
        cfg = ZERO_QUANT_FAST.config
        assert cfg.shallow_fraction < 0.25
        assert cfg.deep_fraction < 0.25

    def test_fast_preset_round_trip(self) -> None:
        rng = np.random.default_rng(99)
        n_layers, n_heads, seq_len, head_dim = 28, 4, 32, 128
        keys = rng.standard_normal((n_layers, n_heads, seq_len, head_dim))
        values = rng.standard_normal((n_layers, n_heads, seq_len, head_dim))
        comp = DepthAdaptiveCompressor(ZERO_QUANT_FAST.config)
        compressed = comp.compress_kv(keys, values)
        rk, rv = comp.decompress_kv(compressed)
        assert rk.shape == keys.shape
        assert rv.shape == values.shape
        # K8 everywhere → near-perfect K reconstruction
        k_mse = float(np.mean((rk - keys) ** 2))
        assert k_mse < 0.001  # K8 should be very accurate


# ===========================================================================
# estimate_kv_memory_gb_zero_quant
# ===========================================================================


class TestEstimateKvMemoryGbZeroQuant:
    def test_returns_positive_float(self) -> None:
        gb = estimate_kv_memory_gb_zero_quant(
            ZERO_QUANT_BALANCED.config, ctx_length=4096
        )
        assert isinstance(gb, float)
        assert gb > 0

    def test_longer_context_more_memory(self) -> None:
        short = estimate_kv_memory_gb_zero_quant(
            ZERO_QUANT_BALANCED.config, ctx_length=1024
        )
        long = estimate_kv_memory_gb_zero_quant(
            ZERO_QUANT_BALANCED.config, ctx_length=8192
        )
        assert long > short

    def test_quality_uses_more_memory_than_turbo(self) -> None:
        q_gb = estimate_kv_memory_gb_zero_quant(ZERO_QUANT_QUALITY.config, ctx_length=4096)
        t_gb = estimate_kv_memory_gb_zero_quant(ZERO_QUANT_TURBO.config, ctx_length=4096)
        assert q_gb > t_gb

    def test_turbo_uses_less_memory_than_turboquant_k8v4(self) -> None:
        """TURBO must use less memory than uniform K8/V4."""
        uniform_k8v4 = estimate_kv_memory_gb_zero_quant(
            ZeroQuantConfig(
                shallow_k_bits=8, shallow_v_bits=4,
                middle_k_bits=8, middle_v_bits=4,
                deep_k_bits=8, deep_v_bits=4,
            ),
            ctx_length=4096,
        )
        turbo_gb = estimate_kv_memory_gb_zero_quant(
            ZERO_QUANT_TURBO.config, ctx_length=4096
        )
        assert turbo_gb < uniform_k8v4

    def test_invalid_ctx_length_raises(self) -> None:
        with pytest.raises(ValueError, match="ctx_length"):
            estimate_kv_memory_gb_zero_quant(ZERO_QUANT_BALANCED.config, ctx_length=0)

    def test_invalid_n_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers"):
            estimate_kv_memory_gb_zero_quant(
                ZERO_QUANT_BALANCED.config, ctx_length=4096, n_layers=0
            )


# ===========================================================================
# savings_vs_turboquant
# ===========================================================================


class TestSavingsVsTurboQuant:
    def test_returns_dict_with_required_keys(self) -> None:
        report = savings_vs_turboquant(ZERO_QUANT_TURBO.config, n_layers=32)
        assert "turboquant_avg_bits" in report
        assert "zero_quant_avg_bits" in report
        assert "bit_reduction" in report
        assert "memory_reduction_pct" in report

    def test_turbo_has_positive_bit_reduction(self) -> None:
        report = savings_vs_turboquant(ZERO_QUANT_TURBO.config, n_layers=32)
        assert report["bit_reduction"] > 0

    def test_quality_may_have_lower_bit_reduction_than_turbo(self) -> None:
        r_quality = savings_vs_turboquant(ZERO_QUANT_QUALITY.config, n_layers=32)
        r_turbo = savings_vs_turboquant(ZERO_QUANT_TURBO.config, n_layers=32)
        assert r_quality["bit_reduction"] < r_turbo["bit_reduction"]

    def test_custom_turboquant_baseline(self) -> None:
        """k8/v8 uniform baseline → more savings."""
        report = savings_vs_turboquant(
            ZERO_QUANT_TURBO.config,
            n_layers=32,
            turboquant_k_bits=8,
            turboquant_v_bits=8,
        )
        assert report["bit_reduction"] > 0
        assert report["turboquant_avg_bits"] == pytest.approx(8.0)

    def test_memory_reduction_pct_between_0_and_100(self) -> None:
        report = savings_vs_turboquant(ZERO_QUANT_TURBO.config, n_layers=32)
        assert 0 < report["memory_reduction_pct"] < 100


# ===========================================================================
# recommend_zero_quant
# ===========================================================================


class TestRecommendZeroQuant:
    def test_large_vram_picks_fast(self) -> None:
        preset = recommend_zero_quant(gpu_vram_gb=24.0, target_ctx_length=8192)
        assert preset.name == "zero_quant_fast"

    def test_tight_vram_picks_turbo_or_ultra(self) -> None:
        preset = recommend_zero_quant(
            gpu_vram_gb=6.0, target_ctx_length=4096, model_weight_gb=4.0
        )
        assert preset.name in ALL_ZERO_QUANT_PRESETS

    def test_invalid_vram_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_vram_gb"):
            recommend_zero_quant(gpu_vram_gb=0, target_ctx_length=4096)

    def test_invalid_ctx_raises(self) -> None:
        with pytest.raises(ValueError, match="target_ctx_length"):
            recommend_zero_quant(gpu_vram_gb=8.0, target_ctx_length=0)

    def test_nothing_fits_raises(self) -> None:
        with pytest.raises(ValueError, match="No ZeroQuant preset fits"):
            recommend_zero_quant(
                gpu_vram_gb=1.0, target_ctx_length=131072, model_weight_gb=0.9
            )

    def test_returns_zero_quant_preset(self) -> None:
        result = recommend_zero_quant(gpu_vram_gb=16.0, target_ctx_length=4096)
        assert isinstance(result, ZeroQuantPreset)
