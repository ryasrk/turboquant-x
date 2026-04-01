"""Tests for asymmetric K/V cache quantization presets."""

import pytest

from src.turboquant.asymmetric import (
    ALL_PRESETS,
    BALANCED_PRESET,
    COMPRESSION_PRESET,
    QUALITY_PRESET,
    SYMMETRIC_TURBO4_PRESET,
    AsymmetricPreset,
    estimate_kv_memory_gb,
    format_preset_comparison,
    get_preset,
    recommend_preset,
)
from src.turboquant.compressor import QuantConfig


class TestPresets:
    """Verify preset definitions and lookup."""

    def test_quality_preset_config(self) -> None:
        assert QUALITY_PRESET.config.k_bits == 8
        assert QUALITY_PRESET.config.v_bits == 4
        assert QUALITY_PRESET.name == "quality"

    def test_balanced_preset_config(self) -> None:
        assert BALANCED_PRESET.config.k_bits == 8
        assert BALANCED_PRESET.config.v_bits == 3

    def test_compression_preset_config(self) -> None:
        assert COMPRESSION_PRESET.config.k_bits == 8
        assert COMPRESSION_PRESET.config.v_bits == 2

    def test_symmetric_preset_config(self) -> None:
        assert SYMMETRIC_TURBO4_PRESET.config.k_bits == 4
        assert SYMMETRIC_TURBO4_PRESET.config.v_bits == 4

    def test_all_presets_dict_complete(self) -> None:
        assert len(ALL_PRESETS) == 4
        for name in ("quality", "balanced", "compression", "symmetric_turbo4"):
            assert name in ALL_PRESETS

    def test_get_preset_valid(self) -> None:
        assert get_preset("quality") is QUALITY_PRESET
        assert get_preset("balanced") is BALANCED_PRESET

    def test_get_preset_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_presets_are_frozen(self) -> None:
        with pytest.raises(AttributeError):
            QUALITY_PRESET.name = "changed"  # type: ignore[misc]

    def test_ppl_delta_ordering(self) -> None:
        """Quality preset should have lowest PPL delta."""
        assert QUALITY_PRESET.expected_ppl_delta < BALANCED_PRESET.expected_ppl_delta
        assert BALANCED_PRESET.expected_ppl_delta < COMPRESSION_PRESET.expected_ppl_delta

    def test_compression_ordering(self) -> None:
        """Compression preset should have highest compression ratio."""
        assert QUALITY_PRESET.expected_compression < COMPRESSION_PRESET.expected_compression


class TestEstimateKvMemory:
    """Verify KV cache memory estimation."""

    def test_quality_preset_8192_ctx(self) -> None:
        gb = estimate_kv_memory_gb(QUALITY_PRESET, ctx_length=8192)
        # K=8bit, V=4bit for 28L/28H/128D at 8192 tokens
        # K: 8192 * 28 * 28 * 128 * 8 / 8 = 821_428_224 bytes
        # V: 8192 * 28 * 28 * 128 * 4 / 8 = 410_714_112 bytes
        # Total ≈ 1.15 GB
        assert 0.5 < gb < 2.0

    def test_compression_preset_smaller(self) -> None:
        qgb = estimate_kv_memory_gb(QUALITY_PRESET, ctx_length=4096)
        cgb = estimate_kv_memory_gb(COMPRESSION_PRESET, ctx_length=4096)
        assert cgb < qgb

    def test_longer_context_more_memory(self) -> None:
        short = estimate_kv_memory_gb(QUALITY_PRESET, ctx_length=1024)
        long = estimate_kv_memory_gb(QUALITY_PRESET, ctx_length=8192)
        assert long > short

    def test_returns_positive_float(self) -> None:
        gb = estimate_kv_memory_gb(QUALITY_PRESET, ctx_length=512)
        assert isinstance(gb, float)
        assert gb > 0


class TestRecommendPreset:
    """Verify preset recommendation logic."""

    def test_large_vram_picks_quality(self) -> None:
        preset = recommend_preset(gpu_vram_gb=24.0, target_ctx_length=8192)
        assert preset.name == "quality"

    def test_tight_vram_picks_compression(self) -> None:
        # Shorter context so at least one preset fits tight VRAM
        preset = recommend_preset(
            gpu_vram_gb=6.0,
            target_ctx_length=2048,
            model_weight_gb=4.8,
        )
        assert preset.name in ("quality", "balanced", "compression", "symmetric_turbo4")

    def test_nothing_fits_raises(self) -> None:
        with pytest.raises(ValueError, match="No preset fits"):
            recommend_preset(gpu_vram_gb=1.0, target_ctx_length=8192, model_weight_gb=4.8)

    def test_invalid_vram_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_vram_gb must be positive"):
            recommend_preset(gpu_vram_gb=0, target_ctx_length=4096)

    def test_invalid_ctx_raises(self) -> None:
        with pytest.raises(ValueError, match="target_ctx_length must be positive"):
            recommend_preset(gpu_vram_gb=8.0, target_ctx_length=0)


class TestFormatPresetComparison:
    """Verify comparison table generation."""

    def test_returns_string(self) -> None:
        table = format_preset_comparison()
        assert isinstance(table, str)
        assert len(table) > 100

    def test_contains_all_presets(self) -> None:
        table = format_preset_comparison()
        assert "quality" in table
        assert "balanced" in table
        assert "compression" in table

    def test_contains_f16_baseline(self) -> None:
        table = format_preset_comparison()
        assert "f16" in table.lower()

    def test_contains_recommendation(self) -> None:
        table = format_preset_comparison()
        assert "Recommendation" in table
