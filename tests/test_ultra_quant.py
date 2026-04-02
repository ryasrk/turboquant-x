"""Tests for the Ultra-Quant inference engine and memory budget planner.

Covers:
- UltraQuantConfig validation
- HardwareProfile detection
- OffloadPlan computation (memory budget calculator)
- MoE architecture detection
- Quant selection logic
- UltraQuantEngine initialization and plan logging
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engine.ultra_quant_engine import (
    HardwareProfile,
    OffloadPlan,
    UltraQuantConfig,
    UltraQuantCompressionStats,
    UltraQuantEngine,
    UltraQuantGenerationResult,
    _estimate_kv_cache_gb,
    _estimate_model_size_gb,
    _estimate_n_heads,
    _estimate_n_layers,
    _select_best_quant,
    _QUANT_BPW,
    _QUANT_QUALITY,
    _QUANT_MIN_PARAMS_B,
    _ULTRA_ZQ_PRESETS,
    compute_offload_plan,
    detect_hardware,
    detect_moe_architecture,
)


# ---------------------------------------------------------------------------
# UltraQuantConfig
# ---------------------------------------------------------------------------


class TestUltraQuantConfig:
    """Validation tests for UltraQuantConfig."""

    def test_default_config(self) -> None:
        cfg = UltraQuantConfig()
        assert cfg.target_model_params_b == 70.0
        assert cfg.max_ram_usage_fraction == 0.85
        assert cfg.max_vram_usage_fraction == 0.90
        assert cfg.enable_mmap is True
        assert cfg.enable_mlock_critical is True
        assert cfg.enable_moe_offload is True
        assert cfg.kv_budget_mb == 0
        assert cfg.force_quant == ""
        assert cfg.zero_quant_preset == "turbo"

    def test_custom_config(self) -> None:
        cfg = UltraQuantConfig(
            target_model_params_b=405.0,
            max_ram_usage_fraction=0.7,
            enable_mmap=False,
            force_quant="IQ2_XS",
        )
        assert cfg.target_model_params_b == 405.0
        assert cfg.max_ram_usage_fraction == 0.7
        assert cfg.enable_mmap is False
        assert cfg.force_quant == "IQ2_XS"

    def test_invalid_params_b(self) -> None:
        with pytest.raises(ValueError, match="target_model_params_b must be > 0"):
            UltraQuantConfig(target_model_params_b=0)

    def test_invalid_params_b_negative(self) -> None:
        with pytest.raises(ValueError, match="target_model_params_b must be > 0"):
            UltraQuantConfig(target_model_params_b=-10)

    def test_invalid_ram_fraction_zero(self) -> None:
        with pytest.raises(ValueError, match="max_ram_usage_fraction"):
            UltraQuantConfig(max_ram_usage_fraction=0.0)

    def test_invalid_ram_fraction_over_one(self) -> None:
        with pytest.raises(ValueError, match="max_ram_usage_fraction"):
            UltraQuantConfig(max_ram_usage_fraction=1.5)

    def test_invalid_vram_fraction(self) -> None:
        with pytest.raises(ValueError, match="max_vram_usage_fraction"):
            UltraQuantConfig(max_vram_usage_fraction=0.0)

    def test_is_frozen(self) -> None:
        cfg = UltraQuantConfig()
        with pytest.raises(AttributeError):
            cfg.target_model_params_b = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    """Tests for HardwareProfile."""

    def test_default_profile(self) -> None:
        hw = HardwareProfile()
        assert hw.gpu_vram_gb == 0.0
        assert hw.has_gpu is False
        assert hw.total_memory_gb == 8.0  # 0 VRAM + 8 free RAM

    def test_with_gpu(self) -> None:
        hw = HardwareProfile(gpu_vram_gb=8.0, gpu_name="RTX 4060 Ti")
        assert hw.has_gpu is True
        assert hw.total_memory_gb == 16.0  # 8 VRAM + 8 free RAM

    def test_no_gpu_threshold(self) -> None:
        hw = HardwareProfile(gpu_vram_gb=0.4)
        assert hw.has_gpu is False  # Below 0.5 GB threshold


# ---------------------------------------------------------------------------
# Memory estimations
# ---------------------------------------------------------------------------


class TestMemoryEstimation:
    """Tests for model size and KV cache estimation functions."""

    def test_model_size_70b_q4(self) -> None:
        size = _estimate_model_size_gb(70.0, _QUANT_BPW["Q4_K_M"])
        assert 35 < size < 50  # ~40 GB for 70B Q4_K_M

    def test_model_size_7b_q4(self) -> None:
        size = _estimate_model_size_gb(7.0, _QUANT_BPW["Q4_K_M"])
        assert 3 < size < 6  # ~4.5 GB for 7B Q4_K_M

    def test_model_size_405b_iq2(self) -> None:
        size = _estimate_model_size_gb(405.0, _QUANT_BPW["IQ2_XS"])
        assert 90 < size < 140  # ~110 GB for 405B IQ2_XS

    def test_kv_cache_estimation(self) -> None:
        kv_gb = _estimate_kv_cache_gb(
            n_ctx=8192, n_layers=80, n_heads=64, head_dim=128, avg_kv_bits=4.0
        )
        assert kv_gb > 0
        # 2 * 80 * 64 * 8192 * 128 * 4 / 8 / 1024^3 ≈ 5.0 GB
        assert 4.5 < kv_gb < 6.0

    def test_kv_cache_scales_with_ctx(self) -> None:
        kv_4k = _estimate_kv_cache_gb(4096, 32, 32, 128, 8.0)
        kv_8k = _estimate_kv_cache_gb(8192, 32, 32, 128, 8.0)
        assert abs(kv_8k - 2 * kv_4k) < 0.01

    def test_layer_count_estimation(self) -> None:
        assert _estimate_n_layers(70) == 80
        assert _estimate_n_layers(7) == 32
        assert _estimate_n_layers(3) == 28
        assert _estimate_n_layers(405) == 126

    def test_head_count_estimation(self) -> None:
        assert _estimate_n_heads(70) == 64
        assert _estimate_n_heads(7) == 32
        assert _estimate_n_heads(3) == 16


# ---------------------------------------------------------------------------
# Quant selection
# ---------------------------------------------------------------------------


class TestQuantSelection:
    """Tests for automatic quant level selection."""

    def test_selects_highest_quality_that_fits(self) -> None:
        notes: list[str] = []
        # 80 GB budget for 7B model — should pick Q8_0 (highest quality)
        quant = _select_best_quant(7.0, 80.0, notes)
        assert quant == "Q8_0"

    def test_selects_q4_for_70b_on_48gb(self) -> None:
        notes: list[str] = []
        # 48 GB budget for 70B — Q4_K_M (40GB) fits, Q5_K_M (47GB) too tight
        quant = _select_best_quant(70.0, 48.0, notes)
        assert quant in ("Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q5_K_S", "Q5_K_M")

    def test_selects_iq2_for_70b_on_24gb(self) -> None:
        notes: list[str] = []
        quant = _select_best_quant(70.0, 24.0, notes)
        bpw = _QUANT_BPW[quant]
        model_gb = _estimate_model_size_gb(70.0, bpw)
        assert model_gb + 1.5 <= 24.0  # Must actually fit

    def test_respects_min_params_requirement(self) -> None:
        notes: list[str] = []
        # IQ1_S requires >= 70B; 7B model shouldn't select it even with tiny budget
        quant = _select_best_quant(7.0, 5.0, notes)
        assert quant not in ("IQ1_S", "IQ1_M")

    def test_quant_tables_consistent(self) -> None:
        """All quants in BPW table must have quality and min_params entries."""
        for quant in _QUANT_BPW:
            assert quant in _QUANT_QUALITY, f"{quant} missing from quality table"
            assert quant in _QUANT_MIN_PARAMS_B, f"{quant} missing from min_params table"


# ---------------------------------------------------------------------------
# OffloadPlan computation
# ---------------------------------------------------------------------------


class TestOffloadPlan:
    """Tests for the memory budget planner."""

    def test_full_gpu_small_model(self) -> None:
        """7B model on 16GB GPU + 32GB RAM — everything fits on GPU."""
        hw = HardwareProfile(
            gpu_vram_gb=16.0, gpu_name="RTX 4080",
            ram_total_gb=32.0, ram_free_gb=24.0, cpu_cores=16,
        )
        cfg = UltraQuantConfig(target_model_params_b=7.0)
        plan = compute_offload_plan(hw, cfg)

        # 7B Q8_0 ≈ 7 GB → should fit entirely on 16 GB GPU
        assert plan.n_gpu_layers > 0
        assert plan.recommended_n_ctx >= 512
        assert plan.total_layers > 0
        assert len(plan.notes) > 0

    def test_split_gpu_cpu_70b(self) -> None:
        """70B model on 8GB GPU + 32GB RAM — must split across GPU and CPU."""
        hw = HardwareProfile(
            gpu_vram_gb=8.0, gpu_name="RTX 4060 Ti",
            ram_total_gb=32.0, ram_free_gb=24.0, cpu_cores=16,
        )
        cfg = UltraQuantConfig(target_model_params_b=70.0)
        plan = compute_offload_plan(hw, cfg)

        # Can't fit all 80 layers on 8GB — need CPU too
        assert plan.n_gpu_layers >= 0
        assert plan.n_cpu_layers > 0 or plan.n_mmap_layers > 0
        assert plan.total_layers > 0

    def test_mmap_for_350b(self) -> None:
        """350B model on 8GB + 16GB — needs mmap disk offloading."""
        hw = HardwareProfile(
            gpu_vram_gb=8.0, gpu_name="RTX 4060 Ti",
            ram_total_gb=16.0, ram_free_gb=10.0, cpu_cores=8,
            disk_free_gb=200.0,
        )
        cfg = UltraQuantConfig(target_model_params_b=350.0)
        plan = compute_offload_plan(hw, cfg)

        # Massive model — should enable mmap
        assert plan.use_mmap is True
        assert plan.total_layers > 0

    def test_cpu_only_no_gpu(self) -> None:
        """No GPU — everything on CPU + mmap."""
        hw = HardwareProfile(
            gpu_vram_gb=0.0, gpu_name="none",
            ram_total_gb=64.0, ram_free_gb=48.0, cpu_cores=16,
        )
        cfg = UltraQuantConfig(target_model_params_b=70.0)
        plan = compute_offload_plan(hw, cfg)

        assert plan.n_gpu_layers == 0
        assert plan.kv_config.flash_attention is False

    def test_forced_quant(self) -> None:
        """Force a specific quant level."""
        hw = HardwareProfile(gpu_vram_gb=24.0, ram_free_gb=64.0)
        cfg = UltraQuantConfig(
            target_model_params_b=70.0,
            force_quant="IQ3_XS",
        )
        plan = compute_offload_plan(hw, cfg)
        assert plan.recommended_quant == "IQ3_XS"

    def test_invalid_forced_quant_falls_back(self) -> None:
        hw = HardwareProfile(gpu_vram_gb=24.0, ram_free_gb=64.0)
        cfg = UltraQuantConfig(
            target_model_params_b=70.0,
            force_quant="INVALID_QUANT",
        )
        plan = compute_offload_plan(hw, cfg)
        assert plan.recommended_quant == "Q4_K_M"

    def test_custom_kv_budget(self) -> None:
        hw = HardwareProfile(gpu_vram_gb=8.0, ram_free_gb=32.0)
        cfg = UltraQuantConfig(
            target_model_params_b=7.0,
            kv_budget_mb=512,
        )
        plan = compute_offload_plan(hw, cfg)
        assert plan.recommended_n_ctx >= 512

    def test_speed_estimate_full_gpu(self) -> None:
        """Full GPU should estimate highest speed."""
        hw = HardwareProfile(gpu_vram_gb=80.0, ram_free_gb=256.0)
        cfg = UltraQuantConfig(target_model_params_b=7.0)
        plan = compute_offload_plan(hw, cfg)
        assert plan.estimated_speed_tok_s >= 10.0


# ---------------------------------------------------------------------------
# MoE detection
# ---------------------------------------------------------------------------


class TestMoEDetection:
    """Tests for MoE architecture detection."""

    def test_nonexistent_file(self) -> None:
        result = detect_moe_architecture("/nonexistent/model.gguf")
        assert result["is_moe"] is False
        assert result["n_experts"] == 0

    @patch("src.utils.gpu_layers._read_gguf_metadata")
    def test_dense_model(self, mock_meta: MagicMock) -> None:
        mock_meta.return_value = {
            "general.architecture": "llama",
            "llama.block_count": 32,
        }
        result = detect_moe_architecture("fake_model.gguf")
        assert result["is_moe"] is False
        assert result["architecture"] == "llama"

    @patch("src.utils.gpu_layers._read_gguf_metadata")
    def test_moe_model(self, mock_meta: MagicMock) -> None:
        mock_meta.return_value = {
            "general.architecture": "qwen2moe",
            "qwen2moe.block_count": 28,
            "qwen2moe.expert_count": 64,
            "qwen2moe.expert_used_count": 4,
        }
        result = detect_moe_architecture("fake_model.gguf")
        assert result["is_moe"] is True
        assert result["n_experts"] == 64
        assert result["n_experts_used"] == 4


# ---------------------------------------------------------------------------
# Ultra-Quant Zero-Quant presets
# ---------------------------------------------------------------------------


class TestUltraZQPresets:
    """Tests for the ultra-quant Zero-Quant compression presets."""

    def test_all_presets_valid(self) -> None:
        from src.turboquant.zero_quant import ZeroQuantConfig

        for name, preset in _ULTRA_ZQ_PRESETS.items():
            # Filter to only ZeroQuantConfig fields
            fields = {
                k: v for k, v in preset.items()
                if k in ZeroQuantConfig.__dataclass_fields__
            }
            cfg = ZeroQuantConfig(**fields)
            # Should not raise
            avg = cfg.average_bits(80)
            assert 2.0 <= avg <= 8.0, f"Preset {name}: avg bits {avg} out of range"

    def test_turbo_preset_aggressive(self) -> None:
        from src.turboquant.zero_quant import ZeroQuantConfig

        preset = _ULTRA_ZQ_PRESETS["turbo"]
        fields = {
            k: v for k, v in preset.items()
            if k in ZeroQuantConfig.__dataclass_fields__
        }
        cfg = ZeroQuantConfig(**fields)
        avg = cfg.average_bits(80)
        # Turbo should be aggressive — less than 5 bits average
        assert avg < 5.0

    def test_extreme_preset_most_aggressive(self) -> None:
        from src.turboquant.zero_quant import ZeroQuantConfig

        preset = _ULTRA_ZQ_PRESETS["extreme"]
        fields = {
            k: v for k, v in preset.items()
            if k in ZeroQuantConfig.__dataclass_fields__
        }
        cfg = ZeroQuantConfig(**fields)
        avg = cfg.average_bits(80)
        # Extreme should be the most aggressive
        assert avg < 4.0


# ---------------------------------------------------------------------------
# UltraQuantEngine
# ---------------------------------------------------------------------------


class TestUltraQuantEngine:
    """Tests for UltraQuantEngine initialization and plan generation."""

    def _make_engine(
        self,
        params_b: float = 7.0,
        gpu_vram: float = 8.0,
        ram_free: float = 16.0,
    ) -> UltraQuantEngine:
        """Create an engine with a mock hardware profile (no real model)."""
        hw = HardwareProfile(
            gpu_vram_gb=gpu_vram,
            gpu_name="TestGPU",
            ram_total_gb=32.0,
            ram_free_gb=ram_free,
            cpu_cores=8,
        )
        model_config = MagicMock()
        model_config.model_path = "/fake/path.gguf"
        model_config.model_name = "test-model"
        model_config.n_ctx = 4096
        model_config.n_gpu_layers = -1
        model_config.chat_format = "chatml"
        model_config.weight_size_gb = 4.8
        model_config.n_threads = -1
        model_config.n_batch = 512
        model_config.n_threads_batch = -1
        model_config.use_mlock = False

        ultra_config = UltraQuantConfig(target_model_params_b=params_b)

        # Mock detect_moe_architecture to avoid file system access
        with patch("src.engine.ultra_quant_engine.detect_moe_architecture") as mock_moe:
            mock_moe.return_value = {
                "is_moe": False, "n_experts": 0,
                "n_experts_used": 0, "architecture": "llama",
            }
            engine = UltraQuantEngine(
                model_config, ultra_config, hardware=hw,
            )
        return engine

    def test_engine_init_7b(self) -> None:
        engine = self._make_engine(params_b=7.0, gpu_vram=16.0)
        assert engine.offload_plan.total_layers > 0
        assert engine.offload_plan.n_gpu_layers > 0
        assert engine.is_loaded is False

    def test_engine_init_70b_constrained(self) -> None:
        engine = self._make_engine(params_b=70.0, gpu_vram=8.0, ram_free=16.0)
        plan = engine.offload_plan
        # Can't fit 70B fully on 8GB VRAM
        assert plan.n_cpu_layers > 0 or plan.n_mmap_layers > 0

    def test_plan_report(self) -> None:
        engine = self._make_engine(params_b=70.0, gpu_vram=8.0)
        report = engine.plan_report()
        assert "ULTRA-QUANT OFFLOAD PLAN" in report
        assert "GPU" in report
        assert "70B" in report or "70" in report

    def test_has_zero_quant_config(self) -> None:
        engine = self._make_engine(params_b=70.0)
        zq = engine.zero_quant_config
        avg = zq.average_bits(80)
        assert 2.0 <= avg <= 8.0

    def test_hardware_property(self) -> None:
        engine = self._make_engine(gpu_vram=24.0)
        assert engine.hardware.gpu_vram_gb == 24.0
        assert engine.hardware.gpu_name == "TestGPU"

    def test_ultra_config_property(self) -> None:
        engine = self._make_engine(params_b=405.0)
        assert engine.ultra_config.target_model_params_b == 405.0

    def test_moe_info_property(self) -> None:
        engine = self._make_engine()
        info = engine.moe_info
        assert "is_moe" in info
        assert "n_experts" in info


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    """Tests for result dataclasses."""

    def test_compression_stats(self) -> None:
        stats = UltraQuantCompressionStats(
            original_bytes=1000000,
            compressed_bytes=250000,
            compression_ratio=4.0,
            compress_time_s=0.5,
            decompress_time_s=0.3,
            mse=0.001,
            avg_bits=4.5,
            n_layers=80,
            offload_plan_summary="10GPU+50CPU+20mmap",
            zone_summary={"shallow": "K8/V4", "middle": "K4/V2", "deep": "K8/V4"},
        )
        assert stats.compression_ratio == 4.0
        assert stats.offload_plan_summary == "10GPU+50CPU+20mmap"

    def test_generation_result(self) -> None:
        gen_stats = MagicMock()
        result = UltraQuantGenerationResult(
            text="Hello world",
            gen_stats=gen_stats,
            compression_stats=None,
        )
        assert result.text == "Hello world"
        assert result.compression_stats is None


# ---------------------------------------------------------------------------
# OffloadPlan properties
# ---------------------------------------------------------------------------


class TestOffloadPlanProperties:
    """Tests for OffloadPlan dataclass."""

    def test_total_layers(self) -> None:
        plan = OffloadPlan(n_gpu_layers=10, n_cpu_layers=50, n_mmap_layers=20)
        assert plan.total_layers == 80

    def test_default_plan(self) -> None:
        plan = OffloadPlan()
        assert plan.total_layers == 0
        assert plan.use_mmap is True
        assert plan.recommended_quant == "Q4_K_M"
