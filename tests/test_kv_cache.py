"""Tests for KV cache configuration and memory estimation."""

from __future__ import annotations

import pytest

from unittest.mock import MagicMock, patch

from src.engine.kv_cache import (
    MODEL_ARCHITECTURES,
    CacheType,
    KVCacheConfig,
    compare_configs,
    estimate_kv_memory_bytes,
    estimate_kv_memory_gb,
    get_baseline_config,
    get_turboquant_config,
    is_turbo_fork_available,
    to_llama_params,
)


class TestCacheType:
    """Tests for CacheType enum."""

    def test_all_enum_values_accessible(self) -> None:
        assert CacheType.F16
        assert CacheType.Q8_0
        assert CacheType.Q4_0
        assert CacheType.TURBO4
        assert CacheType.TURBO3
        assert CacheType.TURBO2

    def test_string_values_match_expected(self) -> None:
        assert CacheType.F16.value == "f16"
        assert CacheType.Q8_0.value == "q8_0"
        assert CacheType.Q4_0.value == "q4_0"
        assert CacheType.TURBO4.value == "turbo4"
        assert CacheType.TURBO3.value == "turbo3"
        assert CacheType.TURBO2.value == "turbo2"

    def test_str_enum_is_string(self) -> None:
        assert isinstance(CacheType.F16, str)
        assert CacheType.TURBO4 == "turbo4"

    def test_enum_count(self) -> None:
        assert len(CacheType) == 6


class TestKVCacheConfig:
    """Tests for KVCacheConfig dataclass."""

    def test_default_config(self) -> None:
        cfg = KVCacheConfig()
        assert cfg.cache_type_k == CacheType.Q8_0
        assert cfg.cache_type_v == CacheType.Q4_0
        assert cfg.flash_attention is True

    def test_frozen_immutable(self) -> None:
        cfg = KVCacheConfig()
        with pytest.raises(AttributeError):
            cfg.cache_type_k = CacheType.F16  # type: ignore[misc]

    def test_turbo_k_without_flash_raises(self) -> None:
        with pytest.raises(ValueError, match="flash_attention must be True"):
            KVCacheConfig(
                cache_type_k=CacheType.TURBO4,
                cache_type_v=CacheType.F16,
                flash_attention=False,
            )

    def test_turbo_v_without_flash_raises(self) -> None:
        with pytest.raises(ValueError, match="flash_attention must be True"):
            KVCacheConfig(
                cache_type_k=CacheType.F16,
                cache_type_v=CacheType.TURBO2,
                flash_attention=False,
            )

    def test_turbo3_without_flash_raises(self) -> None:
        with pytest.raises(ValueError, match="flash_attention must be True"):
            KVCacheConfig(
                cache_type_k=CacheType.TURBO3,
                cache_type_v=CacheType.TURBO3,
                flash_attention=False,
            )

    def test_non_turbo_without_flash_ok(self) -> None:
        cfg = KVCacheConfig(
            cache_type_k=CacheType.F16,
            cache_type_v=CacheType.Q8_0,
            flash_attention=False,
        )
        assert cfg.flash_attention is False

    def test_q4_0_without_flash_ok(self) -> None:
        cfg = KVCacheConfig(
            cache_type_k=CacheType.Q4_0,
            cache_type_v=CacheType.Q4_0,
            flash_attention=False,
        )
        assert cfg.cache_type_k == CacheType.Q4_0

    @pytest.mark.parametrize("ct", list(CacheType))
    def test_all_cache_types_accepted_with_flash(self, ct: CacheType) -> None:
        cfg = KVCacheConfig(cache_type_k=ct, cache_type_v=ct, flash_attention=True)
        assert cfg.cache_type_k == ct
        assert cfg.cache_type_v == ct


class TestGetConfigs:
    """Tests for config factory functions."""

    def test_get_turboquant_config_defaults(self) -> None:
        cfg = get_turboquant_config()
        assert cfg.cache_type_k == CacheType.Q8_0
        assert cfg.cache_type_v == CacheType.TURBO4
        assert cfg.flash_attention is True

    def test_get_baseline_config_f16(self) -> None:
        cfg = get_baseline_config()
        assert cfg.cache_type_k == CacheType.F16
        assert cfg.cache_type_v == CacheType.F16
        assert cfg.flash_attention is True


class TestToLlamaParams:
    """Tests for to_llama_params conversion."""

    def test_baseline_config_returns_dict_with_flash_attn(self) -> None:
        params = to_llama_params(get_baseline_config())
        assert "flash_attn" in params
        assert params["flash_attn"] is True

    def test_baseline_k_and_v_types_are_integers(self) -> None:
        params = to_llama_params(get_baseline_config())
        assert isinstance(params["type_k"], int)
        assert isinstance(params["type_v"], int)

    def test_baseline_config_params(self) -> None:
        params = to_llama_params(get_baseline_config())
        assert params["flash_attn"] is True
        assert params["type_k"] == 1  # GGML_TYPE_F16
        assert params["type_v"] == 1  # GGML_TYPE_F16

    def test_flash_false_in_params(self) -> None:
        cfg = KVCacheConfig(CacheType.F16, CacheType.F16, flash_attention=False)
        params = to_llama_params(cfg)
        assert params["flash_attn"] is False

    def test_turbo_type_raises_without_fork(self) -> None:
        """TurboQuant types require the fork; stock llama-cpp-python must error."""
        with pytest.raises(RuntimeError, match="TurboQuant cache type"):
            to_llama_params(get_turboquant_config())

    def test_turbo_type_works_with_fork(self) -> None:
        """When the fork exposes GGML_TYPE_TURBO4 etc., to_llama_params succeeds."""
        mock_llama = MagicMock()
        mock_llama.GGML_TYPE_F16 = 1
        mock_llama.GGML_TYPE_Q8_0 = 8
        mock_llama.GGML_TYPE_Q4_0 = 2
        mock_llama.GGML_TYPE_TURBO4 = 100

        with patch.dict("sys.modules", {"llama_cpp": mock_llama}):
            params = to_llama_params(get_turboquant_config())
            assert params["type_k"] == 8    # Q8_0
            assert params["type_v"] == 100  # Fork's TURBO4

    def test_standard_config_params(self) -> None:
        """Standard Q8_0/Q4_0 config works with stock llama-cpp-python."""
        cfg = KVCacheConfig(CacheType.Q8_0, CacheType.Q4_0, flash_attention=True)
        params = to_llama_params(cfg)
        assert params["type_k"] == 8  # GGML_TYPE_Q8_0
        assert params["type_v"] == 2  # GGML_TYPE_Q4_0

    def test_is_turbo_fork_available_returns_false_for_stock(self) -> None:
        assert is_turbo_fork_available() is False

    def test_is_turbo_fork_available_returns_true_for_fork(self) -> None:
        mock_llama = MagicMock()
        mock_llama.GGML_TYPE_TURBO4 = 100
        with patch.dict("sys.modules", {"llama_cpp": mock_llama}):
            assert is_turbo_fork_available() is True

    def test_is_turbo_fork_available_returns_false_when_not_installed(self) -> None:
        """ImportError path — llama_cpp not installed at all."""
        with patch.dict("sys.modules", {"llama_cpp": None}):
            assert is_turbo_fork_available() is False

    def test_cache_type_to_ggml_int_importerror_standard(self) -> None:
        """When llama_cpp missing, standard types return well-known IDs."""
        from src.engine.kv_cache import _cache_type_to_ggml_int

        with patch.dict("sys.modules", {"llama_cpp": None}):
            assert _cache_type_to_ggml_int(CacheType.Q8_0) == 8
            assert _cache_type_to_ggml_int(CacheType.F16) == 1
            assert _cache_type_to_ggml_int(CacheType.Q4_0) == 2

    def test_cache_type_to_ggml_int_importerror_turbo(self) -> None:
        """When llama_cpp missing, turbo types return provisional IDs."""
        from src.engine.kv_cache import _cache_type_to_ggml_int

        with patch.dict("sys.modules", {"llama_cpp": None}):
            assert _cache_type_to_ggml_int(CacheType.TURBO4) == 100
            assert _cache_type_to_ggml_int(CacheType.TURBO3) == 101
            assert _cache_type_to_ggml_int(CacheType.TURBO2) == 102


class TestMemoryEstimation:
    """Tests for memory estimation functions."""

    # Qwen2.5-7B architecture
    N_LAYERS = 28
    N_HEADS = 28
    HEAD_DIM = 128
    N_CTX = 8192

    def test_f16_baseline_qwen25_7b(self) -> None:
        """F16 baseline for Qwen2.5-7B at 8192 context should be ~3.06 GB."""
        result = estimate_kv_memory_bytes(
            n_ctx=self.N_CTX,
            n_layers=self.N_LAYERS,
            n_heads=self.N_HEADS,
            head_dim=self.HEAD_DIM,
            config=get_baseline_config(),
        )
        assert 2.5 < result["total_gb"] < 3.5

    def test_turboquant_config_smaller_than_baseline(self) -> None:
        """q8_0/turbo4 should use ~1.22 GB — much less than f16 baseline."""
        result = estimate_kv_memory_bytes(
            n_ctx=self.N_CTX,
            n_layers=self.N_LAYERS,
            n_heads=self.N_HEADS,
            head_dim=self.HEAD_DIM,
            config=get_turboquant_config(),
        )
        assert 1.0 < result["total_gb"] < 1.5

    def test_longer_context_proportionally_more_memory(self) -> None:
        cfg = get_turboquant_config()
        mem_short = estimate_kv_memory_bytes(
            4096, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM, cfg
        )
        mem_long = estimate_kv_memory_bytes(
            8192, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM, cfg
        )
        ratio = mem_long["total_bytes"] / mem_short["total_bytes"]
        assert abs(ratio - 2.0) < 0.01

    def test_returns_all_expected_keys(self) -> None:
        result = estimate_kv_memory_bytes(
            self.N_CTX, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM,
            get_turboquant_config(),
        )
        expected_keys = {"k_bytes", "v_bytes", "total_bytes", "total_mb", "total_gb", "compression_vs_f16"}
        assert set(result.keys()) == expected_keys

    def test_compression_vs_f16_greater_than_one_for_quant(self) -> None:
        result = estimate_kv_memory_bytes(
            self.N_CTX, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM,
            get_turboquant_config(),
        )
        assert result["compression_vs_f16"] > 1.0

    def test_compression_vs_f16_equals_one_for_baseline(self) -> None:
        result = estimate_kv_memory_bytes(
            self.N_CTX, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM,
            get_baseline_config(),
        )
        assert abs(result["compression_vs_f16"] - 1.0) < 0.01

    def test_k_bytes_plus_v_bytes_equals_total(self) -> None:
        result = estimate_kv_memory_bytes(
            self.N_CTX, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM,
            get_turboquant_config(),
        )
        assert result["k_bytes"] + result["v_bytes"] == result["total_bytes"]

    def test_estimate_kv_memory_gb_convenience(self) -> None:
        gb = estimate_kv_memory_gb(self.N_CTX, get_turboquant_config())
        full = estimate_kv_memory_bytes(
            self.N_CTX, self.N_LAYERS, self.N_HEADS, self.HEAD_DIM,
            get_turboquant_config(),
        )
        assert abs(gb - full["total_gb"]) < 1e-9

    def test_estimate_kv_memory_gb_custom_arch(self) -> None:
        arch = MODEL_ARCHITECTURES["llama3-8b"]
        gb = estimate_kv_memory_gb(8192, get_baseline_config(), **arch)
        assert gb > 0


class TestCompareConfigs:
    """Tests for compare_configs table generation."""

    def test_returns_non_empty_string(self) -> None:
        table = compare_configs()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_f16(self) -> None:
        table = compare_configs()
        assert "f16" in table

    def test_contains_turbo4(self) -> None:
        table = compare_configs()
        assert "turbo4" in table

    def test_contains_header_columns(self) -> None:
        table = compare_configs()
        assert "K Type" in table
        assert "V Type" in table
        assert "KV Cache (MB)" in table
        assert "vs f16" in table

    def test_multiple_rows(self) -> None:
        table = compare_configs()
        lines = [l for l in table.strip().split("\n") if l.startswith("|")]
        # header + separator + at least 3 config rows
        assert len(lines) >= 5
