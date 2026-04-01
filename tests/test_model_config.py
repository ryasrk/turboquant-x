"""Tests for model configuration."""

import os

import pytest
from pathlib import Path
from unittest.mock import patch

from src.engine.model_config import (
    MODEL_REGISTRY,
    ModelConfig,
    estimate_total_vram,
    from_env,
    get_default_config,
    validate_model_path,
)


class TestModelConfig:
    """Tests for the ModelConfig dataclass."""

    def test_default_values(self):
        cfg = ModelConfig(model_path="/tmp/model.gguf")
        assert cfg.model_name == "qwen2.5-7b-instruct"
        assert cfg.n_ctx == 8192
        assert cfg.n_gpu_layers == -1
        assert cfg.chat_format == "chatml"
        assert cfg.weight_size_gb == 4.8

    def test_frozen_immutable(self):
        cfg = ModelConfig(model_path="/tmp/model.gguf")
        with pytest.raises(AttributeError):
            cfg.n_ctx = 4096  # type: ignore[misc]

    def test_n_ctx_too_low_raises(self):
        with pytest.raises(ValueError, match="n_ctx must be >= 128"):
            ModelConfig(model_path="/tmp/model.gguf", n_ctx=64)

    def test_n_ctx_too_high_raises(self):
        with pytest.raises(ValueError, match="n_ctx must be <= 131072"):
            ModelConfig(model_path="/tmp/model.gguf", n_ctx=200000)

    def test_n_gpu_layers_too_low_raises(self):
        with pytest.raises(ValueError, match="n_gpu_layers must be >= -1"):
            ModelConfig(model_path="/tmp/model.gguf", n_gpu_layers=-2)

    @pytest.mark.parametrize("layers", [-1, 0, 1, 32])
    def test_valid_n_gpu_layers(self, layers):
        cfg = ModelConfig(model_path="/tmp/model.gguf", n_gpu_layers=layers)
        assert cfg.n_gpu_layers == layers

    def test_custom_values(self):
        cfg = ModelConfig(
            model_path="/opt/models/custom.gguf",
            model_name="qwen2.5-3b-instruct",
            n_ctx=4096,
            n_gpu_layers=20,
            chat_format="chatml",
            weight_size_gb=2.1,
        )
        assert cfg.model_path == "/opt/models/custom.gguf"
        assert cfg.model_name == "qwen2.5-3b-instruct"
        assert cfg.n_ctx == 4096
        assert cfg.n_gpu_layers == 20
        assert cfg.weight_size_gb == 2.1


class TestGetDefaultConfig:
    """Tests for VRAM-based model auto-selection."""

    def test_8gb_selects_7b(self):
        cfg = get_default_config(gpu_vram_gb=8.0)
        assert cfg.model_name == "qwen2.5-7b-instruct"
        assert cfg.weight_size_gb == 4.8

    def test_6gb_selects_7b(self):
        cfg = get_default_config(gpu_vram_gb=6.0)
        assert cfg.model_name == "qwen2.5-7b-instruct"

    def test_4gb_selects_3b(self):
        cfg = get_default_config(gpu_vram_gb=4.0)
        assert cfg.model_name == "qwen2.5-3b-instruct"
        assert cfg.weight_size_gb == 2.1

    def test_3gb_selects_3b(self):
        cfg = get_default_config(gpu_vram_gb=3.0)
        assert cfg.model_name == "qwen2.5-3b-instruct"

    def test_2gb_raises(self):
        with pytest.raises(ValueError, match="No model fits 2.0 GB VRAM"):
            get_default_config(gpu_vram_gb=2.0)

    def test_custom_model_dir(self):
        cfg = get_default_config(gpu_vram_gb=8.0, model_dir="/opt/my-models")
        assert cfg.model_path == "/opt/my-models/qwen2.5-7b-instruct-q4_k_m.gguf"

    def test_default_model_dir(self):
        cfg = get_default_config(gpu_vram_gb=8.0)
        assert "models/" in cfg.model_path or cfg.model_path.startswith("models")


class TestEstimateVram:
    """Tests for VRAM estimation."""

    def test_7b_default_estimate(self):
        cfg = ModelConfig(model_path="/tmp/m.gguf", weight_size_gb=4.8)
        total = estimate_total_vram(cfg)
        assert total == pytest.approx(6.0, abs=0.01)

    def test_3b_default_estimate(self):
        cfg = ModelConfig(model_path="/tmp/m.gguf", weight_size_gb=2.1)
        total = estimate_total_vram(cfg)
        assert total == pytest.approx(3.3, abs=0.01)

    def test_custom_cache_and_overhead(self):
        cfg = ModelConfig(model_path="/tmp/m.gguf", weight_size_gb=4.8)
        total = estimate_total_vram(cfg, kv_cache_gb=1.0, overhead_gb=1.0)
        assert total == pytest.approx(6.8, abs=0.01)

    def test_returns_positive_float(self):
        cfg = ModelConfig(model_path="/tmp/m.gguf", weight_size_gb=0.5)
        total = estimate_total_vram(cfg)
        assert isinstance(total, float)
        assert total > 0


class TestValidateModelPath:
    """Tests for model file existence checking."""

    def test_nonexistent_path_returns_false(self):
        cfg = ModelConfig(model_path="/nonexistent/path/model.gguf")
        assert validate_model_path(cfg) is False

    def test_existing_file_returns_true(self, tmp_path):
        dummy = tmp_path / "test-model.gguf"
        dummy.write_text("fake model data")
        cfg = ModelConfig(model_path=str(dummy))
        assert validate_model_path(cfg) is True


class TestFromEnv:
    """Tests for environment variable–based configuration."""

    def test_explicit_model_path(self):
        env = {"TURBOQUANT_MODEL_PATH": "/custom/model.gguf"}
        with patch.dict(os.environ, env, clear=False):
            cfg = from_env()
        assert cfg.model_path == "/custom/model.gguf"
        assert cfg.model_name == "qwen2.5-7b-instruct"

    def test_explicit_path_with_name(self):
        env = {
            "TURBOQUANT_MODEL_PATH": "/custom/model.gguf",
            "TURBOQUANT_MODEL_NAME": "qwen2.5-3b-instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = from_env()
        assert cfg.model_name == "qwen2.5-3b-instruct"

    def test_n_ctx_override(self):
        env = {
            "TURBOQUANT_MODEL_PATH": "/custom/model.gguf",
            "TURBOQUANT_N_CTX": "4096",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = from_env()
        assert cfg.n_ctx == 4096

    def test_n_gpu_layers_override(self):
        env = {
            "TURBOQUANT_MODEL_PATH": "/custom/model.gguf",
            "TURBOQUANT_N_GPU_LAYERS": "20",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = from_env()
        assert cfg.n_gpu_layers == 20

    def test_default_no_env_vars(self):
        keys_to_remove = [
            "TURBOQUANT_MODEL_PATH",
            "TURBOQUANT_MODEL_NAME",
            "TURBOQUANT_N_CTX",
            "TURBOQUANT_N_GPU_LAYERS",
            "TURBOQUANT_GPU_VRAM_GB",
        ]
        env_clean = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = from_env()
        assert cfg.model_name == "qwen2.5-7b-instruct"
        assert cfg.n_ctx == 8192
        assert cfg.n_gpu_layers == -1

    def test_vram_env_selects_3b(self):
        keys_to_remove = [
            "TURBOQUANT_MODEL_PATH",
            "TURBOQUANT_MODEL_NAME",
            "TURBOQUANT_N_CTX",
            "TURBOQUANT_N_GPU_LAYERS",
        ]
        env_clean = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        env_clean["TURBOQUANT_GPU_VRAM_GB"] = "4.0"
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = from_env()
        assert cfg.model_name == "qwen2.5-3b-instruct"

    def test_auto_select_with_n_ctx_override(self):
        keys_to_remove = [
            "TURBOQUANT_MODEL_PATH",
            "TURBOQUANT_MODEL_NAME",
            "TURBOQUANT_N_GPU_LAYERS",
        ]
        env_clean = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        env_clean["TURBOQUANT_GPU_VRAM_GB"] = "8.0"
        env_clean["TURBOQUANT_N_CTX"] = "16384"
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = from_env()
        assert cfg.model_name == "qwen2.5-7b-instruct"
        assert cfg.n_ctx == 16384


class TestModelRegistry:
    """Tests for the MODEL_REGISTRY data."""

    REQUIRED_KEYS = {
        "hf_repo",
        "filename",
        "chat_format",
        "parameters",
        "weight_quant",
        "weight_size_gb",
        "default_n_ctx",
    }

    def test_all_entries_have_required_keys(self):
        for name, meta in MODEL_REGISTRY.items():
            missing = self.REQUIRED_KEYS - set(meta.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_qwen_7b_entry(self):
        entry = MODEL_REGISTRY["qwen2.5-7b-instruct"]
        assert entry["hf_repo"] == "Qwen/Qwen2.5-7B-Instruct-GGUF"
        assert entry["filename"] == "qwen2.5-7b-instruct-q4_k_m.gguf"
        assert entry["chat_format"] == "chatml"
        assert entry["parameters"] == "7B"
        assert entry["weight_quant"] == "Q4_K_M"
        assert entry["weight_size_gb"] == 4.8
        assert entry["default_n_ctx"] == 8192

    def test_qwen_3b_entry(self):
        entry = MODEL_REGISTRY["qwen2.5-3b-instruct"]
        assert entry["parameters"] == "3B"
        assert entry["weight_size_gb"] == 2.1

    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) >= 2
