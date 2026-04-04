"""Tests for config loading and entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig
from src.main import (
    DEFAULT_CONFIG_PATH,
    apply_env_overrides,
    build_kv_config,
    build_model_config,
    load_config,
    parse_args,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "model": {
        "name": "qwen2.5-7b-instruct",
        "path": "./models/qwen2.5-7b-instruct-q4_k_m.gguf",
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "chat_format": "chatml",
    },
    "kv_cache": {
        "cache_type_k": "q8_0",
        "cache_type_v": "turbo4",
        "flash_attention": True,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "cors_origins": ["http://localhost:3000", "http://localhost:8000"],
    },
    "logging": {"level": "INFO"},
}


@pytest.fixture
def sample_config() -> dict:
    """Return a deep copy of the sample config."""
    import copy
    return copy.deepcopy(SAMPLE_CONFIG)


@pytest.fixture
def yaml_config_file(tmp_path: Path) -> Path:
    """Write sample config to a temp YAML file and return its path."""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(SAMPLE_CONFIG))
    return cfg_file


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config()."""

    def test_loads_yaml_file(self, yaml_config_file: Path) -> None:
        config = load_config(yaml_config_file)
        assert config["model"]["name"] == "qwen2.5-7b-instruct"
        assert config["server"]["port"] == 8000

    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        config = load_config(missing)
        assert config == {}

    def test_valid_yaml_parsed_correctly(self, yaml_config_file: Path) -> None:
        config = load_config(yaml_config_file)
        assert isinstance(config, dict)
        assert "model" in config
        assert "kv_cache" in config
        assert "server" in config
        assert "logging" in config
        assert config["kv_cache"]["flash_attention"] is True
        assert config["server"]["cors_origins"] == [
            "http://localhost:3000",
            "http://localhost:8000",
        ]

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        config = load_config(empty_file)
        assert config == {}

    def test_accepts_string_path(self, yaml_config_file: Path) -> None:
        config = load_config(str(yaml_config_file))
        assert config["model"]["name"] == "qwen2.5-7b-instruct"


# ---------------------------------------------------------------------------
# TestApplyEnvOverrides
# ---------------------------------------------------------------------------


class TestApplyEnvOverrides:
    """Tests for apply_env_overrides()."""

    def test_model_path_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_MODEL_PATH": "/custom/model.gguf"}):
            result = apply_env_overrides(sample_config)
        assert result["model"]["path"] == "/custom/model.gguf"

    def test_port_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_PORT": "9000"}):
            result = apply_env_overrides(sample_config)
        assert result["server"]["port"] == 9000

    def test_n_ctx_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_N_CTX": "4096"}):
            result = apply_env_overrides(sample_config)
        assert result["model"]["n_ctx"] == 4096

    def test_host_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_HOST": "127.0.0.1"}):
            result = apply_env_overrides(sample_config)
        assert result["server"]["host"] == "127.0.0.1"

    def test_log_level_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_LOG_LEVEL": "DEBUG"}):
            result = apply_env_overrides(sample_config)
        assert result["logging"]["level"] == "DEBUG"

    def test_model_name_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_MODEL_NAME": "qwen2.5-3b-instruct"}):
            result = apply_env_overrides(sample_config)
        assert result["model"]["name"] == "qwen2.5-3b-instruct"

    def test_n_gpu_layers_override(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_N_GPU_LAYERS": "20"}):
            result = apply_env_overrides(sample_config)
        assert result["model"]["n_gpu_layers"] == 20

    def test_cache_type_overrides(self, sample_config: dict) -> None:
        with patch.dict(
            "os.environ",
            {"TURBOQUANT_CACHE_TYPE_K": "f16", "TURBOQUANT_CACHE_TYPE_V": "q4_0"},
        ):
            result = apply_env_overrides(sample_config)
        assert result["kv_cache"]["cache_type_k"] == "f16"
        assert result["kv_cache"]["cache_type_v"] == "q4_0"

    def test_no_env_vars_returns_original_values(self, sample_config: dict) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = apply_env_overrides(sample_config)
        assert result["model"]["path"] == sample_config["model"]["path"]
        assert result["server"]["port"] == sample_config["server"]["port"]
        assert result["logging"]["level"] == sample_config["logging"]["level"]

    def test_does_not_mutate_input(self, sample_config: dict) -> None:
        import copy
        original = copy.deepcopy(sample_config)
        with patch.dict("os.environ", {"TURBOQUANT_PORT": "9999"}):
            apply_env_overrides(sample_config)
        assert sample_config == original

    def test_empty_config_with_overrides(self) -> None:
        with patch.dict("os.environ", {"TURBOQUANT_PORT": "7777"}):
            result = apply_env_overrides({})
        assert result["server"]["port"] == 7777


# ---------------------------------------------------------------------------
# TestBuildModelConfig
# ---------------------------------------------------------------------------


class TestBuildModelConfig:
    """Tests for build_model_config()."""

    def test_builds_from_config(self, sample_config: dict) -> None:
        mc = build_model_config(sample_config)
        assert isinstance(mc, ModelConfig)
        assert mc.model_name == "qwen2.5-7b-instruct"
        assert mc.model_path == "./models/qwen2.5-7b-instruct-q4_k_m.gguf"

    def test_defaults_when_empty_config(self) -> None:
        mc = build_model_config({})
        # Path is auto-discovered from models/ directory
        assert mc.model_path.endswith(".gguf")
        assert mc.model_name  # non-empty
        assert mc.n_ctx >= 128
        # GPU layers are computed from available VRAM — assert they are a valid value
        assert mc.n_gpu_layers >= 0 or mc.n_gpu_layers == -1
        assert mc.chat_format

    def test_all_fields_populated(self, sample_config: dict) -> None:
        mc = build_model_config(sample_config)
        assert mc.model_path == sample_config["model"]["path"]
        assert mc.model_name == sample_config["model"]["name"]
        assert mc.n_ctx == sample_config["model"]["n_ctx"]
        # n_gpu_layers=-1 is the "auto" sentinel: build_model_config resolves it
        # to an actual count via VRAM detection, so accept any non-negative value.
        cfg_layers = sample_config["model"]["n_gpu_layers"]
        if cfg_layers == -1:
            assert mc.n_gpu_layers >= 0
        else:
            assert mc.n_gpu_layers == cfg_layers
        assert mc.chat_format == sample_config["model"]["chat_format"]

    def test_custom_values(self, tmp_path: Path) -> None:
        model_file = tmp_path / "custom.gguf"
        model_file.touch()  # file must exist for path validation
        cfg = {
            "model": {
                "path": str(model_file),
                "name": "custom-model",
                "n_ctx": 4096,
                "n_gpu_layers": 10,
                "chat_format": "llama2",
            }
        }
        mc = build_model_config(cfg)
        assert mc.model_path == str(model_file)
        assert mc.n_ctx == 4096
        assert mc.n_gpu_layers == 10
        assert mc.chat_format == "llama2"


# ---------------------------------------------------------------------------
# TestBuildKvConfig
# ---------------------------------------------------------------------------


class TestBuildKvConfig:
    """Tests for build_kv_config()."""

    def test_builds_with_q4_0_defaults(self) -> None:
        kv = build_kv_config({})
        assert isinstance(kv, KVCacheConfig)
        assert kv.cache_type_k == CacheType.Q8_0
        assert kv.cache_type_v == CacheType.Q4_0
        assert kv.flash_attention is True

    def test_builds_from_config(self, sample_config: dict) -> None:
        kv = build_kv_config(sample_config)
        assert kv.cache_type_k == CacheType.Q8_0
        assert kv.cache_type_v == CacheType.TURBO4
        assert kv.flash_attention is True

    def test_custom_cache_types(self) -> None:
        cfg = {
            "kv_cache": {
                "cache_type_k": "f16",
                "cache_type_v": "q4_0",
                "flash_attention": False,
            }
        }
        kv = build_kv_config(cfg)
        assert kv.cache_type_k == CacheType.F16
        assert kv.cache_type_v == CacheType.Q4_0
        assert kv.flash_attention is False

    def test_turbo3_config(self) -> None:
        cfg = {
            "kv_cache": {
                "cache_type_k": "q8_0",
                "cache_type_v": "turbo3",
                "flash_attention": True,
            }
        }
        kv = build_kv_config(cfg)
        assert kv.cache_type_v == CacheType.TURBO3


# ---------------------------------------------------------------------------
# TestSetupLogging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_configures_logging_level(self) -> None:
        config = {"logging": {"level": "DEBUG"}}
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(config)
        mock_basic.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def test_default_is_info(self) -> None:
        with patch("logging.basicConfig") as mock_basic:
            setup_logging({})
        mock_basic.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def test_case_insensitive_level(self) -> None:
        config = {"logging": {"level": "warning"}}
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(config)
        mock_basic.assert_called_once_with(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def test_invalid_level_falls_back_to_info(self) -> None:
        config = {"logging": {"level": "NONEXISTENT"}}
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(config)
        mock_basic.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ---------------------------------------------------------------------------
# TestParseArgs
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for parse_args()."""

    def test_default_config_path(self) -> None:
        args = parse_args([])
        assert args.config == str(DEFAULT_CONFIG_PATH)
        assert args.host is None
        assert args.port is None

    def test_config_flag(self) -> None:
        args = parse_args(["--config", "config/dev.yaml"])
        assert args.config == "config/dev.yaml"

    def test_host_and_port_flags(self) -> None:
        args = parse_args(["--host", "127.0.0.1", "--port", "9000"])
        assert args.host == "127.0.0.1"
        assert args.port == 9000

    def test_all_flags_together(self) -> None:
        args = parse_args([
            "--config", "custom.yaml",
            "--host", "10.0.0.1",
            "--port", "5000",
        ])
        assert args.config == "custom.yaml"
        assert args.host == "10.0.0.1"
        assert args.port == 5000


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() entry point."""

    def test_main_with_defaults(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n"
            "  name: test\n"
            "  path: /tmp/model.gguf\n"
            "  n_ctx: 512\n"
            "server:\n"
            "  host: 0.0.0.0\n"
            "  port: 8000\n"
        )
        with patch("uvicorn.run") as mock_run:
            from src.main import main
            main(argv=["--config", str(config_file)])
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs["host"] == "0.0.0.0"
            assert kwargs["port"] == 8000

    def test_main_host_port_overrides(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n"
            "  name: test\n"
            "  path: /tmp/model.gguf\n"
            "  n_ctx: 512\n"
        )
        with patch("uvicorn.run") as mock_run:
            from src.main import main
            main(argv=[
                "--config", str(config_file),
                "--host", "127.0.0.1",
                "--port", "9000",
            ])
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs["host"] == "127.0.0.1"
            assert kwargs["port"] == 9000
