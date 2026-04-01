"""Entry point for TurboQuant chat server.

Usage:
    python -m src.main                          # Use default.yaml
    python -m src.main --config config/dev.yaml  # Custom config
    TURBOQUANT_PORT=9000 python -m src.main     # Env var override

Environment variable overrides (take precedence over YAML):
    TURBOQUANT_MODEL_PATH  — Path to GGUF model file
    TURBOQUANT_N_CTX       — Context window size
    TURBOQUANT_HOST        — Server bind address
    TURBOQUANT_PORT        — Server port
    TURBOQUANT_LOG_LEVEL   — Logging level
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig
from src.server.app import create_app

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found, using defaults", path)
        return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    return config


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config.

    Environment variables take precedence over YAML values.
    Returns a new dict (does not mutate input).
    """
    result = {
        "model": dict(config.get("model", {})),
        "kv_cache": dict(config.get("kv_cache", {})),
        "server": dict(config.get("server", {})),
        "logging": dict(config.get("logging", {})),
    }

    # Model overrides
    if env_val := os.environ.get("TURBOQUANT_MODEL_PATH"):
        result["model"]["path"] = env_val
    if env_val := os.environ.get("TURBOQUANT_MODEL_NAME"):
        result["model"]["name"] = env_val
    if env_val := os.environ.get("TURBOQUANT_N_CTX"):
        result["model"]["n_ctx"] = int(env_val)
    if env_val := os.environ.get("TURBOQUANT_N_GPU_LAYERS"):
        result["model"]["n_gpu_layers"] = int(env_val)

    # KV cache overrides
    if env_val := os.environ.get("TURBOQUANT_CACHE_TYPE_K"):
        result["kv_cache"]["cache_type_k"] = env_val
    if env_val := os.environ.get("TURBOQUANT_CACHE_TYPE_V"):
        result["kv_cache"]["cache_type_v"] = env_val

    # Server overrides
    if env_val := os.environ.get("TURBOQUANT_HOST"):
        result["server"]["host"] = env_val
    if env_val := os.environ.get("TURBOQUANT_PORT"):
        result["server"]["port"] = int(env_val)

    # Logging overrides
    if env_val := os.environ.get("TURBOQUANT_LOG_LEVEL"):
        result["logging"]["level"] = env_val

    return result


def build_model_config(config: dict[str, Any]) -> ModelConfig:
    """Build ModelConfig from config dict."""
    model = config.get("model", {})
    return ModelConfig(
        model_path=model.get("path", "models/qwen2.5-7b-instruct-q4_k_m.gguf"),
        model_name=model.get("name", "qwen2.5-7b-instruct"),
        n_ctx=model.get("n_ctx", 8192),
        n_gpu_layers=model.get("n_gpu_layers", -1),
        chat_format=model.get("chat_format", "chatml"),
    )


def build_kv_config(config: dict[str, Any]) -> KVCacheConfig:
    """Build KVCacheConfig from config dict."""
    kv = config.get("kv_cache", {})
    return KVCacheConfig(
        cache_type_k=CacheType(kv.get("cache_type_k", "q8_0")),
        cache_type_v=CacheType(kv.get("cache_type_v", "turbo4")),
        flash_attention=kv.get("flash_attention", True),
    )


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TurboQuant Chat Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server bind address (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (overrides config)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Load and merge config
    config = load_config(args.config)
    config = apply_env_overrides(config)

    # CLI args override everything
    if args.host:
        config.setdefault("server", {})["host"] = args.host
    if args.port:
        config.setdefault("server", {})["port"] = args.port

    # Setup
    setup_logging(config)

    model_config = build_model_config(config)
    kv_config = build_kv_config(config)

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    cors_origins = server_cfg.get("cors_origins", ["http://localhost:3000"])

    logger.info("Starting TurboQuant Chat Server")
    logger.info("Model: %s (%s)", model_config.model_name, model_config.model_path)
    logger.info("KV Cache: K=%s, V=%s", kv_config.cache_type_k.value, kv_config.cache_type_v.value)
    logger.info("Server: %s:%d", host, port)

    # Create and run app
    app = create_app(model_config, kv_config, cors_origins)

    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
