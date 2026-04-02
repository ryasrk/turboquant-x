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
from src.server.app import create_app, InferenceMode

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

# Named presets for TurboQuant compression
TURBOQUANT_PRESETS: dict[str, dict[str, int]] = {
    "quality": {"k_bits": 8, "v_bits": 4, "block_size": 128},
    "aggressive": {"k_bits": 8, "v_bits": 2, "block_size": 128},
    "symmetric": {"k_bits": 4, "v_bits": 4, "block_size": 128},
}


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
        "inference_mode": config.get("inference_mode", "standard"),
        "turboquant": dict(config.get("turboquant", {})),
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
    if env_val := os.environ.get("TURBOQUANT_GPU_SAFETY_MARGIN"):
        result["model"]["gpu_safety_margin"] = float(env_val)
    if env_val := os.environ.get("TURBOQUANT_N_THREADS"):
        result["model"]["n_threads"] = int(env_val)
    if env_val := os.environ.get("TURBOQUANT_N_THREADS_BATCH"):
        result["model"]["n_threads_batch"] = int(env_val)
    if env_val := os.environ.get("TURBOQUANT_N_BATCH"):
        result["model"]["n_batch"] = int(env_val)

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

    # Inference mode override
    if env_val := os.environ.get("TURBOQUANT_INFERENCE_MODE"):
        result["inference_mode"] = env_val

    # TurboQuant preset override
    if env_val := os.environ.get("TURBOQUANT_PRESET"):
        result.setdefault("turboquant", {})["preset"] = env_val

    return result


def build_model_config(config: dict[str, Any]) -> ModelConfig:
    """Build ModelConfig from config dict.

    When n_gpu_layers is -1 (the sentinel for "auto"), calls
    compute_optimal_gpu_layers() to derive the best distribution between
    GPU and CPU based on available VRAM and model size.
    """
    model = config.get("model", {})
    model_path = model.get("path", "models/qwen2.5-7b-instruct-q4_k_m.gguf")
    n_ctx = model.get("n_ctx", 8192)
    n_gpu_layers_cfg = model.get("n_gpu_layers", -1)

    if n_gpu_layers_cfg == -1:
        try:
            from src.utils.gpu_layers import compute_optimal_gpu_layers
            safety_margin = model.get("gpu_safety_margin", 0.92)
            n_gpu_layers = compute_optimal_gpu_layers(
                model_path, n_ctx=n_ctx, safety_margin=safety_margin
            )
            logger.info("Auto GPU layers: %d (resolved from VRAM + model size)", n_gpu_layers)
        except Exception as exc:
            logger.warning("GPU layer auto-detection failed: %s. Defaulting to 0.", exc)
            n_gpu_layers = 0
    else:
        n_gpu_layers = n_gpu_layers_cfg

    return ModelConfig(
        model_path=model_path,
        model_name=model.get("name", "qwen2.5-7b-instruct"),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format=model.get("chat_format", "chatml"),
        n_threads=model.get("n_threads", -1),
        n_batch=model.get("n_batch", 512),
        n_threads_batch=model.get("n_threads_batch", -1),
    )


def build_kv_config(config: dict[str, Any]) -> KVCacheConfig:
    """Build KVCacheConfig from config dict."""
    kv = config.get("kv_cache", {})
    return KVCacheConfig(
        cache_type_k=CacheType(kv.get("cache_type_k", "q8_0")),
        cache_type_v=CacheType(kv.get("cache_type_v", "turbo4")),
        flash_attention=kv.get("flash_attention", True),
    )


def build_inference_mode(config: dict[str, Any]) -> InferenceMode:
    """Determine inference mode from config."""
    raw = config.get("inference_mode", "standard").lower()
    try:
        return InferenceMode(raw)
    except ValueError:
        logger.warning("Unknown inference_mode '%s', falling back to standard", raw)
        return InferenceMode.STANDARD


def build_turboquant_config(config: dict[str, Any]) -> dict[str, int]:
    """Build TurboQuant compression settings from config.

    If a preset name is specified, its values are used as the base,
    then any explicit k_bits/v_bits/block_size overrides are applied on top.
    """
    tq = config.get("turboquant", {})
    preset_name = tq.get("preset", "").lower()

    if preset_name and preset_name in TURBOQUANT_PRESETS:
        base = dict(TURBOQUANT_PRESETS[preset_name])
        logger.info("Using TurboQuant preset: %s", preset_name)
    else:
        if preset_name:
            logger.warning(
                "Unknown TurboQuant preset '%s', valid: %s. Using defaults.",
                preset_name,
                ", ".join(TURBOQUANT_PRESETS),
            )
        base = {"k_bits": 8, "v_bits": 4, "block_size": 128}

    # Explicit values override preset defaults
    if "k_bits" in tq:
        base["k_bits"] = tq["k_bits"]
    if "v_bits" in tq:
        base["v_bits"] = tq["v_bits"]
    if "block_size" in tq:
        base["block_size"] = tq["block_size"]

    return base


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
        default=os.environ.get("TURBOQUANT_CONFIG", str(DEFAULT_CONFIG_PATH)),
        help="Path to YAML config file (env: TURBOQUANT_CONFIG)",
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "turboquant", "zero-quant"],
        default=None,
        help="Inference mode (overrides config)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quality", "aggressive", "symmetric"],
        default=None,
        help="TurboQuant preset: quality (K8/V4), aggressive (K8/V2), symmetric (K4/V4)",
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
    if args.mode:
        config["inference_mode"] = args.mode
    if args.preset:
        config.setdefault("turboquant", {})["preset"] = args.preset
        # Auto-enable turboquant mode when a preset is specified
        if not args.mode:
            config["inference_mode"] = "turboquant"

    # Setup
    setup_logging(config)

    model_config = build_model_config(config)
    kv_config = build_kv_config(config)
    inference_mode = build_inference_mode(config)
    turboquant_cfg = build_turboquant_config(config)

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    cors_origins = server_cfg.get("cors_origins", ["http://localhost:3000"])

    logger.info("Starting TurboQuant-X Server")
    logger.info("Model: %s (%s)", model_config.model_name, model_config.model_path)
    logger.info("Inference mode: %s", inference_mode.value)
    logger.info("KV Cache: K=%s, V=%s", kv_config.cache_type_k.value, kv_config.cache_type_v.value)
    if inference_mode == InferenceMode.TURBOQUANT:
        logger.info(
            "TurboQuant: K=%d-bit, V=%d-bit, block_size=%d",
            turboquant_cfg["k_bits"],
            turboquant_cfg["v_bits"],
            turboquant_cfg["block_size"],
        )
    logger.info("Server: %s:%d", host, port)

    # Create and run app
    app = create_app(model_config, kv_config, cors_origins, inference_mode, turboquant_cfg)

    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
