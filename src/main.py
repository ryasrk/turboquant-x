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
        "ultra_quant": dict(config.get("ultra_quant", {})),
        "null_quant": dict(config.get("null_quant", {})),
        "cloud": dict(config.get("cloud", {})),
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

    When n_ctx is -1 (or 0), co-optimises context window and GPU layers
    together via compute_optimal_context().

    When model path is empty or the file does not exist, auto-discovers
    the first available GGUF in the models/ directory.
    """
    from src.engine.model_config import auto_select_model

    model = config.get("model", {})
    model_path: str = model.get("path", "") or ""
    n_ctx = model.get("n_ctx", 8192)

    # Auto-discover model if path is empty or the file does not exist
    if not model_path or not Path(model_path).is_file():
        if model_path and not Path(model_path).is_file():
            logger.warning(
                "Configured model path does not exist: %s — scanning models/ for available GGUF files.",
                model_path,
            )
        discovered = auto_select_model()
        if discovered is None:
            raise FileNotFoundError(
                "No model path configured and no *.gguf files found in models/. "
                "Download a model with: scripts/download_model.sh"
            )
        logger.info(
            "Auto-selected model: %s (%s)",
            discovered.model_name,
            discovered.model_path,
        )
        # Merge discovered defaults with any explicit config overrides
        model_path = discovered.model_path
        if not model.get("name"):
            model = {**model, "name": discovered.model_name}
        if not model.get("chat_format"):
            model = {**model, "chat_format": discovered.chat_format}
        if model.get("n_ctx", -1) == 0 or not model.get("n_ctx"):
            n_ctx = model.get("n_ctx", 8192) or discovered.n_ctx
    n_gpu_layers_cfg = model.get("n_gpu_layers", -1)

    # Build KV config early so auto-ctx can account for actual KV compression
    kv_config_for_ctx = None
    try:
        kv_config_for_ctx = build_kv_config(config)
    except Exception:
        pass

    # Auto-maximise context window: n_ctx = -1 or 0
    if n_ctx <= 0:
        try:
            from src.utils.gpu_layers import compute_optimal_context
            safety_margin = model.get("gpu_safety_margin", 0.92)
            n_ctx, auto_gl = compute_optimal_context(
                model_path,
                safety_margin=safety_margin,
                kv_config=kv_config_for_ctx,
            )
            logger.info("Auto context: n_ctx=%d, n_gpu_layers=%d", n_ctx, auto_gl)
            # If GPU layers are also auto, use the co-optimised value
            if n_gpu_layers_cfg == -1:
                n_gpu_layers_cfg = auto_gl
        except Exception as exc:
            logger.warning("Auto context failed: %s. Defaulting to 8192.", exc)
            n_ctx = 8192

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
        use_mlock=model.get("use_mlock", False),
    )


def build_kv_config(config: dict[str, Any]) -> KVCacheConfig:
    """Build KVCacheConfig from config dict."""
    kv = config.get("kv_cache", {})
    return KVCacheConfig(
        cache_type_k=CacheType(kv.get("cache_type_k", "q8_0")),
        cache_type_v=CacheType(kv.get("cache_type_v", "q4_0")),
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


def build_zero_quant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build Zero-Quant depth-adaptive compression settings from config.

    Reads the ``zero_quant`` section and returns a plain dict that
    ``create_app`` forwards to :class:`~src.turboquant.zero_quant.ZeroQuantConfig`.
    Missing keys fall back to the ``ZeroQuantConfig`` defaults.
    """
    zq = config.get("zero_quant", {})
    return {
        "shallow_fraction": zq.get("shallow_fraction", 0.25),
        "deep_fraction": zq.get("deep_fraction", 0.25),
        "shallow_k_bits": zq.get("shallow_k_bits", 8),
        "shallow_v_bits": zq.get("shallow_v_bits", 8),
        "middle_k_bits": zq.get("middle_k_bits", 4),
        "middle_v_bits": zq.get("middle_v_bits", 2),
        "deep_k_bits": zq.get("deep_k_bits", 8),
        "deep_v_bits": zq.get("deep_v_bits", 8),
        "block_size": zq.get("block_size", 128),
        "use_kv_coquant": zq.get("use_kv_coquant", False),
    }


def build_ultra_quant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build Ultra-Quant settings from config.

    Reads the ``ultra_quant`` section and returns a plain dict that
    ``create_app`` forwards to :class:`~src.engine.ultra_quant_engine.UltraQuantConfig`.
    """
    uq = config.get("ultra_quant", {})
    return {
        "target_model_params_b": uq.get("target_model_params_b", 70.0),
        "max_ram_usage_fraction": uq.get("max_ram_usage_fraction", 0.85),
        "max_vram_usage_fraction": uq.get("max_vram_usage_fraction", 0.90),
        "enable_mmap": uq.get("enable_mmap", True),
        "enable_mlock_critical": uq.get("enable_mlock_critical", True),
        "enable_moe_offload": uq.get("enable_moe_offload", True),
        "kv_budget_mb": uq.get("kv_budget_mb", 0),
        "force_quant": uq.get("force_quant", ""),
        "zero_quant_preset": uq.get("zero_quant_preset", "turbo"),
    }


def build_null_quant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build NullQuant token-eviction + zone-compression settings from config.

    Reads the ``null_quant`` section and returns a plain dict that
    ``create_app`` forwards to :class:`~src.turboquant.null_quant.NullQuantConfig`.
    """
    nq = config.get("null_quant", {})
    return {
        "eviction_ratio": nq.get("eviction_ratio", 0.75),
        "sink_tokens": nq.get("sink_tokens", 256),
        "recent_tokens": nq.get("recent_tokens", 256),
        "scoring_method": nq.get("scoring_method", "l2_norm"),
        "block_size": nq.get("block_size", 64),
        "shallow_fraction": nq.get("shallow_fraction", 0.25),
        "deep_fraction": nq.get("deep_fraction", 0.25),
        "shallow_k_bits": nq.get("shallow_k_bits", 8),
        "shallow_v_bits": nq.get("shallow_v_bits", 8),
        "middle_k_bits": nq.get("middle_k_bits", 4),
        "middle_v_bits": nq.get("middle_v_bits", 2),
        "deep_k_bits": nq.get("deep_k_bits", 8),
        "deep_v_bits": nq.get("deep_v_bits", 8),
        "compress_block_size": nq.get("compress_block_size", 128),
    }


def build_cloud_config(config: dict[str, Any]) -> Any:
    """Build cloud provider config from YAML + env vars.

    Returns a CloudConfig for the default provider, or None if cloud is not configured.
    Environment variables: TURBOQUANT_CLOUD_{PROVIDER}_API_KEY override YAML api_key.
    TURBOQUANT_CLOUD_PROVIDER selects the default provider.
    TURBOQUANT_CLOUD_MODEL overrides the model.
    """
    from src.engine.cloud.registry import build_cloud_configs

    configs = build_cloud_configs(config)
    if not configs:
        return None

    cloud_section = config.get("cloud", {})

    # Determine which provider to use as default
    default_name = os.environ.get(
        "TURBOQUANT_CLOUD_PROVIDER",
        cloud_section.get("default_provider", ""),
    )

    # Model override from env
    model_override = os.environ.get("TURBOQUANT_CLOUD_MODEL", "")

    if default_name and default_name in configs:
        chosen = configs[default_name]
    else:
        # Pick the first configured provider
        chosen = next(iter(configs.values()))

    if model_override:
        from dataclasses import replace as _replace
        chosen = _replace(chosen, model=model_override)

    return chosen


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
        choices=["standard", "turboquant", "zero-quant", "ultra-quant", "null-quant", "cloud"],
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
    parser.add_argument(
        "--doctor",
        nargs="?",
        const="check",
        default=None,
        metavar="ACTION",
        help="Config doctor: check (default), fix, or validate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # ── Doctor mode ──────────────────────────────────────────────────
    if args.doctor is not None:
        from src.utils.doctor import run_doctor
        action = args.doctor.lower()
        fix = action == "fix"
        validate = action == "validate"
        report = run_doctor(fix=fix, validate_keys=validate)
        sys.exit(0 if report.all_ok else 1)

    # ── Load .env file (secrets) ─────────────────────────────────────
    from src.utils.doctor import load_env_file, load_cloud_yaml
    load_env_file()

    # Load and merge config
    config = load_config(args.config)

    # ── Merge cloud.yaml into config ─────────────────────────────────
    cloud_file_config = load_cloud_yaml()
    if cloud_file_config.get("cloud"):
        existing_cloud = config.get("cloud", {})
        file_cloud = cloud_file_config["cloud"]
        # cloud.yaml providers are the source of truth;
        # default.yaml cloud section is a fallback
        merged_cloud = {**existing_cloud, **file_cloud}
        # Merge provider dicts deeply — cloud.yaml wins per-provider
        existing_providers = existing_cloud.get("providers", {})
        file_providers = file_cloud.get("providers", {})
        merged_providers = {**existing_providers, **file_providers}
        merged_cloud["providers"] = merged_providers
        config["cloud"] = merged_cloud

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

    inference_mode = build_inference_mode(config)

    # Cloud mode doesn't need model/kv configs
    cloud_cfg = None
    if inference_mode == InferenceMode.CLOUD:
        cloud_cfg = build_cloud_config(config)
        if cloud_cfg is None:
            logger.error(
                "Cloud mode selected but no cloud provider configured. "
                "Add a 'cloud' section to your config or set "
                "TURBOQUANT_CLOUD_OPENAI_API_KEY (or other provider) env var."
            )
            sys.exit(1)

    model_config = build_model_config(config)
    kv_config = build_kv_config(config)
    turboquant_cfg = build_turboquant_config(config)
    zero_quant_cfg = build_zero_quant_config(config)
    ultra_quant_cfg = build_ultra_quant_config(config)
    null_quant_cfg = build_null_quant_config(config)

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)
    cors_origins = server_cfg.get("cors_origins", ["http://localhost:3000", "http://localhost:8000"])

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
    elif inference_mode == InferenceMode.ZERO_QUANT:
        logger.info(
            "ZeroQuant: shallow K%d/V%d | middle K%d/V%d | deep K%d/V%d | coquant=%s",
            zero_quant_cfg["shallow_k_bits"],
            zero_quant_cfg["shallow_v_bits"],
            zero_quant_cfg["middle_k_bits"],
            zero_quant_cfg["middle_v_bits"],
            zero_quant_cfg["deep_k_bits"],
            zero_quant_cfg["deep_v_bits"],
            zero_quant_cfg["use_kv_coquant"],
        )
    elif inference_mode == InferenceMode.ULTRA_QUANT:
        logger.info(
            "UltraQuant: target=%.0fB, quant_preset=%s, mmap=%s, mlock=%s",
            ultra_quant_cfg["target_model_params_b"],
            ultra_quant_cfg["zero_quant_preset"],
            ultra_quant_cfg["enable_mmap"],
            ultra_quant_cfg["enable_mlock_critical"],
        )
    elif inference_mode == InferenceMode.NULL_QUANT:
        logger.info(
            "NullQuant: eviction=%.0f%% sink=%d recent=%d scoring=%s | "
            "zones: shallow K%d/V%d | middle K%d/V%d | deep K%d/V%d",
            null_quant_cfg["eviction_ratio"] * 100,
            null_quant_cfg["sink_tokens"],
            null_quant_cfg["recent_tokens"],
            null_quant_cfg["scoring_method"],
            null_quant_cfg["shallow_k_bits"],
            null_quant_cfg["shallow_v_bits"],
            null_quant_cfg["middle_k_bits"],
            null_quant_cfg["middle_v_bits"],
            null_quant_cfg["deep_k_bits"],
            null_quant_cfg["deep_v_bits"],
        )
    elif inference_mode == InferenceMode.CLOUD and cloud_cfg is not None:
        logger.info(
            "Cloud mode: provider=%s, model=%s, base_url=%s",
            cloud_cfg.provider,
            cloud_cfg.model or "(default)",
            cloud_cfg.base_url or "(default)",
        )
    logger.info("Server: %s:%d", host, port)

    # Create and run app
    app = create_app(
        model_config, kv_config, cors_origins, inference_mode,
        turboquant_cfg, zero_quant_cfg, ultra_quant_cfg, null_quant_cfg,
        thought_log_path=config.get("logging", {}).get("thought_log"),
        cloud_config=cloud_cfg,
        cloud_yaml_config=config.get("cloud", {}),
    )

    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
