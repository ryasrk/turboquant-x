"""Model configuration and selection for TurboQuant-X LLM inference.

Manages GGUF model paths, context sizes, and GPU layer offloading.
Primary target: Qwen2.5-7B-Instruct Q4_K_M GGUF for 8GB GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Default model directory (relative to project root)
DEFAULT_MODEL_DIR = Path("models")

# Supported models with their metadata
MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-7b-instruct": {
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "chat_format": "chatml",
        "parameters": "7B",
        "weight_quant": "Q4_K_M",
        "weight_size_gb": 4.8,
        "default_n_ctx": 8192,
    },
    "qwen2.5-3b-instruct": {
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "chat_format": "chatml",
        "parameters": "3B",
        "weight_quant": "Q4_K_M",
        "weight_size_gb": 2.1,
        "default_n_ctx": 8192,
    },
}


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for an LLM model.

    Attributes:
        model_path: Absolute or relative path to the GGUF model file.
        model_name: Name key from MODEL_REGISTRY.
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers to offload to GPU (-1 = all).
        chat_format: Chat template format (e.g., "chatml").
        weight_size_gb: Approximate model weight size in GB.
        n_threads: CPU threads for token generation (-1 = auto: half of cpu_count).
        n_batch: Batch size for prompt evaluation (larger = faster prompt, more RAM).
    """

    model_path: str
    model_name: str = "qwen2.5-7b-instruct"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    chat_format: str = "chatml"
    weight_size_gb: float = 4.8
    n_threads: int = -1
    n_batch: int = 512
    n_threads_batch: int = -1  # CPU threads for prompt eval (-1 = auto: cpu_count, all cores)
    use_mlock: bool = False    # Lock model weights in RAM (prevents OS paging to SSD)
                               # Set True if model fits in RAM and you want consistent speed

    def __post_init__(self) -> None:
        if self.n_ctx < 128:
            raise ValueError(f"n_ctx must be >= 128, got {self.n_ctx}")
        if self.n_ctx > 131072:
            raise ValueError(f"n_ctx must be <= 131072, got {self.n_ctx}")
        if self.n_gpu_layers < -1:
            raise ValueError(f"n_gpu_layers must be >= -1, got {self.n_gpu_layers}")
        if self.n_batch < 1:
            raise ValueError(f"n_batch must be >= 1, got {self.n_batch}")


def get_default_config(
    gpu_vram_gb: float = 8.0,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> ModelConfig:
    """Auto-select model configuration based on available VRAM.

    Selection logic:
    - >= 6 GB VRAM: Qwen2.5-7B-Instruct Q4_K_M (4.8 GB weights)
    - >= 3 GB VRAM: Qwen2.5-3B-Instruct Q4_K_M (2.1 GB weights)
    - < 3 GB: raise ValueError

    The model file MUST exist at model_dir/filename.

    Args:
        gpu_vram_gb: Available GPU VRAM in GB.
        model_dir: Directory where GGUF files are stored.

    Returns:
        ModelConfig with appropriate settings.

    Raises:
        ValueError: If no model fits the available VRAM.
    """
    model_dir = Path(model_dir)

    if gpu_vram_gb >= 6.0:
        key = "qwen2.5-7b-instruct"
    elif gpu_vram_gb >= 3.0:
        key = "qwen2.5-3b-instruct"
    else:
        raise ValueError(
            f"No model fits {gpu_vram_gb:.1f} GB VRAM. Minimum 3 GB required."
        )

    meta = MODEL_REGISTRY[key]
    model_path = str(model_dir / meta["filename"])

    return ModelConfig(
        model_path=model_path,
        model_name=key,
        n_ctx=meta["default_n_ctx"],
        n_gpu_layers=-1,
        chat_format=meta["chat_format"],
        weight_size_gb=meta["weight_size_gb"],
    )


def estimate_total_vram(
    config: ModelConfig,
    kv_cache_gb: float = 0.7,
    overhead_gb: float = 0.5,
) -> float:
    """Estimate total VRAM needed for model + KV cache + overhead.

    Args:
        config: Model configuration.
        kv_cache_gb: Estimated KV cache size (depends on cache type and n_ctx).
        overhead_gb: CUDA runtime + framework overhead.

    Returns:
        Total estimated VRAM in GB.
    """
    return config.weight_size_gb + kv_cache_gb + overhead_gb


def validate_model_path(config: ModelConfig) -> bool:
    """Check if the model file exists at the configured path."""
    return Path(config.model_path).is_file()


def from_env(model_dir: str | Path = DEFAULT_MODEL_DIR) -> ModelConfig:
    """Create ModelConfig from environment variables.

    Environment variables (all optional, with defaults from get_default_config):
    - TURBOQUANT_MODEL_PATH: Full path to GGUF file (overrides auto-detection)
    - TURBOQUANT_MODEL_NAME: Model registry key
    - TURBOQUANT_N_CTX: Context window size
    - TURBOQUANT_N_GPU_LAYERS: GPU layer count (-1 = all)
    - TURBOQUANT_GPU_VRAM_GB: Available VRAM for auto-selection
    """
    explicit_path = os.environ.get("TURBOQUANT_MODEL_PATH")

    if explicit_path:
        return ModelConfig(
            model_path=explicit_path,
            model_name=os.environ.get("TURBOQUANT_MODEL_NAME", "qwen2.5-7b-instruct"),
            n_ctx=int(os.environ.get("TURBOQUANT_N_CTX", "8192")),
            n_gpu_layers=int(os.environ.get("TURBOQUANT_N_GPU_LAYERS", "-1")),
        )

    vram = float(os.environ.get("TURBOQUANT_GPU_VRAM_GB", "8.0"))
    config = get_default_config(gpu_vram_gb=vram, model_dir=model_dir)

    # Allow env var overrides on top of auto-selected config
    n_ctx = int(os.environ.get("TURBOQUANT_N_CTX", str(config.n_ctx)))
    n_gpu_layers = int(
        os.environ.get("TURBOQUANT_N_GPU_LAYERS", str(config.n_gpu_layers))
    )

    return ModelConfig(
        model_path=config.model_path,
        model_name=config.model_name,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format=config.chat_format,
        weight_size_gb=config.weight_size_gb,
    )
