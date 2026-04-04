"""KV cache configuration for llama-cpp-python with TurboQuant support.

Maps cache type names to llama-cpp-python constructor parameters.
Provides memory estimation for different KV cache configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheType(str, Enum):
    """Valid KV cache quantization types for llama.cpp.

    Standard llama.cpp types:
    - F16: 16-bit float (baseline, no quantization)
    - Q8_0: 8-bit uniform quantization
    - Q4_0: 4-bit uniform quantization (llama.cpp native)

    TurboQuant types (requires TurboQuant-enabled llama.cpp fork):
    - TURBO4: 4-bit PolarQuant (Lloyd-Max optimal for Gaussian)
    - TURBO3: 3-bit PolarQuant
    - TURBO2: 2-bit PolarQuant
    """

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    TURBO4 = "turbo4"
    TURBO3 = "turbo3"
    TURBO2 = "turbo2"


# Bits per element for each cache type (including any per-block overhead)
_BITS_PER_ELEMENT: dict[CacheType, float] = {
    CacheType.F16: 16.0,
    CacheType.Q8_0: 8.5,  # 8 bits + 0.5 bits block overhead
    CacheType.Q4_0: 4.5,  # 4 bits + 0.5 bits block overhead
    CacheType.TURBO4: 4.25,  # 4 bits + 0.25 bits block overhead (block_size=128)
    CacheType.TURBO3: 3.5,  # 3 + 0.5 for block overhead
    CacheType.TURBO2: 2.5,  # 2 + 0.5 for block overhead
}

_TURBO_TYPES = frozenset({CacheType.TURBO4, CacheType.TURBO3, CacheType.TURBO2})


@dataclass(frozen=True)
class KVCacheConfig:
    """Configuration for KV cache quantization.

    Attributes:
        cache_type_k: Cache type for keys.
        cache_type_v: Cache type for values.
        flash_attention: Whether to enable flash attention (required for TurboQuant).
    """

    cache_type_k: CacheType = CacheType.Q8_0
    cache_type_v: CacheType = CacheType.Q4_0
    flash_attention: bool = True

    def __post_init__(self) -> None:
        if not self.flash_attention:
            if self.cache_type_k in _TURBO_TYPES or self.cache_type_v in _TURBO_TYPES:
                raise ValueError(
                    "flash_attention must be True when using TurboQuant cache types "
                    f"(K={self.cache_type_k.value}, V={self.cache_type_v.value})"
                )


def get_turboquant_config() -> KVCacheConfig:
    """Return the recommended TurboQuant KV cache config.

    Default: K=q8_0, V=turbo4, flash_attention=True
    """
    return KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO4,
        flash_attention=True,
    )


def get_baseline_config() -> KVCacheConfig:
    """Return f16 baseline config (no KV cache compression)."""
    return KVCacheConfig(
        cache_type_k=CacheType.F16,
        cache_type_v=CacheType.F16,
        flash_attention=True,
    )


def to_llama_params(config: KVCacheConfig) -> dict[str, object]:
    """Convert KVCacheConfig to llama-cpp-python Llama() constructor kwargs.

    Returns dict suitable for unpacking into Llama(**params):
    {
        "flash_attn": True,
        "type_k": "q8_0",
        "type_v": "turbo4",
    }

    Note: The actual parameter names depend on the llama-cpp-python version.
    The TurboQuant fork uses type_k/type_v. Some versions use
    cache_type_k/cache_type_v. We use the most common convention.

    type_k/type_v require integer GGML type IDs. For standard types
    we lazily import llama_cpp constants. TurboQuant types are mapped
    to their fork-specific integer values.
    """
    return {
        "flash_attn": config.flash_attention,
        "type_k": _cache_type_to_ggml_int(config.cache_type_k),
        "type_v": _cache_type_to_ggml_int(config.cache_type_v),
    }


def is_turbo_fork_available() -> bool:
    """Check whether the installed llama-cpp-python is the TurboQuant fork.

    Returns True if ``llama_cpp`` exposes ``GGML_TYPE_TURBO4``.
    Returns False if llama-cpp-python is missing or is the stock build.
    """
    try:
        import llama_cpp
        return hasattr(llama_cpp, "GGML_TYPE_TURBO4")
    except ImportError:
        return False


def ensure_compatible_config(config: KVCacheConfig) -> KVCacheConfig:
    """Return a config with turbo types downgraded if the fork is missing.

    When the TurboQuant llama.cpp fork is installed, returns *config* unchanged.
    Otherwise, any turbo cache types (turbo4/turbo3/turbo2) are replaced with
    q4_0 and a warning is logged.
    """
    import logging
    _log = logging.getLogger(__name__)

    if is_turbo_fork_available():
        return config

    new_k = config.cache_type_k
    new_v = config.cache_type_v
    downgraded = False

    if new_k in _TURBO_TYPES:
        _log.warning(
            "KV cache K type '%s' requires TurboQuant fork; falling back to q4_0",
            new_k.value,
        )
        new_k = CacheType.Q4_0
        downgraded = True

    if new_v in _TURBO_TYPES:
        _log.warning(
            "KV cache V type '%s' requires TurboQuant fork; falling back to q4_0",
            new_v.value,
        )
        new_v = CacheType.Q4_0
        downgraded = True

    if downgraded:
        return KVCacheConfig(
            cache_type_k=new_k,
            cache_type_v=new_v,
            flash_attention=config.flash_attention,
        )
    return config


# Provisional GGML type IDs for TurboQuant cache types.
# Used ONLY when llama-cpp-python is absent (e.g. memory estimation in tests).
# Values are intentionally high (100+) to avoid collisions with upstream
# GGML types (e.g. GGML_TYPE_NVFP4=40 in llama-cpp-python ≥0.3.x).
_TURBO_PROVISIONAL_IDS: dict[CacheType, int] = {
    CacheType.TURBO4: 100,
    CacheType.TURBO3: 101,
    CacheType.TURBO2: 102,
}


def _cache_type_to_ggml_int(cache_type: CacheType) -> int:
    """Convert CacheType enum to llama-cpp-python GGML integer type ID.

    Standard types use llama_cpp constants. TurboQuant types use
    fork-specific IDs (must match the TurboQuant llama.cpp fork).

    When llama-cpp-python IS installed but the TurboQuant fork is NOT
    detected, requesting a TurboQuant type raises RuntimeError to prevent
    silent misuse of conflicting upstream type IDs.

    When llama-cpp-python is NOT installed, returns provisional IDs
    (safe for offline memory estimation only).
    """
    # Well-known GGML type IDs for standard types (stable across versions)
    _STANDARD_WELL_KNOWN: dict[CacheType, int] = {
        CacheType.F16: 1,    # GGML_TYPE_F16
        CacheType.Q8_0: 8,   # GGML_TYPE_Q8_0
        CacheType.Q4_0: 2,   # GGML_TYPE_Q4_0
    }

    try:
        import llama_cpp

        # Use runtime constants for standard types
        _RUNTIME_MAP: dict[CacheType, int] = {
            CacheType.F16: llama_cpp.GGML_TYPE_F16,
            CacheType.Q8_0: llama_cpp.GGML_TYPE_Q8_0,
            CacheType.Q4_0: llama_cpp.GGML_TYPE_Q4_0,
        }

        if cache_type in _RUNTIME_MAP:
            return _RUNTIME_MAP[cache_type]

        # TurboQuant types — check if the fork exposes them
        turbo_attr = f"GGML_TYPE_{cache_type.value.upper()}"
        if hasattr(llama_cpp, turbo_attr):
            return getattr(llama_cpp, turbo_attr)

        # Fork NOT detected — refuse to return a conflicting provisional ID.
        raise RuntimeError(
            f"TurboQuant cache type {cache_type.value!r} requires the "
            f"TurboQuant llama.cpp fork (GGML_TYPE_{cache_type.value.upper()} "
            f"not found in llama_cpp). Install the fork or use a standard "
            f"cache type (f16, q8_0, q4_0)."
        )
    except ImportError:
        # llama-cpp-python not installed — return well-known / provisional IDs
        # (safe for offline memory estimation and testing only)
        if cache_type in _STANDARD_WELL_KNOWN:
            return _STANDARD_WELL_KNOWN[cache_type]
        if cache_type in _TURBO_PROVISIONAL_IDS:
            return _TURBO_PROVISIONAL_IDS[cache_type]
        raise ValueError(
            f"Cannot resolve GGML type for {cache_type.value!r} without llama-cpp-python"
        )


def _compute_cache_bytes(
    n_ctx: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    cache_type: CacheType,
) -> int:
    """Compute cache bytes for one of K or V."""
    bits = _BITS_PER_ELEMENT[cache_type]
    total_elements = n_ctx * n_layers * n_heads * head_dim
    return int(total_elements * bits / 8)


def estimate_kv_memory_bytes(
    n_ctx: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    config: KVCacheConfig,
) -> dict[str, int | float]:
    """Estimate KV cache memory usage.

    Returns:
        {
            "k_bytes": int,
            "v_bytes": int,
            "total_bytes": int,
            "total_mb": float,
            "total_gb": float,
            "compression_vs_f16": float,  # ratio (f16_size / config_size)
        }

    Formula: bytes = n_ctx * n_layers * n_heads * head_dim * bits_per_element / 8
    """
    k_bytes = _compute_cache_bytes(n_ctx, n_layers, n_heads, head_dim, config.cache_type_k)
    v_bytes = _compute_cache_bytes(n_ctx, n_layers, n_heads, head_dim, config.cache_type_v)
    total_bytes = k_bytes + v_bytes
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)

    f16_bytes = _compute_cache_bytes(
        n_ctx, n_layers, n_heads, head_dim, CacheType.F16
    ) * 2  # K + V both f16
    compression_vs_f16 = f16_bytes / total_bytes if total_bytes > 0 else float("inf")

    return {
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "total_gb": total_gb,
        "compression_vs_f16": compression_vs_f16,
    }


def estimate_kv_memory_gb(
    n_ctx: int,
    config: KVCacheConfig,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
) -> float:
    """Convenience function returning total KV cache size in GB."""
    result = estimate_kv_memory_bytes(n_ctx, n_layers, n_heads, head_dim, config)
    return result["total_gb"]


# Common model architectures for memory estimation
MODEL_ARCHITECTURES: dict[str, dict[str, int]] = {
    "qwen2.5-7b": {"n_layers": 28, "n_heads": 28, "head_dim": 128},
    "qwen2.5-3b": {"n_layers": 36, "n_heads": 16, "head_dim": 128},
    "llama3-8b": {"n_layers": 32, "n_heads": 32, "head_dim": 128},
}


def compare_configs(
    n_ctx: int = 8192,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
) -> str:
    """Generate a comparison table of common KV cache configurations.

    Returns formatted string table with:
    | Config        | K Type | V Type | KV Cache (MB) | vs f16 |
    """
    configs: list[tuple[str, KVCacheConfig]] = [
        ("f16 baseline", get_baseline_config()),
        ("q8_0 / q8_0", KVCacheConfig(CacheType.Q8_0, CacheType.Q8_0, True)),
        ("q4_0 / q4_0", KVCacheConfig(CacheType.Q4_0, CacheType.Q4_0, True)),
        ("q8_0 / turbo4", get_turboquant_config()),
        ("q8_0 / turbo2", KVCacheConfig(CacheType.Q8_0, CacheType.TURBO2, True)),
        ("turbo4 / turbo4", KVCacheConfig(CacheType.TURBO4, CacheType.TURBO4, True)),
        ("turbo2 / turbo2", KVCacheConfig(CacheType.TURBO2, CacheType.TURBO2, True)),
    ]

    header = f"| {'Config':<16} | {'K Type':<8} | {'V Type':<8} | {'KV Cache (MB)':>14} | {'vs f16':>8} |"
    separator = f"|{'-' * 18}|{'-' * 10}|{'-' * 10}|{'-' * 16}|{'-' * 10}|"
    lines = [header, separator]

    for name, cfg in configs:
        result = estimate_kv_memory_bytes(n_ctx, n_layers, n_heads, head_dim, cfg)
        ratio = result["compression_vs_f16"]
        lines.append(
            f"| {name:<16} | {cfg.cache_type_k.value:<8} | {cfg.cache_type_v.value:<8} "
            f"| {result['total_mb']:>13.1f} | {ratio:>7.2f}x |"
        )

    return "\n".join(lines)
