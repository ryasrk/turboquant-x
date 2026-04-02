"""Automatic GPU layer distribution for TurboQuant-X.

Reads GGUF metadata (block count, architecture) without loading model weights,
queries free VRAM, and computes the optimal n_gpu_layers for a given model so
that the GPU stays within a configurable memory budget.

Usage::

    from src.utils.gpu_layers import compute_optimal_gpu_layers

    n = compute_optimal_gpu_layers(
        model_path="./models/Qwen3.5-35B-A3B-q4_k_m.gguf",
        n_ctx=4096,
        safety_margin=0.90,   # use at most 90 % of free VRAM
    )
    print(f"Recommended n_gpu_layers: {n}")
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── GGUF binary constants ────────────────────────────────────────────────
_GGUF_MAGIC = b"GGUF"
_VALUE_SIZES: dict[int, int] = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    # 8 = STRING  (variable)
    # 9 = ARRAY   (variable)
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}

# KV types that hold integers we might care about
_INTEGER_TYPES = frozenset([0, 1, 2, 3, 4, 5, 10, 11])


def _read_gguf_metadata(model_path: str | Path) -> dict[str, Any]:
    """Parse GGUF header KV metadata without loading model weights.

    Reads only the header section (typically < 1 MB), returning a flat dict
    of all string-keyed values (int, float, bool, str types only; arrays
    are skipped for simplicity).

    Returns an empty dict if the file is not a valid GGUF or on any I/O error.
    """
    path = Path(model_path)
    if not path.exists():
        logger.warning("Model file not found: %s", path)
        return {}

    try:
        with open(path, "rb") as f:
            return _parse_header(f)
    except Exception as exc:
        logger.debug("GGUF header parse failed (%s): %s", path.name, exc)
        return {}


def _parse_header(f) -> dict[str, Any]:  # noqa: ANN001
    """Parse GGUF header from an open binary file."""
    magic = f.read(4)
    if magic != _GGUF_MAGIC:
        return {}

    (version,) = struct.unpack("<I", f.read(4))
    if version not in (1, 2, 3):
        logger.debug("Unsupported GGUF version %d", version)
        return {}

    (_n_tensors,) = struct.unpack("<Q", f.read(8))
    (n_kv,) = struct.unpack("<Q", f.read(8))

    result: dict[str, Any] = {"_gguf_version": version}

    for _ in range(n_kv):
        # Key
        (key_len,) = struct.unpack("<Q", f.read(8))
        key = f.read(key_len).decode("utf-8", errors="replace")

        # Value type
        (vtype,) = struct.unpack("<I", f.read(4))

        value = _read_value(f, vtype)
        if value is not None:
            result[key] = value

    return result


def _read_value(f, vtype: int) -> Any:  # noqa: ANN001
    """Read a single GGUF value. Returns None for arrays (skipped)."""
    if vtype in _VALUE_SIZES:
        size = _VALUE_SIZES[vtype]
        raw = f.read(size)
        if vtype in (0, 2, 4, 10):   # unsigned ints
            fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[size]
            return struct.unpack(fmt, raw)[0]
        if vtype in (1, 3, 5, 11):   # signed ints
            fmt = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}[size]
            return struct.unpack(fmt, raw)[0]
        if vtype == 6:                # FLOAT32
            return struct.unpack("<f", raw)[0]
        if vtype == 12:               # FLOAT64
            return struct.unpack("<d", raw)[0]
        if vtype == 7:                # BOOL
            return bool(raw[0])

    if vtype == 8:                    # STRING
        (str_len,) = struct.unpack("<Q", f.read(8))
        return f.read(str_len).decode("utf-8", errors="replace")

    if vtype == 9:                    # ARRAY — skip but must advance file pos
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        _skip_array(f, elem_type, count)
        return None

    return None  # Unknown type — cannot advance safely; parsing stops here


def _skip_array(f, elem_type: int, count: int) -> None:
    """Skip over an array in the GGUF stream."""
    if elem_type in _VALUE_SIZES:
        f.read(_VALUE_SIZES[elem_type] * count)
        return
    if elem_type == 8:  # STRING array
        for _ in range(count):
            (s_len,) = struct.unpack("<Q", f.read(8))
            f.read(s_len)
        return
    # Nested arrays or unknown: cannot skip safely — stop


def _get_block_count(meta: dict[str, Any]) -> int | None:
    """Extract transformer block count from GGUF metadata."""
    arch = meta.get("general.architecture", "")
    if arch:
        candidate = meta.get(f"{arch}.block_count")
        if isinstance(candidate, int):
            return candidate

    # Fallback: scan for any *block_count key
    for key, val in meta.items():
        if key.endswith(".block_count") and isinstance(val, int):
            return val
    return None


def _get_free_vram_bytes() -> int | None:
    """Return free GPU VRAM in bytes, or None if no GPU info is available."""
    try:
        from src.utils.memory import get_gpu_memory

        info = get_gpu_memory()
        if info is not None:
            return info.free
    except Exception:
        pass
    return None


def compute_optimal_gpu_layers(
    model_path: str | Path,
    n_ctx: int = 4096,
    safety_margin: float = 0.90,
    kv_bytes_per_ctx_per_layer: int = 1024,
) -> int:
    """Compute the optimal number of layers to offload to the GPU.

    Algorithm
    ---------
    1. Read GGUF metadata to get total layer count.
    2. Query free VRAM.
    3. Reserve a fixed overhead budget (compute buffers, etc.).
    4. Use remaining VRAM for model weights at a per-layer cost derived
       from the model file size.

    Parameters
    ----------
    model_path:
        Path to the GGUF model file.
    n_ctx:
        Context window size (affects KV cache VRAM usage).
    safety_margin:
        Fraction of free VRAM to use (0.0–1.0). Default 0.90 keeps a
        10 % buffer for runtime overhead.
    kv_bytes_per_ctx_per_layer:
        Bytes of KV cache VRAM consumed per context token per GPU layer.
        A conservative default is 1024 (suits Q8_0 KV with typical
        head dimensions). Reduce for smaller architectures.

    Returns
    -------
    int
        Recommended n_gpu_layers. Returns 0 if no GPU, or the total
        layer count if everything fits.
    """
    path = Path(model_path)
    meta = _read_gguf_metadata(path)
    n_layers = _get_block_count(meta)

    if n_layers is None:
        logger.warning(
            "Could not determine layer count from %s. "
            "Returning n_gpu_layers=0 (CPU-only).",
            path.name,
        )
        return 0

    free_vram = _get_free_vram_bytes()
    if free_vram is None:
        logger.info("No GPU detected — using CPU-only (n_gpu_layers=0).")
        return 0

    # Budget available after applying safety margin (single deduction — no double reserve)
    # safety_margin already encapsulates all overhead: KV cache growth, runtime buffers,
    # activation memory.  Default 0.92 → use 92 % of free VRAM.
    budget = int(free_vram * safety_margin)

    # KV cache cost for layers we put on GPU
    # We'll solve: budget - kv_cost(gl) - weight_cost(gl) >= 0
    # kv_cost(gl) = gl * kv_bytes_per_ctx_per_layer * n_ctx
    # weight_cost(gl) = gl * bytes_per_layer
    try:
        model_bytes = path.stat().st_size
    except OSError:
        logger.warning("Cannot stat model file %s", path)
        return 0

    bytes_per_layer = model_bytes / n_layers
    kv_cost_per_layer = kv_bytes_per_ctx_per_layer * n_ctx

    cost_per_gpu_layer = bytes_per_layer + kv_cost_per_layer
    if cost_per_gpu_layer <= 0:
        return 0

    optimal = min(n_layers, int(budget / cost_per_gpu_layer))
    optimal = max(0, optimal)

    logger.info(
        "GPU layer distribution — model: %s | layers: %d | "
        "free VRAM: %.1f GB | budget: %.1f GB | "
        "bytes/layer: %.0f MB | kv/layer: %.0f MB | "
        "optimal n_gpu_layers: %d",
        path.name,
        n_layers,
        free_vram / 1e9,
        budget / 1e9,
        bytes_per_layer / 1e6,
        kv_cost_per_layer / 1e6,
        optimal,
    )

    return optimal


def compute_optimal_context(
    model_path: str | Path,
    *,
    min_gpu_layers: int = 4,
    safety_margin: float = 0.90,
    kv_bytes_per_ctx_per_layer: int = 1024,
    kv_config: Any | None = None,
) -> tuple[int, int]:
    """Co-optimise n_ctx and n_gpu_layers for maximum context window.

    Finds the largest power-of-two n_ctx (up to the model's native max)
    that still keeps at least *min_gpu_layers* on the GPU.

    Parameters
    ----------
    model_path:
        Path to the GGUF model file.
    min_gpu_layers:
        Minimum acceptable GPU layers (below this, reduce n_ctx instead).
    safety_margin:
        Fraction of free VRAM to use.
    kv_bytes_per_ctx_per_layer:
        Per-token per-GPU-layer KV cost in bytes.
    kv_config:
        Optional ``KVCacheConfig`` — if supplied, actual per-token KV
        bytes are computed from the model's architecture metadata.

    Returns
    -------
    (n_ctx, n_gpu_layers)
        The co-optimised values.
    """
    path = Path(model_path)
    meta = _read_gguf_metadata(path)
    n_layers = _get_block_count(meta)
    if n_layers is None:
        return 8192, 0  # safe fallback

    # Native context ceiling from the GGUF architecture
    arch = meta.get("general.architecture", "")
    native_ctx = 131072  # default ceiling
    for key in (f"{arch}.context_length", "llama.context_length"):
        if key in meta:
            native_ctx = int(meta[key])
            break

    free_vram = _get_free_vram_bytes()
    if free_vram is None:
        # CPU-only: use RAM budget instead
        import psutil
        ram_avail = psutil.virtual_memory().available
        try:
            model_bytes = path.stat().st_size
        except OSError:
            return 8192, 0
        bytes_per_layer = model_bytes / n_layers
        ram_for_kv = ram_avail * 0.80 - model_bytes
        if ram_for_kv <= 0:
            return 4096, 0
        _kv_per_token = _estimate_kv_bytes_per_token(meta, kv_config)
        max_ctx = int(ram_for_kv / _kv_per_token) if _kv_per_token > 0 else 8192
        max_ctx = min(max_ctx, native_ctx)
        # Round down to nearest power of 2
        n_ctx = 1
        while n_ctx * 2 <= max_ctx:
            n_ctx *= 2
        return max(4096, n_ctx), 0

    budget = int(free_vram * safety_margin)

    try:
        model_bytes = path.stat().st_size
    except OSError:
        return 8192, 0

    bytes_per_layer = model_bytes / n_layers
    _kv_per_token = _estimate_kv_bytes_per_token(meta, kv_config)

    # Sweep from largest viable n_ctx down until min_gpu_layers is satisfied.
    # Candidate n_ctx values: powers of 2 from native_ctx down to 4096.
    candidates = []
    ctx = 4096
    while ctx <= native_ctx:
        candidates.append(ctx)
        ctx *= 2

    best_ctx = 4096
    best_gl = 0

    for ctx in reversed(candidates):
        kv_vram = _kv_per_token * ctx  # total KV cache in VRAM
        remaining = budget - kv_vram
        if remaining <= 0:
            continue
        gl = min(n_layers, int(remaining / bytes_per_layer))
        gl = max(0, gl)
        if gl >= min_gpu_layers:
            best_ctx = ctx
            best_gl = gl
            break
        # Even if below threshold, remember best option
        if gl > best_gl:
            best_gl = gl
            best_ctx = ctx

    logger.info(
        "Auto context — model: %s | native_ctx: %d | "
        "VRAM budget: %.1f GB | KV/token: %d B | "
        "optimal: n_ctx=%d, n_gpu_layers=%d",
        path.name, native_ctx, budget / 1e9, _kv_per_token,
        best_ctx, best_gl,
    )

    return best_ctx, best_gl


def _estimate_kv_bytes_per_token(meta: dict[str, Any], kv_config: Any | None) -> int:
    """Estimate total KV cache bytes per context token from GGUF metadata."""
    arch = meta.get("general.architecture", "llama")

    # Try architecture-specific keys first, then fall back to llama.*
    def _get(suffix: str, default: int) -> int:
        for prefix in (arch, "llama"):
            key = f"{prefix}.{suffix}"
            if key in meta:
                return int(meta[key])
        return default

    n_layers = _get("block_count", 32)
    n_kv_heads = _get("attention.head_count_kv", 4)
    # head_dim = key_length (or rope.dimension_count, or embedding / head_count)
    head_dim = _get("attention.key_length", 128)

    # bits per element (default Q8_0 = 8.5 bpe for both K and V)
    k_bpe = 8.5
    v_bpe = 8.5
    if kv_config is not None:
        from src.engine.kv_cache import _BITS_PER_ELEMENT
        k_bpe = _BITS_PER_ELEMENT.get(kv_config.cache_type_k, 8.5)
        v_bpe = _BITS_PER_ELEMENT.get(kv_config.cache_type_v, 8.5)

    k_bytes_per_token = n_layers * n_kv_heads * head_dim * k_bpe / 8
    v_bytes_per_token = n_layers * n_kv_heads * head_dim * v_bpe / 8
    return int(k_bytes_per_token + v_bytes_per_token)


def layer_distribution_report(
    model_path: str | Path,
    n_gpu_layers: int,
    n_ctx: int = 4096,
) -> dict:
    """Generate a human-readable GPU/CPU distribution report.

    Returns a dict suitable for logging or the /health endpoint.
    """
    path = Path(model_path)
    meta = _read_gguf_metadata(path)
    n_layers = _get_block_count(meta) or 0
    arch = meta.get("general.architecture", "unknown")

    gpu_layers = min(n_gpu_layers, n_layers)
    cpu_layers = max(0, n_layers - gpu_layers)

    free_vram = _get_free_vram_bytes()
    try:
        model_mb = path.stat().st_size / 1e6
    except OSError:
        model_mb = 0.0

    return {
        "architecture": arch,
        "total_layers": n_layers,
        "gpu_layers": gpu_layers,
        "cpu_layers": cpu_layers,
        "gpu_fraction_pct": round(gpu_layers / max(n_layers, 1) * 100, 1),
        "model_size_mb": round(model_mb, 0),
        "free_vram_mb": round(free_vram / 1e6, 0) if free_vram else None,
        "n_ctx": n_ctx,
    }
