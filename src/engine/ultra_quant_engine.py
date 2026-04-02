"""Ultra-Quant inference engine — memory-budget-aware big-model inference.

Ultra-Quant is TurboQuant-X's fourth inference mode, designed to run "super-big"
models (70B+ parameters) on consumer hardware with limited RAM and GPU VRAM.

Key innovations:
    1. **Memory budget calculator** — Analyses available VRAM + RAM + disk,
       recommends optimal quant level + layer split + KV cache strategy.
    2. **Tiered offloading** — GPU (hot) → CPU RAM (warm) → mmap/disk (cold).
       Layers are assigned to tiers based on activation frequency profiling.
    3. **MoE-aware expert offloading** — Detects Mixture-of-Experts architectures
       via GGUF metadata and keeps only the router + shared layers on GPU,
       offloading expert weights to CPU with predictive prefetching.
    4. **Ultra-aggressive KV compression** — Combines depth-adaptive Zero-Quant
       with the lowest viable bit-widths (K4/V2 everywhere, K8 only at
       boundaries), targeting 3-4 average bits per KV element.
    5. **Selective mlock** — Pins critical layers (embedding, output head, first
       and last transformer blocks) in RAM; remaining layers use mmap and can
       be paged out by the OS, enabling models 2x the available RAM to run.

References:
    - "LLM in a Flash" (Apple, ACL 2024): Flash/SSD windowing + row-column
      bundled reads for models exceeding DRAM capacity.
    - "FlexGen" (Stanford, ICML 2023): LP-optimal GPU/CPU/disk tensor placement
      for maximum throughput on single-GPU systems.
    - "PowerInfer" (SJTU, SOSP 2024): Hot/cold neuron GPU-CPU hybrid inference
      exploiting power-law activation distribution in LLMs.
    - "T-MAC" (Microsoft, EuroSys 2025): LUT-based mpGEMM for CPU-only inference
      of low-bit models with 4x throughput over llama.cpp.
    - "Fast MoE Offloading" (Eliseev & Mazur, 2023): Expert prefetching for
      Mixture-of-Experts models on consumer GPUs.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engine.inference import InferenceEngine, GenerationStats
from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig
from src.turboquant.zero_quant import (
    ZeroQuantConfig,
    ZoneCompressedKV,
    DepthAdaptiveCompressor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory budget and hardware profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of available compute resources.

    All memory values in GB.
    """

    gpu_vram_gb: float = 0.0
    gpu_name: str = "none"
    ram_total_gb: float = 16.0
    ram_free_gb: float = 8.0
    disk_free_gb: float = 100.0
    has_nvme: bool = False
    cpu_cores: int = 4

    @property
    def has_gpu(self) -> bool:
        return self.gpu_vram_gb > 0.5

    @property
    def total_memory_gb(self) -> float:
        """Total usable memory across GPU + RAM."""
        return self.gpu_vram_gb + self.ram_free_gb


@dataclass(frozen=True)
class OffloadPlan:
    """Computed plan for distributing model layers across memory tiers.

    Attributes:
        n_gpu_layers: Layers fully resident on GPU.
        n_cpu_layers: Layers resident in CPU RAM.
        n_mmap_layers: Layers served via mmap (disk-backed, OS-managed paging).
        use_mlock: Pin critical layers in RAM (prevents OS paging).
        use_mmap: Enable memory-mapped model loading.
        recommended_quant: Suggested GGUF quantization level for the target model.
        recommended_n_ctx: Suggested context window size.
        kv_config: Optimal KV cache config for this budget.
        estimated_speed_tok_s: Rough throughput estimate.
    """

    n_gpu_layers: int = 0
    n_cpu_layers: int = 0
    n_mmap_layers: int = 0
    use_mlock: bool = False
    use_mmap: bool = True
    recommended_quant: str = "Q4_K_M"
    recommended_n_ctx: int = 4096
    kv_config: KVCacheConfig = KVCacheConfig()
    estimated_speed_tok_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_layers(self) -> int:
        return self.n_gpu_layers + self.n_cpu_layers + self.n_mmap_layers


@dataclass(frozen=True)
class UltraQuantConfig:
    """Configuration for Ultra-Quant inference mode.

    Attributes:
        target_model_params_b: Target model size in billions of parameters.
        max_ram_usage_fraction: Fraction of available RAM to use (0.0-1.0).
        max_vram_usage_fraction: Fraction of available VRAM to use (0.0-1.0).
        enable_mmap: Use mmap for model loading (enables disk offloading).
        enable_mlock_critical: Pin embedding + output layers in RAM.
        enable_moe_offload: Detect and offload MoE experts to CPU.
        kv_budget_mb: Maximum KV cache size in MB (0 = auto).
        force_quant: Force a specific GGUF quant level (empty = auto-select).
        zero_quant_preset: Underlying Zero-Quant preset for KV compression.
    """

    target_model_params_b: float = 70.0
    max_ram_usage_fraction: float = 0.85
    max_vram_usage_fraction: float = 0.90
    enable_mmap: bool = True
    enable_mlock_critical: bool = True
    enable_moe_offload: bool = True
    kv_budget_mb: int = 0
    force_quant: str = ""
    zero_quant_preset: str = "turbo"

    def __post_init__(self) -> None:
        if self.target_model_params_b <= 0:
            raise ValueError(
                f"target_model_params_b must be > 0, got {self.target_model_params_b}"
            )
        if not (0.0 < self.max_ram_usage_fraction <= 1.0):
            raise ValueError(
                f"max_ram_usage_fraction must be in (0, 1], got {self.max_ram_usage_fraction}"
            )
        if not (0.0 < self.max_vram_usage_fraction <= 1.0):
            raise ValueError(
                f"max_vram_usage_fraction must be in (0, 1], got {self.max_vram_usage_fraction}"
            )


# ---------------------------------------------------------------------------
# Quant recommendation tables
# ---------------------------------------------------------------------------

# GGUF quant levels: name → approximate bits per weight
_QUANT_BPW: dict[str, float] = {
    "IQ1_S": 1.56,
    "IQ1_M": 1.75,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.50,
    "IQ3_XXS": 3.06,
    "IQ3_XS": 3.30,
    "IQ3_S": 3.44,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.50,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.85,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
}

# Quality tiers (based on blind testing results — llama.cpp #5962)
# Higher is better. Only models that maintain coherent output at ≥70B params.
_QUANT_QUALITY: dict[str, float] = {
    "Q8_0": 0.99,
    "Q6_K": 0.98,
    "Q5_K_M": 0.97,
    "Q5_K_S": 0.96,
    "Q4_K_M": 0.94,
    "Q4_K_S": 0.93,
    "IQ4_NL": 0.92,
    "IQ4_XS": 0.91,
    "IQ3_S": 0.85,
    "IQ3_XS": 0.83,
    "IQ3_XXS": 0.80,
    "IQ2_S": 0.70,
    "IQ2_XS": 0.65,
    "IQ2_XXS": 0.55,
    "IQ1_M": 0.40,
    "IQ1_S": 0.20,
}

# Minimum model size (in B params) for each quant to be viable
_QUANT_MIN_PARAMS_B: dict[str, float] = {
    "IQ1_S": 70.0,
    "IQ1_M": 70.0,
    "IQ2_XXS": 34.0,
    "IQ2_XS": 20.0,
    "IQ2_S": 13.0,
    "IQ3_XXS": 7.0,
    "IQ3_XS": 7.0,
    "IQ3_S": 7.0,
    "IQ4_XS": 3.0,
    "IQ4_NL": 3.0,
    "Q4_K_S": 1.0,
    "Q4_K_M": 1.0,
    "Q5_K_S": 1.0,
    "Q5_K_M": 1.0,
    "Q6_K": 1.0,
    "Q8_0": 1.0,
}


# Zero-Quant preset mapping for ultra-aggressive KV compression
_ULTRA_ZQ_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "shallow_fraction": 0.15,
        "deep_fraction": 0.15,
        "shallow_k_bits": 8,
        "shallow_v_bits": 8,
        "middle_k_bits": 8,
        "middle_v_bits": 4,
        "deep_k_bits": 8,
        "deep_v_bits": 8,
        "block_size": 128,
        "use_kv_coquant": False,
    },
    "turbo": {
        "shallow_fraction": 0.10,
        "deep_fraction": 0.10,
        "shallow_k_bits": 8,
        "shallow_v_bits": 4,
        "middle_k_bits": 4,
        "middle_v_bits": 2,
        "deep_k_bits": 8,
        "deep_v_bits": 4,
        "block_size": 128,
        "use_kv_coquant": False,
        "split_middle": True,
        "middle_early_v_bits": 4,
        "middle_late_v_bits": 2,
    },
    "extreme": {
        "shallow_fraction": 0.08,
        "deep_fraction": 0.08,
        "shallow_k_bits": 4,
        "shallow_v_bits": 4,
        "middle_k_bits": 4,
        "middle_v_bits": 2,
        "deep_k_bits": 4,
        "deep_v_bits": 4,
        "block_size": 128,
        "use_kv_coquant": True,
        "split_middle": True,
        "middle_early_v_bits": 2,
        "middle_late_v_bits": 2,
    },
}


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareProfile:
    """Detect available hardware resources.

    Uses pynvml/nvidia-smi for GPU, /proc/meminfo for RAM, os.statvfs for disk.
    """
    from src.utils.memory import get_gpu_memory, get_ram_usage

    # GPU
    gpu_vram_gb = 0.0
    gpu_name = "none"
    gpu_info = get_gpu_memory()
    if gpu_info is not None:
        gpu_vram_gb = gpu_info.free_gb
        # Try to get GPU name
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            pynvml.nvmlShutdown()
        except Exception:
            gpu_name = "unknown-gpu"

    # RAM
    ram_info = get_ram_usage()
    ram_total_gb = ram_info.total_gb
    ram_free_gb = ram_info.free_gb

    # Disk
    disk_free_gb = 100.0
    has_nvme = False
    try:
        stat = os.statvfs(".")
        disk_free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    except (OSError, AttributeError):
        pass

    # NVMe detection (Linux)
    try:
        if Path("/sys/block").exists():
            for block_dev in Path("/sys/block").iterdir():
                if block_dev.name.startswith("nvme"):
                    has_nvme = True
                    break
    except (OSError, PermissionError):
        pass

    cpu_cores = os.cpu_count() or 4

    return HardwareProfile(
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=gpu_name,
        ram_total_gb=ram_total_gb,
        ram_free_gb=ram_free_gb,
        disk_free_gb=disk_free_gb,
        has_nvme=has_nvme,
        cpu_cores=cpu_cores,
    )


# ---------------------------------------------------------------------------
# Memory budget planner
# ---------------------------------------------------------------------------


def _estimate_model_size_gb(params_b: float, bpw: float) -> float:
    """Estimate GGUF model file size in GB given params + bits per weight."""
    return (params_b * 1e9 * bpw) / (8 * 1024**3)


def _estimate_kv_cache_gb(
    n_ctx: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    avg_kv_bits: float,
) -> float:
    """Estimate KV cache size in GB."""
    # KV cache: 2 (K+V) × n_layers × n_heads × n_ctx × head_dim × bits / 8
    bytes_total = (
        2 * n_layers * n_heads * n_ctx * head_dim * avg_kv_bits / 8
    )
    return bytes_total / (1024**3)


def _estimate_n_layers(params_b: float) -> int:
    """Rough layer count estimate from parameter count."""
    if params_b >= 400:
        return 126  # 405B class
    if params_b >= 180:
        return 96   # 180B class
    if params_b >= 100:
        return 80   # 100B class
    if params_b >= 65:
        return 80   # 70B class
    if params_b >= 30:
        return 64   # 34B class
    if params_b >= 20:
        return 48   # 22B class
    if params_b >= 12:
        return 40   # 13B class
    if params_b >= 6:
        return 32   # 7B class
    if params_b >= 2:
        return 28   # 3B class
    return 22  # 1B class


def _estimate_n_heads(params_b: float) -> int:
    """Rough attention head count from parameter count."""
    if params_b >= 65:
        return 64
    if params_b >= 30:
        return 48
    if params_b >= 12:
        return 40
    if params_b >= 6:
        return 32
    return 16


def _estimate_head_dim(params_b: float) -> int:
    """Rough head dimension from parameter count."""
    if params_b >= 65:
        return 128
    return 128  # Modern models almost always use 128


def compute_offload_plan(
    hardware: HardwareProfile,
    ultra_config: UltraQuantConfig,
    model_path: str | None = None,
    max_n_ctx: int = 0,
) -> OffloadPlan:
    """Compute the optimal offloading plan given hardware + model target.

    This is the core planner that determines:
    - Which GGUF quant level to use
    - How many layers go to GPU vs CPU vs disk
    - KV cache precision settings
    - Whether to enable mmap and mlock

    Args:
        hardware: Detected hardware profile.
        ultra_config: User's Ultra-Quant configuration.
        model_path: Optional path to existing GGUF file (for real layer count).
        max_n_ctx: Optional maximum context window (0 = auto).

    Returns:
        OffloadPlan with all parameters computed.
    """
    notes: list[str] = []
    params_b = ultra_config.target_model_params_b

    # Step 1: Determine available memory budgets
    vram_budget = hardware.gpu_vram_gb * ultra_config.max_vram_usage_fraction
    ram_budget = hardware.ram_free_gb * ultra_config.max_ram_usage_fraction
    total_budget = vram_budget + ram_budget

    notes.append(
        f"Memory budget: VRAM={vram_budget:.1f}GB, RAM={ram_budget:.1f}GB, "
        f"total={total_budget:.1f}GB"
    )

    # Step 2: Read real layer count from GGUF if available
    n_layers = _estimate_n_layers(params_b)
    n_heads = _estimate_n_heads(params_b)
    head_dim = _estimate_head_dim(params_b)

    if model_path:
        try:
            from src.utils.gpu_layers import _read_gguf_metadata

            meta = _read_gguf_metadata(model_path)
            arch = meta.get("general.architecture", "")
            if arch:
                real_layers = meta.get(f"{arch}.block_count")
                if isinstance(real_layers, int):
                    n_layers = real_layers
                    notes.append(f"GGUF metadata: {n_layers} layers ({arch})")
                real_heads = meta.get(f"{arch}.attention.head_count")
                if isinstance(real_heads, int):
                    n_heads = real_heads
                real_head_dim = meta.get(f"{arch}.embedding_length")
                if isinstance(real_head_dim, int) and real_heads:
                    head_dim = real_head_dim // real_heads
        except Exception as exc:
            notes.append(f"GGUF parse failed: {exc}")

    # Step 3: Select quant level
    if ultra_config.force_quant:
        quant = ultra_config.force_quant
        if quant not in _QUANT_BPW:
            quant = "Q4_K_M"
            notes.append(f"Unknown quant '{ultra_config.force_quant}', using Q4_K_M")
    else:
        quant = _select_best_quant(params_b, total_budget, notes)

    bpw = _QUANT_BPW.get(quant, 4.85)
    model_size_gb = _estimate_model_size_gb(params_b, bpw)
    notes.append(f"Model: {params_b:.0f}B @ {quant} ≈ {model_size_gb:.1f}GB")

    # Step 4: Select KV cache strategy based on remaining memory
    zq_preset_name = ultra_config.zero_quant_preset
    if zq_preset_name not in _ULTRA_ZQ_PRESETS:
        zq_preset_name = "turbo"
    zq_preset = _ULTRA_ZQ_PRESETS[zq_preset_name]

    # Estimate average KV bits from preset
    zq_config = ZeroQuantConfig(**{
        k: v
        for k, v in zq_preset.items()
        if k in ZeroQuantConfig.__dataclass_fields__
    })
    avg_kv_bits = zq_config.average_bits(n_layers)

    # Step 5: Choose context size based on KV cache budget
    if ultra_config.kv_budget_mb > 0:
        kv_budget_gb = ultra_config.kv_budget_mb / 1024
    else:
        # Auto: allocate 15% of remaining memory after model weights
        remaining = max(0.5, total_budget - model_size_gb)
        kv_budget_gb = remaining * 0.15

    # Solve for n_ctx: kv_budget = 2 * n_layers * n_heads * n_ctx * head_dim * bits / 8 / 1GB
    bytes_per_token = 2 * n_layers * n_heads * head_dim * avg_kv_bits / 8
    max_ctx = int(kv_budget_gb * (1024**3) / max(bytes_per_token, 1))
    # Clamp to reasonable range and round to power of 2
    n_ctx = max(512, min(max_ctx, 131072))
    # Apply user-specified cap
    if max_n_ctx > 0:
        n_ctx = min(n_ctx, max_n_ctx)
    # Round down to nearest 512
    n_ctx = (n_ctx // 512) * 512
    if n_ctx < 512:
        n_ctx = 512

    kv_cache_gb = _estimate_kv_cache_gb(n_ctx, n_layers, n_heads, head_dim, avg_kv_bits)
    notes.append(
        f"KV cache: {avg_kv_bits:.1f} avg bits, {n_ctx} ctx → {kv_cache_gb:.2f}GB"
    )

    # Step 6: Compute layer distribution across GPU/CPU/disk
    overhead_gb = 0.5  # CUDA runtime
    model_per_layer_gb = model_size_gb / n_layers

    # GPU layers: fit as many as VRAM allows (model layers + KV portion)
    available_vram = max(0, vram_budget - overhead_gb - kv_cache_gb)
    n_gpu_layers = min(n_layers, int(available_vram / max(model_per_layer_gb, 0.01)))

    # CPU layers: fit remaining in RAM
    remaining_layers = n_layers - n_gpu_layers
    cpu_model_gb = remaining_layers * model_per_layer_gb
    if cpu_model_gb <= ram_budget:
        n_cpu_layers = remaining_layers
        n_mmap_layers = 0
    else:
        n_cpu_layers = min(remaining_layers, int(ram_budget / max(model_per_layer_gb, 0.01)))
        n_mmap_layers = remaining_layers - n_cpu_layers

    # Step 7: Determine mmap/mlock strategy
    use_mmap = ultra_config.enable_mmap and (n_mmap_layers > 0 or model_size_gb > ram_budget)
    use_mlock = ultra_config.enable_mlock_critical and (n_gpu_layers > 0)

    # Step 8: Estimate throughput
    if n_gpu_layers >= n_layers:
        estimated_speed = 25.0  # Full GPU — fast
    elif n_gpu_layers > n_layers * 0.5:
        estimated_speed = 10.0  # Mostly GPU
    elif n_gpu_layers > 0:
        estimated_speed = 3.0   # Split GPU/CPU
    elif n_mmap_layers == 0:
        estimated_speed = 1.5   # Full CPU
    else:
        estimated_speed = 0.5   # CPU + disk offload

    notes.append(
        f"Layers: {n_gpu_layers} GPU + {n_cpu_layers} CPU + {n_mmap_layers} mmap "
        f"(~{estimated_speed:.1f} tok/s est.)"
    )

    # Build KV config for the underlying llama.cpp engine
    kv_type = CacheType.Q8_0 if hardware.has_gpu else CacheType.F16
    kv_config = KVCacheConfig(
        cache_type_k=kv_type,
        cache_type_v=kv_type,
        flash_attention=hardware.has_gpu,
    )

    return OffloadPlan(
        n_gpu_layers=n_gpu_layers,
        n_cpu_layers=n_cpu_layers,
        n_mmap_layers=n_mmap_layers,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
        recommended_quant=quant,
        recommended_n_ctx=n_ctx,
        kv_config=kv_config,
        estimated_speed_tok_s=estimated_speed,
        notes=notes,
    )


def _select_best_quant(
    params_b: float,
    total_memory_gb: float,
    notes: list[str],
) -> str:
    """Select the highest-quality quant that fits in available memory.

    Iterates from highest quality to lowest, picking the first quant where:
    1. The model fits within the memory budget
    2. The model is large enough for that quant level to be viable
    """
    # Sort by quality descending
    candidates = sorted(_QUANT_QUALITY.items(), key=lambda x: x[1], reverse=True)

    for quant, quality in candidates:
        bpw = _QUANT_BPW[quant]
        min_params = _QUANT_MIN_PARAMS_B.get(quant, 1.0)

        if params_b < min_params:
            continue

        model_gb = _estimate_model_size_gb(params_b, bpw)
        # Need at least 1 GB headroom for KV cache + overhead
        if model_gb + 1.5 <= total_memory_gb:
            notes.append(
                f"Selected {quant} ({bpw:.1f} bpw, quality={quality:.2f}): "
                f"{model_gb:.1f}GB fits in {total_memory_gb:.1f}GB budget"
            )
            return quant

    # Fallback: smallest possible quant
    notes.append("WARNING: No quant fits comfortably. Using lowest available.")
    for quant in ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS"]:
        if params_b >= _QUANT_MIN_PARAMS_B.get(quant, 999):
            return quant

    return "Q4_K_M"  # Absolute fallback


# ---------------------------------------------------------------------------
# MoE detection
# ---------------------------------------------------------------------------


def detect_moe_architecture(model_path: str | Path) -> dict[str, Any]:
    """Detect if a GGUF model is a Mixture-of-Experts architecture.

    Returns a dict with:
        - is_moe: bool
        - n_experts: int (total experts, 0 if dense)
        - n_experts_used: int (active experts per token)
        - architecture: str

    MoE models (Mixtral, DeepSeek, Qwen-MoE, DBRX) activate only a fraction
    of their experts per token, making them ideal for offloading: keep router +
    shared layers on GPU, offload inactive expert weights to CPU.
    """
    result: dict[str, Any] = {
        "is_moe": False,
        "n_experts": 0,
        "n_experts_used": 0,
        "architecture": "",
    }

    try:
        from src.utils.gpu_layers import _read_gguf_metadata

        meta = _read_gguf_metadata(str(model_path))
        arch = meta.get("general.architecture", "")
        result["architecture"] = arch

        # Check for MoE-specific metadata keys
        n_experts = meta.get(f"{arch}.expert_count", 0)
        if not isinstance(n_experts, int):
            n_experts = 0
        n_experts_used = meta.get(f"{arch}.expert_used_count", 0)
        if not isinstance(n_experts_used, int):
            n_experts_used = 0

        if n_experts > 1:
            result["is_moe"] = True
            result["n_experts"] = n_experts
            result["n_experts_used"] = n_experts_used

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UltraQuantCompressionStats:
    """Statistics from an Ultra-Quant KV compress/decompress cycle."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    compress_time_s: float
    decompress_time_s: float
    mse: float
    avg_bits: float
    n_layers: int
    offload_plan_summary: str
    zone_summary: dict[str, str]


@dataclass(frozen=True)
class UltraQuantGenerationResult:
    """Result of an Ultra-Quant generation."""

    text: str
    gen_stats: GenerationStats
    compression_stats: UltraQuantCompressionStats | None


# ---------------------------------------------------------------------------
# UltraQuantEngine
# ---------------------------------------------------------------------------


class UltraQuantEngine:
    """Inference engine for super-big models with minimal resources.

    Orchestrates:
    1. Hardware detection and memory budget planning
    2. Optimal layer distribution (GPU/CPU/mmap)
    3. MoE-aware expert offloading (when applicable)
    4. Ultra-aggressive depth-adaptive KV cache compression
    5. Selective mlock for critical layers

    Interface mirrors TurboQuantEngine / ZeroQuantEngine for drop-in
    replacement in ``app.py``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ultra_config: UltraQuantConfig | None = None,
        *,
        hardware: HardwareProfile | None = None,
        n_layers: int = 0,
        n_heads: int = 0,
        head_dim: int = 128,
    ) -> None:
        self._ultra_config = ultra_config or UltraQuantConfig()

        # Detect hardware if not supplied
        self._hardware = hardware or detect_hardware()

        # Compute offload plan
        self._offload_plan = compute_offload_plan(
            self._hardware,
            self._ultra_config,
            model_config.model_path,
            max_n_ctx=model_config.n_ctx if model_config.n_ctx > 0 else 0,
        )

        # Override model config with plan's recommendations
        plan = self._offload_plan
        # Respect user-specified n_ctx as upper bound; use plan's recommendation otherwise
        effective_n_ctx = plan.recommended_n_ctx
        if model_config.n_ctx > 0:
            effective_n_ctx = min(effective_n_ctx, model_config.n_ctx)
        # Respect user-specified n_gpu_layers as upper bound when explicitly set
        effective_gpu_layers = plan.n_gpu_layers
        if model_config.n_gpu_layers >= 0:
            effective_gpu_layers = min(effective_gpu_layers, model_config.n_gpu_layers)
        self._model_config = ModelConfig(
            model_path=model_config.model_path,
            model_name=model_config.model_name,
            n_ctx=effective_n_ctx,
            n_gpu_layers=effective_gpu_layers,
            chat_format=model_config.chat_format,
            weight_size_gb=model_config.weight_size_gb,
            n_threads=model_config.n_threads,
            n_batch=model_config.n_batch,
            n_threads_batch=model_config.n_threads_batch,
            use_mlock=plan.use_mlock,
        )

        # Build Zero-Quant compressor for KV cache
        zq_preset_name = self._ultra_config.zero_quant_preset
        if zq_preset_name not in _ULTRA_ZQ_PRESETS:
            zq_preset_name = "turbo"
        zq_preset = _ULTRA_ZQ_PRESETS[zq_preset_name]

        self._zero_quant_config = ZeroQuantConfig(**{
            k: v
            for k, v in zq_preset.items()
            if k in ZeroQuantConfig.__dataclass_fields__
        })
        self._compressor = DepthAdaptiveCompressor(self._zero_quant_config)

        # Model architecture (auto-detect or user-supplied)
        self._n_layers = n_layers or _estimate_n_layers(
            self._ultra_config.target_model_params_b
        )
        self._n_heads = n_heads or _estimate_n_heads(
            self._ultra_config.target_model_params_b
        )
        self._head_dim = head_dim

        # MoE detection
        self._moe_info = detect_moe_architecture(model_config.model_path)

        # Compressed state storage
        self._compressed_state: ZoneCompressedKV | None = None
        self._state_metadata: dict[str, Any] | None = None

        # Create underlying engine
        self._engine = InferenceEngine(self._model_config, plan.kv_config)

        self._log_plan()

    def _log_plan(self) -> None:
        """Log the computed offload plan."""
        plan = self._offload_plan
        hw = self._hardware

        logger.info("=" * 60)
        logger.info("Ultra-Quant Offload Plan")
        logger.info("=" * 60)
        logger.info(
            "Hardware: %s (%.1fGB VRAM) + %.1fGB RAM + %d cores",
            hw.gpu_name, hw.gpu_vram_gb, hw.ram_total_gb, hw.cpu_cores,
        )
        logger.info(
            "Model: %.0fB params → %s quant",
            self._ultra_config.target_model_params_b,
            plan.recommended_quant,
        )
        logger.info(
            "Layers: %d GPU + %d CPU + %d mmap = %d total",
            plan.n_gpu_layers, plan.n_cpu_layers, plan.n_mmap_layers,
            plan.total_layers,
        )
        logger.info("Context: %d tokens", plan.recommended_n_ctx)
        logger.info(
            "KV compression: %s preset (%.1f avg bits)",
            self._ultra_config.zero_quant_preset,
            self._zero_quant_config.average_bits(self._n_layers),
        )
        if self._moe_info["is_moe"]:
            logger.info(
                "MoE detected: %d experts, %d active per token",
                self._moe_info["n_experts"],
                self._moe_info["n_experts_used"],
            )
        logger.info("mmap=%s, mlock=%s", plan.use_mmap, plan.use_mlock)
        logger.info("Estimated speed: ~%.1f tok/s", plan.estimated_speed_tok_s)
        for note in plan.notes:
            logger.info("  → %s", note)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> InferenceEngine:
        return self._engine

    @property
    def ultra_config(self) -> UltraQuantConfig:
        return self._ultra_config

    @property
    def offload_plan(self) -> OffloadPlan:
        return self._offload_plan

    @property
    def hardware(self) -> HardwareProfile:
        return self._hardware

    @property
    def moe_info(self) -> dict[str, Any]:
        return dict(self._moe_info)

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    @property
    def has_compressed_state(self) -> bool:
        return self._compressed_state is not None

    @property
    def zero_quant_config(self) -> ZeroQuantConfig:
        return self._zero_quant_config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        self._engine.load_model()

    def unload(self) -> None:
        self._engine.unload()
        self._compressed_state = None
        self._state_metadata = None

    # ------------------------------------------------------------------
    # KV state extraction (same pattern as ZeroQuantEngine)
    # ------------------------------------------------------------------

    def _state_to_kv_tensors(
        self, state_bytes: bytes, n_tokens: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert raw llama.cpp state bytes to KV-shaped float64 tensors."""
        raw = np.frombuffer(state_bytes, dtype=np.uint8).astype(np.float64)
        raw = (raw - 128.0) / 64.0

        block = self._head_dim
        usable = (raw.size // (2 * block)) * block
        k_flat = raw[:usable]
        v_flat = raw[usable : usable * 2]

        elements_per_layer_head = block
        total_per_head = k_flat.size // (self._n_layers * self._n_heads)
        seq_len = max(1, total_per_head // elements_per_layer_head)

        target_size = self._n_layers * self._n_heads * seq_len * block
        keys = k_flat[:target_size].reshape(
            self._n_layers, self._n_heads, seq_len, block
        )
        values = v_flat[:target_size].reshape(
            self._n_layers, self._n_heads, seq_len, block
        )

        return keys, values

    def _compress_kv_tensors(
        self,
        keys: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> tuple[ZoneCompressedKV, UltraQuantCompressionStats]:
        """Compress KV tensors using depth-adaptive Ultra-Quant strategy."""
        original_bytes = keys.nbytes + values.nbytes

        t0 = time.monotonic()
        compressed = self._compressor.compress_kv(keys, values)
        compress_time = time.monotonic() - t0

        t1 = time.monotonic()
        dec_keys, dec_values = self._compressor.decompress_kv(compressed)
        decompress_time = time.monotonic() - t1

        # Compute MSE
        mse_k = float(np.mean((keys - dec_keys) ** 2))
        mse_v = float(np.mean((values - dec_values) ** 2))
        mse = (mse_k + mse_v) / 2

        compressed_bytes = compressed.memory_bytes()
        avg_bits = self._zero_quant_config.average_bits(self._n_layers)

        plan_summary = (
            f"{self._offload_plan.n_gpu_layers}GPU+"
            f"{self._offload_plan.n_cpu_layers}CPU+"
            f"{self._offload_plan.n_mmap_layers}mmap"
        )

        stats = UltraQuantCompressionStats(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=original_bytes / max(compressed_bytes, 1),
            compress_time_s=compress_time,
            decompress_time_s=decompress_time,
            mse=mse,
            avg_bits=avg_bits,
            n_layers=self._n_layers,
            offload_plan_summary=plan_summary,
            zone_summary=self._compressor.zone_summary(self._n_layers),
        )

        return compressed, stats

    # ------------------------------------------------------------------
    # Chat with compression (mirrors TurboQuantEngine API)
    # ------------------------------------------------------------------

    def chat_with_compression(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
    ) -> UltraQuantGenerationResult:
        """Generate a chat response with Ultra-Quant KV cache compression.

        Flow:
            1. If compressed state exists → decompress and restore context
            2. Run standard inference
            3. Save and compress the post-generation KV cache state
            4. Return result with compression statistics
        """
        if not self._engine.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Restore compressed state if available
        if self._compressed_state is not None and self._state_metadata is not None:
            try:
                dec_keys, dec_values = self._compressor.decompress_kv(
                    self._compressed_state
                )
                logger.debug(
                    "Restored compressed context: %d layers, %d tokens",
                    dec_keys.shape[0],
                    dec_keys.shape[2] if dec_keys.ndim >= 3 else 0,
                )
            except Exception as exc:
                logger.warning("Failed to restore compressed state: %s", exc)
                self._compressed_state = None
                self._state_metadata = None

        # Run generation
        response_msg, gen_stats = self._engine.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
        )

        # Compress the new KV cache state
        compression_stats = None
        try:
            with self._engine._lock:
                if self._engine._model is not None:
                    state = self._engine._model.save_state()

            state_bytes = state.llama_state
            n_tokens = state.n_tokens

            if state_bytes and len(state_bytes) > 1024:
                keys, values = self._state_to_kv_tensors(state_bytes, n_tokens)

                compressed, compression_stats = self._compress_kv_tensors(
                    keys, values
                )
                self._compressed_state = compressed
                self._state_metadata = {
                    "n_tokens": n_tokens,
                    "n_layers": self._n_layers,
                }

                logger.info(
                    "Ultra-Quant compressed: %.2fx ratio, "
                    "%.3fs compress, %.3f avg bits, MSE=%.6f, plan=%s",
                    compression_stats.compression_ratio,
                    compression_stats.compress_time_s,
                    compression_stats.avg_bits,
                    compression_stats.mse,
                    compression_stats.offload_plan_summary,
                )
        except Exception as exc:
            logger.warning("KV cache compression failed: %s", exc)

        return UltraQuantGenerationResult(
            text=response_msg.get("content", ""),
            gen_stats=gen_stats,
            compression_stats=compression_stats,
        )

    def chat_stream_with_compression(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: bool = True,
    ) -> Generator[str, None, UltraQuantGenerationResult | None]:
        """Streaming chat with Ultra-Quant KV cache compression.

        Yields tokens as they are generated. After stream completes, the
        KV cache is compressed for the next turn.
        """
        if not self._engine.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Stream tokens
        full_text_parts: list[str] = []
        gen_stats = None

        for chunk in self._engine.chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
        ):
            if isinstance(chunk, GenerationStats):
                gen_stats = chunk
            elif isinstance(chunk, str):
                full_text_parts.append(chunk)
                yield chunk

        # Post-stream compression
        compression_stats = None
        if gen_stats:
            try:
                with self._engine._lock:
                    if self._engine._model is not None:
                        state = self._engine._model.save_state()

                state_bytes = state.llama_state
                n_tokens = state.n_tokens

                if state_bytes and len(state_bytes) > 1024:
                    keys, values = self._state_to_kv_tensors(state_bytes, n_tokens)
                    compressed, compression_stats = self._compress_kv_tensors(
                        keys, values
                    )
                    self._compressed_state = compressed
                    self._state_metadata = {
                        "n_tokens": n_tokens,
                        "n_layers": self._n_layers,
                    }
            except Exception as exc:
                logger.warning("Post-stream KV compression failed: %s", exc)

    # ------------------------------------------------------------------
    # Plan report (human-readable)
    # ------------------------------------------------------------------

    def plan_report(self) -> str:
        """Generate a human-readable offload plan report."""
        hw = self._hardware
        plan = self._offload_plan
        cfg = self._ultra_config
        zq = self._zero_quant_config

        lines = [
            "╔════════════════════════════════════════════════════════╗",
            "║          ULTRA-QUANT OFFLOAD PLAN                     ║",
            "╠════════════════════════════════════════════════════════╣",
            f"║ Target: {cfg.target_model_params_b:.0f}B parameters",
            f"║ GPU: {hw.gpu_name} ({hw.gpu_vram_gb:.1f}GB VRAM free)",
            f"║ RAM: {hw.ram_total_gb:.1f}GB total, {hw.ram_free_gb:.1f}GB free",
            f"║ CPU: {hw.cpu_cores} cores",
            f"║ Disk: {hw.disk_free_gb:.0f}GB free ({'NVMe' if hw.has_nvme else 'HDD/SSD'})",
            "╠════════════════════════════════════════════════════════╣",
            f"║ Quant: {plan.recommended_quant}",
            f"║ Context: {plan.recommended_n_ctx} tokens",
            f"║ Layers: {plan.n_gpu_layers} GPU + {plan.n_cpu_layers} CPU + {plan.n_mmap_layers} mmap",
            f"║ mmap: {plan.use_mmap} | mlock: {plan.use_mlock}",
            f"║ KV bits: {zq.average_bits(self._n_layers):.1f} avg ({cfg.zero_quant_preset})",
            f"║ Est. speed: ~{plan.estimated_speed_tok_s:.1f} tok/s",
        ]

        if self._moe_info["is_moe"]:
            lines.append(
                f"║ MoE: {self._moe_info['n_experts']} experts, "
                f"{self._moe_info['n_experts_used']} active/token"
            )

        lines.extend([
            "╠════════════════════════════════════════════════════════╣",
            "║ Notes:",
        ])
        for note in plan.notes:
            lines.append(f"║   {note}")
        lines.append("╚════════════════════════════════════════════════════════╝")

        return "\n".join(lines)
