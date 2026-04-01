"""Asymmetric K/V cache quantization presets and analysis for TurboQuant.

Community research finding: ALL quality degradation comes from K (key) compression.
V (value) cache can be compressed to 2-bit with near-zero quality loss if K precision
is maintained at q8_0 or higher.

Why asymmetric works:
  - Keys are used in attention score computation (dot product with queries).
    Small errors in K → large errors in attention weights after softmax.
  - Values are just weighted-averaged — quantization noise averages out.

This module provides preset configurations and a utility to analyse the trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.turboquant.compressor import QuantConfig


# ---------------------------------------------------------------------------
# Preset dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AsymmetricPreset:
    """Named preset for asymmetric K/V quantization."""

    name: str
    description: str
    config: QuantConfig
    expected_ppl_delta: float    # Expected PPL increase vs f16 baseline (percentage)
    expected_compression: float  # Expected compression ratio vs f16


# ---------------------------------------------------------------------------
# Preset configurations
# Based on TurboQuant paper + community benchmarks (6+ independent teams)
# ---------------------------------------------------------------------------

QUALITY_PRESET = AsymmetricPreset(
    name="quality",
    description="Best quality after f16 baseline. K at q8_0, V at turbo4.",
    config=QuantConfig(k_bits=8, v_bits=4),
    expected_ppl_delta=0.23,
    expected_compression=2.5,
)

BALANCED_PRESET = AsymmetricPreset(
    name="balanced",
    description="Good balance of quality and compression. K at q8_0, V at turbo3.",
    config=QuantConfig(k_bits=8, v_bits=3),
    expected_ppl_delta=1.06,
    expected_compression=2.9,
)

COMPRESSION_PRESET = AsymmetricPreset(
    name="compression",
    description="Maximum compression. K at q8_0, V at turbo2. Some quality loss.",
    config=QuantConfig(k_bits=8, v_bits=2),
    expected_ppl_delta=6.48,
    expected_compression=3.2,
)

SYMMETRIC_TURBO4_PRESET = AsymmetricPreset(
    name="symmetric_turbo4",
    description=(
        "Symmetric turbo4 for both K and V. "
        "NOT recommended — K quality matters."
    ),
    config=QuantConfig(k_bits=4, v_bits=4),
    expected_ppl_delta=0.52,
    expected_compression=3.8,
)

ALL_PRESETS: dict[str, AsymmetricPreset] = {
    "quality": QUALITY_PRESET,
    "balanced": BALANCED_PRESET,
    "compression": COMPRESSION_PRESET,
    "symmetric_turbo4": SYMMETRIC_TURBO4_PRESET,
}

# Ordered from highest quality to most aggressive compression.
_PREFERENCE_ORDER: list[str] = ["quality", "balanced", "compression", "symmetric_turbo4"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_preset(name: str) -> AsymmetricPreset:
    """Get a preset by name.

    Raises:
        KeyError: If *name* is not a recognised preset.
    """
    if name not in ALL_PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {list(ALL_PRESETS.keys())}"
        )
    return ALL_PRESETS[name]


def estimate_kv_memory_gb(
    preset: AsymmetricPreset,
    ctx_length: int,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
) -> float:
    """Estimate KV cache memory in GB for a given preset and context length.

    Formula per cache (K or V)::

        bytes = ctx_length * n_layers * n_heads * head_dim * bits / 8

    Total = K_bytes + V_bytes
    """
    per_token_elements = n_layers * n_heads * head_dim

    k_bytes = ctx_length * per_token_elements * preset.config.k_bits / 8
    v_bytes = ctx_length * per_token_elements * preset.config.v_bits / 8

    total_bytes = k_bytes + v_bytes
    return total_bytes / (1024 ** 3)


def recommend_preset(
    gpu_vram_gb: float,
    target_ctx_length: int,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
    model_weight_gb: float = 4.8,
) -> AsymmetricPreset:
    """Recommend the best preset based on hardware constraints.

    Logic:
      1. Calculate KV cache size for each preset at *target_ctx_length*.
      2. Filter presets where ``model_weights + kv_cache`` fits in GPU VRAM.
      3. Return highest-quality preset that fits.

    Args:
        gpu_vram_gb: Available GPU VRAM in GB.
        target_ctx_length: Desired context window length (tokens).
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        model_weight_gb: Model weight size in GB (post weight-quantization).

    Returns:
        Best fitting :class:`AsymmetricPreset` (quality > balanced > compression).

    Raises:
        ValueError: If no preset fits the available VRAM.
    """
    if gpu_vram_gb <= 0:
        raise ValueError("gpu_vram_gb must be positive")
    if target_ctx_length <= 0:
        raise ValueError("target_ctx_length must be positive")

    available_for_kv = gpu_vram_gb - model_weight_gb

    for preset_name in _PREFERENCE_ORDER:
        preset = ALL_PRESETS[preset_name]
        kv_gb = estimate_kv_memory_gb(
            preset,
            target_ctx_length,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        if kv_gb <= available_for_kv:
            return preset

    # Nothing fits — build a helpful error message.
    lines = [
        f"No preset fits in {gpu_vram_gb:.1f} GB VRAM "
        f"(model weights: {model_weight_gb:.1f} GB, "
        f"available for KV: {available_for_kv:.2f} GB).",
        "",
        "KV cache requirements at "
        f"{target_ctx_length} tokens ({n_layers}L/{n_heads}H/{head_dim}D):",
    ]
    for preset_name in _PREFERENCE_ORDER:
        preset = ALL_PRESETS[preset_name]
        kv_gb = estimate_kv_memory_gb(
            preset,
            target_ctx_length,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        lines.append(f"  {preset_name:20s} → {kv_gb:.3f} GB")

    lines.append("")
    lines.append(
        "Try reducing target_ctx_length or using a smaller model."
    )
    raise ValueError("\n".join(lines))


def format_preset_comparison(
    ctx_length: int = 8192,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
) -> str:
    """Generate a formatted comparison table of all presets.

    Returns a string table showing preset name, K/V bit widths,
    estimated KV cache memory, compression ratio, and expected PPL delta.
    """
    header = (
        f"{'Preset':<20s} | {'K bits':>6s} | {'V bits':>6s} | "
        f"{'KV Cache (GB)':>13s} | {'Compression':>11s} | {'PPL Delta':>9s}"
    )
    separator = "-" * len(header)

    rows: list[str] = [
        f"Context length: {ctx_length} tokens "
        f"({n_layers}L / {n_heads}H / {head_dim}D)",
        "",
        header,
        separator,
    ]

    # f16 baseline — computed directly (16 bits not a valid QuantConfig)
    per_token_elements = n_layers * n_heads * head_dim
    f16_kv_gb = (ctx_length * per_token_elements * 16 / 8 * 2) / (1024 ** 3)
    rows.append(
        f"{'f16_baseline':<20s} | {'16':>6s} | {'16':>6s} | "
        f"{f16_kv_gb:>13.3f} | {'1.0':>10s}x | +{'0.00':>7s}%"
    )

    for preset_name in _PREFERENCE_ORDER:
        preset = ALL_PRESETS[preset_name]
        kv_gb = estimate_kv_memory_gb(
            preset,
            ctx_length,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        rows.append(
            f"{preset.name:<20s} | {preset.config.k_bits:>6d} | "
            f"{preset.config.v_bits:>6d} | {kv_gb:>13.3f} | "
            f"{preset.expected_compression:>10.1f}x | "
            f"+{preset.expected_ppl_delta:>7.2f}%"
        )

    rows.append(separator)
    rows.append("")
    rows.append("Recommendation: 'quality' (K=q8_0, V=turbo4) for most use cases.")

    return "\n".join(rows)
