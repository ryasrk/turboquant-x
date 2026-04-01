#!/usr/bin/env python3
"""Benchmark report generator for TurboQuant KV cache evaluation.

Aggregates results from perplexity, speed, and NIAH benchmarks into a
comprehensive markdown report with matplotlib/seaborn visualizations.

Usage::

    python -m benchmarks.generate_report

    python -m benchmarks.generate_report \
        --results-dir benchmarks/results \
        --output benchmarks/results/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow execution from project root or benchmarks/ directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.kv_cache import (  # noqa: E402
    MODEL_ARCHITECTURES,
    CacheType,
    KVCacheConfig,
    estimate_kv_memory_bytes,
)
from src.utils.memory import get_gpu_memory, get_ram_usage  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "REPORT.md"
DEFAULT_ARCH = "qwen2.5-7b"
CHART_DPI = 150

# Expected paper values for comparison
PAPER_VALUES: dict[str, dict[str, float]] = {
    "q8_0/turbo4": {"ppl_delta_percent": 0.23, "compression": 2.5},
    "q8_0/turbo3": {"ppl_delta_percent": 1.06, "compression": 2.9},
    "q8_0/turbo2": {"ppl_delta_percent": 6.48, "compression": 3.2},
    "q4_0/q4_0":   {"ppl_delta_percent": 0.52, "compression": 3.6},
}

# Consistent color palette across all charts — keyed by config name.
CONFIG_COLORS: dict[str, str] = {
    "f16/f16":     "#4C72B0",
    "q8_0/q8_0":   "#55A868",
    "q4_0/q4_0":   "#C44E52",
    "q8_0/turbo4": "#8172B3",
    "q8_0/turbo3": "#CCB974",
    "q8_0/turbo2": "#64B5CD",
}
_FALLBACK_PALETTE = [
    "#DD8452", "#DA8BC3", "#8C8C8C", "#A1C9F4", "#FFB482",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class HardwareInfo:
    """Auto-detected hardware specifications."""

    cpu: str = "Unknown"
    gpu: str = "Unknown"
    vram_gb: float = 0.0
    ram_gb: float = 0.0

    def to_markdown_table(self) -> str:
        """Render hardware info as a markdown table."""
        from tabulate import tabulate as _tabulate

        rows = [
            ["CPU", self.cpu],
            ["GPU", self.gpu],
            ["VRAM", f"{self.vram_gb:.1f} GB"],
            ["System RAM", f"{self.ram_gb:.1f} GB"],
        ]
        return _tabulate(rows, headers=["Component", "Detail"], tablefmt="github")


@dataclass
class BenchmarkData:
    """Container for all loaded benchmark results."""

    ppl: dict[str, Any] | None = None
    speed: dict[str, Any] | None = None
    niah: dict[str, Any] | None = None

    @property
    def model_name(self) -> str:
        """Best-effort model name from any available result file."""
        for source in (self.ppl, self.speed, self.niah):
            if source and "model" in source:
                return source["model"]
        return "Unknown"

    @property
    def n_ctx(self) -> int:
        """Context size from ppl results or default."""
        if self.ppl and "n_ctx" in self.ppl:
            return int(self.ppl["n_ctx"])
        return 8192


@dataclass
class ConfigScore:
    """Aggregated scoring for recommendation logic."""

    config_name: str
    ppl_delta_percent: float = float("inf")
    niah_accuracy: float = 0.0
    decode_tok_s: float = 0.0
    peak_vram_gb: float = float("inf")

    @property
    def viable(self) -> bool:
        """Whether this config meets minimum viability thresholds."""
        return self.niah_accuracy >= 0.95


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def detect_hardware() -> HardwareInfo:
    """Auto-detect CPU, GPU, VRAM, and system RAM."""
    info = HardwareInfo()

    # CPU
    info.cpu = _detect_cpu()

    # GPU + VRAM
    gpu_name = _detect_gpu_name()
    if gpu_name:
        info.gpu = gpu_name
    gpu_mem = get_gpu_memory()
    if gpu_mem is not None:
        info.vram_gb = gpu_mem.total_gb

    # RAM
    ram = get_ram_usage()
    info.ram_gb = ram.total_gb

    return info


def _detect_cpu() -> str:
    """Read CPU model string from /proc/cpuinfo or platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def _detect_gpu_name() -> str | None:
    """Get GPU name via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_benchmark_data(results_dir: Path) -> BenchmarkData:
    """Load all benchmark JSON files from *results_dir*.

    Missing files are logged as warnings and the corresponding field is
    left as ``None``.
    """
    data = BenchmarkData()

    file_map: dict[str, str] = {
        "ppl": "ppl_results.json",
        "speed": "speed_results.json",
        "niah": "niah_results.json",
    }

    for attr, filename in file_map.items():
        path = results_dir / filename
        if not path.exists():
            logger.warning("Result file not found, section will be skipped: %s", path)
            continue
        try:
            with open(path) as f:
                setattr(data, attr, json.load(f))
            logger.info("Loaded %s (%d bytes)", path, path.stat().st_size)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    return data


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def _color_for(config_name: str, idx: int = 0) -> str:
    """Return a consistent color for a config name."""
    if config_name in CONFIG_COLORS:
        return CONFIG_COLORS[config_name]
    return _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]


def _ordered_colors(config_names: list[str]) -> list[str]:
    """Return a color list matching the order of *config_names*."""
    fallback_idx = 0
    colors: list[str] = []
    for name in config_names:
        if name in CONFIG_COLORS:
            colors.append(CONFIG_COLORS[name])
        else:
            colors.append(_FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)])
            fallback_idx += 1
    return colors


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
def generate_charts(data: BenchmarkData, results_dir: Path) -> dict[str, Path]:
    """Generate all visualization charts.

    Returns a mapping of chart name → file path for charts that were
    successfully created.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping chart generation")
        return {}

    sns.set_theme(style="whitegrid", palette="muted")
    charts: dict[str, Path] = {}

    if data.ppl:
        path = _chart_ppl(data.ppl, results_dir, plt, sns)
        if path:
            charts["ppl"] = path

    if data.speed:
        for name, path in _chart_speed(data.speed, results_dir, plt, sns).items():
            charts[name] = path

    if data.speed:
        path = _chart_memory(data.speed, results_dir, plt, sns)
        if path:
            charts["memory"] = path

    plt.close("all")
    return charts


def _chart_ppl(
    ppl_data: dict[str, Any],
    results_dir: Path,
    plt: Any,
    sns: Any,
) -> Path | None:
    """Perplexity delta bar chart comparing measured vs paper values."""
    results = ppl_data.get("results", [])
    if not results:
        return None

    # Exclude f16 baseline (delta = 0)
    entries = [r for r in results if r.get("ppl_delta_percent", 0.0) != 0.0]
    if not entries:
        entries = results  # fallback: show everything

    names = [e["config_name"] for e in entries]
    measured = [e.get("ppl_delta_percent", 0.0) for e in entries]
    paper = [PAPER_VALUES.get(n, {}).get("ppl_delta_percent", 0.0) for n in names]
    colors = _ordered_colors(names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    bar_w = 0.35
    bars_m = ax.bar(
        [i - bar_w / 2 for i in x], measured, bar_w,
        label="Measured", color=colors, edgecolor="white", linewidth=0.5,
    )
    bars_p = ax.bar(
        [i + bar_w / 2 for i in x], paper, bar_w,
        label="Paper", color=colors, alpha=0.45, edgecolor="white", linewidth=0.5,
    )

    ax.set_xlabel("KV Cache Configuration")
    ax.set_ylabel("PPL Delta vs f16 (%)")
    ax.set_title("Perplexity Impact — Measured vs Paper")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")

    # Value labels
    for bar in bars_m:
        height = bar.get_height()
        ax.annotate(
            f"{height:+.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    path = results_dir / "chart_ppl.png"
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    logger.info("Saved PPL chart → %s", path)
    return path


def _chart_speed(
    speed_data: dict[str, Any],
    results_dir: Path,
    plt: Any,
    sns: Any,
) -> dict[str, Path]:
    """Generate decode tok/s and TTFT charts, grouped by context length."""
    results = speed_data.get("results", [])
    if not results:
        return {}

    charts: dict[str, Path] = {}

    # ----- Decode tok/s chart -----
    configs = sorted({r["config_name"] for r in results})
    ctx_lengths = sorted({r["context_length"] for r in results})

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.8 / max(len(configs), 1)

    for i, cfg in enumerate(configs):
        vals = []
        errs = []
        for ctx in ctx_lengths:
            entry = _find_result(results, cfg, ctx)
            if entry:
                vals.append(entry["decode_tokens_per_sec"]["mean"])
                errs.append(entry["decode_tokens_per_sec"]["std"])
            else:
                vals.append(0)
                errs.append(0)
        positions = [j + i * bar_w for j in range(len(ctx_lengths))]
        ax.bar(
            positions, vals, bar_w, yerr=errs,
            label=cfg, color=_color_for(cfg, i),
            edgecolor="white", linewidth=0.5, capsize=3,
        )

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Decode Tokens/s")
    ax.set_title("Decode Throughput by Configuration and Context Length")
    ax.set_xticks([j + bar_w * (len(configs) - 1) / 2 for j in range(len(ctx_lengths))])
    ax.set_xticklabels([str(c) for c in ctx_lengths])
    ax.legend(loc="upper right", fontsize=8)

    path = results_dir / "chart_speed_decode.png"
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    charts["speed_decode"] = path
    logger.info("Saved decode speed chart → %s", path)

    # ----- TTFT chart -----
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cfg in enumerate(configs):
        ttft_vals = []
        ttft_errs = []
        for ctx in ctx_lengths:
            entry = _find_result(results, cfg, ctx)
            if entry:
                ttft_vals.append(entry["ttft_ms"]["mean"])
                ttft_errs.append(entry["ttft_ms"]["std"])
            else:
                ttft_vals.append(0)
                ttft_errs.append(0)
        ax.errorbar(
            ctx_lengths, ttft_vals, yerr=ttft_errs,
            marker="o", label=cfg, color=_color_for(cfg, i),
            linewidth=1.5, capsize=3,
        )

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Time to First Token (ms)")
    ax.set_title("TTFT by Configuration and Context Length")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks(ctx_lengths)

    path = results_dir / "chart_ttft.png"
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    charts["ttft"] = path
    logger.info("Saved TTFT chart → %s", path)

    return charts


def _chart_memory(
    speed_data: dict[str, Any],
    results_dir: Path,
    plt: Any,
    sns: Any,
) -> Path | None:
    """Peak VRAM bar chart grouped by config (max across context lengths)."""
    results = speed_data.get("results", [])
    if not results:
        return None

    # Aggregate peak VRAM per config (take max across all context lengths)
    vram_by_config: dict[str, float] = {}
    for r in results:
        name = r["config_name"]
        peak = r.get("peak_vram_gb", 0.0)
        vram_by_config[name] = max(vram_by_config.get(name, 0.0), peak)

    configs = sorted(vram_by_config.keys())
    values = [vram_by_config[c] for c in configs]
    colors = _ordered_colors(configs)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(configs, values, color=colors, edgecolor="white", linewidth=0.5)

    # Total VRAM line
    hw = speed_data.get("hardware", {})
    total_vram = hw.get("vram_gb", 0)
    if total_vram > 0:
        ax.axhline(y=total_vram, color="red", linewidth=1.2, linestyle="--", label=f"Total VRAM ({total_vram:.1f} GB)")
        ax.legend(fontsize=8)

    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.1f} GB",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("KV Cache Configuration")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Peak VRAM Usage by Configuration")
    ax.set_xticklabels(configs, rotation=30, ha="right")

    path = results_dir / "chart_memory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    logger.info("Saved memory chart → %s", path)
    return path


def _find_result(
    results: list[dict[str, Any]], config_name: str, context_length: int
) -> dict[str, Any] | None:
    """Find a speed result matching config + context length."""
    for r in results:
        if r["config_name"] == config_name and r["context_length"] == context_length:
            return r
    return None


# ---------------------------------------------------------------------------
# KV cache memory table (computed, not from benchmark results)
# ---------------------------------------------------------------------------
def _kv_memory_table(n_ctx_list: list[int] | None = None) -> str:
    """Generate a KV cache memory comparison table using estimates."""
    from tabulate import tabulate as _tabulate

    if n_ctx_list is None:
        n_ctx_list = [2048, 4096, 8192]

    arch = MODEL_ARCHITECTURES.get(DEFAULT_ARCH, {})
    n_layers = arch.get("n_layers", 28)
    n_heads = arch.get("n_heads", 28)
    head_dim = arch.get("head_dim", 128)

    configs: list[tuple[str, KVCacheConfig]] = [
        ("f16/f16", KVCacheConfig(CacheType.F16, CacheType.F16, True)),
        ("q8_0/q8_0", KVCacheConfig(CacheType.Q8_0, CacheType.Q8_0, True)),
        ("q4_0/q4_0", KVCacheConfig(CacheType.Q4_0, CacheType.Q4_0, True)),
        ("q8_0/turbo4", KVCacheConfig(CacheType.Q8_0, CacheType.TURBO4, True)),
        ("q8_0/turbo3", KVCacheConfig(CacheType.Q8_0, CacheType.TURBO3, True)),
        ("q8_0/turbo2", KVCacheConfig(CacheType.Q8_0, CacheType.TURBO2, True)),
    ]

    headers = ["Config"] + [f"{ctx} ctx (MB)" for ctx in n_ctx_list] + ["Compression"]
    rows: list[list[str]] = []

    for name, cfg in configs:
        row: list[str] = [name]
        compression = 1.0
        for n_ctx in n_ctx_list:
            est = estimate_kv_memory_bytes(n_ctx, n_layers, n_heads, head_dim, cfg)
            row.append(f"{est['total_mb']:.1f}")
            compression = est["compression_vs_f16"]
        row.append(f"{compression:.2f}x")
        rows.append(row)

    return _tabulate(rows, headers=headers, tablefmt="github")


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def recommend_config(
    data: BenchmarkData, available_vram_gb: float
) -> ConfigScore | None:
    """Determine the best KV cache configuration for the available hardware.

    Best config = lowest PPL delta among configs where:
    - Peak VRAM ≤ available VRAM
    - NIAH accuracy ≥ 95%
    - Decode speed ≥ q4_0/q4_0 baseline (if available)
    """
    scores: dict[str, ConfigScore] = {}

    # Seed with PPL data
    if data.ppl:
        for r in data.ppl.get("results", []):
            name = r["config_name"]
            scores.setdefault(name, ConfigScore(config_name=name))
            scores[name].ppl_delta_percent = r.get("ppl_delta_percent", float("inf"))

    # Merge speed data
    if data.speed:
        for r in data.speed.get("results", []):
            name = r["config_name"]
            scores.setdefault(name, ConfigScore(config_name=name))
            peak = r.get("peak_vram_gb", 0.0)
            scores[name].peak_vram_gb = max(scores[name].peak_vram_gb if scores[name].peak_vram_gb != float("inf") else 0.0, peak)
            decode = r.get("decode_tokens_per_sec", {}).get("mean", 0.0)
            scores[name].decode_tok_s = max(scores[name].decode_tok_s, decode)

    # Merge NIAH data
    if data.niah:
        for r in data.niah.get("results", []):
            name = r["config_name"]
            scores.setdefault(name, ConfigScore(config_name=name))
            scores[name].niah_accuracy = r.get("average_accuracy", 0.0)

    if not scores:
        return None

    # Decode speed baseline (q4_0/q4_0)
    q4_baseline = scores.get("q4_0/q4_0")
    min_decode = q4_baseline.decode_tok_s if q4_baseline else 0.0

    # Filter viable candidates
    candidates = [
        s for s in scores.values()
        if s.viable
        and (s.peak_vram_gb <= available_vram_gb or s.peak_vram_gb == float("inf"))
        and s.decode_tok_s >= min_decode
    ]

    if not candidates:
        # Relax constraints — just filter by VRAM
        candidates = [
            s for s in scores.values()
            if s.peak_vram_gb <= available_vram_gb or s.peak_vram_gb == float("inf")
        ]

    if not candidates:
        return None

    # Sort by PPL delta (lower is better)
    candidates.sort(key=lambda s: s.ppl_delta_percent)
    return candidates[0]


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------
def generate_report(
    data: BenchmarkData,
    hardware: HardwareInfo,
    charts: dict[str, Path],
    results_dir: Path,
) -> str:
    """Build the full markdown report string."""
    from tabulate import tabulate as _tabulate

    sections: list[str] = []

    # Header
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        f"# TurboQuant Benchmark Report\n\n"
        f"**Model:** {data.model_name}  \n"
        f"**Date:** {timestamp}  \n"
        f"**Context Size:** {data.n_ctx}\n"
    )

    # Hardware
    sections.append(f"## Hardware Specifications\n\n{hardware.to_markdown_table()}\n")

    # Model information
    sections.append(_section_model_info(data))

    # KV Cache memory table
    sections.append(f"## KV Cache Memory Usage (Estimated)\n\n{_kv_memory_table()}\n")

    # Perplexity
    sections.append(_section_ppl(data, charts, _tabulate))

    # Speed
    sections.append(_section_speed(data, charts, _tabulate))

    # NIAH
    sections.append(_section_niah(data, charts, _tabulate))

    # Memory summary
    sections.append(_section_memory(data, charts, _tabulate))

    # Recommendation
    available_vram = hardware.vram_gb
    if available_vram <= 0 and data.speed:
        available_vram = data.speed.get("hardware", {}).get("vram_gb", 0)
    rec = recommend_config(data, available_vram)
    sections.append(_section_recommendation(rec, available_vram))

    # Conclusion
    sections.append(_section_conclusion(data))

    return "\n---\n\n".join(sections)


def _section_model_info(data: BenchmarkData) -> str:
    """Model information section."""
    from tabulate import tabulate as _tabulate

    arch = MODEL_ARCHITECTURES.get(DEFAULT_ARCH, {})
    rows = [
        ["Model", data.model_name],
        ["Architecture", DEFAULT_ARCH],
        ["Layers", arch.get("n_layers", "—")],
        ["Attention Heads", arch.get("n_heads", "—")],
        ["Head Dimension", arch.get("head_dim", "—")],
        ["Context Size", data.n_ctx],
    ]

    if data.ppl and "dataset" in data.ppl:
        rows.append(["PPL Dataset", data.ppl["dataset"]])

    return f"## Model Information\n\n{_tabulate(rows, headers=['Parameter', 'Value'], tablefmt='github')}\n"


def _section_ppl(
    data: BenchmarkData,
    charts: dict[str, Path],
    tabulate_fn: Any,
) -> str:
    """Perplexity results section."""
    if data.ppl is None:
        return "## Perplexity Results\n\n> **Note:** Perplexity benchmark results not found — skipped.\n"

    results = data.ppl.get("results", [])
    if not results:
        return "## Perplexity Results\n\n> **Note:** No perplexity data available.\n"

    headers = [
        "Config", "Perplexity", "Δ vs f16", "Δ %",
        "Paper Δ %", "KV Memory (MB)", "Compression",
    ]
    rows: list[list[str]] = []

    for r in results:
        name = r["config_name"]
        paper_delta = PAPER_VALUES.get(name, {}).get("ppl_delta_percent")
        paper_str = f"{paper_delta:+.2f}%" if paper_delta is not None else "—"
        rows.append([
            name,
            f"{r.get('perplexity', 0):.2f}",
            f"{r.get('ppl_delta_vs_f16', 0):+.3f}",
            f"{r.get('ppl_delta_percent', 0):+.2f}%",
            paper_str,
            f"{r.get('kv_memory_mb', 0):.1f}",
            f"{r.get('compression_vs_f16', 0):.2f}x",
        ])

    table = tabulate_fn(rows, headers=headers, tablefmt="github")

    # Analysis
    analysis_lines: list[str] = []
    for r in results:
        name = r["config_name"]
        measured = r.get("ppl_delta_percent", 0.0)
        paper = PAPER_VALUES.get(name, {}).get("ppl_delta_percent")
        if paper is not None and paper != 0:
            diff = measured - paper
            direction = "higher" if diff > 0 else "lower"
            analysis_lines.append(
                f"- **{name}**: Measured Δ {measured:+.2f}% vs paper {paper:+.2f}% "
                f"({abs(diff):.2f}pp {direction})"
            )

    analysis = "\n".join(analysis_lines) if analysis_lines else ""

    chart_ref = ""
    if "ppl" in charts:
        chart_ref = f"\n![PPL Comparison](chart_ppl.png)\n"

    return (
        f"## Perplexity Results\n\n{table}\n"
        f"{chart_ref}\n"
        f"### Analysis\n\n{analysis}\n"
    )


def _section_speed(
    data: BenchmarkData,
    charts: dict[str, Path],
    tabulate_fn: Any,
) -> str:
    """Speed results section."""
    if data.speed is None:
        return "## Speed Results\n\n> **Note:** Speed benchmark results not found — skipped.\n"

    results = data.speed.get("results", [])
    if not results:
        return "## Speed Results\n\n> **Note:** No speed data available.\n"

    headers = [
        "Config", "Context", "Prefill (tok/s)", "Decode (tok/s)",
        "TTFT (ms)", "Total (s)", "Peak VRAM (GB)", "KV Mem (MB)",
    ]
    rows: list[list[str]] = []

    for r in results:
        prefill = r.get("prefill_tokens_per_sec", {})
        decode = r.get("decode_tokens_per_sec", {})
        ttft = r.get("ttft_ms", {})
        total = r.get("total_time_s", {})
        rows.append([
            r["config_name"],
            str(r.get("context_length", "—")),
            f"{prefill.get('mean', 0):.1f} ± {prefill.get('std', 0):.1f}",
            f"{decode.get('mean', 0):.1f} ± {decode.get('std', 0):.1f}",
            f"{ttft.get('mean', 0):.1f} ± {ttft.get('std', 0):.1f}",
            f"{total.get('mean', 0):.2f} ± {total.get('std', 0):.2f}",
            f"{r.get('peak_vram_gb', 0):.1f}",
            f"{r.get('kv_memory_mb', 0):.1f}",
        ])

    table = tabulate_fn(rows, headers=headers, tablefmt="github")

    chart_refs: list[str] = []
    if "speed_decode" in charts:
        chart_refs.append("![Decode Speed](chart_speed_decode.png)")
    if "ttft" in charts:
        chart_refs.append("![TTFT](chart_ttft.png)")
    charts_md = "\n\n".join(chart_refs)

    return f"## Speed Results\n\n{table}\n\n{charts_md}\n"


def _section_niah(
    data: BenchmarkData,
    charts: dict[str, Path],
    tabulate_fn: Any,
) -> str:
    """Needle-in-a-Haystack results section."""
    if data.niah is None:
        return "## NIAH Results\n\n> **Note:** NIAH benchmark results not found — skipped.\n"

    results = data.niah.get("results", [])
    if not results:
        return "## NIAH Results\n\n> **Note:** No NIAH data available.\n"

    # Summary table
    headers = ["Config", "Avg Accuracy", "Retrieval Latency (s)"]
    rows: list[list[str]] = []
    for r in results:
        latency = r.get("retrieval_latency_s", {})
        rows.append([
            r["config_name"],
            f"{r.get('average_accuracy', 0) * 100:.1f}%",
            f"{latency.get('mean', 0):.2f} ± {latency.get('std', 0):.2f}",
        ])

    summary_table = tabulate_fn(rows, headers=headers, tablefmt="github")

    # Per-config score grids
    grids: list[str] = []
    settings = data.niah.get("settings", {})
    depths = settings.get("depths", [])
    ctx_lengths = settings.get("context_lengths", [])

    for r in results:
        scores = r.get("scores", {})
        if not scores:
            continue

        grid_headers = ["Depth \\ Context"] + [str(c) for c in ctx_lengths]
        grid_rows: list[list[str]] = []
        for d in depths:
            row: list[str] = [str(d)]
            for c in ctx_lengths:
                val = scores.get(str(c), {}).get(str(d))
                if val is not None:
                    cell = f"{val * 100:.0f}%" if isinstance(val, float) else str(val)
                else:
                    cell = "—"
                row.append(cell)
            grid_rows.append(row)

        grid_table = tabulate_fn(grid_rows, headers=grid_headers, tablefmt="github")
        grids.append(f"### {r['config_name']}\n\n{grid_table}\n")

    grids_md = "\n".join(grids) if grids else ""

    return f"## NIAH Results (Needle-in-a-Haystack)\n\n{summary_table}\n\n{grids_md}\n"


def _section_memory(
    data: BenchmarkData,
    charts: dict[str, Path],
    tabulate_fn: Any,
) -> str:
    """Memory usage summary section."""
    if data.speed is None:
        return "## Memory Usage Summary\n\n> **Note:** No speed data available for memory analysis.\n"

    results = data.speed.get("results", [])
    if not results:
        return "## Memory Usage Summary\n\n> **Note:** No memory data available.\n"

    # Aggregate peak VRAM per config
    vram_by_config: dict[str, float] = {}
    kv_by_config: dict[str, float] = {}
    for r in results:
        name = r["config_name"]
        vram_by_config[name] = max(vram_by_config.get(name, 0.0), r.get("peak_vram_gb", 0.0))
        kv_by_config[name] = max(kv_by_config.get(name, 0.0), r.get("kv_memory_mb", 0.0))

    headers = ["Config", "Peak VRAM (GB)", "Max KV Cache (MB)"]
    rows = [
        [name, f"{vram_by_config[name]:.1f}", f"{kv_by_config.get(name, 0):.1f}"]
        for name in sorted(vram_by_config.keys())
    ]

    table = tabulate_fn(rows, headers=headers, tablefmt="github")

    chart_ref = ""
    if "memory" in charts:
        chart_ref = "\n![Memory Usage](chart_memory.png)\n"

    return f"## Memory Usage Summary\n\n{table}\n{chart_ref}\n"


def _section_recommendation(rec: ConfigScore | None, available_vram: float) -> str:
    """Recommended configuration section."""
    section = "## Recommended Configuration\n\n"

    if rec is None:
        return section + (
            "> **Note:** Unable to determine recommendation — "
            "insufficient benchmark data or no config fits hardware constraints.\n"
        )

    section += (
        f"**Best Configuration: `{rec.config_name}`**\n\n"
        f"| Criterion | Value |\n"
        f"|-----------|-------|\n"
        f"| PPL Δ vs f16 | {rec.ppl_delta_percent:+.2f}% |\n"
        f"| NIAH Accuracy | {rec.niah_accuracy * 100:.1f}% |\n"
        f"| Decode Speed | {rec.decode_tok_s:.1f} tok/s |\n"
        f"| Peak VRAM | {rec.peak_vram_gb:.1f} GB / {available_vram:.1f} GB available |\n"
        f"\n"
    )

    criteria: list[str] = []
    criteria.append(f"- VRAM headroom: {available_vram - rec.peak_vram_gb:.1f} GB remaining")
    if rec.niah_accuracy >= 0.98:
        criteria.append("- Near-perfect retrieval accuracy (≥ 98%)")
    elif rec.niah_accuracy >= 0.95:
        criteria.append("- Good retrieval accuracy (≥ 95%)")
    if rec.ppl_delta_percent < 1.0:
        criteria.append("- Minimal perplexity impact (< 1%)")

    paper = PAPER_VALUES.get(rec.config_name)
    if paper:
        criteria.append(
            f"- Paper claims: Δ {paper['ppl_delta_percent']:+.2f}%, "
            f"{paper['compression']:.1f}x compression"
        )

    section += "\n".join(criteria) + "\n"
    return section


def _section_conclusion(data: BenchmarkData) -> str:
    """Summary and conclusion section."""
    section = "## Conclusion\n\n"
    findings: list[str] = []

    if data.ppl:
        results = data.ppl.get("results", [])
        turbo_results = [
            r for r in results
            if "turbo" in r.get("config_name", "")
        ]
        if turbo_results:
            avg_delta = sum(r.get("ppl_delta_percent", 0) for r in turbo_results) / len(turbo_results)
            findings.append(
                f"- **Perplexity:** TurboQuant configs show an average "
                f"PPL delta of {avg_delta:+.2f}% vs f16 baseline."
            )

    if data.speed:
        results = data.speed.get("results", [])
        if results:
            max_decode = max(
                (r.get("decode_tokens_per_sec", {}).get("mean", 0) for r in results),
                default=0,
            )
            findings.append(
                f"- **Speed:** Peak decode throughput reached {max_decode:.1f} tokens/s."
            )

    if data.niah:
        results = data.niah.get("results", [])
        if results:
            min_acc = min(
                (r.get("average_accuracy", 0) for r in results),
                default=0,
            )
            findings.append(
                f"- **Accuracy:** Minimum NIAH accuracy across configs: "
                f"{min_acc * 100:.1f}%."
            )

    # Paper comparison summary
    if data.ppl:
        matches: list[str] = []
        for r in data.ppl.get("results", []):
            name = r.get("config_name", "")
            paper = PAPER_VALUES.get(name)
            if paper:
                measured = r.get("ppl_delta_percent", 0)
                expected = paper["ppl_delta_percent"]
                if abs(measured - expected) < 1.0:
                    matches.append(name)
        if matches:
            findings.append(
                f"- **Paper Validation:** Configs within 1pp of paper claims: "
                f"{', '.join(matches)}."
            )

    if not findings:
        findings.append("- Insufficient data for comprehensive conclusions.")

    section += "\n".join(findings) + "\n"
    return section


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run(results_dir: Path, output: Path) -> None:
    """Load data, generate charts, write markdown report."""
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark data from %s", results_dir)
    data = load_benchmark_data(results_dir)

    logger.info("Detecting hardware...")
    hardware = detect_hardware()

    # Override hardware from speed results if auto-detect is incomplete
    if hardware.vram_gb <= 0 and data.speed:
        hw_data = data.speed.get("hardware", {})
        if hw_data.get("vram_gb"):
            hardware.vram_gb = hw_data["vram_gb"]
        if hw_data.get("gpu"):
            hardware.gpu = hw_data["gpu"]
        if hw_data.get("ram_gb"):
            hardware.ram_gb = hw_data["ram_gb"]

    logger.info("Generating charts...")
    charts = generate_charts(data, results_dir)

    logger.info("Building markdown report...")
    report = generate_report(data, hardware, charts, results_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    logger.info("Report written to %s (%d bytes)", output, output.stat().st_size)

    # Summary
    chart_count = len(charts)
    section_count = report.count("\n## ")
    print(f"\n✓ Report saved: {output}")
    print(f"  Sections: {section_count} | Charts: {chart_count}")
    if charts:
        for name, path in sorted(charts.items()):
            print(f"    • {path.name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate TurboQuant benchmark report with charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing benchmark JSON results (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the markdown report (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    run(results_dir=args.results_dir, output=args.output)


if __name__ == "__main__":
    main()
