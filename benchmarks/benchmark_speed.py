#!/usr/bin/env python3
"""Speed/throughput benchmark for evaluating LLM inference across KV cache configs.

Measures prefill speed (prompt processing), decode speed (token generation),
time-to-first-token (TTFT), and peak VRAM across different KV cache
quantization configurations and context lengths.

Each (config, context_length) pair is warmed up and then measured over
multiple runs, reporting mean ± std for all metrics.

Usage::

    python -m benchmarks.benchmark_speed \
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf

    python -m benchmarks.benchmark_speed \
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \
        --configs q8_0/turbo4 f16/f16 --n-ctx 8192 --max-gen-tokens 64

    python -m benchmarks.benchmark_speed \
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \
        --warmup-runs 1 --measure-runs 3  # quick sanity run

Requires the ``bench`` extras::

    pip install -e ".[bench,gpu]"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow execution as both ``python benchmarks/benchmark_speed.py``
# and ``python -m benchmarks.benchmark_speed`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.kv_cache import (  # noqa: E402
    MODEL_ARCHITECTURES,
    CacheType,
    KVCacheConfig,
    estimate_kv_memory_bytes,
    to_llama_params,
)
from src.engine.model_config import ModelConfig  # noqa: E402
from src.utils.memory import get_gpu_memory, get_ram_usage  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000

DEFAULT_CONTEXT_LENGTHS: list[int] = [512, 2048, 4096, 8192]
DEFAULT_ARCH = "qwen2.5-7b"

BENCHMARK_CONFIGS: dict[str, KVCacheConfig] = {
    "f16/f16": KVCacheConfig(
        cache_type_k=CacheType.F16,
        cache_type_v=CacheType.F16,
        flash_attention=True,
    ),
    "q8_0/q8_0": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.Q8_0,
        flash_attention=True,
    ),
    "q4_0/q4_0": KVCacheConfig(
        cache_type_k=CacheType.Q4_0,
        cache_type_v=CacheType.Q4_0,
        flash_attention=True,
    ),
    "q8_0/turbo4": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO4,
        flash_attention=True,
    ),
    "q8_0/turbo3": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO3,
        flash_attention=True,
    ),
    "q8_0/turbo2": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO2,
        flash_attention=True,
    ),
}


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StatSummary:
    """Mean ± std for a metric across measurement runs."""

    mean: float
    std: float


@dataclass(frozen=True)
class SingleRunTiming:
    """Raw timing from a single benchmark run."""

    prefill_time_ns: int
    prefill_tokens: int
    decode_time_ns: int
    decode_tokens: int
    ttft_ns: int
    total_time_ns: int
    peak_vram_gb: float


@dataclass(frozen=True)
class SpeedResult:
    """Aggregated speed benchmark result for one (config, context_length) pair."""

    config_name: str
    cache_type_k: str
    cache_type_v: str
    context_length: int
    prefill_tokens_per_sec: StatSummary
    decode_tokens_per_sec: StatSummary
    ttft_ms: StatSummary
    total_time_s: StatSummary
    peak_vram_gb: float
    kv_memory_mb: float


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# Deterministic seed phrase repeated and truncated to exact token counts.
# Using a factual, low-entropy passage ensures reproducible tokenization.
_SEED_TEXT = (
    "The Transformer architecture, introduced in 2017 by Vaswani et al. in "
    "the paper 'Attention Is All You Need', revolutionized natural language "
    "processing by replacing recurrence with self-attention mechanisms. "
    "Unlike recurrent neural networks that process tokens sequentially, "
    "Transformers compute attention scores across all positions in parallel, "
    "enabling significantly faster training on modern hardware. The key "
    "innovation is the scaled dot-product attention, where queries, keys, "
    "and values are projected from the input embeddings, and attention "
    "weights are computed as softmax of the dot product of queries and keys "
    "divided by the square root of the dimension. Multi-head attention "
    "extends this by running multiple attention functions in parallel, each "
    "with different learned projections, allowing the model to jointly "
    "attend to information from different representation subspaces. The "
    "original architecture uses an encoder-decoder structure with residual "
    "connections and layer normalization, though decoder-only variants have "
    "become dominant for language modeling tasks. "
)


def generate_prompt(model: Any, target_tokens: int) -> str:
    """Generate a prompt with exactly *target_tokens* tokens.

    Repeats a deterministic seed text and tokenizes/truncates to the exact
    count, then decodes back to a string. This guarantees reproducible
    prompt lengths across runs.

    Args:
        model: A ``llama_cpp.Llama`` instance for tokenization.
        target_tokens: Desired number of tokens.

    Returns:
        A string that tokenizes to exactly *target_tokens* tokens.
    """
    # Build a long-enough raw string by repeating the seed
    repeats = (target_tokens // 50) + 2  # ~50 tokens per repetition
    raw_text = (_SEED_TEXT * repeats)

    # Tokenize, truncate, decode
    tokens = model.tokenize(raw_text.encode(), add_bos=True)
    tokens = tokens[:target_tokens]

    # Decode back to string — this guarantees exact token count
    text = model.detokenize(tokens).decode("utf-8", errors="replace")
    return text


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------
def _collect_hardware_info() -> dict[str, Any]:
    """Collect GPU and RAM information for the report."""
    gpu_mem = get_gpu_memory()
    ram = get_ram_usage()

    hw: dict[str, Any] = {
        "ram_gb": round(ram.total_gb, 1),
    }

    if gpu_mem is not None:
        hw["vram_gb"] = round(gpu_mem.total_gb, 1)

    # Try to get GPU name
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        hw["gpu"] = pynvml.nvmlDeviceGetName(handle)
        if isinstance(hw["gpu"], bytes):
            hw["gpu"] = hw["gpu"].decode()
        pynvml.nvmlShutdown()
    except Exception:
        hw["gpu"] = "unknown"

    return hw


def _snapshot_peak_vram() -> float:
    """Return current GPU memory usage in GB, or 0.0 if unavailable."""
    mem = get_gpu_memory()
    return round(mem.used_gb, 2) if mem is not None else 0.0


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
def _create_llama_model(
    model_config: ModelConfig,
    kv_config: KVCacheConfig,
) -> Any:
    """Create a ``llama_cpp.Llama`` instance for speed benchmarking.

    Args:
        model_config: Model path, context size, GPU layer configuration.
        kv_config: KV cache quantization parameters.

    Returns:
        A ``llama_cpp.Llama`` instance.

    Raises:
        ImportError: If ``llama-cpp-python`` is not installed.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is required for this benchmark. "
            "Install with: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
        )

    kv_params = to_llama_params(kv_config)

    logger.info(
        "Loading model %s (n_ctx=%d, K=%s, V=%s)",
        model_config.model_name,
        model_config.n_ctx,
        kv_config.cache_type_k.value,
        kv_config.cache_type_v.value,
    )

    start = time.monotonic()
    model = Llama(
        model_path=model_config.model_path,
        n_ctx=model_config.n_ctx,
        n_gpu_layers=model_config.n_gpu_layers,
        chat_format=model_config.chat_format,
        verbose=False,
        **kv_params,
    )
    logger.info("Model loaded in %.1fs", time.monotonic() - start)
    return model


def _unload_model(model: Any) -> None:
    """Release model memory and trigger garbage collection."""
    del model
    gc.collect()
    logger.info("Model unloaded")


# ---------------------------------------------------------------------------
# Core timing
# ---------------------------------------------------------------------------
def _run_single_timing(
    model: Any,
    prompt_text: str,
    max_gen_tokens: int,
) -> SingleRunTiming:
    """Execute one timed prefill + decode pass and return raw timings.

    Prefill is measured by calling ``model.eval()`` on the tokenized prompt.
    Decode is measured by generating ``max_gen_tokens`` via ``model()`` with
    the prompt already in the KV cache (we use the full ``model()`` call
    which handles both, and separate the timings using the returned usage).

    For precise prefill/decode separation we:
    1. Reset the model's KV cache.
    2. Tokenize the prompt.
    3. Time ``model.eval(tokens)`` for prefill.
    4. Time token-by-token generation for decode.

    Args:
        model: A ``llama_cpp.Llama`` instance.
        prompt_text: The input prompt string.
        max_gen_tokens: Number of tokens to generate after prefill.

    Returns:
        A :class:`SingleRunTiming` with nanosecond-precision timings.
    """
    tokens = model.tokenize(prompt_text.encode(), add_bos=False)
    n_prompt_tokens = len(tokens)

    # Reset KV cache for a clean measurement
    model.reset()

    # --- Prefill (prompt processing) ---
    prefill_start = time.perf_counter_ns()
    model.eval(tokens)
    prefill_end = time.perf_counter_ns()
    prefill_ns = prefill_end - prefill_start

    # TTFT = time from prompt submission to first generated token
    # After prefill, sample the first token
    ttft_start = prefill_start  # TTFT includes prefill

    # --- Decode (token generation) ---
    decode_start = time.perf_counter_ns()
    generated_count = 0

    for _i in range(max_gen_tokens):
        # Sample next token from the model
        token = model.sample()
        model.eval([token])
        generated_count += 1

        # Record TTFT after first token is sampled
        if generated_count == 1:
            ttft_ns = time.perf_counter_ns() - ttft_start

        # Check for EOS
        if token == model.token_eos():
            break

    decode_end = time.perf_counter_ns()
    decode_ns = decode_end - decode_start

    # If we never generated a token, TTFT is just prefill time
    if generated_count == 0:
        ttft_ns = prefill_ns

    total_ns = prefill_ns + decode_ns
    peak_vram = _snapshot_peak_vram()

    return SingleRunTiming(
        prefill_time_ns=prefill_ns,
        prefill_tokens=n_prompt_tokens,
        decode_time_ns=decode_ns,
        decode_tokens=generated_count,
        ttft_ns=ttft_ns,
        total_time_ns=total_ns,
        peak_vram_gb=peak_vram,
    )


def _aggregate_runs(runs: list[SingleRunTiming]) -> dict[str, Any]:
    """Compute mean ± std from a list of single-run timings.

    Args:
        runs: List of raw timing results from measurement runs.

    Returns:
        Dict with StatSummary objects for each metric and peak VRAM.
    """
    prefill_tps = np.array([
        r.prefill_tokens / (r.prefill_time_ns / _NS_PER_S)
        for r in runs
        if r.prefill_time_ns > 0
    ])
    decode_tps = np.array([
        r.decode_tokens / (r.decode_time_ns / _NS_PER_S)
        for r in runs
        if r.decode_time_ns > 0 and r.decode_tokens > 0
    ])
    ttft_ms = np.array([r.ttft_ns / _NS_PER_MS for r in runs])
    total_s = np.array([r.total_time_ns / _NS_PER_S for r in runs])
    peak_vram = max(r.peak_vram_gb for r in runs)

    return {
        "prefill_tokens_per_sec": StatSummary(
            mean=round(float(np.mean(prefill_tps)), 1) if len(prefill_tps) else 0.0,
            std=round(float(np.std(prefill_tps)), 1) if len(prefill_tps) else 0.0,
        ),
        "decode_tokens_per_sec": StatSummary(
            mean=round(float(np.mean(decode_tps)), 1) if len(decode_tps) else 0.0,
            std=round(float(np.std(decode_tps)), 1) if len(decode_tps) else 0.0,
        ),
        "ttft_ms": StatSummary(
            mean=round(float(np.mean(ttft_ms)), 1),
            std=round(float(np.std(ttft_ms)), 1),
        ),
        "total_time_s": StatSummary(
            mean=round(float(np.mean(total_s)), 2),
            std=round(float(np.std(total_s)), 2),
        ),
        "peak_vram_gb": peak_vram,
    }


# ---------------------------------------------------------------------------
# Single-config evaluation
# ---------------------------------------------------------------------------
def benchmark_config_at_context(
    model: Any,
    config_name: str,
    kv_config: KVCacheConfig,
    context_length: int,
    max_gen_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    arch: dict[str, int],
) -> SpeedResult:
    """Run warmup + measurement for one (config, context_length) pair.

    The model must already be loaded. Prompt is generated to fill exactly
    *context_length* tokens.

    Args:
        model: A loaded ``llama_cpp.Llama`` instance.
        config_name: Human-readable label (e.g. ``"q8_0/turbo4"``).
        kv_config: KV cache quantization configuration.
        context_length: Number of prompt tokens to use.
        max_gen_tokens: Tokens to generate per run.
        warmup_runs: Number of warmup iterations (results discarded).
        measure_runs: Number of measurement iterations.
        arch: Model architecture dict for KV memory estimation.

    Returns:
        A frozen :class:`SpeedResult`.
    """
    prompt_text = generate_prompt(model, context_length)

    # Verify prompt length
    actual_tokens = len(model.tokenize(prompt_text.encode(), add_bos=False))
    logger.info(
        "  ctx=%d: prompt has %d tokens, generating up to %d tokens",
        context_length,
        actual_tokens,
        max_gen_tokens,
    )

    # --- Warmup ---
    for i in range(warmup_runs):
        logger.debug("  warmup run %d/%d", i + 1, warmup_runs)
        _run_single_timing(model, prompt_text, max_gen_tokens)

    # --- Measurement ---
    runs: list[SingleRunTiming] = []
    for i in range(measure_runs):
        logger.debug("  measure run %d/%d", i + 1, measure_runs)
        timing = _run_single_timing(model, prompt_text, max_gen_tokens)
        runs.append(timing)
        logger.info(
            "    run %d: prefill=%.0f tok/s, decode=%.1f tok/s, ttft=%.1fms",
            i + 1,
            timing.prefill_tokens / (timing.prefill_time_ns / _NS_PER_S)
            if timing.prefill_time_ns > 0
            else 0.0,
            timing.decode_tokens / (timing.decode_time_ns / _NS_PER_S)
            if timing.decode_time_ns > 0 and timing.decode_tokens > 0
            else 0.0,
            timing.ttft_ns / _NS_PER_MS,
        )

    agg = _aggregate_runs(runs)

    # KV memory estimate
    mem_est = estimate_kv_memory_bytes(
        n_ctx=context_length,
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        head_dim=arch["head_dim"],
        config=kv_config,
    )

    return SpeedResult(
        config_name=config_name,
        cache_type_k=kv_config.cache_type_k.value,
        cache_type_v=kv_config.cache_type_v.value,
        context_length=context_length,
        prefill_tokens_per_sec=agg["prefill_tokens_per_sec"],
        decode_tokens_per_sec=agg["decode_tokens_per_sec"],
        ttft_ms=agg["ttft_ms"],
        total_time_s=agg["total_time_s"],
        peak_vram_gb=agg["peak_vram_gb"],
        kv_memory_mb=round(mem_est["total_mb"], 1),
    )


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    model_config: ModelConfig,
    configs: dict[str, KVCacheConfig],
    context_lengths: list[int],
    max_gen_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    arch_name: str = DEFAULT_ARCH,
) -> list[SpeedResult]:
    """Run the speed benchmark across all configs and context lengths.

    For each config, the model is loaded once, benchmarked across all
    context lengths, then unloaded. This avoids repeated load/unload
    overhead while ensuring each config gets a clean memory state.

    Args:
        model_config: Model path and context configuration.
        configs: Mapping of config_name → KVCacheConfig.
        context_lengths: List of prompt token counts to test.
        max_gen_tokens: Tokens to generate per run.
        warmup_runs: Warmup iterations per (config, ctx) pair.
        measure_runs: Measurement iterations per (config, ctx) pair.
        arch_name: Key into MODEL_ARCHITECTURES for KV estimation.

    Returns:
        List of :class:`SpeedResult`, one per (config, context_length) pair.

    Raises:
        ValueError: If *arch_name* is not in ``MODEL_ARCHITECTURES``.
    """
    arch = MODEL_ARCHITECTURES.get(arch_name)
    if arch is None:
        available = ", ".join(sorted(MODEL_ARCHITECTURES))
        raise ValueError(
            f"Unknown architecture '{arch_name}'. Available: {available}"
        )

    # Filter context lengths that exceed the model's n_ctx
    valid_ctx = [c for c in context_lengths if c <= model_config.n_ctx]
    if len(valid_ctx) < len(context_lengths):
        skipped = [c for c in context_lengths if c > model_config.n_ctx]
        logger.warning(
            "Skipping context lengths %s (exceeds n_ctx=%d)",
            skipped,
            model_config.n_ctx,
        )

    results: list[SpeedResult] = []

    for cfg_idx, (name, kv_cfg) in enumerate(configs.items(), 1):
        logger.info(
            "=== Config %d/%d: %s ===", cfg_idx, len(configs), name,
        )

        # Create model config with max context from the list
        cfg_model_config = ModelConfig(
            model_path=model_config.model_path,
            model_name=model_config.model_name,
            n_ctx=model_config.n_ctx,
            n_gpu_layers=model_config.n_gpu_layers,
            chat_format=model_config.chat_format,
            weight_size_gb=model_config.weight_size_gb,
        )

        try:
            model = _create_llama_model(cfg_model_config, kv_cfg)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping config %s: %s", name, exc)
            continue
        try:
            for ctx_idx, ctx_len in enumerate(valid_ctx, 1):
                logger.info(
                    "--- Context %d/%d: %d tokens ---",
                    ctx_idx,
                    len(valid_ctx),
                    ctx_len,
                )
                result = benchmark_config_at_context(
                    model=model,
                    config_name=name,
                    kv_config=kv_cfg,
                    context_length=ctx_len,
                    max_gen_tokens=max_gen_tokens,
                    warmup_runs=warmup_runs,
                    measure_runs=measure_runs,
                    arch=arch,
                )
                results.append(result)
        finally:
            _unload_model(model)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _stat_str(s: StatSummary) -> str:
    """Format a StatSummary as 'mean ± std'."""
    if s.mean >= 100:
        return f"{s.mean:.0f} ± {s.std:.0f}"
    if s.mean >= 10:
        return f"{s.mean:.1f} ± {s.std:.1f}"
    return f"{s.mean:.2f} ± {s.std:.2f}"


def print_results_table(results: list[SpeedResult]) -> None:
    """Print a human-readable results table to stdout."""
    from tabulate import tabulate

    headers = [
        "Config",
        "Ctx Len",
        "Prefill (tok/s)",
        "Decode (tok/s)",
        "TTFT (ms)",
        "Total (s)",
        "VRAM (GB)",
        "KV (MB)",
    ]
    rows = [
        [
            r.config_name,
            f"{r.context_length:,}",
            _stat_str(r.prefill_tokens_per_sec),
            _stat_str(r.decode_tokens_per_sec),
            _stat_str(r.ttft_ms),
            _stat_str(r.total_time_s),
            f"{r.peak_vram_gb:.2f}",
            f"{r.kv_memory_mb:.1f}",
        ]
        for r in results
    ]

    print("\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n")


def _stat_to_dict(s: StatSummary) -> dict[str, float]:
    """Serialize a StatSummary for JSON output."""
    return {"mean": s.mean, "std": s.std}


def save_results(
    results: list[SpeedResult],
    model_config: ModelConfig,
    settings: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Persist benchmark results as JSON.

    Args:
        results: Speed benchmark results.
        model_config: The model configuration used.
        settings: Benchmark settings (warmup_runs, measure_runs, etc.).
        output_dir: Target directory (created if absent).

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    hw = _collect_hardware_info()

    report = {
        "model": model_config.model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "settings": settings,
        "results": [
            {
                "config_name": r.config_name,
                "cache_type_k": r.cache_type_k,
                "cache_type_v": r.cache_type_v,
                "context_length": r.context_length,
                "prefill_tokens_per_sec": _stat_to_dict(r.prefill_tokens_per_sec),
                "decode_tokens_per_sec": _stat_to_dict(r.decode_tokens_per_sec),
                "ttft_ms": _stat_to_dict(r.ttft_ms),
                "total_time_s": _stat_to_dict(r.total_time_s),
                "peak_vram_gb": r.peak_vram_gb,
                "kv_memory_mb": r.kv_memory_mb,
            }
            for r in results
        ],
    }

    out_path = output_dir / "speed_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Results saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_configs(
    selected: list[str] | None,
) -> dict[str, KVCacheConfig]:
    """Validate and return the requested config names.

    When ``f16/f16`` is in the selection it is moved to the front as the
    baseline reference.
    """
    if selected is None:
        return dict(BENCHMARK_CONFIGS)

    resolved: dict[str, KVCacheConfig] = {}
    for name in selected:
        if name not in BENCHMARK_CONFIGS:
            available = ", ".join(BENCHMARK_CONFIGS)
            raise SystemExit(
                f"Error: unknown config '{name}'. Available: {available}"
            )
        resolved[name] = BENCHMARK_CONFIGS[name]

    # Ensure baseline comes first when present
    if "f16/f16" in resolved:
        baseline = resolved.pop("f16/f16")
        resolved = {"f16/f16": baseline, **resolved}

    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Speed/throughput benchmark for KV cache quantization configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf --configs f16/f16 q8_0/turbo4
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf --n-ctx 4096 --max-gen-tokens 64
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf --warmup-runs 1 --measure-runs 3
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--model-name",
        default="qwen2.5-7b-instruct",
        help="Model name label for reporting (default: %(default)s)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Maximum context window size in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Layers to offload to GPU; -1 = all (default: %(default)s)",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=None,
        metavar="LEN",
        help=(
            "Context lengths to benchmark (default: 512 2048 4096 8192). "
            "Values exceeding --n-ctx are skipped."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="CFG",
        help=(
            "KV cache configs to benchmark (default: all). "
            f"Choices: {', '.join(BENCHMARK_CONFIGS)}"
        ),
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        choices=sorted(MODEL_ARCHITECTURES),
        help="Model architecture for memory estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for the output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Warmup iterations per (config, ctx) pair (default: %(default)s)",
    )
    parser.add_argument(
        "--measure-runs",
        type=int,
        default=5,
        help="Measurement iterations per (config, ctx) pair (default: %(default)s)",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=128,
        help="Tokens to generate per run (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the speed benchmark."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        raise SystemExit(1)

    model_config = ModelConfig(
        model_path=str(model_path),
        model_name=args.model_name,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    configs = _resolve_configs(args.configs)
    context_lengths = args.context_lengths or DEFAULT_CONTEXT_LENGTHS

    settings = {
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "max_gen_tokens": args.max_gen_tokens,
        "context_lengths": context_lengths,
    }

    logger.info(
        "Starting speed benchmark: model=%s, n_ctx=%d, configs=%s, "
        "ctx_lengths=%s, warmup=%d, measure=%d, gen_tokens=%d",
        model_config.model_name,
        model_config.n_ctx,
        list(configs),
        context_lengths,
        args.warmup_runs,
        args.measure_runs,
        args.max_gen_tokens,
    )

    results = run_benchmark(
        model_config=model_config,
        configs=configs,
        context_lengths=context_lengths,
        max_gen_tokens=args.max_gen_tokens,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        arch_name=args.arch,
    )

    print_results_table(results)

    out_path = save_results(results, model_config, settings, args.output_dir)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
