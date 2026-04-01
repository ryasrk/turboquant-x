#!/usr/bin/env python3
"""Compare Standard Inference vs TurboQuant Inference.

Runs the same prompts through:
  1. Standard inference (Q8_0 KV cache — no TurboQuant compression)
  2. TurboQuant inference (Q8_0 base + Python-level TurboQuant compression)

Reports: model name, RAM usage, token speed, KV cache compression,
MSE quality, and timing breakdowns.

Usage:
    python -m benchmarks.benchmark_compare
    python -m benchmarks.benchmark_compare --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.inference import InferenceEngine
from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig
from src.engine.turbo_engine import TurboQuantEngine
from src.turboquant.compressor import QuantConfig
from src.utils.memory import get_gpu_memory, get_ram_usage


def get_ram_mb() -> float:
    info = get_ram_usage()
    return info.used / (1024 * 1024)


def get_gpu_mb() -> float | None:
    info = get_gpu_memory()
    if info is None:
        return None
    return info.used / (1024 * 1024)


def _detect_gpu_name() -> str:
    """Detect GPU name via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        pynvml.nvmlShutdown()
        return name if isinstance(name, str) else name.decode()
    except Exception:
        return "NVIDIA GPU"


def parse_args():
    parser = argparse.ArgumentParser(description="Standard vs TurboQuant benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        help="Path to GGUF model",
    )
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers to offload (-1=all, 0=CPU-only)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


# Test prompts for benchmarking
PROMPTS = [
    [{"role": "user", "content": "What is 195 + 39?"}],
    [{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
    [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
]


def run_standard_benchmark(model_config, prompts, max_tokens, runs):
    """Benchmark standard inference (no TurboQuant)."""
    kv_config = KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.Q8_0,
        flash_attention=True,
    )
    engine = InferenceEngine(model_config, kv_config)

    ram_before = get_ram_mb()
    gpu_before = get_gpu_mb()
    engine.load_model()
    ram_after = get_ram_mb()
    gpu_after = get_gpu_mb()

    results = {
        "mode": "Standard (Q8_0/Q8_0)",
        "model_name": model_config.model_name,
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": round(ram_after, 1),
        "model_ram_mb": round(ram_after - ram_before, 1),
        "gpu_before_mb": round(gpu_before, 1) if gpu_before is not None else None,
        "gpu_after_mb": round(gpu_after, 1) if gpu_after is not None else None,
        "gpu_used_mb": round(gpu_after - gpu_before, 1) if gpu_before is not None and gpu_after is not None else None,
        "kv_cache_k": "q8_0",
        "kv_cache_v": "q8_0",
        "compression_ratio": 1.0,
        "mse": 0.0,
        "compress_time_ms": 0.0,
        "decompress_time_ms": 0.0,
        "runs": [],
    }

    for run_idx in range(runs):
        run_stats = []
        for prompt in prompts:
            t0 = time.monotonic()
            msg, stats = engine.chat(
                prompt, max_tokens=max_tokens, temperature=0.0,
            )
            elapsed = time.monotonic() - t0

            run_stats.append({
                "prompt": prompt[0]["content"][:50],
                "response": msg["content"][:80],
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.total_tokens,
                "elapsed_s": round(elapsed, 3),
                "tokens_per_second": round(stats.tokens_per_second, 2),
            })
        results["runs"].append(run_stats)

    engine.unload()
    gc.collect()

    return results


def run_turbo_benchmark(model_config, prompts, max_tokens, runs, quant_config):
    """Benchmark TurboQuant inference."""
    turbo = TurboQuantEngine(
        model_config, quant_config,
        n_layers=28, n_heads=28, head_dim=128,
    )

    ram_before = get_ram_mb()
    gpu_before = get_gpu_mb()
    turbo.load_model()
    ram_after = get_ram_mb()
    gpu_after = get_gpu_mb()

    results = {
        "mode": f"TurboQuant (K{quant_config.k_bits}/V{quant_config.v_bits})",
        "model_name": model_config.model_name,
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": round(ram_after, 1),
        "model_ram_mb": round(ram_after - ram_before, 1),
        "gpu_before_mb": round(gpu_before, 1) if gpu_before is not None else None,
        "gpu_after_mb": round(gpu_after, 1) if gpu_after is not None else None,
        "gpu_used_mb": round(gpu_after - gpu_before, 1) if gpu_before is not None and gpu_after is not None else None,
        "kv_cache_k": f"q8_0 + turbo{quant_config.k_bits}",
        "kv_cache_v": f"q8_0 + turbo{quant_config.v_bits}",
        "compression_ratios": [],
        "mses": [],
        "compress_times_ms": [],
        "decompress_times_ms": [],
        "runs": [],
    }

    for run_idx in range(runs):
        run_stats = []
        for prompt in prompts:
            t0 = time.monotonic()
            result = turbo.chat_with_compression(
                prompt, max_tokens=max_tokens, temperature=0.0,
            )
            elapsed = time.monotonic() - t0

            stats = result.gen_stats
            comp = result.compression_stats

            entry = {
                "prompt": prompt[0]["content"][:50],
                "response": result.text[:80],
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.total_tokens,
                "elapsed_s": round(elapsed, 3),
                "tokens_per_second": round(stats.tokens_per_second, 2),
            }

            if comp:
                entry["compression_ratio"] = round(comp.compression_ratio, 2)
                entry["mse"] = round(comp.mse, 6)
                entry["compress_ms"] = round(comp.compress_time_s * 1000, 2)
                entry["decompress_ms"] = round(comp.decompress_time_s * 1000, 2)
                results["compression_ratios"].append(comp.compression_ratio)
                results["mses"].append(comp.mse)
                results["compress_times_ms"].append(comp.compress_time_s * 1000)
                results["decompress_times_ms"].append(comp.decompress_time_s * 1000)

            run_stats.append(entry)
        results["runs"].append(run_stats)

    turbo.unload()
    gc.collect()

    # Aggregate compression stats
    if results["compression_ratios"]:
        results["avg_compression_ratio"] = round(np.mean(results["compression_ratios"]), 2)
        results["avg_mse"] = round(np.mean(results["mses"]), 6)
        results["avg_compress_ms"] = round(np.mean(results["compress_times_ms"]), 2)
        results["avg_decompress_ms"] = round(np.mean(results["decompress_times_ms"]), 2)

    return results


def print_comparison(standard, turbo_configs):
    """Print formatted comparison table."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  STANDARD INFERENCE vs TURBOQUANT INFERENCE — BENCHMARK COMPARISON")
    print(sep)

    # GPU info
    gpu_info = get_gpu_memory()
    gpu_name = _detect_gpu_name()
    if gpu_info:
        gpu_total_mb = gpu_info.total / (1024 * 1024)
        gpu_str = f"{gpu_name} ({gpu_total_mb:.0f} MiB)"
    else:
        gpu_str = "N/A (CPU-only)"

    n_gpu_layers = standard.get('n_gpu_layers', 0)

    print(f"\n{'Model:':<25} {standard['model_name']}")
    print(f"{'GPU:':<25} {gpu_str}")
    print(f"{'GPU Layers Offloaded:':<25} {n_gpu_layers if n_gpu_layers >= 0 else 'ALL'}")
    print(f"{'Context Window:':<25} {standard.get('n_ctx', 'default')}")
    print(f"{'Max Gen Tokens:':<25} {standard['runs'][0][0]['completion_tokens']}")
    print(f"{'Benchmark Runs:':<25} {len(standard['runs'])}")

    # RAM comparison
    print(f"\n{'─' * 80}")
    print(f"  {'Metric':<30} {'Standard':<20} ", end="")
    for tc in turbo_configs:
        print(f"{tc['mode']:<25} ", end="")
    print()
    print(f"{'─' * 80}")

    print(f"  {'Model RAM (MB)':<30} {standard['model_ram_mb']:<20} ", end="")
    for tc in turbo_configs:
        print(f"{tc['model_ram_mb']:<25} ", end="")
    print()

    # GPU VRAM row
    std_gpu = standard.get('gpu_used_mb')
    print(f"  {'GPU VRAM Used (MB)':<30} {std_gpu if std_gpu is not None else 'N/A':<20} ", end="")
    for tc in turbo_configs:
        tg = tc.get('gpu_used_mb')
        print(f"{tg if tg is not None else 'N/A':<25} ", end="")
    print()

    print(f"  {'KV Cache K Type':<30} {standard['kv_cache_k']:<20} ", end="")
    for tc in turbo_configs:
        print(f"{tc['kv_cache_k']:<25} ", end="")
    print()

    print(f"  {'KV Cache V Type':<30} {standard['kv_cache_v']:<20} ", end="")
    for tc in turbo_configs:
        print(f"{tc['kv_cache_v']:<25} ", end="")
    print()

    # Speed comparison
    std_speeds = [
        entry["tokens_per_second"]
        for run in standard["runs"]
        for entry in run
    ]
    avg_std_speed = np.mean(std_speeds)
    print(f"  {'Avg Token Speed (tok/s)':<30} {avg_std_speed:<20.2f} ", end="")

    for tc in turbo_configs:
        t_speeds = [
            entry["tokens_per_second"]
            for run in tc["runs"]
            for entry in run
        ]
        avg_t_speed = np.mean(t_speeds)
        overhead = ((avg_t_speed / avg_std_speed) - 1) * 100
        print(f"{avg_t_speed:<12.2f} ({overhead:+.1f}%) ", end="")
    print()

    # Compression stats
    print(f"  {'Compression Ratio':<30} {'1.00x (baseline)':<20} ", end="")
    for tc in turbo_configs:
        ratio = tc.get("avg_compression_ratio", 1.0)
        print(f"{ratio:.2f}x{'':<19} ", end="")
    print()

    print(f"  {'MSE (quality loss)':<30} {'0.000000':<20} ", end="")
    for tc in turbo_configs:
        mse = tc.get("avg_mse", 0.0)
        print(f"{mse:<25.6f} ", end="")
    print()

    print(f"  {'Compress Time (ms)':<30} {'N/A':<20} ", end="")
    for tc in turbo_configs:
        ct = tc.get("avg_compress_ms", 0.0)
        print(f"{ct:<25.2f} ", end="")
    print()

    print(f"  {'Decompress Time (ms)':<30} {'N/A':<20} ", end="")
    for tc in turbo_configs:
        dt = tc.get("avg_decompress_ms", 0.0)
        print(f"{dt:<25.2f} ", end="")
    print()

    # Sample responses
    print(f"\n{'─' * 80}")
    print("  SAMPLE RESPONSES (Run 1)")
    print(f"{'─' * 80}")

    for i, prompt in enumerate(PROMPTS):
        p_text = prompt[0]["content"]
        print(f"\n  Prompt: {p_text}")
        print(f"  Standard: {standard['runs'][0][i]['response']}")
        for tc in turbo_configs:
            print(f"  {tc['mode']}: {tc['runs'][0][i]['response']}")

    print(f"\n{sep}")


def main():
    args = parse_args()

    model_config = ModelConfig(
        model_path=args.model_path,
        model_name="qwen2.5-7b-instruct",
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        chat_format="chatml",
    )

    print(f"Running Standard Inference benchmark (GPU layers: {args.n_gpu_layers})...")
    std_results = run_standard_benchmark(
        model_config, PROMPTS, args.max_tokens, args.runs,
    )
    std_results["n_ctx"] = args.n_ctx
    std_results["n_gpu_layers"] = args.n_gpu_layers

    turbo_configs_results = []

    for k_bits, v_bits, label in [
        (8, 4, "Quality"),
        (8, 2, "Aggressive"),
        (4, 4, "Symmetric"),
    ]:
        qcfg = QuantConfig(k_bits=k_bits, v_bits=v_bits, block_size=128)
        print(f"\nRunning TurboQuant K{k_bits}/V{v_bits} ({label}) benchmark...")
        turbo_results = run_turbo_benchmark(
            model_config, PROMPTS, args.max_tokens, args.runs, qcfg,
        )
        turbo_configs_results.append(turbo_results)

    print_comparison(std_results, turbo_configs_results)


if __name__ == "__main__":
    main()
