#!/usr/bin/env python3
"""Multi-turn conversation benchmark: Standard vs TurboQuant.

Simulates a realistic multi-turn conversation where KV cache compression
matters most — each subsequent turn benefits from compressed prior context.

Usage:
    python -m benchmarks.benchmark_multiturn
    python -m benchmarks.benchmark_multiturn --n-gpu-layers -1 --turns 5
"""

from __future__ import annotations

import argparse
import gc
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
    return info.used / (1024 * 1024) if info else None


def _detect_gpu_name() -> str:
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
    parser = argparse.ArgumentParser(description="Multi-turn benchmark")
    parser.add_argument(
        "--model-path", type=str,
        default="models/qwen2.5-7b-instruct-q4_k_m.gguf",
    )
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--turns", type=int, default=5)
    return parser.parse_args()


# Multi-turn conversation — each turn builds on prior context
CONVERSATION = [
    "What is a binary search tree? Explain briefly.",
    "What is the time complexity of search, insert, and delete operations?",
    "How does a self-balancing BST (like AVL or Red-Black tree) differ?",
    "Write a Python class for a simple BST with insert and search methods.",
    "Now add a delete method to the BST class you wrote.",
    "What are real-world applications where BSTs are preferred over hash tables?",
    "Summarize everything we discussed about BSTs in 3 bullet points.",
]


def run_standard_multiturn(model_config, n_turns, max_tokens):
    """Standard inference multi-turn conversation."""
    kv_config = KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.Q8_0,
        flash_attention=True,
    )
    engine = InferenceEngine(model_config, kv_config)
    engine.load_model()

    messages = []
    turn_results = []

    for i in range(min(n_turns, len(CONVERSATION))):
        user_msg = CONVERSATION[i]
        messages.append({"role": "user", "content": user_msg})

        ram_before = get_ram_mb()
        gpu_before = get_gpu_mb()
        t0 = time.monotonic()

        response, stats = engine.chat(
            messages, max_tokens=max_tokens, temperature=0.0,
        )
        elapsed = time.monotonic() - t0

        ram_after = get_ram_mb()
        gpu_after = get_gpu_mb()

        messages.append({"role": "assistant", "content": response["content"]})

        turn_results.append({
            "turn": i + 1,
            "prompt": user_msg[:60],
            "response": response["content"][:80],
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "total_tokens": stats.total_tokens,
            "tokens_per_second": round(stats.tokens_per_second, 2),
            "elapsed_s": round(elapsed, 3),
            "ram_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
            "gpu_mb": round(gpu_after, 1) if gpu_after else None,
            "gpu_delta_mb": round(gpu_after - gpu_before, 1) if gpu_before and gpu_after else None,
        })

    engine.unload()
    gc.collect()
    return turn_results


def run_turbo_multiturn(model_config, n_turns, max_tokens, quant_config):
    """TurboQuant inference multi-turn conversation."""
    turbo = TurboQuantEngine(
        model_config, quant_config,
        n_layers=28, n_heads=28, head_dim=128,
    )
    turbo.load_model()

    messages = []
    turn_results = []

    for i in range(min(n_turns, len(CONVERSATION))):
        user_msg = CONVERSATION[i]
        messages.append({"role": "user", "content": user_msg})

        ram_before = get_ram_mb()
        gpu_before = get_gpu_mb()
        t0 = time.monotonic()

        result = turbo.chat_with_compression(
            messages, max_tokens=max_tokens, temperature=0.0,
        )
        elapsed = time.monotonic() - t0

        ram_after = get_ram_mb()
        gpu_after = get_gpu_mb()

        messages.append({"role": "assistant", "content": result.text})

        comp = result.compression_stats
        turn_results.append({
            "turn": i + 1,
            "prompt": user_msg[:60],
            "response": result.text[:80],
            "prompt_tokens": result.gen_stats.prompt_tokens,
            "completion_tokens": result.gen_stats.completion_tokens,
            "total_tokens": result.gen_stats.total_tokens,
            "tokens_per_second": round(result.gen_stats.tokens_per_second, 2),
            "elapsed_s": round(elapsed, 3),
            "ram_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
            "gpu_mb": round(gpu_after, 1) if gpu_after else None,
            "gpu_delta_mb": round(gpu_after - gpu_before, 1) if gpu_before and gpu_after else None,
            "compression_ratio": round(comp.compression_ratio, 2) if comp else None,
            "mse": round(comp.mse, 6) if comp else None,
            "compress_ms": round(comp.compress_time_s * 1000, 2) if comp else None,
            "decompress_ms": round(comp.decompress_time_s * 1000, 2) if comp else None,
        })

    turbo.unload()
    gc.collect()
    return turn_results


def print_multiturn_results(std_turns, turbo_turns, turbo_label, n_gpu_layers, n_ctx):
    """Print formatted multi-turn comparison."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("  MULTI-TURN CONVERSATION BENCHMARK")
    print(sep)

    gpu_info = get_gpu_memory()
    gpu_name = _detect_gpu_name() if gpu_info else "N/A"
    gpu_total = f"{gpu_info.total / (1024**2):.0f} MiB" if gpu_info else "N/A"

    print(f"\n  GPU: {gpu_name} ({gpu_total})")
    print(f"  GPU Layers: {'ALL' if n_gpu_layers < 0 else n_gpu_layers}")
    print(f"  Context: {n_ctx} tokens")
    print(f"  Turns: {len(std_turns)}")

    # Per-turn table
    print(f"\n{'─' * 90}")
    print(f"  {'Turn':<5} {'Tokens':<10} {'Standard tok/s':<18} {turbo_label + ' tok/s':<22} {'Δ%':<8} {'Comp. Ratio':<14} {'MSE':<12}")
    print(f"{'─' * 90}")

    for s, t in zip(std_turns, turbo_turns):
        delta = ((t["tokens_per_second"] / max(s["tokens_per_second"], 0.01)) - 1) * 100
        cr = f"{t.get('compression_ratio', 'N/A')}x" if t.get("compression_ratio") else "N/A"
        mse = f"{t.get('mse', 'N/A')}" if t.get("mse") is not None else "N/A"
        print(
            f"  {s['turn']:<5} {s['total_tokens']:<10} "
            f"{s['tokens_per_second']:<18} {t['tokens_per_second']:<22} "
            f"{delta:+.1f}%{'':<3} {cr:<14} {mse:<12}"
        )

    # Aggregates
    std_avg_speed = np.mean([t["tokens_per_second"] for t in std_turns])
    turbo_avg_speed = np.mean([t["tokens_per_second"] for t in turbo_turns])
    avg_delta = ((turbo_avg_speed / std_avg_speed) - 1) * 100
    ratios = [t["compression_ratio"] for t in turbo_turns if t.get("compression_ratio")]
    mses = [t["mse"] for t in turbo_turns if t.get("mse") is not None]

    print(f"{'─' * 90}")
    print(f"  {'AVG':<5} {'':10} {std_avg_speed:<18.2f} {turbo_avg_speed:<22.2f} {avg_delta:+.1f}%{'':<3} ", end="")
    print(f"{np.mean(ratios):.2f}x{'':<10} {np.mean(mses):.6f}" if ratios else "")

    # Context growth
    print(f"\n  CONTEXT GROWTH (prompt tokens per turn):")
    for s in std_turns:
        bar = "█" * (s["prompt_tokens"] // 20)
        print(f"    Turn {s['turn']}: {s['prompt_tokens']:>5} tokens  {bar}")

    # RAM comparison
    print(f"\n  RAM USAGE (MB):")
    print(f"    {'Turn':<6} {'Standard':<12} {turbo_label:<18}")
    for s, t in zip(std_turns, turbo_turns):
        print(f"    {s['turn']:<6} {s['ram_mb']:<12} {t['ram_mb']:<18}")

    if std_turns[0].get("gpu_mb") is not None:
        print(f"\n  GPU VRAM (MB):")
        print(f"    {'Turn':<6} {'Standard':<12} {turbo_label:<18}")
        for s, t in zip(std_turns, turbo_turns):
            print(f"    {s['turn']:<6} {s.get('gpu_mb', 'N/A'):<12} {t.get('gpu_mb', 'N/A'):<18}")

    # Sample conversation
    print(f"\n{'─' * 90}")
    print("  CONVERSATION SAMPLE (first & last turn)")
    print(f"{'─' * 90}")
    for idx in [0, len(std_turns) - 1]:
        s, t = std_turns[idx], turbo_turns[idx]
        print(f"\n  [Turn {s['turn']}] User: {s['prompt']}")
        print(f"    Standard:  {s['response']}")
        print(f"    {turbo_label}: {t['response']}")

    print(f"\n{sep}\n")


def main():
    args = parse_args()

    model_config = ModelConfig(
        model_path=args.model_path,
        model_name="qwen2.5-7b-instruct",
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        chat_format="chatml",
    )

    print(f"Running multi-turn Standard benchmark ({args.turns} turns)...")
    std_turns = run_standard_multiturn(model_config, args.turns, args.max_tokens)

    # Run TurboQuant K8/V4 (best quality/speed tradeoff)
    qcfg = QuantConfig(k_bits=8, v_bits=4, block_size=128)
    label = "TurboQuant K8/V4"
    print(f"\nRunning multi-turn {label} benchmark ({args.turns} turns)...")
    turbo_turns = run_turbo_multiturn(model_config, args.turns, args.max_tokens, qcfg)

    print_multiturn_results(
        std_turns, turbo_turns, label, args.n_gpu_layers, args.n_ctx,
    )


if __name__ == "__main__":
    main()
