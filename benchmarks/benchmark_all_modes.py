#!/usr/bin/env python3
"""Three-mode benchmark: Standard vs TurboQuant vs Zero-Quant.

Runs identical prompts through all three inference modes on the same model
and reports: token speed, KV compression ratio, MSE, compress/decompress
latency, and avg bits per value.

Usage:
    python -m benchmarks.benchmark_all_modes
    python -m benchmarks.benchmark_all_modes \\
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --max-tokens 64 --runs 2
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
from src.engine.zero_quant_engine import ZeroQuantEngine
from src.turboquant.compressor import QuantConfig
from src.turboquant.zero_quant import ZeroQuantConfig
from src.utils.memory import get_gpu_memory, get_ram_usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ram_mb() -> float:
    return get_ram_usage().used / (1024 * 1024)


def _gpu_mb() -> float | None:
    info = get_gpu_memory()
    return info.used / (1024 * 1024) if info else None


def _gpu_name() -> str:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        pynvml.nvmlShutdown()
        return name if isinstance(name, str) else name.decode()
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    [{"role": "user", "content": "What is 195 + 39?"}],
    [{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
    [
        {
            "role": "user",
            "content": "Write a Python function to check if a number is prime.",
        }
    ],
]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_standard(model_config: ModelConfig, max_tokens: int, runs: int, flash_attention: bool = True) -> dict:
    # Q8_0 KV requires flash_attn; fall back to F16 when flash_attn is disabled
    if flash_attention:
        kv_type = CacheType.Q8_0
        kv_label = "q8_0"
    else:
        kv_type = CacheType.F16
        kv_label = "f16"
    kv_config = KVCacheConfig(
        cache_type_k=kv_type,
        cache_type_v=kv_type,
        flash_attention=flash_attention,
    )
    engine = InferenceEngine(model_config, kv_config)

    r_before, g_before = _ram_mb(), _gpu_mb()
    engine.load_model()
    r_after, g_after = _ram_mb(), _gpu_mb()

    result = _collect_runs(engine, "direct", max_tokens, runs)
    engine.unload()
    gc.collect()

    result.update(
        {
            "label": f"Standard ({kv_label}/{kv_label})",
            "kv_k": kv_label,
            "kv_v": kv_label,
            "avg_bits": 8.0,
            "compression_ratio": 1.0,
            "mse": 0.0,
            "avg_compress_ms": 0.0,
            "avg_decompress_ms": 0.0,
            "model_ram_mb": round(r_after - r_before, 1),
            "gpu_mb": (
                round(g_after - g_before, 1)
                if g_before is not None and g_after is not None
                else None
            ),
        }
    )
    return result


def run_turboquant(
    model_config: ModelConfig,
    max_tokens: int,
    runs: int,
    k_bits: int = 8,
    v_bits: int = 4,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
    flash_attention: bool = True,
) -> dict:
    q_cfg = QuantConfig(k_bits=k_bits, v_bits=v_bits, block_size=128)
    engine = TurboQuantEngine(
        model_config, q_cfg, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
        flash_attention=flash_attention,
    )

    r_before, g_before = _ram_mb(), _gpu_mb()
    engine.load_model()
    r_after, g_after = _ram_mb(), _gpu_mb()

    result = _collect_runs(engine, "turbo", max_tokens, runs)
    engine.unload()
    gc.collect()

    result.update(
        {
            "label": f"TurboQuant K{k_bits}/V{v_bits}",
            "kv_k": f"q8_0+turbo{k_bits}",
            "kv_v": f"q8_0+turbo{v_bits}",
            "avg_bits": (k_bits + v_bits) / 2,
            "model_ram_mb": round(r_after - r_before, 1),
            "gpu_mb": (
                round(g_after - g_before, 1)
                if g_before is not None and g_after is not None
                else None
            ),
        }
    )
    return result


def run_zero_quant(
    model_config: ModelConfig,
    max_tokens: int,
    runs: int,
    zq_cfg: ZeroQuantConfig | None = None,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
    flash_attention: bool = True,
) -> dict:
    cfg = zq_cfg or ZeroQuantConfig()
    engine = ZeroQuantEngine(model_config, cfg, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
                             flash_attention=flash_attention)

    r_before, g_before = _ram_mb(), _gpu_mb()
    engine.load_model()
    r_after, g_after = _ram_mb(), _gpu_mb()

    result = _collect_runs(engine, "zero", max_tokens, runs)
    engine.unload()
    gc.collect()

    avg_bits = cfg.average_bits(n_layers)
    result.update(
        {
            "label": f"Zero-Quant (avg {avg_bits:.1f} bits)",
            "kv_k": f"q8_0+zq-adaptive",
            "kv_v": f"q8_0+zq-adaptive",
            "avg_bits": avg_bits,
            "model_ram_mb": round(r_after - r_before, 1),
            "gpu_mb": (
                round(g_after - g_before, 1)
                if g_before is not None and g_after is not None
                else None
            ),
        }
    )
    return result


def _collect_runs(engine, mode: str, max_tokens: int, runs: int) -> dict:
    """Run benchmark prompts and collect per-run stats."""
    tps_list: list[float] = []
    compression_ratios: list[float] = []
    mses: list[float] = []
    compress_ms_list: list[float] = []
    decompress_ms_list: list[float] = []
    sample_responses: list[str] = []

    for run_i in range(runs):
        for prompt in PROMPTS:
            t0 = time.monotonic()

            if mode == "direct":
                msg, stats = engine.chat(prompt, max_tokens=max_tokens, temperature=0.0)
                response_text = msg["content"]
                comp_stats = None
            elif mode == "turbo":
                res = engine.chat_with_compression(
                    prompt, max_tokens=max_tokens, temperature=0.0
                )
                response_text = res.text
                stats = res.gen_stats
                comp_stats = res.compression_stats
            else:  # zero
                res = engine.chat_with_compression(
                    prompt, max_tokens=max_tokens, temperature=0.0
                )
                response_text = res.text
                stats = res.gen_stats
                comp_stats = res.compression_stats

            elapsed = time.monotonic() - t0
            tps_list.append(stats.tokens_per_second)

            if comp_stats:
                compression_ratios.append(comp_stats.compression_ratio)
                mses.append(comp_stats.mse)
                compress_ms_list.append(comp_stats.compress_time_s * 1000)
                decompress_ms_list.append(comp_stats.decompress_time_s * 1000)

            if run_i == 0:
                sample_responses.append(response_text[:90])

    result: dict = {
        "avg_tps": round(float(np.mean(tps_list)), 2),
        "sample_responses": sample_responses,
    }
    if compression_ratios:
        result["compression_ratio"] = round(float(np.mean(compression_ratios)), 2)
        result["mse"] = round(float(np.mean(mses)), 6)
        result["avg_compress_ms"] = round(float(np.mean(compress_ms_list)), 2)
        result["avg_decompress_ms"] = round(float(np.mean(decompress_ms_list)), 2)
    return result


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def print_report(results: list[dict], model_path: str, n_ctx: int) -> None:
    sep = "=" * 84
    line = "-" * 84

    print(f"\n{sep}")
    print("  INFERENCE MODE COMPARISON — Standard / TurboQuant / Zero-Quant")
    print(sep)

    gpu_info = get_gpu_memory()
    gpu_name = _gpu_name()
    if gpu_info:
        gpu_total = gpu_info.total / (1024 * 1024)
        print(f"  GPU:          {gpu_name} ({gpu_total:.0f} MiB)")
    else:
        print("  GPU:          N/A (CPU-only)")

    print(f"  Model:        {Path(model_path).name}")
    print(f"  Context:      {n_ctx} tokens")
    print(f"  Prompts:      {len(PROMPTS)}")
    print(f"\n{line}")

    # Header row
    hdr_labels = [r["label"] for r in results]
    col_w = 22
    print(f"  {'Metric':<30}", end="")
    for lbl in hdr_labels:
        print(f"  {lbl[:col_w]:<{col_w}}", end="")
    print()
    print(f"  {line}")

    metrics = [
        ("Model RAM delta (MB)", "model_ram_mb", "{:.0f}"),
        ("GPU VRAM delta (MB)", "gpu_mb", "{:.0f}"),
        ("Avg token speed (tok/s)", "avg_tps", "{:.2f}"),
        ("Avg bits / value", "avg_bits", "{:.1f}"),
        ("Compression ratio", "compression_ratio", "{:.2f}x"),
        ("MSE (quality loss)", "mse", "{:.6f}"),
        ("Compress latency (ms)", "avg_compress_ms", "{:.1f}"),
        ("Decompress latency (ms)", "avg_decompress_ms", "{:.1f}"),
    ]

    for label, key, fmt in metrics:
        print(f"  {label:<30}", end="")
        for r in results:
            val = r.get(key)
            if val is None:
                cell = "N/A"
            elif key == "compression_ratio":
                cell = fmt.format(val)
            else:
                cell = fmt.format(val)
            print(f"  {cell:<{col_w}}", end="")
        print()

    # Speed overhead relative to standard
    std_tps = results[0]["avg_tps"]
    print(f"  {'Speed vs Standard':<30}", end="")
    for r in results:
        delta = ((r["avg_tps"] / std_tps) - 1.0) * 100
        cell = "baseline" if delta == 0 else f"{delta:+.1f}%"
        print(f"  {cell:<{col_w}}", end="")
    print()

    # KV memory savings relative to standard (avg_bits)
    std_bits = results[0]["avg_bits"]
    print(f"  {'KV memory vs Standard':<30}", end="")
    for r in results:
        saving = (1.0 - r["avg_bits"] / std_bits) * 100
        cell = "baseline" if saving == 0 else f"-{saving:.0f}%"
        print(f"  {cell:<{col_w}}", end="")
    print()

    print(f"\n  {line}")
    print("  SAMPLE RESPONSES (Prompt 1)")
    print(f"  {line}")
    prompt_text = PROMPTS[0][0]["content"]
    print(f"  Prompt: {prompt_text}")
    for r in results:
        resp = r["sample_responses"][0] if r.get("sample_responses") else "N/A"
        print(f"  [{r['label'][:20]}]: {resp}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standard vs TurboQuant vs Zero-Quant benchmark")
    p.add_argument(
        "--model-path",
        default="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        help="Path to GGUF model",
    )
    p.add_argument("--model-name", default="", help="Model name (auto-detected from path if empty)")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--n-layers", type=int, default=28, help="Transformer layer count (architecture)")
    p.add_argument("--n-heads", type=int, default=28, help="Attention head count (architecture)")
    p.add_argument("--head-dim", type=int, default=128, help="Attention head dimension (architecture)")
    p.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention (needed for some MoE models)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_name = args.model_name or Path(args.model_path).stem
    flash_attn = not args.no_flash_attn

    model_config = ModelConfig(
        model_path=args.model_path,
        model_name=model_name,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        chat_format="chatml",
    )

    zq_cfg = ZeroQuantConfig()  # uses defaults from ZeroQuantConfig (middle_v_bits=3)

    all_results = []

    print(f"\n[1/3] Standard inference...")
    all_results.append(run_standard(model_config, args.max_tokens, args.runs, flash_attention=flash_attn))

    print(f"[2/3] TurboQuant K8/V4 (quality preset)...")
    all_results.append(run_turboquant(
        model_config, args.max_tokens, args.runs,
        k_bits=8, v_bits=4,
        n_layers=args.n_layers, n_heads=args.n_heads, head_dim=args.head_dim,
        flash_attention=flash_attn,
    ))

    avg_bits = zq_cfg.average_bits(args.n_layers)
    print(f"[3/3] Zero-Quant depth-adaptive (avg {avg_bits:.1f} bits)...")
    all_results.append(run_zero_quant(
        model_config, args.max_tokens, args.runs, zq_cfg,
        n_layers=args.n_layers, n_heads=args.n_heads, head_dim=args.head_dim,
        flash_attention=flash_attn,
    ))

    print_report(all_results, args.model_path, args.n_ctx)


if __name__ == "__main__":
    main()
