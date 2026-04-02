#!/usr/bin/env python3
"""KV cache compression quality benchmark.

Measures how faithfully TurboQuant preserves attention computation by
simulating realistic KV tensors and computing four fidelity metrics:

    Compression     — actual bytes ratio vs float16 baseline
    Cosine Sim      — cosine similarity between original and reconstructed
                      attention output vectors (1.0 = perfect)
    Top-1 Match     — fraction of queries where the highest-attention
                      token position is unchanged after compression (%)
    Top-5 Match     — fraction of queries where the original top-1
                      attention position still falls in top-5 after
                      compression (%)

No model loading required — synthetic KV tensors of realistic dimensions
are used so the benchmark completes in seconds.

Methodology
-----------
For each trial:
  1. Sample random K (n_layers, n_heads, seq_len, head_dim) and
     V (n_layers, n_heads, seq_len, head_dim) tensors.
  2. Compress with TurboQuant, then decompress.
  3. For 10 random query vectors per (layer, head):
       - Compute attention scores with original K  → softmax → attn_orig
       - Compute attention scores with decompressed K → softmax → attn_recon
       - Compute attention output: out = attn @ V  (original) and
                                   out_r = attn_r @ V_r (reconstructed)
       - Top-1 Match: argmax(attn_orig) == argmax(attn_recon)
       - Top-5 Match: argmax(attn_orig) in top-5 of attn_recon
       - Cosine Sim: cos(out, out_r)
  4. Average across all (trial, layer, head, query) combinations.

Usage
-----
    python -m benchmarks.benchmark_quality
    python -m benchmarks.benchmark_quality --n-layers 40 --seq-len 512
    python -m benchmarks.benchmark_quality --trials 100 --json results/quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.turboquant.compressor import QuantConfig, TurboQuantCompressor  # noqa: E402
from src.turboquant.zero_quant import ZeroQuantConfig, DepthAdaptiveCompressor  # noqa: E402

# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------
PRESETS: dict[str, QuantConfig] = {
    "Quality (K8/V4)":    QuantConfig(k_bits=8, v_bits=4),
    "Aggressive (K8/V2)": QuantConfig(k_bits=8, v_bits=2),
    "Symmetric (K4/V4)":  QuantConfig(k_bits=4, v_bits=4),
}

# Zero-Quant depth-adaptive presets
ZQ_PRESETS: dict[str, ZeroQuantConfig] = {
    "ZQ-Default (sh8/mi4-2/dp8)": ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=2,
        deep_k_bits=8, deep_v_bits=8,
    ),
    "ZQ-Balanced (sh8/mi4-4/dp8)": ZeroQuantConfig(
        shallow_k_bits=8, shallow_v_bits=8,
        middle_k_bits=4, middle_v_bits=4,
        deep_k_bits=8, deep_v_bits=8,
    ),
    "ZQ-Aggressive (sh4/mi4-2/dp4)": ZeroQuantConfig(
        shallow_k_bits=4, shallow_v_bits=4,
        middle_k_bits=4, middle_v_bits=2,
        deep_k_bits=4, deep_v_bits=4,
    ),
}

# Process layers one-at-a-time when tensor would exceed this threshold
# to avoid OOM on large seq_len (e.g. 8K context × 40 layers × 32 heads).
_LAYER_BY_LAYER_THRESHOLD_BYTES = 2 * 1024 ** 3  # 2 GB


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cosine(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def _attention_metrics(
    K_orig: NDArray[np.float64],   # (seq_len, head_dim)
    V_orig: NDArray[np.float64],   # (seq_len, head_dim)
    K_recon: NDArray[np.float64],  # (seq_len, head_dim)
    V_recon: NDArray[np.float64],  # (seq_len, head_dim)
    n_queries: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Compute (cosine_sim, attn_score_acc, top1_match, top5_match)."""
    head_dim = K_orig.shape[1]
    scale = 1.0 / (head_dim ** 0.5)

    cos_sims, attn_accs, top1_hits, top5_hits = [], [], [], []

    for _ in range(n_queries):
        q = rng.standard_normal(head_dim)

        # --- original attention ---
        attn_orig = _softmax(scale * (K_orig @ q))   # (seq_len,)
        out_orig  = attn_orig @ V_orig                # (head_dim,)
        top1_idx  = int(np.argmax(attn_orig))

        # --- reconstructed attention ---
        attn_recon = _softmax(scale * (K_recon @ q))
        out_recon  = attn_recon @ V_recon

        top5_idx = set(np.argsort(attn_recon)[-5:].tolist())

        cos_sims.append(_cosine(out_orig, out_recon))
        attn_accs.append(_cosine(attn_orig, attn_recon))  # score distribution similarity
        top1_hits.append(1.0 if int(np.argmax(attn_recon)) == top1_idx else 0.0)
        top5_hits.append(1.0 if top1_idx in top5_idx else 0.0)

    return (
        float(np.mean(cos_sims)),
        float(np.mean(attn_accs)),
        float(np.mean(top1_hits)),
        float(np.mean(top5_hits)),
    )


# ---------------------------------------------------------------------------
# Per-preset benchmark
# ---------------------------------------------------------------------------

def run_preset(
    preset_name: str,
    config: QuantConfig,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    n_trials: int,
    n_queries: int,
    rng: np.random.Generator,
) -> dict:
    """Run quality benchmark for a single QuantConfig.

    Uses layer-by-layer processing when the full KV tensor would exceed
    _LAYER_BY_LAYER_THRESHOLD_BYTES to avoid OOM at long seq_len.
    """
    compressor = TurboQuantCompressor(config)

    full_tensor_bytes = 2 * n_layers * n_heads * seq_len * head_dim * 8  # float64 K+V
    layer_by_layer = full_tensor_bytes > _LAYER_BY_LAYER_THRESHOLD_BYTES

    all_cos, all_attn_acc, all_top1, all_top5 = [], [], [], []
    compress_times, decompress_times = [], []
    actual_ratios: list[float] = []

    for _trial in range(n_trials):
        if layer_by_layer:
            # ---- memory-efficient: one layer at a time ----
            t_compress_total = 0.0
            t_decompress_total = 0.0
            ratio_accum = 0.0

            for _layer in range(n_layers):
                K = rng.standard_normal((1, n_heads, seq_len, head_dim)).astype(np.float64)
                V = rng.standard_normal((1, n_heads, seq_len, head_dim)).astype(np.float64)

                t0 = time.perf_counter()
                ckv = compressor.compress_kv(K, V)
                t_compress_total += time.perf_counter() - t0

                orig_b = K.nbytes + V.nbytes
                comp_b = sum(
                    blk.indices.nbytes + 8
                    for ct in (*ckv.keys, *ckv.values)
                    for blk in ct.blocks
                )
                if comp_b > 0:
                    ratio_accum += orig_b / comp_b

                t0 = time.perf_counter()
                K_r, V_r = compressor.decompress_kv(ckv)
                t_decompress_total += time.perf_counter() - t0

                for h in range(n_heads):
                    cos, acc, t1, t5 = _attention_metrics(
                        K[0, h], V[0, h], K_r[0, h], V_r[0, h],
                        n_queries=n_queries, rng=rng,
                    )
                    all_cos.append(cos)
                    all_attn_acc.append(acc)
                    all_top1.append(t1)
                    all_top5.append(t5)

            compress_times.append(t_compress_total)
            decompress_times.append(t_decompress_total)
            actual_ratios.append(ratio_accum / n_layers)
        else:
            # ---- all layers at once ----
            K = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float64)
            V = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float64)

            t0 = time.perf_counter()
            ckv = compressor.compress_kv(K, V)
            compress_times.append(time.perf_counter() - t0)

            orig_b = K.nbytes + V.nbytes
            comp_b = sum(
                blk.indices.nbytes + 8
                for ct in (*ckv.keys, *ckv.values)
                for blk in ct.blocks
            )
            if orig_b > 0 and comp_b > 0:
                actual_ratios.append(orig_b / comp_b)

            t0 = time.perf_counter()
            K_r, V_r = compressor.decompress_kv(ckv)
            decompress_times.append(time.perf_counter() - t0)

            for l in range(n_layers):
                for h in range(n_heads):
                    cos, acc, t1, t5 = _attention_metrics(
                        K[l, h], V[l, h], K_r[l, h], V_r[l, h],
                        n_queries=n_queries, rng=rng,
                    )
                    all_cos.append(cos)
                    all_attn_acc.append(acc)
                    all_top1.append(t1)
                    all_top5.append(t5)

    return {
        "preset": preset_name,
        "k_bits": config.k_bits,
        "v_bits": config.v_bits,
        "compression_ratio": round(float(np.mean(actual_ratios)), 2) if actual_ratios else None,
        "cosine_similarity": round(float(np.mean(all_cos)), 4),
        "attn_score_accuracy": round(float(np.mean(all_attn_acc)), 4),
        "top1_match_pct": round(float(np.mean(all_top1)) * 100, 1),
        "top5_match_pct": round(float(np.mean(all_top5)) * 100, 1),
        "avg_compress_ms": round(float(np.mean(compress_times)) * 1000, 1),
        "avg_decompress_ms": round(float(np.mean(decompress_times)) * 1000, 1),
        "trials": n_trials,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "layer_by_layer": layer_by_layer,
    }


# ---------------------------------------------------------------------------
# Zero-Quant quality benchmark
# ---------------------------------------------------------------------------


def run_zero_quant_preset(
    preset_name: str,
    config: ZeroQuantConfig,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    n_trials: int,
    n_queries: int,
    rng: np.random.Generator,
) -> dict:
    """Run quality benchmark for a single ZeroQuantConfig.

    Same metrics as ``run_preset`` (cosine_similarity, top1/top5 match),
    but uses ``DepthAdaptiveCompressor`` so each zone is compressed at its
    configured bit-width.
    """
    compressor = DepthAdaptiveCompressor(config)

    full_tensor_bytes = 2 * n_layers * n_heads * seq_len * head_dim * 8
    layer_by_layer = full_tensor_bytes > _LAYER_BY_LAYER_THRESHOLD_BYTES

    all_cos, all_attn_acc, all_top1, all_top5 = [], [], [], []
    compress_times, decompress_times = [], []
    actual_ratios: list[float] = []

    for _trial in range(n_trials):
        K = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float64)
        V = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float64)

        t0 = time.perf_counter()
        ckv = compressor.compress_kv(K, V)
        compress_times.append(time.perf_counter() - t0)

        orig_b = K.nbytes + V.nbytes
        comp_b = ckv.memory_bytes()
        if orig_b > 0 and comp_b > 0:
            actual_ratios.append(orig_b / comp_b)

        t0 = time.perf_counter()
        K_r, V_r = compressor.decompress_kv(ckv)
        decompress_times.append(time.perf_counter() - t0)

        for layer_i in range(n_layers):
            for h in range(n_heads):
                cos, acc, t1, t5 = _attention_metrics(
                    K[layer_i, h], V[layer_i, h],
                    K_r[layer_i, h], V_r[layer_i, h],
                    n_queries=n_queries, rng=rng,
                )
                all_cos.append(cos)
                all_attn_acc.append(acc)
                all_top1.append(t1)
                all_top5.append(t5)

    avg_bits = config.average_bits(n_layers)
    return {
        "preset": preset_name,
        "k_bits": f"sh{config.shallow_k_bits}/mi{config.middle_k_bits}/dp{config.deep_k_bits}",
        "v_bits": f"sh{config.shallow_v_bits}/mi{config.middle_v_bits}/dp{config.deep_v_bits}",
        "avg_bits": round(avg_bits, 2),
        "compression_ratio": round(float(np.mean(actual_ratios)), 2) if actual_ratios else None,
        "cosine_similarity": round(float(np.mean(all_cos)), 4),
        "attn_score_accuracy": round(float(np.mean(all_attn_acc)), 4),
        "top1_match_pct": round(float(np.mean(all_top1)) * 100, 1),
        "top5_match_pct": round(float(np.mean(all_top5)) * 100, 1),
        "avg_compress_ms": round(float(np.mean(compress_times)) * 1000, 1),
        "avg_decompress_ms": round(float(np.mean(decompress_times)) * 1000, 1),
        "trials": n_trials,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "layer_by_layer": layer_by_layer,
    }


# ---------------------------------------------------------------------------
# CLI + output
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboQuant KV cache compression quality benchmark"
    )
    parser.add_argument(
        "--n-layers", type=int, default=40,
        help="Number of transformer layers (default: 40, Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--n-heads", type=int, default=32,
        help="Number of attention heads per layer (default: 32)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=128,
        help="Attention head dimension (default: 128)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256,
        help="KV cache sequence length to simulate (default: 256)",
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Random KV tensor trials per preset (default: 3)",
    )
    parser.add_argument(
        "--queries", type=int, default=8,
        help="Query vectors per (layer, head) per trial (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="NumPy random seed (default: 42)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Optional path to save results as JSON",
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=list(PRESETS.keys()),
        help="Run only this preset (default: all three)",
    )
    return parser.parse_args()


def print_table(results: list[dict], title: str = "TurboQuant Presets") -> None:
    """Print a formatted results table."""
    print(f"\n  {title}")
    header = (
        f"\n  {'Preset':<32} {'Avg Bits':>9} {'Compression':>13} {'Cosine Sim':>12} "
        f"{'Attn Acc':>10} {'Top-1 Match':>12} {'Top-5 Match':>12} "
        f"{'Compress':>10} {'Decompress':>11}"
    )
    sep = "  " + "-" * (len(header.rstrip()) - 2)
    print(header)
    print(sep)
    for r in results:
        ratio = f"{r['compression_ratio']:.2f}x" if r["compression_ratio"] else "N/A"
        avg_bits = r.get("avg_bits", (r.get("k_bits", 0) + r.get("v_bits", 0)) / 2)
        if isinstance(avg_bits, (int, float)):
            avg_bits_str = f"{avg_bits:.1f}"
        else:
            avg_bits_str = str(avg_bits)
        print(
            f"  {r['preset']:<32} {avg_bits_str:>9} {ratio:>13} "
            f"{r['cosine_similarity']:>12.4f} {r['attn_score_accuracy']:>10.4f} "
            f"{r['top1_match_pct']:>11.1f}% {r['top5_match_pct']:>11.1f}% "
            f"{r['avg_compress_ms']:>9.1f}ms {r['avg_decompress_ms']:>10.1f}ms"
        )
    print()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    presets_to_run = (
        {args.preset: PRESETS[args.preset]} if args.preset else PRESETS
    )

    full_bytes = 2 * args.n_layers * args.n_heads * args.seq_len * args.head_dim * 8
    strategy = "layer-by-layer" if full_bytes > _LAYER_BY_LAYER_THRESHOLD_BYTES else "batched"

    print(
        f"\nTurboQuant Quality Benchmark\n"
        f"  Architecture : {args.n_layers} layers \u00d7 {args.n_heads} heads \u00d7 "
        f"head_dim={args.head_dim}\n"
        f"  Sequence len : {args.seq_len:,} tokens\n"
        f"  Trials       : {args.trials}  |  Queries/head: {args.queries}\n"
        f"  Strategy     : {strategy}\n"
    )

    results: list[dict] = []
    for name, config in presets_to_run.items():
        print(f"  Running {name} ...", end="", flush=True)
        t_start = time.perf_counter()
        r = run_preset(
            name, config,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            seq_len=args.seq_len,
            n_trials=args.trials,
            n_queries=args.queries,
            rng=rng,
        )
        elapsed = time.perf_counter() - t_start
        print(f" done ({elapsed:.1f}s)")
        results.append(r)

    print_table(results, title="TurboQuant Presets (flat per-layer quantization)")

    # Zero-Quant depth-adaptive benchmarks
    print("\n  Running Zero-Quant depth-adaptive presets...")
    zq_results: list[dict] = []
    for name, zq_config in ZQ_PRESETS.items():
        print(f"  Running {name} ...", end="", flush=True)
        t_start = time.perf_counter()
        r = run_zero_quant_preset(
            name, zq_config,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            seq_len=args.seq_len,
            n_trials=args.trials,
            n_queries=args.queries,
            rng=rng,
        )
        elapsed = time.perf_counter() - t_start
        print(f" done ({elapsed:.1f}s)")
        zq_results.append(r)

    print_table(zq_results, title="Zero-Quant Presets (depth-adaptive: shallow/middle/deep zones)")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "turboquant_results": results,
            "zero_quant_results": zq_results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
