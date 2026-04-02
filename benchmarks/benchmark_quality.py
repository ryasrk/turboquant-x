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
from src.turboquant.polar_quant import compression_ratio  # noqa: E402


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------
PRESETS: dict[str, QuantConfig] = {
    "Quality (K8/V4)":    QuantConfig(k_bits=8, v_bits=4),
    "Aggressive (K8/V2)": QuantConfig(k_bits=8, v_bits=2),
    "Symmetric (K4/V4)":  QuantConfig(k_bits=4, v_bits=4),
}


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
) -> tuple[float, float, float]:
    """Compute (cosine_sim, top1_match, top5_match) for random queries."""
    seq_len, head_dim = K_orig.shape
    scale = 1.0 / (head_dim ** 0.5)

    cos_sims, top1_hits, top5_hits = [], [], []

    for _ in range(n_queries):
        q = rng.standard_normal(head_dim)

        # --- original attention ---
        scores_orig = scale * (K_orig @ q)          # (seq_len,)
        attn_orig   = _softmax(scores_orig)
        out_orig    = attn_orig @ V_orig             # (head_dim,)
        top1_idx    = int(np.argmax(attn_orig))

        # --- reconstructed attention ---
        scores_recon = scale * (K_recon @ q)
        attn_recon   = _softmax(scores_recon)
        out_recon    = attn_recon @ V_recon

        top5_idx = set(np.argsort(attn_recon)[-5:].tolist())

        cos_sims.append(_cosine(out_orig, out_recon))
        top1_hits.append(1.0 if int(np.argmax(attn_recon)) == top1_idx else 0.0)
        top5_hits.append(1.0 if top1_idx in top5_idx else 0.0)

    return (
        float(np.mean(cos_sims)),
        float(np.mean(top1_hits)),
        float(np.mean(top5_hits)),
    )


# ---------------------------------------------------------------------------
# Compression size helper
# ---------------------------------------------------------------------------

def _compressed_bytes(compressed_kv, config: QuantConfig) -> int:
    """Approximate compressed byte count from CompressedKV."""
    total = 0
    for ct in (*compressed_kv.keys, *compressed_kv.values):
        for block in ct.blocks:
            if block.indices.size > 0:
                # Each index uses n_bits rounded up to storage dtype
                bits = config.k_bits if ct in compressed_kv.keys else config.v_bits
                total += block.indices.nbytes
    return total


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
    """Run quality benchmark for a single QuantConfig."""
    compressor = TurboQuantCompressor(config)

    all_cos, all_top1, all_top5 = [], [], []
    compress_times, decompress_times = [], []
    actual_ratios: list[float] = []

    for trial in range(n_trials):
        # Sample realistic KV tensors in (n_layers, n_heads, seq_len, head_dim)
        K = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float64)
        V = rng.standard_normal((n_layers, n_heads, seq_len, head_dim)).astype(np.float64)

        # --- Compress ---
        t0 = time.perf_counter()
        ckv = compressor.compress_kv(K, V)
        compress_times.append(time.perf_counter() - t0)

        # --- Actual compression ratio (vs float64 original, matching turbo_engine) ---
        original_bytes = K.nbytes + V.nbytes          # float64 baseline
        compressed_sz = 0
        for ct in (*ckv.keys, *ckv.values):
            for block in ct.blocks:
                compressed_sz += block.indices.nbytes  # uint8 indices
                compressed_sz += 8                     # norm (float32) + seed overhead
        if original_bytes > 0 and compressed_sz > 0:
            actual_ratios.append(original_bytes / compressed_sz)

        # --- Decompress ---
        t0 = time.perf_counter()
        K_r, V_r = compressor.decompress_kv(ckv)
        decompress_times.append(time.perf_counter() - t0)

        # --- Attention metrics per (layer, head) ---
        for l in range(n_layers):
            for h in range(n_heads):
                cos, t1, t5 = _attention_metrics(
                    K[l, h], V[l, h], K_r[l, h], V_r[l, h],
                    n_queries=n_queries,
                    rng=rng,
                )
                all_cos.append(cos)
                all_top1.append(t1)
                all_top5.append(t5)

    return {
        "preset": preset_name,
        "k_bits": config.k_bits,
        "v_bits": config.v_bits,
        "compression_ratio": round(float(np.mean(actual_ratios)), 2) if actual_ratios else None,
        "cosine_similarity": round(float(np.mean(all_cos)), 4),
        "top1_match_pct": round(float(np.mean(all_top1)) * 100, 1),
        "top5_match_pct": round(float(np.mean(all_top5)) * 100, 1),
        "avg_compress_ms": round(float(np.mean(compress_times)) * 1000, 1),
        "avg_decompress_ms": round(float(np.mean(decompress_times)) * 1000, 1),
        "trials": n_trials,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
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


def print_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    header = (
        f"\n{'Preset':<24} {'Compression':>13} {'Cosine Sim':>12} "
        f"{'Top-1 Match':>12} {'Top-5 Match':>12} "
        f"{'Compress':>10} {'Decompress':>11}"
    )
    sep = "-" * len(header.rstrip())
    print(header)
    print(sep)
    for r in results:
        ratio = f"{r['compression_ratio']:.2f}x" if r["compression_ratio"] else "N/A"
        print(
            f"{r['preset']:<24} {ratio:>13} {r['cosine_similarity']:>12.4f} "
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

    print(
        f"\nTurboQuant Quality Benchmark\n"
        f"  Architecture : {args.n_layers} layers × {args.n_heads} heads × "
        f"head_dim={args.head_dim}\n"
        f"  Sequence len : {args.seq_len} tokens\n"
        f"  Trials       : {args.trials}  |  Queries/head: {args.queries}\n"
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

    print_table(results)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
