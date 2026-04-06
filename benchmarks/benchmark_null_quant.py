"""NullQuant benchmark — token eviction + zone compression performance.

Measures compression ratio, eviction overhead, MSE degradation, and
throughput across different NullQuant configurations.

Usage:
    python -m benchmarks.benchmark_null_quant
    python -m benchmarks.benchmark_null_quant --runs 5 --seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.turboquant.null_quant import (
    NullQuantConfig,
    NullQuantCompressor,
    NullQuantCompressedKV,
    score_tokens_l2_norm,
    select_survivors,
)

logger = logging.getLogger(__name__)


@dataclass
class NullQuantBenchResult:
    """Result of a single NullQuant benchmark run."""
    config_name: str
    n_layers: int
    n_heads: int
    head_dim: int
    seq_len: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    original_tokens: int
    surviving_tokens: int
    eviction_ratio: float
    eviction_time_s: float
    compression_time_s: float
    decompress_time_s: float
    total_time_s: float
    mse: float
    total_reduction: float
    estimated_reduction: float  # from config.estimated_reduction()


# Default configs to benchmark
CONFIGS = {
    "default": NullQuantConfig(),  # 75% eviction, K8V8/K4V2/K8V8
    "mild": NullQuantConfig(eviction_ratio=0.50, middle_v_bits=4),
    "aggressive": NullQuantConfig(eviction_ratio=0.90, middle_v_bits=2),
    "extreme": NullQuantConfig(
        eviction_ratio=0.9375,  # keep 6.25%
        sink_tokens=128,
        recent_tokens=128,
        middle_k_bits=4,
        middle_v_bits=2,
    ),
    "random_baseline": NullQuantConfig(scoring_method="random"),
    "stride_baseline": NullQuantConfig(scoring_method="uniform_stride"),
}


def generate_synthetic_kv(
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic KV tensors resembling normalised llama.cpp state."""
    rng = np.random.default_rng(seed)
    # Match the normalisation in _state_to_kv_tensors: roughly N(0, 1)
    keys = rng.standard_normal((n_layers, n_heads, seq_len, head_dim))
    values = rng.standard_normal((n_layers, n_heads, seq_len, head_dim))
    return keys, values


def run_single(
    config_name: str,
    config: NullQuantConfig,
    keys: np.ndarray,
    values: np.ndarray,
) -> NullQuantBenchResult:
    """Run one NullQuant compress → decompress cycle and report stats."""
    n_layers, n_heads, seq_len, head_dim = keys.shape
    original_bytes = keys.nbytes + values.nbytes

    compressor = NullQuantCompressor(config)

    # Compress
    t0 = time.monotonic()
    compressed = compressor.compress_kv(keys, values)
    compress_time = time.monotonic() - t0

    # Decompress
    t1 = time.monotonic()
    dec_keys, dec_values = compressor.decompress_kv(compressed)
    decompress_time = time.monotonic() - t1

    # MSE on survivors
    all_positions = np.arange(seq_len, dtype=np.int32)
    survivor_positions = np.setdiff1d(all_positions, compressed.evicted_positions)
    orig_survivor_keys = keys[:, :, survivor_positions, :]
    orig_survivor_values = values[:, :, survivor_positions, :]
    k_mse = float(np.mean((orig_survivor_keys - dec_keys) ** 2))
    v_mse = float(np.mean((orig_survivor_values - dec_values) ** 2))
    avg_mse = (k_mse + v_mse) / 2.0

    compressed_bytes = compressed.memory_bytes
    total_time = compress_time + decompress_time

    return NullQuantBenchResult(
        config_name=config_name,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=original_bytes / max(compressed_bytes, 1),
        original_tokens=seq_len,
        surviving_tokens=compressed.n_surviving_tokens,
        eviction_ratio=compressed.eviction_ratio_actual,
        eviction_time_s=compressed.eviction_time_s,
        compression_time_s=compressed.compression_time_s,
        decompress_time_s=decompress_time,
        total_time_s=total_time,
        mse=avg_mse,
        total_reduction=compressed.total_reduction,
        estimated_reduction=config.estimated_reduction(seq_len, n_layers),
    )


def run_benchmark(
    runs: int = 3,
    seq_len: int = 2048,
    n_layers: int = 28,
    n_heads: int = 28,
    head_dim: int = 128,
    output_path: str | None = None,
) -> list[NullQuantBenchResult]:
    """Run NullQuant benchmark across all configs."""
    results: list[NullQuantBenchResult] = []

    print(f"\n{'='*80}")
    print(f"  NullQuant Benchmark — {runs} runs × {len(CONFIGS)} configs")
    print(f"  Model dims: {n_layers}L × {n_heads}H × {head_dim}D | seq_len={seq_len}")
    print(f"{'='*80}\n")

    keys, values = generate_synthetic_kv(n_layers, n_heads, head_dim, seq_len)
    original_mb = (keys.nbytes + values.nbytes) / (1024 * 1024)
    print(f"  Original KV size: {original_mb:.1f} MB\n")

    for config_name, config in CONFIGS.items():
        run_results: list[NullQuantBenchResult] = []

        for run_idx in range(runs):
            result = run_single(config_name, config, keys, values)
            run_results.append(result)

        # Average results
        avg = NullQuantBenchResult(
            config_name=config_name,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            original_bytes=run_results[0].original_bytes,
            compressed_bytes=int(np.mean([r.compressed_bytes for r in run_results])),
            compression_ratio=float(np.mean([r.compression_ratio for r in run_results])),
            original_tokens=seq_len,
            surviving_tokens=int(np.mean([r.surviving_tokens for r in run_results])),
            eviction_ratio=float(np.mean([r.eviction_ratio for r in run_results])),
            eviction_time_s=float(np.mean([r.eviction_time_s for r in run_results])),
            compression_time_s=float(np.mean([r.compression_time_s for r in run_results])),
            decompress_time_s=float(np.mean([r.decompress_time_s for r in run_results])),
            total_time_s=float(np.mean([r.total_time_s for r in run_results])),
            mse=float(np.mean([r.mse for r in run_results])),
            total_reduction=float(np.mean([r.total_reduction for r in run_results])),
            estimated_reduction=run_results[0].estimated_reduction,
        )
        results.append(avg)

        # Print row
        print(f"  {config_name:<18s} | "
              f"tokens {avg.original_tokens:>5d} → {avg.surviving_tokens:>5d} "
              f"({avg.eviction_ratio*100:5.1f}% evicted) | "
              f"compressed {avg.compressed_bytes/1024:>8.1f} KB | "
              f"ratio {avg.compression_ratio:>6.1f}x | "
              f"total {avg.total_reduction:>6.1f}x | "
              f"MSE {avg.mse:.6f} | "
              f"time {avg.total_time_s*1000:>6.1f} ms")

    # Summary
    print(f"\n{'─'*80}")
    print("  Summary (avg bits vs fp16=16):")
    for r in results:
        est = r.estimated_reduction
        act = r.total_reduction
        delta = ((act - est) / est * 100) if est > 0 else 0
        print(f"    {r.config_name:<18s}: est {est:>6.1f}x | actual {act:>6.1f}x | "
              f"delta {delta:+.1f}%")
    print()

    # Save to JSON
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"  Results saved to: {output_path}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="NullQuant benchmark")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--n-heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="benchmarks/results/null_quant.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_benchmark(
        runs=args.runs,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
