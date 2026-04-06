# TurboQuant-X — KV Cache Compression Benchmark

> Full benchmark of all 13 compression configurations across 5 inference modes.
> Generated with synthetic KV tensors matching Qwen2.5-7B architecture.

## Test Configuration

| Parameter | Value |
|---|---|
| Layers × Heads × Head Dim | 8L × 8H × 64D (scaled) |
| Sequence Length | 1024 tokens |
| Original KV Size | 65,536 KB |
| Data Type | float64 |
| Runs Per Config | 3 (averaged) |
| Scoring Method | L2 norm (NullQuant) |

## Accuracy Ranking (by MSE — lower is better)

| Rank | Mode | Compression Ratio | MSE | MAE | Cosine Similarity | PSNR (dB) | K MSE | V MSE | Tier |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | Standard Q8/Q8 | 8.0× | 0.000285 | 0.0126 | 0.999832 | 34.3 | 0.000498 | 0.000072 | ★★★ |
| 2 | ZeroQuant conservative | 7.9× | 0.000862 | 0.0210 | 0.999370 | 29.5 | 0.000498 | 0.001226 | ★★★ |
| 3 | TurboQuant K8/V4 | 10.7× | 0.003159 | 0.0379 | 0.997526 | 23.8 | 0.000498 | 0.005820 | ★★★ |
| 4 | ZeroQuant default | 7.8× | 0.003197 | 0.0377 | 0.997691 | 23.8 | 0.002082 | 0.004312 | ★★★ |
| 5 | TurboQuant K4/V4 | 16.0× | 0.007078 | 0.0612 | 0.995338 | 20.3 | 0.008335 | 0.005820 | ★★★ |
| 6 | ZeroQuant aggressive | 7.6× | 0.010617 | 0.0701 | 0.991694 | 18.6 | 0.002082 | 0.019152 | ★★ |
| 7 | TurboQuant K8/V3 | 11.6× | 0.010843 | 0.0641 | 0.991307 | 18.5 | 0.000498 | 0.021189 | ★★ |
| 8 | TurboQuant K8/V2 | 12.8× | 0.036500 | 0.1109 | 0.969946 | 13.2 | 0.000498 | 0.072502 | ★★ |
| 9 | TurboQuant K4/V2 | 21.3× | 0.040418 | 0.1342 | 0.967758 | 12.7 | 0.008335 | 0.072502 | ★ |
| 10 | NullQuant mild | 15.5× | 0.358731 | 0.3310 | 0.801156 | 3.3 | 0.468966 | 0.248495 | — |
| 11 | NullQuant default | 31.0× | 0.462350 | 0.3664 | 0.804992 | 2.2 | 0.688558 | 0.236142 | — |
| 12 | NullQuant aggressive | 77.5× | 0.790090 | 0.4670 | 0.804502 | −0.2 | 1.343742 | 0.236438 | — |
| 13 | NullQuant extreme | 123.2× | 0.920313 | 0.4960 | 0.807136 | −0.8 | 1.607459 | 0.233168 | — |

### Tier Definitions

| Tier | PSNR Range | Use Case |
|:---:|---|---|
| ★★★ | > 20 dB | Production quality — negligible accuracy loss |
| ★★ | 13–20 dB | Good for drafting, summarization, non-critical tasks |
| ★ | 10–13 dB | Aggressive compression — visible degradation |
| — | < 10 dB | Maximum compression — trades accuracy for KV memory savings |

## Speed Ranking (total compress + decompress time)

| Rank | Mode | Speed (ms) | Compress (ms) | Decompress (ms) | Ratio |
|:---:|---|---:|---:|---:|---:|
| 1 | Standard Q8/Q8 | 100.2 | — | — | 8.0× |
| 2 | NullQuant extreme | 159.2 | — | — | 123.2× |
| 3 | NullQuant default | 177.1 | — | — | 31.0× |
| 4 | NullQuant aggressive | 197.6 | — | — | 77.5× |
| 5 | NullQuant mild | 244.1 | — | — | 15.5× |
| 6 | ZeroQuant conservative | 258.4 | — | — | 7.9× |
| 7 | TurboQuant K8/V4 | 323.8 | — | — | 10.7× |
| 8 | TurboQuant K8/V2 | 347.0 | — | — | 12.8× |
| 9 | ZeroQuant default | 350.2 | — | — | 7.8× |
| 10 | TurboQuant K4/V2 | 450.7 | — | — | 21.3× |
| 11 | ZeroQuant aggressive | 453.4 | — | — | 7.6× |
| 12 | TurboQuant K8/V3 | 526.2 | — | — | 11.6× |
| 13 | TurboQuant K4/V4 | 568.2 | — | — | 16.0× |

## KV Cache Compression Ratio Ranking (higher = more memory saved)

| Rank | Mode | Ratio | Surviving Tokens | KV Size Reduction |
|:---:|---|---:|---:|---|
| 1 | NullQuant extreme | 123.2× | 64 / 1024 | 99.2% |
| 2 | NullQuant aggressive | 77.5× | 102 / 1024 | 98.7% |
| 3 | NullQuant default | 31.0× | 256 / 1024 | 96.8% |
| 4 | TurboQuant K4/V2 | 21.3× | 1024 / 1024 | 95.3% |
| 5 | TurboQuant K4/V4 | 16.0× | 1024 / 1024 | 93.8% |
| 6 | NullQuant mild | 15.5× | 512 / 1024 | 93.5% |
| 7 | TurboQuant K8/V2 | 12.8× | 1024 / 1024 | 92.2% |
| 8 | TurboQuant K8/V3 | 11.6× | 1024 / 1024 | 91.4% |
| 9 | TurboQuant K8/V4 | 10.7× | 1024 / 1024 | 90.6% |
| 10 | Standard Q8/Q8 | 8.0× | 1024 / 1024 | 87.5% |
| 11 | ZeroQuant conservative | 7.9× | 1024 / 1024 | 87.3% |
| 12 | ZeroQuant default | 7.8× | 1024 / 1024 | 87.2% |
| 13 | ZeroQuant aggressive | 7.6× | 1024 / 1024 | 86.8% |

## Top 1 — Best Per Category

| Category | Winner | Value |
|---|---|---|
| **Best Accuracy (MSE)** | Standard Q8/Q8 | 0.000285 |
| **Best Accuracy (PSNR)** | Standard Q8/Q8 | 34.3 dB |
| **Best Cosine Similarity** | Standard Q8/Q8 | 0.999832 |
| **Best Compression Ratio** | NullQuant extreme | 123.2× |
| **Fastest Compression** | Standard Q8/Q8 | 100.2 ms |
| **Best Accuracy-per-Ratio** | TurboQuant K8/V4 | 10.7× @ 0.003 MSE |
| **Best Balance** | TurboQuant K4/V4 | 16.0× @ ★★★ tier |

## Top 5 — Recommended Configurations

| Rank | Mode | Why |
|:---:|---|---|
| 1 | **TurboQuant K8/V4** | Best balance of 10.7× ratio with ★★★ accuracy (MSE 0.003) |
| 2 | **TurboQuant K4/V4** | Highest ★★★ ratio at 16.0× — production-viable compression |
| 3 | **ZeroQuant conservative** | Excellent accuracy (MSE 0.0009) with depth-adaptive zones |
| 4 | **NullQuant default** | Maximum KV memory reduction (31.0×) when accuracy is secondary |
| 5 | **Standard Q8/Q8** | Baseline reference — best accuracy, moderate compression |

## Mode Comparison Summary

### TurboQuant (PolarQuant Quantization)

Fixed bit-width quantization of all KV cache tokens. Every token is preserved; precision is reduced.

- **Approach**: Quantize K and V tensors to configurable bit widths (2–8 bits)
- **Strengths**: Predictable quality, all tokens retained, ★★★ tier achievable
- **Trade-off**: Compression ratio limited by bit width (max ~21× at K4/V2)
- **Best config**: K8/V4 (10.7×, MSE 0.003, ★★★)

### ZeroQuant (Depth-Adaptive Zones)

Different quantization levels for shallow, middle, and deep transformer layers.

- **Approach**: Assign per-zone (depth-based) bit widths — higher precision for attention-critical layers
- **Strengths**: Matches layer importance, competitive with fixed-width at lower ratios
- **Trade-off**: Ratio stays ~7–8× regardless of aggressiveness (all tokens kept)
- **Best config**: Conservative (7.9×, MSE 0.0009, ★★★)

### NullQuant (Token Eviction + Zone Compression)

Two-stage pipeline: first evict low-importance tokens (L2 norm scoring), then quantize survivors with depth-adaptive zones.

- **Approach**: Score tokens → evict bottom N% → compress survivors with ZeroQuant zones
- **Strengths**: Extreme ratios (31–124×), fastest compression, minimal KV memory footprint
- **Trade-off**: Evicted tokens are lost — higher MSE, lower PSNR
- **Best config**: Default (31.0×, 75% eviction, 256 surviving tokens)
- **Inspired by**: SlimInfer (AAAI 2026) dynamic token pruning concept

## Metric Definitions

| Metric | Description |
|---|---|
| **MSE** | Mean Squared Error — average squared difference between original and reconstructed KV values |
| **MAE** | Mean Absolute Error — average absolute difference |
| **Cosine Similarity** | Directional alignment between original and reconstructed vectors (1.0 = identical) |
| **PSNR** | Peak Signal-to-Noise Ratio in dB — higher is better, derived from MSE vs signal power |
| **K MSE / V MSE** | Per-component MSE for Key and Value tensors separately |
| **Compression Ratio** | Original size ÷ compressed size |
| **Surviving Tokens** | Number of tokens kept after eviction (NullQuant only) |

## Reproducing

```bash
cd turboquant-x

# Full benchmark (all modes, averaged over 3 runs)
env/bin/python3 benchmarks/benchmark_all_modes.py

# NullQuant-specific benchmark with more configs
env/bin/python3 -m benchmarks.benchmark_null_quant --runs 5 --seq-len 4096

# Speed benchmark
env/bin/python3 benchmarks/benchmark_speed.py

# Quality benchmark
env/bin/python3 benchmarks/benchmark_quality.py
```
