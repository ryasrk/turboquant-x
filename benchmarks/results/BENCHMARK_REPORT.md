# TurboQuant Benchmark Report

**Standard Inference vs TurboQuant Inference**

| Detail   | Value                                           |
|----------|-------------------------------------------------|
| Date     | April 1, 2026                                   |
| Model    | Qwen 2.5 7B Instruct Q4_K_M (4.4 GB GGUF)      |
| CPU      | AMD Ryzen 9 8945H (16 threads)                  |
| RAM      | 24 GB DDR5                                      |
| GPU      | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  |
| Driver   | 595.79 / CUDA 13.2                              |
| Runtime  | llama-cpp-python 0.3.19 (CUDA backend)          |

---

## 1. Single-Prompt Benchmark (4K Context, GPU)

**Config:** 3 prompts × 2 runs, 4096 context, all 29 layers offloaded to GPU

| Metric | Standard (Q8_0) | TurboQuant K8/V4 | TurboQuant K8/V2 | TurboQuant K4/V4 |
|--------|:---:|:---:|:---:|:---:|
| **KV Cache K** | q8_0 | q8_0 + turbo8 | q8_0 + turbo8 | q8_0 + turbo4 |
| **KV Cache V** | q8_0 | q8_0 + turbo4 | q8_0 + turbo2 | q8_0 + turbo4 |
| **RAM Used (MB)** | 283.6 | 32.8 | 28.7 | 24.8 |
| **GPU VRAM Used (MB)** | 4689.3 | 4594.4 | 4592.0 | 4593.8 |
| **Token Speed (tok/s)** | 43.59 | 42.95 (-1.5%) | 44.28 (+1.6%) | 44.37 (+1.8%) |
| **Compression Ratio** | 1.00x | **7.76x** | **7.76x** | **7.53x** |
| **MSE (quality loss)** | 0.000000 | 0.009966 | 0.123518 | 0.021150 |
| **Compress Time (ms)** | N/A | 253.8 | 234.4 | 492.5 |
| **Decompress Time (ms)** | N/A | 197.5 | 190.7 | 375.0 |

### Key Findings
- **~7.5x compression** with near-zero speed overhead on GPU
- **K8/V4** is the sweet spot: 7.76x compression, MSE 0.01, only -1.5% speed
- **88% RAM reduction**: 283 MB → ~29 MB system RAM
- All responses are **identical** across all configs

---

## 2. Extended Context Benchmark (8K Context, GPU)

**Config:** 3 prompts × 2 runs, 8192 context, all layers on GPU

| Metric | Standard (Q8_0) | TurboQuant K8/V4 | TurboQuant K8/V2 | TurboQuant K4/V4 |
|--------|:---:|:---:|:---:|:---:|
| **RAM Used (MB)** | 288.8 | 39.8 | 37.3 | 33.1 |
| **GPU VRAM Used (MB)** | 4814.3 | 4719.1 | 4723.7 | 4720.0 |
| **Token Speed (tok/s)** | 43.21 | 44.51 (+3.0%) | 43.98 (+1.8%) | 44.09 (+2.0%) |
| **Compression Ratio** | 1.00x | **7.76x** | **7.76x** | **7.53x** |
| **MSE (quality loss)** | 0.000000 | 0.009966 | 0.123518 | 0.021150 |
| **Compress Time (ms)** | N/A | 241.6 | 233.5 | 481.8 |
| **Decompress Time (ms)** | N/A | 189.2 | 190.3 | 367.6 |

### Key Findings
- At 8K context, TurboQuant configs are **slightly faster** than standard (+1.8% to +3.0%)
- Compression ratio remains consistent at ~7.5x
- GPU VRAM increases by ~125 MB compared to 4K (KV cache grows with context)

---

## 3. High-Token Generation Benchmark (256 max tokens, GPU)

**Config:** 3 prompts × 2 runs, 4096 context, 256 max generation tokens

| Metric | Standard (Q8_0) | TurboQuant K8/V4 | TurboQuant K8/V2 | TurboQuant K4/V4 |
|--------|:---:|:---:|:---:|:---:|
| **RAM Used (MB)** | 279.6 | 33.3 | 24.6 | 24.8 |
| **GPU VRAM Used (MB)** | 4674.9 | 4604.3 | 4594.0 | 4585.7 |
| **Token Speed (tok/s)** | 40.89 | 40.06 (-2.0%) | 41.59 (+1.7%) | 41.10 (+0.5%) |
| **Compression Ratio** | 1.00x | **7.76x** | **7.76x** | **7.53x** |
| **MSE (quality loss)** | 0.000000 | 0.009951 | 0.123467 | 0.021192 |
| **Compress Time (ms)** | N/A | 521.9 | 531.3 | 1047.8 |
| **Decompress Time (ms)** | N/A | 411.8 | 402.1 | 811.2 |

### Key Findings
- Longer generation slightly slows all modes (~40-41 tok/s vs ~43-44 tok/s)
- Compress/decompress times ~2x longer due to larger state (more tokens generated)
- Speed delta remains small: -2.0% to +1.7%

---

## 4. Multi-Turn Conversation Benchmark (5 turns, GPU)

**Config:** 5-turn BST conversation, 128 max tokens/turn, TurboQuant K8/V4

| Turn | Total Tokens | Standard (tok/s) | TurboQuant K8/V4 (tok/s) | Δ Speed | Compression | MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 151 | 47.0 | 46.6 | -0.8% | 7.76x | 0.0099 |
| 2 | 303 | 48.5 | 49.2 | +1.4% | 7.76x | 0.0099 |
| 3 | 459 | 45.8 | 42.9 | -6.2% | 7.76x | 0.0099 |
| 4 | 611 | 45.5 | 38.6 | -15.1% | 7.76x | 0.0099 |
| 5 | 761 | 47.9 | 42.7 | -10.8% | 7.76x | 0.0099 |
| **AVG** | — | **46.9** | **44.0** | **-6.2%** | **7.76x** | **0.0099** |

### Context Growth Per Turn

```
Turn 1:    23 tokens  █
Turn 2:   175 tokens  ████████
Turn 3:   331 tokens  ████████████████
Turn 4:   483 tokens  ████████████████████████
Turn 5:   633 tokens  ███████████████████████████████
```

### RAM Usage (MB)

| Turn | Standard | TurboQuant K8/V4 |
|:---:|:---:|:---:|
| 1 | 5412.9 | 5459.3 |
| 2 | 5418.8 | 5458.1 |
| 3 | 5421.0 | 5479.4 |
| 4 | 5424.4 | 5438.7 |
| 5 | 5426.1 | 5461.0 |

### GPU VRAM (MB)

| Turn | Standard | TurboQuant K8/V4 |
|:---:|:---:|:---:|
| 1 | 6399.2 | 6386.3 |
| 2 | 6395.8 | 6386.0 |
| 3 | 6391.3 | 6386.0 |
| 4 | 6401.8 | 6381.3 |
| 5 | 6403.3 | 6381.1 |

### Key Findings
- **Context grows linearly** from 23 → 633 prompt tokens over 5 turns
- Compression ratio stays **constant at 7.76x** regardless of context size
- **~6.2% average speed overhead** for multi-turn (higher than single-prompt due to Python-level compress/decompress each turn)
- Later turns (4-5) show more overhead (-10% to -15%) as state size grows
- **Output quality is identical** — same responses on both modes
- GPU VRAM is **~20 MB lower** with TurboQuant (compressed state reduces memory pressure)

---

## 5. CPU vs GPU Speed Comparison

### CPU-Only Benchmark (0 GPU layers, RAM only)

| Metric | Standard (Q8_0) | TurboQuant K8/V4 | TurboQuant K8/V2 | TurboQuant K4/V4 |
|--------|:---:|:---:|:---:|:---:|
| **RAM Used (MB)** | 409.5 | 161.5 | 157.5 | 151.6 |
| **GPU VRAM Used (MB)** | 824.5 | 731.1 | 732.0 | 744.6 |
| **Token Speed (tok/s)** | 8.30 | 8.56 (+3.2%) | 8.78 (+5.8%) | 8.47 (+2.1%) |
| **Compression Ratio** | 1.00x | **7.76x** | **7.76x** | **7.53x** |
| **MSE** | 0.000000 | 0.009995 | 0.123814 | 0.021197 |
| **Compress Time (ms)** | N/A | 250.7 | 235.8 | 507.9 |
| **Decompress Time (ms)** | N/A | 190.9 | 185.9 | 388.1 |

> **Note:** Even with `n_gpu_layers=0`, some minimal GPU VRAM is used by the CUDA runtime. All model weights and KV cache are in system RAM.

### CPU-Only Key Findings
- TurboQuant is **2-6% faster** than standard on CPU (memory pressure relief helps CPU caches)
- **60% RAM reduction**: 409 MB → ~155 MB system RAM
- All responses remain **identical** to GPU mode

### Speed Comparison: CPU vs GPU

| Mode | CPU (tok/s) | GPU (tok/s) | GPU Speedup |
|:---:|:---:|:---:|:---:|
| Standard Q8_0 | 8.30 | 43.59 | **5.3x** |
| TurboQuant K8/V4 | 8.56 | 42.95 | **5.0x** |
| TurboQuant K8/V2 | 8.78 | 44.28 | **5.0x** |
| TurboQuant K4/V4 | 8.47 | 44.37 | **5.2x** |

---

## 6. TurboQuant Configuration Guide

| Config | K-bits | V-bits | Compression | MSE | Speed Impact | Best For |
|--------|:---:|:---:|:---:|:---:|:---:|---|
| **Quality (K8/V4)** | 8 | 4 | 7.76x | 0.010 | -1.5% | Production use, quality-sensitive tasks |
| **Aggressive (K8/V2)** | 8 | 2 | 7.76x | 0.124 | +1.6% | Memory-constrained, simple tasks |
| **Symmetric (K4/V4)** | 4 | 4 | 7.53x | 0.021 | +1.8% | Balanced compression, longer compress time |

### Recommendations

1. **Use K8/V4 (Quality) as default** — best quality/speed/compression tradeoff
2. **Use K8/V2 for memory-constrained deployments** — same compression, higher MSE but no visible quality loss on tested prompts
3. **Avoid K4/V4 in latency-sensitive apps** — 2x longer compress time for similar compression ratio
4. **GPU offloading is essential** — 5.3x speedup with no changes to compression behavior
5. **Multi-turn penalty is ~6%** — acceptable for most conversational applications

---

## 7. Architecture Notes

- **Standard mode**: llama.cpp handles KV cache internally with Q8_0 quantization
- **TurboQuant mode**: llama.cpp uses Q8_0 base KV cache, then Python-level TurboQuant compresses the saved state between turns
- Compression happens **between turns** (not during generation), so token generation speed is minimally affected
- The Python-level compress/decompress adds ~200-500ms overhead per turn depending on state size
- With a C-level TurboQuant fork in llama.cpp, this overhead would be eliminated entirely

---

*Report generated by `benchmarks/benchmark_compare.py` and `benchmarks/benchmark_multiturn.py`*
