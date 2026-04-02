# TurboQuant-X

Local LLM inference server with **7.76x KV cache compression** and near-zero speed overhead. Runs quantized GGUF models on GPU or CPU with a FastAPI server exposing an OpenAI-compatible chat API.

## Key Results

| | Standard (Q8_0) | TurboQuant K8/V4 |
|---|:---:|:---:|
| **GPU Speed** | 43.10 tok/s | 44.85 tok/s (**+4.1%**) |
| **CPU Speed** | 9.05 tok/s | 9.08 tok/s (+0.2%) |
| **KV Cache RAM** | 284 MB | 33 MB (**-88%**) |
| **Compression** | 1.0x | **7.76x** |
| **Compress / Decompress** | — | **30 ms / 24 ms** |
| **Quality (MSE)** | 0.000 | 0.010 |
| **Multi-Turn Overhead** | — | **+0.1%** (was -6.2% with Python) |

*Tested on RTX 4060 Ti 8 GB, Ryzen 9 8945H, Qwen2.5-7B Q4_K_M, 4096 context, C++ backend enabled*

## Features

- **TurboQuant KV cache compression** — asymmetric K/V precision (K8/V4 default) via PolarQuant pipeline (WHT rotation + Lloyd-Max codebook quantization)
- **C++ acceleration backend** — 8x faster compression via fast Walsh-Hadamard O(d log d) + OpenMP threading — makes TurboQuant **faster** than standard inference (auto-detected, optional)
- **OpenAI-compatible API** — drop-in `/v1/chat/completions` endpoint with streaming (SSE) support
- **Browser Chat UI** — Sci-Fi HUD chatbot at `GET /`; real-time SSE streaming, collapsible thinking blocks, settings panel
- **Qwen3 thinking mode** — toggle chain-of-thought reasoning per request via `thinking` field or `chat_template_kwargs.enable_thinking`
- **Auto GPU layer distribution** — GGUF binary parser computes optimal `n_gpu_layers` from free VRAM at startup (`n_gpu_layers: -1`)
- **GPU + CPU inference** — llama.cpp backend with configurable GPU layer offloading
- **3 compression presets** — Quality (K8/V4), Aggressive (K8/V2), Symmetric (K4/V4)
- **YAML + env var + CLI config** — layered configuration with clear precedence
- **Benchmarking suite** — speed, perplexity, multi-turn, and needle-in-a-haystack benchmarks

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11+ | 3.12 |
| GPU | — (CPU works) | NVIDIA 8GB+ VRAM (CUDA) |
| RAM | 6 GB | 16 GB+ |
| Disk | 5 GB (model) | 10 GB |
| OS | Linux / macOS | Ubuntu 22.04+ |

Build tools for C++ backend (optional): `cmake >= 3.20`, `g++ >= 11` or `clang >= 14`, `pybind11`

---

## Setup (Zero to Running)

### Step 1: Clone and create virtual environment

```bash
cd /path/to/your/workspace
git clone <repo-url> turboquant-x
cd turboquant-x

python3 -m venv env
source env/bin/activate
```

### Step 2: Install Python dependencies

```bash
# Core only (FastAPI, NumPy, SciPy, Pydantic, uvicorn)
pip install -e .

# For GPU inference (llama-cpp-python + CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -e ".[gpu]"

# For development (pytest, coverage, httpx)
pip install -e ".[dev]"

# For benchmarking (datasets, matplotlib, tabulate)
pip install -e ".[bench]"

# Everything at once
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -e ".[dev,gpu,bench]"
```

> **CPU-only llama.cpp:** If you don't have an NVIDIA GPU, skip the `CMAKE_ARGS` line and install llama-cpp-python normally:
> ```bash
> pip install llama-cpp-python
> ```

### Step 3: Download a model

Download **Qwen3.5-35B-A3B Q4_K_M** (~20 GB) manually from HuggingFace or via `huggingface-cli`, then place the `.gguf` file in `models/`:

```bash
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-q4_k_m.gguf --local-dir models/
```

Alternatively, use the legacy downloader for Qwen2.5-7B (smaller, faster):

```bash
./scripts/download_model.sh
```

### Step 4: Build C++ backend (optional — 8x faster compression)

```bash
pip install pybind11

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build build -j$(nproc)

cp build/_turboquant_cpp.cpython-*.so src/turboquant_cpp/
```

Verify:
```bash
python3 -c "from src.turboquant_cpp import CPP_AVAILABLE; print('C++ backend:', 'enabled' if CPP_AVAILABLE else 'disabled')"
```

> The server runs without the C++ backend — compression just uses pure Python/NumPy (~5-9x slower).

### Step 5: Start the server

```bash
# Standard mode (no TurboQuant compression)
python -m src.main

# TurboQuant mode (recommended)
python -m src.main --mode turboquant

# TurboQuant with a named preset
python -m src.main --preset quality      # K8/V4 — best quality/speed (default)
python -m src.main --preset aggressive   # K8/V2 — max compression
python -m src.main --preset symmetric    # K4/V4 — equal K/V precision

# CPU-only (no GPU)
TURBOQUANT_N_GPU_LAYERS=0 python -m src.main --mode turboquant

# Custom port
python -m src.main --mode turboquant --port 9000
```

Expected startup output:
```
2026-04-01 20:37:50 [INFO] Starting TurboQuant-X Server
2026-04-01 20:37:50 [INFO] Model: qwen3.5-35b-a3b
2026-04-01 20:37:50 [INFO] Inference mode: turboquant
2026-04-01 20:37:50 [INFO] n_gpu_layers=-1 → auto-detecting from VRAM ...
2026-04-01 20:37:50 [INFO] Optimal n_gpu_layers: 12 (30.0% on GPU, 28 CPU layers)
2026-04-01 20:37:51 [INFO] Using C++ backend for PolarQuant compression
2026-04-01 20:37:58 [INFO] Model loaded in 7.2s
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Send a message

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, what is TurboQuant?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

---

## API Reference

### Chat Completions

```
POST /v1/chat/completions
```

#### Non-streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain KV cache compression in 3 sentences."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": false
  }'
```

Response:
```json
{
  "id": "chatcmpl-d5dbada23915",
  "object": "chat.completion",
  "model": "qwen3.5-35b-a3b",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "..." },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 25, "completion_tokens": 42, "total_tokens": 67 }
}
```

#### Streaming (SSE)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

Yields Server-Sent Events:
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

#### OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | array | required | 1–100 messages |
| `max_tokens` | int | 512 | 1–4096 tokens |
| `temperature` | float | 0.7 | 0.0–2.0 |
| `top_p` | float | 0.95 | 0.0–1.0 |
| `stream` | bool | false | SSE streaming |
| `thinking` | bool | true | Enable Qwen3 chain-of-thought reasoning |
| `chat_template_kwargs` | object | null | vLLM-compatible override — `{"enable_thinking": false}` takes precedence over `thinking` |

### Other Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser Chat UI (Sci-Fi HUD, streaming, thinking mode toggle) |
| `GET` | `/health` | Server health, model status, GPU memory, TurboQuant config, layer distribution |
| `GET` | `/v1/models` | List loaded models |

```bash
# Health check (includes layer_distribution when n_gpu_layers=-1)
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

---

## Chat UI

Open a browser at `http://localhost:8000/` to access the built-in Sci-Fi HUD chatbot.

**Features:**
- Real-time SSE streaming with blinking cursor
- **◈ Thinking** toggle — enable/disable Qwen3 chain-of-thought reasoning
- Collapsible `<think>…</think>` blocks rendered inline
- Settings panel: system prompt, `max_tokens`, `temperature`, `top_p` (persisted in `localStorage`)
- Model name and inference mode badges pulled from `/health`

No build step required — served as a single HTML file from `src/static/chat.html`.

---

## Qwen3 Thinking Mode

Qwen3-family models (e.g. `Qwen3.5-35B-A3B`) support chain-of-thought reasoning via a special `<think>…</think>` block.

**Enable thinking (default):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 17 * 43?"}], "thinking": true}'
```

**Disable thinking (fast mode):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "thinking": false}'
```

**vLLM-compatible override** (`chat_template_kwargs` takes precedence over `thinking`):
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [...], "chat_template_kwargs": {"enable_thinking": false}}'
```

When `thinking: false`, the server prefills an empty `<think>\n\n</think>` block, telling the model to skip chain-of-thought and respond directly.

---

## Compression Presets

| Preset | K-bits | V-bits | Compression | MSE | GPU Speed | CPU Speed | Best For |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|----------|
| **Quality** | 8 | 4 | 7.76x | 0.010 | +4.1% | +0.2% | Production default — best quality/speed tradeoff |
| **Aggressive** | 8 | 2 | 7.76x | 0.124 | +3.8% | +0.9% | Memory-constrained, simple tasks |
| **Symmetric** | 4 | 4 | 7.53x | 0.021 | +3.5% | +1.4% | Balanced K/V compression |

**Why asymmetric K/V?** Keys are used in attention score computation (dot product + softmax amplifies errors). Values are just weighted-averaged — quantization noise averages out. Keeping K at 8-bit while compressing V to 4-bit gives optimal quality/compression.

---

## C++ Turbo Engine

Optional C++ backend that accelerates the PolarQuant compression pipeline:

| Operation | Python (NumPy) | C++ Backend | Speedup |
|-----------|:---:|:---:|:---:|
| Compress (K8/V4) | 254 ms | 30 ms | **8.4x** |
| Decompress (K8/V4) | 198 ms | 24 ms | **8.2x** |
| Compress (K4/V4) | 493 ms | 58 ms | **8.4x** |
| Decompress (K4/V4) | 375 ms | 33 ms | **11.3x** |

Optimizations:
- **Fast Walsh-Hadamard Transform** — O(d log d) butterfly algorithm replaces O(d²) matrix multiply (18x fewer FLOPs for d=128)
- **OpenMP threading** — parallel block compression/decompression
- **Native codebook** — precomputed Lloyd-Max centroids compiled in, zero allocation overhead
- **`-O3 -march=native`** — full compiler optimization with AVX2/AVX-512

With the C++ backend, compression overhead drops from ~250ms to ~31ms per turn, making TurboQuant **strictly faster** than standard inference on all presets.

Build instructions in [Step 4](#step-4-build-c-backend-optional--8x-faster-compression) above. Without C++, everything works — just slower compression.

---

## Performance

### KV Cache Compression Quality

#### 256-token context (40 layers × 32 heads)

| Preset | Compression | Cosine Sim | Attn Score Acc | Top-1 Match | Top-5 Match |
|--------|:-----------:|:----------:|:--------------:|:-----------:|:-----------:|
| Quality K8/V4 | **7.76x** | 0.9952 | 1.0000 | 98.3% | **100.0%** |
| Aggressive K8/V2 | **7.76x** | 0.9397 | 1.0000 | 98.2% | **100.0%** |
| Symmetric K4/V4 | **7.53x** | 0.9908 | 0.9945 | 84.5% | 99.9% |

#### 8K-token context (40 layers × 32 heads, layer-by-layer)

| Preset | Compression | Cosine Sim | Attn Score Acc | Top-1 Match | Top-5 Match |
|--------|:-----------:|:----------:|:--------------:|:-----------:|:-----------:|
| Quality K8/V4 | **7.76x** | 0.9952 | **0.9999** | 97.4% | **100.0%** |
| Aggressive K8/V2 | **7.76x** | 0.9395 | **0.9999** | 97.2% | **100.0%** |
| Symmetric K4/V4 | **7.53x** | 0.9903 | 0.9951 | 79.7% | 99.3% |

**Metrics:** `Cosine Sim` = cosine similarity of attention output vectors before/after decompression. `Attn Score Acc` = cosine similarity between attention score *distributions* (softmax over all positions). `Top-1/Top-5 Match` = fraction of queries where the top-attended position is unchanged / still in top-5.

Reproduce:
```bash
# 256-token (fast, ~25s)
python -m benchmarks.benchmark_quality

# 8K-token (layer-by-layer processing, ~4min)
python -m benchmarks.benchmark_quality --seq-len 8192 --trials 1 --queries 4
```

### GPU Mode (RTX 4060 Laptop, 8 GB VRAM, C++ Backend)

| Config | Speed (tok/s) | System RAM | Compress | Decompress | MSE | Compression |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard Q8_0 | 43.10 | 284 MB | — | — | 0.000 | 1.0x |
| **TurboQuant K8/V4** | **44.85 (+4.1%)** | 33 MB (-88%) | 30 ms | 24 ms | 0.010 | **7.76x** |
| TurboQuant K8/V2 | 44.73 (+3.8%) | 23 MB (-92%) | 33 ms | 22 ms | 0.124 | **7.76x** |
| TurboQuant K4/V4 | 44.59 (+3.5%) | 24 MB (-92%) | 58 ms | 33 ms | 0.021 | **7.53x** |

### CPU Mode (Ryzen 9 8945H, 16 threads)

| Config | Speed (tok/s) | System RAM | Compress | Decompress | MSE | Compression |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard Q8_0 | 9.05 | 410 MB | — | — | 0.000 | 1.0x |
| **TurboQuant K8/V4** | 9.08 (+0.2%) | 160 MB (-61%) | 33 ms | 24 ms | 0.010 | **7.76x** |
| TurboQuant K8/V2 | 9.13 (+0.9%) | 153 MB (-63%) | 31 ms | 21 ms | 0.124 | **7.76x** |
| TurboQuant K4/V4 | 9.18 (+1.4%) | 151 MB (-63%) | 59 ms | 33 ms | 0.021 | **7.53x** |

> TurboQuant is **faster** on CPU because the smaller compressed KV cache improves L2/L3 cache utilization. Compression times are identical (~30ms) regardless of GPU/CPU mode since the C++ engine runs on CPU.

### GPU vs CPU

| Mode | CPU | GPU | GPU Speedup |
|------|:---:|:---:|:---:|
| Standard Q8_0 | 9.05 tok/s | 43.10 tok/s | **4.8x** |
| TurboQuant K8/V4 | 9.08 tok/s | 44.85 tok/s | **4.9x** |

### Multi-Turn (5-turn conversation, GPU)

| Metric | Standard | TurboQuant K8/V4 |
|--------|:---:|:---:|
| Avg Speed | 46.21 tok/s | 46.25 tok/s (**+0.1%**) |
| Turn 4 (worst) | 46.28 tok/s | 46.14 tok/s (-0.3%) |
| Compression | 1.0x | 7.76x (constant) |
| Avg MSE | 0.000 | 0.010 |
| Output Quality | baseline | identical responses |

With the C++ backend, multi-turn overhead is **eliminated** — compress/decompress adds only ~30ms per turn. The old Python backend had -6.2% average overhead (up to -15% on later turns).

---

## Configuration

### YAML (`config/default.yaml`)

```yaml
model:
  name: "qwen3.5-35b-a3b"                              # Registry key shown in /v1/models
  path: "./models/Qwen3.5-35B-A3B-q4_k_m.gguf"        # Path to GGUF file
  n_ctx: 4096
  n_gpu_layers: -1      # -1 = auto-detect from VRAM, 0 = all CPU, N = exactly N layers
  chat_format: "chatml"

inference_mode: "turboquant"   # "standard" or "turboquant"

kv_cache:
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  flash_attention: true

turboquant:
  preset: "quality"       # quality | aggressive | symmetric
  block_size: 128          # PolarQuant block size

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8000"

logging:
  level: "INFO"
```

> **`n_gpu_layers: -1` (auto-detect):** At startup, TurboQuant-X reads the GGUF binary header to get architecture and layer count, queries free VRAM via `nvidia-smi`, and computes the optimal number of layers that fit in GPU memory (leaving 10% headroom for compute). The resolved value is logged and exposed in `/health` → `layer_distribution`.\n\n> **LLaMA-2-70B on 8 GB VRAM:** With 80 layers and a 37 GB model, only ~12 layers fit on GPU (15%). For better throughput, tune `n_threads` (set to physical core count, e.g. 14) and `n_batch: 1024`. Reduce `n_ctx` to `2048` to free 1–2 more GPU layers:\n> ```yaml\n> model:\n>   name: \"llama-2-70b-chat\"\n>   path: \"./models/llama-2-70b-chat.Q4_K_S.gguf\"\n>   n_ctx: 2048              # Halved from 4096 → fits 1-2 more GPU layers\n>   n_gpu_layers: -1         # Auto: resolves to ~13 layers\n>   chat_format: \"llama-2\"\n>   n_threads: 14            # Physical core count (28 logical / 2)\n>   n_batch: 1024            # Larger batch = faster prompt phase\n> ```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TURBOQUANT_MODEL_PATH` | Path to GGUF model file | `./models/Qwen3.5-35B-A3B-q4_k_m.gguf` |
| `TURBOQUANT_MODEL_NAME` | Model registry key | `qwen3.5-35b-a3b` |
| `TURBOQUANT_N_CTX` | Context window size | `8192` |
| `TURBOQUANT_N_GPU_LAYERS` | GPU layer offload (-1=auto-detect, 0=CPU, N=exact) | `-1` |
| `TURBOQUANT_N_THREADS` | CPU threads for generation (-1=auto: cpu_count//2) | `-1` |
| `TURBOQUANT_N_BATCH` | Prompt eval batch size (1024 recommended for 70B) | `512` |
| `TURBOQUANT_CACHE_TYPE_K` | KV cache K type | `q8_0` |
| `TURBOQUANT_CACHE_TYPE_V` | KV cache V type | `q8_0` |
| `TURBOQUANT_HOST` | Server bind address | `0.0.0.0` |
| `TURBOQUANT_PORT` | Server port | `8000` |
| `TURBOQUANT_LOG_LEVEL` | Logging level | `INFO` |
| `TURBOQUANT_INFERENCE_MODE` | Inference mode | `standard` |
| `TURBOQUANT_PRESET` | TurboQuant preset | `quality` |

### CLI Arguments

```bash
python -m src.main [OPTIONS]

  --config PATH    YAML config file (default: config/default.yaml)
  --host HOST      Server bind address
  --port PORT      Server port
  --mode MODE      standard | turboquant
  --preset PRESET  quality | aggressive | symmetric
```

**Precedence:** CLI args > environment variables > YAML config > code defaults

---

## Project Structure

```
turboquant-x/
├── config/
│   └── default.yaml              # Server configuration
├── models/                       # GGUF model files (.gitignored)
├── scripts/
│   └── download_model.sh         # Model downloader
├── docs/
│   └── ARCHITECTURE.md           # Full architecture documentation
├── src/
│   ├── main.py                   # CLI entry point + config loading
│   ├── engine/
│   │   ├── inference.py          # LLM inference engine (llama-cpp-python)
│   │   ├── kv_cache.py           # KV cache config + memory estimation
│   │   ├── model_config.py       # Model config + auto-selection
│   │   └── turbo_engine.py       # TurboQuant-enhanced engine wrapper
│   ├── server/
│   │   ├── app.py                # FastAPI application factory + lifespan
│   │   ├── routes.py             # API routes (chat, health, models)
│   │   └── schemas.py            # Pydantic request/response models
│   ├── turboquant/               # Python compression pipeline
│   │   ├── compressor.py         # Main compressor (auto-selects C++ backend)
│   │   ├── polar_quant.py        # PolarQuant block + tensor operations
│   │   ├── rotation.py           # Walsh-Hadamard transform + random signs
│   │   ├── codebook.py           # Lloyd-Max codebook generation
│   │   ├── asymmetric.py         # Named presets (Quality/Aggressive/Symmetric)
│   │   └── boundary.py           # Boundary layer protection
│   ├── turboquant_cpp/           # C++ acceleration backend (optional)
│   │   ├── __init__.py           # Auto-import with CPP_AVAILABLE flag
│   │   ├── rotation.h/cpp        # Fast WHT O(d log d) butterfly + OpenMP
│   │   ├── codebook.h/cpp        # Precomputed Lloyd-Max centroids
│   │   ├── polar_quant.h/cpp     # Block + tensor compress (OpenMP parallel)
│   │   └── bindings.cpp          # pybind11 → Python interface
│   ├── utils/
│   │   ├── memory.py             # GPU memory monitoring (pynvml / nvidia-smi)
│   │   └── gpu_layers.py         # GGUF binary parser + optimal GPU layer calculator
│   └── static/
│       └── chat.html             # Sci-Fi HUD browser chatbot (served at GET /)
├── tests/                        # 536 tests (80%+ coverage)
├── benchmarks/                   # Performance benchmarks
│   ├── benchmark_compare.py      # Standard vs TurboQuant comparison
│   ├── benchmark_multiturn.py    # Multi-turn conversation
│   ├── benchmark_niah.py         # Needle-in-a-haystack quality
│   ├── benchmark_ppl.py          # Perplexity measurement
│   ├── benchmark_speed.py        # Token throughput
│   └── generate_report.py        # Aggregate report generator
├── CMakeLists.txt                # C++ backend build (cmake)
└── pyproject.toml                # Python packaging + test config
```

---

## Testing

```bash
# All 536 tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Specific modules
pytest tests/test_compressor.py -v
pytest tests/test_turbo_engine.py -v
pytest tests/test_api.py -v
```

Coverage minimum: **80%** (enforced in `pyproject.toml`).

## Benchmarks

```bash
# Standard vs TurboQuant (GPU)
python -m benchmarks.benchmark_compare --runs 2 --max-tokens 64 --n-ctx 4096 --n-gpu-layers -1

# Standard vs TurboQuant (CPU only)
python -m benchmarks.benchmark_compare --runs 2 --max-tokens 64 --n-ctx 4096 --n-gpu-layers 0

# Multi-turn conversation
python -m benchmarks.benchmark_multiturn --turns 5 --max-tokens 128 --n-gpu-layers -1 --n-ctx 4096

# KV cache compression quality (Compression, Cosine Sim, Top-1/Top-5 Match)
# No model loading required — runs in ~25s on CPU
python -m benchmarks.benchmark_quality
python -m benchmarks.benchmark_quality --n-layers 40 --seq-len 512 --trials 5 --json benchmarks/results/quality_report.json

# Perplexity impact
python -m benchmarks.benchmark_ppl --n-ctx 4096

# Generate full report
python -m benchmarks.generate_report
```

Results → `benchmarks/results/BENCHMARK_REPORT.md`

## Documentation

Full architecture documentation: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Covers CPU/GPU inference modes, TurboQuant compression pipeline internals, C++ engine algorithms (fast WHT, OpenMP, Lloyd-Max), server architecture, and complete configuration reference.

## License

Private — all rights reserved.
