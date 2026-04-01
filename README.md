# TurboQuant-X

Local LLM inference server with **7.76x KV cache compression** and near-zero speed overhead. Runs quantized GGUF models on GPU or CPU with a FastAPI server exposing an OpenAI-compatible chat API.

## Key Results

| | Standard (Q8_0) | TurboQuant K8/V4 |
|---|:---:|:---:|
| **GPU Speed** | 44.78 tok/s | 47.10 tok/s (**+5.2%**) |
| **CPU Speed** | 8.3 tok/s | 8.6 tok/s (+3.2%) |
| **KV Cache RAM** | 281 MB | 35 MB (**-87%**) |
| **Compression** | 1.0x | **7.76x** |
| **Compress / Decompress** | — | **31 ms / 26 ms** |
| **Quality (MSE)** | 0.000 | 0.010 |

*Tested on RTX 4060 Laptop (8 GB), Ryzen 9 8945H, Qwen2.5-7B Q4_K_M, 4096 context, C++ backend enabled*

## Features

- **TurboQuant KV cache compression** — asymmetric K/V precision (K8/V4 default) via PolarQuant pipeline (WHT rotation + Lloyd-Max codebook quantization)
- **C++ acceleration backend** — 8x faster compression via fast Walsh-Hadamard O(d log d) + OpenMP threading — makes TurboQuant **faster** than standard inference (auto-detected, optional)
- **OpenAI-compatible API** — drop-in `/v1/chat/completions` endpoint with streaming (SSE) support
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

```bash
./scripts/download_model.sh
```

This downloads **Qwen2.5-7B-Instruct Q4_K_M** (~4.4 GB) to `models/`. You can also download manually from HuggingFace and place the `.gguf` file in `models/`.

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
2026-04-01 20:37:50 [INFO] Model: qwen2.5-7b-instruct
2026-04-01 20:37:50 [INFO] Inference mode: turboquant
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
  "model": "qwen2.5-7b-instruct",
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
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

#### Request Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `messages` | array | required | 1–100 messages |
| `max_tokens` | int | 512 | 1–4096 |
| `temperature` | float | 0.7 | 0.0–2.0 |
| `top_p` | float | 0.95 | 0.0–1.0 |
| `stream` | bool | false | — |

### Other Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health, model status, GPU memory, TurboQuant config |
| `GET` | `/v1/models` | List loaded models |

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

---

## Compression Presets

| Preset | K-bits | V-bits | Compression | MSE | Speed Impact | Best For |
|--------|:---:|:---:|:---:|:---:|:---:|----------|
| **Quality** | 8 | 4 | 7.76x | 0.010 | -1.5% (GPU) | Production default — best quality/speed tradeoff |
| **Aggressive** | 8 | 2 | 7.76x | 0.124 | +1.6% (GPU) | Memory-constrained, simple tasks |
| **Symmetric** | 4 | 4 | 7.53x | 0.021 | +1.8% (GPU) | Research, balanced K/V compression |

**Why asymmetric K/V?** Keys are used in attention score computation (dot product + softmax amplifies errors). Values are just weighted-averaged — quantization noise averages out. Keeping K at 8-bit while compressing V to 4-bit gives optimal quality/compression.

---

## C++ Turbo Engine

Optional C++ backend that accelerates the PolarQuant compression pipeline:

| Operation | Python (NumPy) | C++ Backend | Speedup |
|-----------|:---:|:---:|:---:|
| Compress (K8/V4) | 254 ms | 31 ms | **8.1x** |
| Decompress (K8/V4) | 198 ms | 26 ms | **7.6x** |

Optimizations:
- **Fast Walsh-Hadamard Transform** — O(d log d) butterfly algorithm replaces O(d²) matrix multiply (18x fewer FLOPs for d=128)
- **OpenMP threading** — parallel block compression/decompression
- **Native codebook** — precomputed Lloyd-Max centroids compiled in, zero allocation overhead
- **`-O3 -march=native`** — full compiler optimization with AVX2/AVX-512

With the C++ backend, compression overhead drops from ~250ms to ~31ms per turn, making TurboQuant **strictly faster** than standard inference on all presets.

Build instructions in [Step 4](#step-4-build-c-backend-optional--8x-faster-compression) above. Without C++, everything works — just slower compression.

---

## Performance

### GPU Mode (RTX 4060 Laptop, 8 GB VRAM, C++ Backend)

| Config | Speed (tok/s) | System RAM | Compress | Decompress | Compression |
|--------|:---:|:---:|:---:|:---:|:---:|
| Standard Q8_0 | 44.78 | 281 MB | — | — | 1.0x |
| **TurboQuant K8/V4** | **47.10 (+5.2%)** | 35 MB (-87%) | 31 ms | 26 ms | **7.76x** |
| TurboQuant K8/V2 | 46.74 (+4.4%) | 26 MB (-91%) | 31 ms | 26 ms | **7.76x** |
| TurboQuant K4/V4 | 45.59 (+1.8%) | 24 MB (-91%) | 58 ms | 33 ms | **7.53x** |

### CPU Mode (Ryzen 9 8945H, 16 threads)

| Config | Speed (tok/s) | System RAM | Compression |
|--------|:---:|:---:|:---:|
| Standard Q8_0 | 8.30 | 410 MB | 1.0x |
| **TurboQuant K8/V4** | 8.56 (+3.2%) | 162 MB (-60%) | **7.76x** |
| TurboQuant K8/V2 | 8.78 (+5.8%) | 158 MB (-61%) | **7.76x** |

> TurboQuant is **faster** on CPU because the smaller compressed KV cache improves L2/L3 cache utilization.

### GPU vs CPU

| Mode | CPU | GPU | GPU Speedup |
|------|:---:|:---:|:---:|
| Standard Q8_0 | 8.30 tok/s | 44.78 tok/s | **5.4x** |
| TurboQuant K8/V4 | 8.56 tok/s | 47.10 tok/s | **5.5x** |

### Multi-Turn (5-turn conversation, GPU)

| Metric | Standard | TurboQuant K8/V4 |
|--------|:---:|:---:|
| Avg Speed | 46.9 tok/s | 44.0 tok/s (-6.2%) |
| Compression | 1.0x | 7.76x (constant) |
| Output Quality | baseline | identical responses |

Multi-turn overhead is from compress/decompress between turns. With the C++ backend, this drops to ~31ms compress + ~26ms decompress per turn.

---

## Configuration

### YAML (`config/default.yaml`)

```yaml
model:
  name: "qwen2.5-7b-instruct"
  path: "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
  n_ctx: 8192
  n_gpu_layers: -1      # -1 = all GPU, 0 = all CPU
  chat_format: "chatml"

inference_mode: "standard"   # "standard" or "turboquant"

kv_cache:
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  flash_attention: true

turboquant:
  k_bits: 8               # Key cache: 2, 3, 4, or 8
  v_bits: 4               # Value cache: 2, 3, 4, or 8
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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TURBOQUANT_MODEL_PATH` | Path to GGUF model file | `./models/qwen2.5-7b-instruct-q4_k_m.gguf` |
| `TURBOQUANT_MODEL_NAME` | Model registry key | `qwen2.5-7b-instruct` |
| `TURBOQUANT_N_CTX` | Context window size | `8192` |
| `TURBOQUANT_N_GPU_LAYERS` | GPU layer offload (-1=all, 0=CPU) | `-1` |
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
│   └── utils/
│       └── memory.py             # GPU memory monitoring (pynvml / nvidia-smi)
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
