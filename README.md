# TurboQuant-X

TurboQuant-powered LLM inference server with KV cache compression. Runs quantized GGUF models locally with a FastAPI server exposing an OpenAI-compatible chat completions API.

## Features

- **TurboQuant KV cache compression** — asymmetric K/V precision (K8/V4 default) using PolarQuant pipeline (rotation + codebook quantization)
- **OpenAI-compatible API** — drop-in `/v1/chat/completions` endpoint with streaming (SSE) support
- **llama.cpp backend** — GPU-accelerated inference via llama-cpp-python
- **YAML + env var config** — layered configuration with environment variable overrides
- **Benchmarking suite** — speed, perplexity, multi-turn, and needle-in-a-haystack benchmarks

## Requirements

- Python >= 3.11
- NVIDIA GPU with CUDA support (recommended, 8GB+ VRAM for 7B models)
- A GGUF model file (default: Qwen2.5-7B-Instruct Q4_K_M)

## Setup

### 1. Create virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 2. Install dependencies

```bash
# Core dependencies
pip install -e .

# Development (pytest, coverage, httpx)
pip install -e ".[dev]"

# GPU inference (llama-cpp-python, pynvml, huggingface-hub)
pip install -e ".[gpu]"

# Benchmarking (datasets, matplotlib, tabulate)
pip install -e ".[bench]"

# All extras
pip install -e ".[dev,gpu,bench]"
```

> **Note:** `llama-cpp-python` requires a C++ compiler and CMake. For CUDA support, install with:
> ```bash
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
> ```

### 3. Download a model

```bash
./scripts/download_model.sh
```

This downloads **Qwen2.5-7B-Instruct Q4_K_M** (~4.4GB) to `models/`.

## Usage

### Start the server

```bash
# Standard inference (no TurboQuant compression)
python -m src.main

# TurboQuant with default "quality" preset (K8/V4)
python -m src.main --mode turboquant

# TurboQuant with a named preset
python -m src.main --preset quality      # K8/V4 — best quality/speed tradeoff
python -m src.main --preset aggressive   # K8/V2 — max compression, lower quality
python -m src.main --preset symmetric    # K4/V4 — equal K/V precision
```

> Using `--preset` automatically enables TurboQuant mode.

With a custom config:

```bash
python -m src.main --config config/dev.yaml
```

With environment variable overrides:

```bash
TURBOQUANT_PORT=9000 TURBOQUANT_N_CTX=4096 python -m src.main
TURBOQUANT_PRESET=aggressive python -m src.main --mode turboquant
```

### Send a message

#### Non-streaming request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, what is TurboQuant?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'
```

#### Streaming request (SSE)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain KV cache compression in 3 sentences."}
    ],
    "max_tokens": 256,
    "stream": true
  }'
```

#### Health check

```bash
curl http://localhost:8000/health
```

#### List models

```bash
curl http://localhost:8000/v1/models
```

#### Using the OpenAI Python client

The API is OpenAI-compatible, so you can use the official client:

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

#### Request parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `messages` | array | required | 1–100 messages |
| `max_tokens` | int | 512 | 1–4096 |
| `temperature` | float | 0.7 | 0.0–2.0 |
| `top_p` | float | 0.95 | 0.0–1.0 |
| `stream` | bool | false | — |

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TURBOQUANT_MODEL_PATH` | Path to GGUF model file | `./models/qwen2.5-7b-instruct-q4_k_m.gguf` |
| `TURBOQUANT_N_CTX` | Context window size | `8192` |
| `TURBOQUANT_HOST` | Server bind address | `0.0.0.0` |
| `TURBOQUANT_PORT` | Server port | `8000` |
| `TURBOQUANT_LOG_LEVEL` | Logging level | `INFO` |
| `TURBOQUANT_INFERENCE_MODE` | Inference mode (`standard` / `turboquant`) | `standard` |
| `TURBOQUANT_PRESET` | TurboQuant preset (`quality` / `aggressive` / `symmetric`) | `quality` |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Server health check |

The chat completions endpoint is compatible with the OpenAI API format.

## Configuration

Edit `config/default.yaml`:

```yaml
model:
  name: "qwen2.5-7b-instruct"
  path: "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
  n_ctx: 8192
  n_gpu_layers: -1      # -1 = offload all layers to GPU
  chat_format: "chatml"

inference_mode: "standard"   # "standard" or "turboquant"

kv_cache:
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  flash_attention: true

# TurboQuant presets: "quality" (K8/V4), "aggressive" (K8/V2), "symmetric" (K4/V4)
turboquant:
  preset: "quality"
  block_size: 128

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
```

## Project Structure

```
turboquant-x/
├── config/
│   └── default.yaml          # Server configuration
├── models/                   # GGUF model files
├── scripts/
│   └── download_model.sh     # Model downloader
├── src/
│   ├── main.py               # Entry point
│   ├── engine/
│   │   ├── inference.py      # LLM inference engine
│   │   ├── kv_cache.py       # KV cache config and memory estimation
│   │   ├── model_config.py   # Model configuration
│   │   └── turbo_engine.py   # TurboQuant-enhanced engine
│   ├── server/
│   │   ├── app.py            # FastAPI application factory
│   │   ├── routes.py         # API routes
│   │   └── schemas.py        # Pydantic request/response models
│   ├── turboquant/
│   │   ├── asymmetric.py     # Asymmetric K/V quantization
│   │   ├── boundary.py       # Boundary detection
│   │   ├── codebook.py       # Codebook quantization
│   │   ├── compressor.py     # Main compressor (MSE-only)
│   │   ├── polar_quant.py    # PolarQuant pipeline
│   │   └── rotation.py       # Rotation matrices
│   └── utils/
│       └── memory.py         # GPU memory utilities
├── tests/                    # Unit and integration tests
├── benchmarks/               # Performance benchmarks
│   ├── benchmark_compare.py  # Standard vs TurboQuant comparison
│   ├── benchmark_multiturn.py
│   ├── benchmark_niah.py     # Needle-in-a-haystack
│   ├── benchmark_ppl.py      # Perplexity
│   ├── benchmark_speed.py    # Token throughput
│   └── generate_report.py
└── pyproject.toml
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_compressor.py -v
```

Coverage minimum: **80%** (enforced in `pyproject.toml`).

## Benchmarks

```bash
# Compare standard vs TurboQuant inference
python -m benchmarks.benchmark_compare --runs 2 --max-tokens 64 --n-ctx 4096 --n-gpu-layers -1

# Multi-turn conversation benchmark
python -m benchmarks.benchmark_multiturn --turns 5 --max-tokens 128 --n-gpu-layers -1 --n-ctx 4096

# Generate full report
python -m benchmarks.generate_report
```

Results are written to `benchmarks/results/`.

## License

Private — all rights reserved.
