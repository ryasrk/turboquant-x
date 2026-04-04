# TurboQuant-X

TurboQuant-X is a FastAPI-based LLM inference server for local GGUF models and cloud providers. It combines OpenAI-compatible chat APIs, KV-cache compression research, a browser chat UI, runtime model and provider switching, agent tools, document generation, and reverse-proxy-friendly deployment.

It supports two broad operating modes:

- Local inference through llama.cpp with `standard`, `turboquant`, `zero-quant`, and `ultra-quant` execution modes
- Cloud inference through provider adapters for OpenAI, NVIDIA, Anthropic, Moonshot, Zhipu, DeepSeek, and Groq

## What This Repository Contains

- FastAPI server with chat, health, model switching, cloud switching, auth, sessions, and document download endpoints
- Browser chat UI at `/` with streaming responses, settings, auth overlay, sessions, model/provider selection, and agent mode
- TurboQuant KV-cache compression pipeline plus optional C++ acceleration backend
- Agent tool system with file, code, shell, SQL, RAG, web, memory, MCP, and document-generation tools
- Benchmark suite for speed, quality, perplexity, multi-turn, and needle-in-a-haystack evaluation

## Key Capabilities

- OpenAI-compatible `POST /v1/chat/completions` API with streaming and non-streaming responses
- Runtime switching between local inference modes and cloud providers
- Runtime switching between GGUF model files in `models/`
- JWT auth plus SQLite-backed chat session persistence
- Document generation tools for Word, PDF, and CSV output with download links
- Reverse-proxy-safe frontend paths and downloadable documents
- Optional cloud model browsing for supported providers
- Browser UI with self-hosted KaTeX assets

## Repository Layout

```text
turboquant-x/
├── config/                # default.yaml, cloud.yaml, runtime secrets in config/.env
├── docs/                  # architecture and deeper technical documentation
├── benchmarks/            # benchmarking scripts and report generation
├── models/                # local GGUF model files
├── scripts/               # helper scripts such as model download
├── src/
│   ├── agent/             # tool definitions and agent loop support
│   ├── engine/            # local inference, cloud providers, compression engines
│   ├── server/            # FastAPI app, routes, auth, database
│   ├── static/            # browser UI assets
│   ├── turboquant/        # compression pipeline code
│   ├── turboquant_cpp/    # optional compiled extension target
│   └── utils/             # doctor, GPU helpers, memory utilities
├── tests/                 # API, engine, config, tool, and compression tests
├── ngrok_run.sh           # helper to run server + ngrok tunnel
├── pyproject.toml         # package metadata and dependencies
└── README.md
```

## Requirements

| Component | Minimum | Recommended |
|---|---:|---:|
| Python | 3.11 | 3.12 |
| RAM | 8 GB | 16 GB+ |
| GPU | optional | NVIDIA CUDA GPU with 8 GB+ VRAM |
| Disk | 5 GB+ | model-dependent |
| OS | Linux/macOS | Linux |

Optional native build tools for the C++ backend:

- `cmake >= 3.20`
- `g++ >= 11` or `clang >= 14`
- `pybind11`

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 2. Install dependencies

Base install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e ".[dev]"
```

Benchmarking extras:

```bash
pip install -e ".[bench]"
```

GPU path with llama.cpp CUDA build:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -e ".[gpu]"
```

If you are running CPU-only, install `llama-cpp-python` without the CUDA flag.

## Configuration

TurboQuant-X uses layered configuration:

1. `config/default.yaml`
2. `config/cloud.yaml`
3. environment variables
4. CLI flags

### Important config files

- `config/default.yaml`: server, local inference, mode, KV-cache, and general defaults
- `config/cloud.yaml`: cloud provider defaults, model names, and provider-specific settings
- `config/.env`: secrets and server/runtime environment overrides actually loaded by the app
- `.env.example`: example shell-style variables for helper scripts such as `ngrok_run.sh`

### Server secrets and env vars

The application loads runtime secrets from `config/.env`, not from the repository root `.env`.

Typical entries:

```dotenv
TQ_JWT_SECRET=replace-with-a-random-secret
TURBOQUANT_CORS_ORIGINS=https://your-domain.example
NGROK_DOMAIN=your-subdomain.ngrok-free.dev

TURBOQUANT_CLOUD_OPENAI_API_KEY=
TURBOQUANT_CLOUD_NVIDIA_API_KEY=
TURBOQUANT_CLOUD_ANTHROPIC_API_KEY=
TURBOQUANT_CLOUD_MOONSHOT_API_KEY=
TURBOQUANT_CLOUD_ZHIPU_API_KEY=
TURBOQUANT_CLOUD_DEEPSEEK_API_KEY=
TURBOQUANT_CLOUD_GROQ_API_KEY=
```

### Useful environment overrides

| Variable | Purpose |
|---|---|
| `TURBOQUANT_HOST` | bind host |
| `TURBOQUANT_PORT` | bind port |
| `TURBOQUANT_MODEL_PATH` | override GGUF model path |
| `TURBOQUANT_N_CTX` | override context length |
| `TURBOQUANT_N_GPU_LAYERS` | override GPU layer count |
| `TURBOQUANT_INFERENCE_MODE` | force startup mode |
| `TURBOQUANT_CORS_ORIGINS` | comma-separated extra CORS origins |
| `TQ_JWT_SECRET` | JWT signing secret |
| `TURBOQUANT_DOC_OUTPUT` | document output directory |

## Running The Server

### Start locally

```bash
env/bin/python3 -m src.main
```

Examples:

```bash
env/bin/python3 -m src.main --host 0.0.0.0 --port 8000
env/bin/python3 -m src.main --mode turboquant
env/bin/python3 -m src.main --mode zero-quant
env/bin/python3 -m src.main --mode ultra-quant
```

### Download a model

Small helper script:

```bash
./scripts/download_model.sh
```

Or place your `.gguf` files directly in `models/`.

### Browser UI

Once the server is up, open:

```text
http://localhost:8000/
```

The UI includes:

- local/cloud toggle
- mode switching
- model and provider switching
- auth overlay
- session sidebar
- agent mode
- streamed answers
- tool call visibility controls

## Local Inference Modes

Local inference is backed by llama.cpp and can be switched at runtime.

| Mode | Purpose |
|---|---|
| `standard` | baseline local inference without Python-side KV compression |
| `turboquant` | TurboQuant compression flow |
| `zero-quant` | depth-adaptive compression |
| `ultra-quant` | memory-budget-aware large-model configuration |
| `cloud` | routes completions to the active cloud provider |

Runtime mode switching endpoint:

```text
POST /v1/switch-mode
```

Runtime local model switching endpoint:

```text
POST /v1/switch-model
```

## Cloud Inference

Cloud provider defaults live in `config/cloud.yaml`.

Configured providers in this repo:

- OpenAI
- NVIDIA NIM
- Anthropic
- Moonshot
- Zhipu
- DeepSeek
- Groq

Cloud flow:

1. Put API keys in `config/.env`
2. Start the server
3. Switch to cloud mode from the UI or API
4. Optionally change provider and cloud model at runtime

Relevant endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /v1/cloud-providers` | list configured and active providers |
| `GET /v1/cloud-providers/{provider}/models` | list models when supported |
| `POST /v1/switch-provider` | switch active cloud provider |
| `POST /v1/switch-cloud-model` | switch model on the active provider |

Example:

```bash
curl -X POST http://localhost:8000/v1/switch-provider \
  -H "Content-Type: application/json" \
  -d '{"provider":"zhipu"}'
```

## Reverse Proxy And ngrok

The application is set up to work behind a reverse proxy:

- frontend uses relative URLs
- document download links are relative
- FastAPI trusts forwarded headers
- extra proxy origins can be added with `TURBOQUANT_CORS_ORIGINS`

### ngrok helper

Run both the server and ngrok tunnel:

```bash
./ngrok_run.sh
```

Other modes:

```bash
./ngrok_run.sh --server
./ngrok_run.sh --ngrok
```

The helper script reads root-level `.env` for ngrok settings. The application itself loads runtime secrets from `config/.env`.

Recommended reverse-proxy settings in `config/.env`:

```dotenv
TURBOQUANT_CORS_ORIGINS=https://your-public-domain.example
TQ_JWT_SECRET=replace-with-a-random-secret
```

## API Overview

### Chat API

Primary completion endpoint:

```text
POST /v1/chat/completions
```

Supports:

- OpenAI-style `messages`
- streaming responses
- `max_tokens`, `temperature`, `top_p`
- agent mode via `tools`
- local or cloud routing depending on active inference mode

Example:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Explain TurboQuant in 3 sentences."}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

### Health and model endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | health, model status, inference mode, context, GPU memory |
| `GET /v1/models` | OpenAI-style model list |
| `GET /v1/available-models` | GGUF files in `models/` |
| `POST /v1/switch-model` | switch local GGUF model |
| `POST /v1/switch-mode` | switch inference mode |

### Auth and sessions

Auth uses JWT in the `Authorization: Bearer ...` header.

| Endpoint | Purpose |
|---|---|
| `POST /v1/auth/register` | create a user and return a token |
| `POST /v1/auth/login` | log in and return a token |
| `GET /v1/auth/me` | validate token and fetch current user |
| `GET /v1/sessions` | list chat sessions |
| `POST /v1/sessions` | create a chat session |
| `PATCH /v1/sessions/{id}` | rename a session |
| `DELETE /v1/sessions/{id}` | delete a session |
| `GET /v1/sessions/{id}/messages` | load session messages |
| `POST /v1/sessions/{id}/messages` | save a message |

## Agent Tools

TurboQuant-X includes a server-side tool registry for agent mode. Tool families visible in `src/agent/tools/` include:

- code tools
- data tools
- diff tool
- document tools
- file tools
- MCP bridge tools
- memory tools
- RAG tools
- shell and terminal tools
- SQL tools
- system and sysinfo tools
- web search and web tools

Agent metadata endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /v1/agent/tools` | list registered tools |
| `POST /v1/agent/approve-tool` | approve a pending tool action |
| `POST /v1/agent/mcp/reload` | reload MCP-backed tools |

## Document Generation

The repository includes document-generation tools for Word, PDF, and CSV output.

Available tool names:

- `generate_word`
- `generate_pdf`
- `generate_csv`

Generated files are stored under `data/generated_docs/` by default and served by:

| Endpoint | Purpose |
|---|---|
| `GET /v1/documents/list` | list generated files |
| `GET /v1/documents/download/{filename}` | download a generated file |

The download links returned by tools are relative, so they work through reverse proxies.

## Optional C++ Backend

TurboQuant can use a compiled C++ backend for faster compression.

Build steps:

```bash
pip install pybind11

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(env/bin/python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build build -j$(nproc)
cp build/_turboquant_cpp.cpython-*.so src/turboquant_cpp/
```

Verify:

```bash
env/bin/python3 -c "from src.turboquant_cpp import CPP_AVAILABLE; print(CPP_AVAILABLE)"
```

## Benchmarking

The `benchmarks/` directory contains scripts for:

- all-mode comparison
- speed benchmarking
- quality benchmarking
- perplexity benchmarking
- multi-turn evaluation
- needle-in-a-haystack evaluation
- report generation

Examples:

```bash
env/bin/python3 benchmarks/benchmark_speed.py
env/bin/python3 benchmarks/benchmark_quality.py
env/bin/python3 benchmarks/generate_report.py
```

## Testing

Run the full test suite:

```bash
env/bin/python3 -m pytest tests/
```

Verified subset used during recent changes:

```bash
env/bin/python3 -m pytest \
  tests/test_document_tools.py \
  tests/test_api.py \
  tests/test_main.py \
  tests/test_schemas.py \
  tests/test_model_config.py -q
```

## Architecture And Internal Docs

For deeper implementation detail, see:

- `docs/ARCHITECTURE.md`

That document goes deeper into:

- inference architecture
- TurboQuant pipeline internals
- Zero-Quant behavior
- CPU and GPU execution paths
- C++ backend design

## Troubleshooting

### The server ignores my root `.env`

That is expected for application secrets. Put runtime secrets in `config/.env`.

### Cloud mode says no provider is configured

Add the relevant `TURBOQUANT_CLOUD_<PROVIDER>_API_KEY` to `config/.env`, then restart the server.

### Reverse proxy works but downloads break

Generated document links are relative. If downloads still fail, verify:

- `TURBOQUANT_CORS_ORIGINS` includes your public origin
- your proxy forwards standard `X-Forwarded-*` headers
- the generated file exists in `data/generated_docs/`

### I get auth failures after changing JWT secret

Existing browser tokens become invalid. Log out or clear local storage and log in again.

### The app starts but no local model loads

Check that:

- a `.gguf` file exists in `models/`
- `TURBOQUANT_MODEL_PATH` points to a real file if set
- llama.cpp dependencies are installed in the active virtual environment

## Maintenance Note

This README is intended to be the top-level operational reference for the repository. Update it when API routes, setup steps, runtime config, or deployment behavior changes.