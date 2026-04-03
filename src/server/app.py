"""FastAPI application factory for TurboQuant-X server."""

from __future__ import annotations

import logging
import time
import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.engine.inference import InferenceEngine
from src.engine.kv_cache import KVCacheConfig
from src.engine.model_config import ModelConfig
from src.server.schemas import ErrorResponse

logger = logging.getLogger(__name__)


class InferenceMode(str, Enum):
    """Available inference modes."""

    STANDARD = "standard"
    TURBOQUANT = "turboquant"
    ZERO_QUANT = "zero-quant"
    ULTRA_QUANT = "ultra-quant"
    CLOUD = "cloud"


# Module-level engine references (set during lifespan)
_engine: InferenceEngine | None = None
_turbo_engine: Any = None  # TurboQuantEngine | ZeroQuantEngine | None
_cloud_engine: Any = None  # CloudEngine | None
_inference_mode: InferenceMode = InferenceMode.STANDARD
_start_time: float = 0.0
_switch_lock = threading.Lock()  # prevents concurrent mode/model switches
_loading: bool = False  # True while a model is being loaded/switched
_agent_registry: Any = None  # ToolRegistry instance


def get_engine() -> InferenceEngine:
    """Get the global inference engine. Raises RuntimeError if not initialized."""
    if _engine is None:
        raise RuntimeError("Inference engine not initialized")
    return _engine


def get_cloud_engine():
    """Get the CloudEngine (None if not in cloud mode)."""
    return _cloud_engine


def get_turbo_engine():
    """Get the TurboQuant engine (None if in standard mode)."""
    return _turbo_engine


def get_inference_mode() -> InferenceMode:
    """Get the active inference mode."""
    return _inference_mode


def get_uptime() -> float:
    """Get server uptime in seconds."""
    if _start_time == 0.0:
        return 0.0
    return time.monotonic() - _start_time


def is_loading() -> bool:
    """Return True if a model/mode switch is in progress."""
    return _loading


def get_agent_registry():
    """Get the global agent tool registry."""
    return _agent_registry


def init_agent_registry() -> None:
    """Initialize the agent tool registry with built-in tools."""
    global _agent_registry
    from src.agent.registry import ToolRegistry
    from src.agent.tools import (
        ReadFileTool, WriteFileTool, ListDirTool, FindFilesTool, ReplaceInFileTool,
        ExecTool, WebSearchTool,
        FetchWebpageTool, HttpRequestTool,
        GetEnvTool, CurrentTimeTool, SystemInfoTool,
        GrepCodeTool, PythonEvalTool, CountLinesTool,
        JsonQueryTool, CalculateTool, CsvReadTool,
        ReadPdfTool, IndexDocumentTool, SearchDocumentTool,
        SaveNoteTool, RecallNoteTool, DeleteNoteTool,
        SqlQueryTool,
        DiffFilesTool, EncodeDecodeTool,
    )

    _agent_registry = ToolRegistry()

    # File System
    _agent_registry.register(ReadFileTool())
    _agent_registry.register(WriteFileTool())
    _agent_registry.register(ListDirTool())
    _agent_registry.register(FindFilesTool())
    _agent_registry.register(ReplaceInFileTool())

    # System
    _agent_registry.register(ExecTool())
    _agent_registry.register(GetEnvTool())
    _agent_registry.register(CurrentTimeTool())
    _agent_registry.register(SystemInfoTool())

    # Code Analysis
    _agent_registry.register(GrepCodeTool())
    _agent_registry.register(PythonEvalTool())
    _agent_registry.register(CountLinesTool())

    # Data & Math
    _agent_registry.register(JsonQueryTool())
    _agent_registry.register(CalculateTool())
    _agent_registry.register(CsvReadTool())

    # RAG / Document tools (LLM fn wired lazily via engine at call time)
    _agent_registry.register(ReadPdfTool())
    _agent_registry.register(IndexDocumentTool())
    _agent_registry.register(SearchDocumentTool())

    # Memory
    _agent_registry.register(SaveNoteTool())
    _agent_registry.register(RecallNoteTool())
    _agent_registry.register(DeleteNoteTool())

    # Database
    _agent_registry.register(SqlQueryTool())

    # Utility
    _agent_registry.register(DiffFilesTool())
    _agent_registry.register(EncodeDecodeTool())

    # Web (optional deps)
    try:
        _agent_registry.register(WebSearchTool())
    except Exception:
        logger.warning("WebSearchTool unavailable (install ddgs)")
    try:
        _agent_registry.register(FetchWebpageTool())
        _agent_registry.register(HttpRequestTool())
    except Exception:
        logger.warning("Web tools unavailable (install aiohttp)")

    logger.info("Agent registry initialized with %d tools: %s", len(_agent_registry.list_tools()), _agent_registry.list_tools())


def _unload_engines() -> None:
    """Unload existing engine(s) and clear global refs."""
    global _engine, _turbo_engine, _cloud_engine
    if _cloud_engine is not None:
        _cloud_engine.unload()
        _cloud_engine = None
        _engine = None
    elif _turbo_engine is not None:
        _turbo_engine.unload()
        _turbo_engine = None
    elif _engine is not None:
        _engine.unload()
    _engine = None


def _create_and_load_engine(app: FastAPI, mode: InferenceMode) -> None:
    """Create an engine for *mode* using configs from app.state, then load it."""
    global _engine, _turbo_engine, _cloud_engine

    if mode == InferenceMode.CLOUD:
        from src.engine.cloud_engine import CloudEngine

        cloud_config = getattr(app.state, "cloud_config", None)
        if cloud_config is None:
            raise ValueError(
                "Cloud mode requires cloud provider configuration. "
                "Add a 'cloud' section to your config YAML."
            )
        _cloud_engine = CloudEngine(cloud_config)
        _cloud_engine.load_model()
        # _engine stays None in cloud mode — routes check _cloud_engine first
        _engine = None
        _turbo_engine = None
        return

    model_config: ModelConfig = app.state.model_config
    kv_config: KVCacheConfig = app.state.kv_config

    if mode == InferenceMode.TURBOQUANT:
        from src.engine.turbo_engine import TurboQuantEngine
        from src.turboquant.compressor import QuantConfig

        tq_cfg = app.state.turboquant_config
        quant_config = QuantConfig(
            k_bits=tq_cfg.get("k_bits", 8),
            v_bits=tq_cfg.get("v_bits", 4),
            block_size=tq_cfg.get("block_size", 128),
        )
        _turbo_engine = TurboQuantEngine(model_config, quant_config)
        _engine = _turbo_engine.engine
    elif mode == InferenceMode.ZERO_QUANT:
        from src.engine.zero_quant_engine import ZeroQuantEngine
        from src.turboquant.zero_quant import ZeroQuantConfig

        zq_dict = getattr(app.state, "zero_quant_config", {})
        zq_config = ZeroQuantConfig(
            shallow_fraction=zq_dict.get("shallow_fraction", 0.25),
            deep_fraction=zq_dict.get("deep_fraction", 0.25),
            shallow_k_bits=zq_dict.get("shallow_k_bits", 8),
            shallow_v_bits=zq_dict.get("shallow_v_bits", 8),
            middle_k_bits=zq_dict.get("middle_k_bits", 4),
            middle_v_bits=zq_dict.get("middle_v_bits", 2),
            deep_k_bits=zq_dict.get("deep_k_bits", 8),
            deep_v_bits=zq_dict.get("deep_v_bits", 8),
            block_size=zq_dict.get("block_size", 128),
            use_kv_coquant=zq_dict.get("use_kv_coquant", False),
        )
        _turbo_engine = ZeroQuantEngine(model_config, zq_config)
        _engine = _turbo_engine.engine
    elif mode == InferenceMode.ULTRA_QUANT:
        from src.engine.ultra_quant_engine import UltraQuantEngine, UltraQuantConfig

        uq_dict = getattr(app.state, "ultra_quant_config", {})
        uq_config = UltraQuantConfig(
            target_model_params_b=uq_dict.get("target_model_params_b", 70.0),
            max_ram_usage_fraction=uq_dict.get("max_ram_usage_fraction", 0.85),
            max_vram_usage_fraction=uq_dict.get("max_vram_usage_fraction", 0.90),
            enable_mmap=uq_dict.get("enable_mmap", True),
            enable_mlock_critical=uq_dict.get("enable_mlock_critical", True),
            enable_moe_offload=uq_dict.get("enable_moe_offload", True),
            kv_budget_mb=uq_dict.get("kv_budget_mb", 0),
            force_quant=uq_dict.get("force_quant", ""),
            zero_quant_preset=uq_dict.get("zero_quant_preset", "turbo"),
        )
        _turbo_engine = UltraQuantEngine(model_config, uq_config)
        _engine = _turbo_engine.engine
    else:
        _engine = InferenceEngine(model_config, kv_config)
        _turbo_engine = None

    # Load model
    if _turbo_engine is not None:
        _turbo_engine.load_model()
    elif _engine is not None:
        _engine.load_model()


def switch_inference_mode(app: FastAPI, new_mode: InferenceMode) -> None:
    """Switch inference mode at runtime (unloads current engine, loads new one).

    This is a blocking operation — call via ``asyncio.to_thread`` from async context.
    """
    global _inference_mode, _loading

    if new_mode == _inference_mode:
        return

    if not _switch_lock.acquire(blocking=False):
        raise RuntimeError("Another model operation is already in progress")

    try:
        _loading = True
        logger.info("Switching inference mode: %s → %s", _inference_mode.value, new_mode.value)
        _unload_engines()
        _create_and_load_engine(app, new_mode)
        _inference_mode = new_mode
        logger.info("Mode switch complete: now running %s", new_mode.value)
    finally:
        _loading = False
        _switch_lock.release()


def switch_model(app: FastAPI, model_path: str) -> dict:
    """Switch to a different GGUF model file at runtime.

    Returns dict with model info on success.
    This is a blocking operation — call via ``asyncio.to_thread`` from async context.
    """
    global _loading

    resolved = Path(model_path)
    if not resolved.is_absolute():
        resolved = Path("models") / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")

    current_config: ModelConfig = app.state.model_config
    new_path_str = str(resolved)

    if current_config.model_path == new_path_str:
        return {"status": "ok", "message": "Already using this model"}

    if not _switch_lock.acquire(blocking=False):
        raise RuntimeError("Another model operation is already in progress")

    try:
        _loading = True
        logger.info("Switching model: %s → %s", current_config.model_path, new_path_str)
        _unload_engines()

        # Build new ModelConfig preserving user settings except path/name
        stem = resolved.stem.lower()
        new_size_gb = resolved.stat().st_size / (1024**3)
        from dataclasses import replace as _replace
        new_config = _replace(
            current_config,
            model_path=new_path_str,
            model_name=stem,
            weight_size_gb=round(new_size_gb, 2),
        )
        app.state.model_config = new_config

        _create_and_load_engine(app, _inference_mode)
        logger.info("Model switch complete: %s (%.1f GB)", stem, new_size_gb)
        return {
            "status": "ok",
            "model_name": stem,
            "size_gb": round(new_size_gb, 2),
            "mode": _inference_mode.value,
        }
    finally:
        _loading = False
        _switch_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load model on startup, unload on shutdown.

    The model config is read from the app.state attributes set by create_app().
    If model loading fails (e.g., no GPU, missing model file), the server
    starts in degraded mode — health endpoint reports unhealthy, chat returns 503.
    """
    global _inference_mode, _start_time
    _start_time = time.monotonic()

    _inference_mode = app.state.inference_mode

    try:
        _create_and_load_engine(app, _inference_mode)
        model_config: ModelConfig = app.state.model_config
        logger.info("Model loaded successfully: %s (%s)", model_config.model_name, _inference_mode.value)
    except Exception as e:
        logger.warning("Failed to load model: %s. Running in degraded mode.", e)
        _inference_mode = InferenceMode.STANDARD

    try:
        init_agent_registry()
    except Exception as e:
        logger.warning("Failed to initialize agent registry: %s", e)

    # Initialize thought log for chain-of-thought capture
    try:
        from src.agent.loop import init_thought_log
        thought_path = getattr(app.state, "thought_log_path", None)
        init_thought_log(thought_path)
        logger.info("Thought log initialized: %s", thought_path or "logs/agent-thoughts.log")
    except Exception as e:
        logger.warning("Failed to initialize thought log: %s", e)

    # Initialize database
    try:
        from src.server.database import init_db
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Failed to initialize database: %s", e)

    yield

    # Shutdown
    _unload_engines()
    logger.info("Server shutdown complete")


def create_app(
    model_config: ModelConfig | None = None,
    kv_config: KVCacheConfig | None = None,
    cors_origins: list[str] | None = None,
    inference_mode: InferenceMode = InferenceMode.STANDARD,
    turboquant_config: dict[str, Any] | None = None,
    zero_quant_config: dict[str, Any] | None = None,
    ultra_quant_config: dict[str, Any] | None = None,
    thought_log_path: str | None = None,
    cloud_config: Any | None = None,
    cloud_yaml_config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_config: Model configuration. Defaults to env-based config.
        kv_config: KV cache configuration. Defaults to Q8_0/Q8_0.
        cors_origins: Allowed CORS origins. Defaults to localhost only.
        inference_mode: "standard", "turboquant", "zero-quant", "ultra-quant", or "cloud".
        turboquant_config: TurboQuant compression settings (k_bits, v_bits, block_size).
        cloud_config: CloudConfig for cloud mode.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="TurboQuant-X API",
        description="LLM inference server with TurboQuant KV cache compression",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config in app.state for lifespan access
    app.state.model_config = model_config or ModelConfig(
        model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf"
    )
    app.state.kv_config = kv_config or KVCacheConfig()
    app.state.inference_mode = inference_mode
    app.state.turboquant_config = turboquant_config or {}
    app.state.zero_quant_config = zero_quant_config or {}
    app.state.ultra_quant_config = ultra_quant_config or {}
    app.state.thought_log_path = thought_log_path
    app.state.cloud_config = cloud_config
    app.state._cloud_yaml_config = cloud_yaml_config or {}

    # CORS middleware
    origins = cors_origins or ["http://localhost:3000", "http://localhost:8000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID + timing middleware
    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.monotonic()

        response = await call_next(request)

        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
            "[%s] %s %s → %d (%.1fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Unhandled error: %s", exc, exc_info=True)
        error = ErrorResponse(
            error="Internal server error",
            detail=str(exc) if logger.isEnabledFor(logging.DEBUG) else None,
            status_code=500,
        )
        return JSONResponse(
            status_code=500,
            content=error.model_dump(),
        )

    # Import routes (may not exist during testing)
    try:
        from src.server.routes import router

        app.include_router(router)
    except ImportError:
        logger.warning("Routes module not found — running without API endpoints")

    # Auth + chat-memory routes
    try:
        from src.server.auth_routes import router as auth_router

        app.include_router(auth_router)
    except ImportError:
        logger.warning("Auth routes module not found — running without auth")

    # Serve chat UI at root
    _static_dir = Path(__file__).parent.parent / "static"
    if _static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_chat_ui() -> FileResponse:
            return FileResponse(str(_static_dir / "chat.html"))

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            from fastapi.responses import Response

            return Response(status_code=204)

    return app
