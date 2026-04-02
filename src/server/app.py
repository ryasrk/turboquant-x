"""FastAPI application factory for TurboQuant-X server."""

from __future__ import annotations

import logging
import time
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


# Module-level engine references (set during lifespan)
_engine: InferenceEngine | None = None
_turbo_engine: Any = None  # TurboQuantEngine | ZeroQuantEngine | None
_inference_mode: InferenceMode = InferenceMode.STANDARD
_start_time: float = 0.0


def get_engine() -> InferenceEngine:
    """Get the global inference engine. Raises RuntimeError if not initialized."""
    if _engine is None:
        raise RuntimeError("Inference engine not initialized")
    return _engine


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load model on startup, unload on shutdown.

    The model config is read from the app.state attributes set by create_app().
    If model loading fails (e.g., no GPU, missing model file), the server
    starts in degraded mode — health endpoint reports unhealthy, chat returns 503.
    """
    global _engine, _turbo_engine, _inference_mode, _start_time
    _start_time = time.monotonic()

    model_config: ModelConfig = app.state.model_config
    kv_config: KVCacheConfig = app.state.kv_config
    _inference_mode = app.state.inference_mode

    if _inference_mode == InferenceMode.TURBOQUANT:
        try:
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
            logger.info("Using TurboQuant inference mode (K=%d-bit, V=%d-bit)",
                        quant_config.k_bits, quant_config.v_bits)
        except Exception as e:
            logger.warning(
                "Failed to create TurboQuant engine: %s. Falling back to standard mode.", e
            )
            _inference_mode = InferenceMode.STANDARD
            _turbo_engine = None
            _engine = InferenceEngine(model_config, kv_config)
    elif _inference_mode == InferenceMode.ZERO_QUANT:
        try:
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
            avg_bits = zq_config.average_bits(28)  # rough estimate for log
            logger.info(
                "Using Zero-Quant inference mode (depth-adaptive, avg %.1f bits/value)",
                avg_bits,
            )
        except Exception as e:
            logger.warning(
                "Failed to create Zero-Quant engine: %s. Falling back to standard mode.", e
            )
            _inference_mode = InferenceMode.STANDARD
            _turbo_engine = None
            _engine = InferenceEngine(model_config, kv_config)
    else:
        _engine = InferenceEngine(model_config, kv_config)
        logger.info("Using standard inference mode")

    try:
        if _turbo_engine is not None:
            _turbo_engine.load_model()
        elif _engine is not None:
            _engine.load_model()
        logger.info("Model loaded successfully: %s", model_config.model_name)
    except Exception as e:
        logger.warning("Failed to load model: %s. Running in degraded mode.", e)

    yield

    # Shutdown
    if _turbo_engine is not None:
        _turbo_engine.unload()
        _turbo_engine = None
    elif _engine is not None:
        _engine.unload()
    _engine = None
    logger.info("Server shutdown complete")


def create_app(
    model_config: ModelConfig | None = None,
    kv_config: KVCacheConfig | None = None,
    cors_origins: list[str] | None = None,
    inference_mode: InferenceMode = InferenceMode.STANDARD,
    turboquant_config: dict[str, Any] | None = None,
    zero_quant_config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_config: Model configuration. Defaults to env-based config.
        kv_config: KV cache configuration. Defaults to Q8_0/Q8_0.
        cors_origins: Allowed CORS origins. Defaults to localhost only.
        inference_mode: "standard" or "turboquant".
        turboquant_config: TurboQuant compression settings (k_bits, v_bits, block_size).

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

    # Serve chat UI at root
    _static_dir = Path(__file__).parent.parent / "static"
    if _static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_chat_ui() -> FileResponse:
            return FileResponse(str(_static_dir / "chat.html"))

    return app
