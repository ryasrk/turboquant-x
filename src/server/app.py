"""FastAPI application factory for TurboQuant chat server."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.engine.inference import InferenceEngine
from src.engine.kv_cache import KVCacheConfig
from src.engine.model_config import ModelConfig
from src.server.schemas import ErrorResponse

logger = logging.getLogger(__name__)

# Module-level engine reference (set during lifespan)
_engine: InferenceEngine | None = None
_start_time: float = 0.0


def get_engine() -> InferenceEngine:
    """Get the global inference engine. Raises RuntimeError if not initialized."""
    if _engine is None:
        raise RuntimeError("Inference engine not initialized")
    return _engine


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
    global _engine, _start_time
    _start_time = time.monotonic()

    model_config: ModelConfig = app.state.model_config
    kv_config: KVCacheConfig = app.state.kv_config

    _engine = InferenceEngine(model_config, kv_config)

    try:
        _engine.load_model()
        logger.info("Model loaded successfully: %s", model_config.model_name)
    except Exception as e:
        logger.warning("Failed to load model: %s. Running in degraded mode.", e)
        # Don't re-raise — server starts in degraded mode

    yield

    # Shutdown
    if _engine is not None:
        _engine.unload()
        _engine = None
    logger.info("Server shutdown complete")


def create_app(
    model_config: ModelConfig | None = None,
    kv_config: KVCacheConfig | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_config: Model configuration. Defaults to env-based config.
        kv_config: KV cache configuration. Defaults to TurboQuant recommended.
        cors_origins: Allowed CORS origins. Defaults to localhost only.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="TurboQuant Chat API",
        description="LLM chat server with TurboQuant KV cache compression",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config in app.state for lifespan access
    app.state.model_config = model_config or ModelConfig(
        model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf"
    )
    app.state.kv_config = kv_config or KVCacheConfig()

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

    return app
