"""API routes for TurboQuant chat server."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.server.app import get_engine, get_turbo_engine, get_inference_mode, get_uptime, InferenceMode
from src.server.schemas import (
    ChatRequest,
    ChatResponse,
    ChatChoice,
    ChatMessage,
    UsageStats,
    StreamChunk,
    StreamChoice,
    StreamDelta,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    ErrorResponse,
)
from src.utils.memory import get_gpu_memory

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint — OpenAI-compatible.

    Supports both streaming and non-streaming modes.
    """
    engine = get_engine()

    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is in degraded mode.",
        )

    # Convert Pydantic messages to dicts for engine
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return _stream_response(messages, request)

    return _sync_response(messages, request)


def _sync_response(messages: list[dict], request: ChatRequest) -> ChatResponse:
    """Handle non-streaming chat completion."""
    engine = get_engine()

    response_msg, stats = engine.chat(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return ChatResponse(
        id=completion_id,
        model=engine.model_config.model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role=response_msg["role"],
                    content=response_msg["content"],
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageStats(
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            total_tokens=stats.total_tokens,
        ),
    )


def _stream_response(
    messages: list[dict], request: ChatRequest
) -> EventSourceResponse:
    """Handle streaming chat completion via SSE."""

    async def event_generator():
        engine = get_engine()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = engine.model_config.model_name

        # First chunk: role
        first_chunk = StreamChunk(
            id=completion_id,
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(role="assistant"),
                )
            ],
        )
        yield {"data": first_chunk.model_dump_json()}

        # Content chunks
        stream = engine.chat_stream(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        for token in stream:
            chunk = StreamChunk(
                id=completion_id,
                model=model_name,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=token),
                    )
                ],
            )
            yield {"data": chunk.model_dump_json()}

        # Final chunk: finish_reason
        final_chunk = StreamChunk(
            id=completion_id,
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield {"data": final_chunk.model_dump_json()}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model and GPU status."""
    try:
        engine = get_engine()
        model_loaded = engine.is_loaded
        turbo = get_turbo_engine()
        if turbo is not None:
            stats = turbo.get_stats()
        else:
            stats = engine.get_stats()
        status = "healthy" if model_loaded else "degraded"
    except RuntimeError:
        model_loaded = False
        stats = {}
        status = "unhealthy"

    # GPU memory (may be None if no GPU)
    gpu_mem = get_gpu_memory()
    gpu_info = None
    if gpu_mem is not None:
        gpu_info = {
            "total_gb": round(gpu_mem.total_gb, 2),
            "used_gb": round(gpu_mem.used_gb, 2),
            "free_gb": round(gpu_mem.free_gb, 2),
            "used_percent": round(gpu_mem.used_percent, 1),
        }

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_name=stats.get("model_name", ""),
        inference_mode=get_inference_mode().value,
        gpu_memory=gpu_info,
        kv_cache_config={
            "cache_type_k": stats.get("kv_cache_k", "unknown"),
            "cache_type_v": stats.get("kv_cache_v", "unknown"),
            "flash_attention": stats.get("flash_attention", False),
        }
        if stats
        else None,
        turboquant_config=stats.get("turbo_quant") if stats else None,
        uptime_s=round(get_uptime(), 1),
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    try:
        engine = get_engine()
        models = [ModelInfo(id=engine.model_config.model_name)]
    except RuntimeError:
        models = []

    return ModelListResponse(data=models)
