"""API routes for TurboQuant chat server."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.server.app import get_engine, get_turbo_engine, get_cloud_engine, get_inference_mode, get_uptime, is_loading, InferenceMode
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
_thought_logger = logging.getLogger("agent.thoughts")

router = APIRouter()


class _ThinkStreamFilter:
    """Token-level filter that intercepts ``<think>…</think>`` blocks.

    * Tokens inside a think block are buffered and written to the
      thought log file — they are **not** yielded to the client.
    * Tokens outside think blocks pass through unchanged.
    * Handles partial tag boundaries across token splits by keeping a
      small look-behind buffer.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""          # partial-tag look-behind
        self._thought_buf = ""  # accumulated thought text

    def feed(self, token: str) -> str | None:
        """Process one token. Return text to send, or *None* to suppress."""
        self._buf += token

        if not self._inside:
            idx = self._buf.find(self._OPEN)
            if idx != -1:
                # Emit everything before the tag, enter think mode
                before = self._buf[:idx]
                self._buf = self._buf[idx + len(self._OPEN):]
                self._inside = True
                # Continue processing the buffer for </think> in same call
                after = self._process_inside()
                parts = [p for p in (before, after) if p]
                return "".join(parts) or None
            # Guard against partial "<thi" at end of buffer
            if len(self._buf) > len(self._OPEN):
                emit = self._buf[:-len(self._OPEN)]
                self._buf = self._buf[-len(self._OPEN):]
                return emit
            return None
        else:
            return self._process_inside()

    def _process_inside(self) -> str | None:
        """Consume buffer while inside a ``<think>`` block."""
        idx = self._buf.find(self._CLOSE)
        if idx != -1:
            # Capture thought, exit think mode
            self._thought_buf += self._buf[:idx]
            self._buf = self._buf[idx + len(self._CLOSE):]
            self._inside = False
            # Log the thought
            thought = self._thought_buf.strip()
            if thought:
                _thought_logger.debug("[stream] %s", thought)
            self._thought_buf = ""
            # Process anything remaining after </think>
            if self._buf:
                leftover = self._buf
                self._buf = ""
                return self.feed(leftover)
            return None
        # Check for partial </think> at end of buffer — keep it there
        # so the next feed() can complete the match.
        for i in range(1, len(self._CLOSE)):
            if self._buf.endswith(self._CLOSE[:i]):
                safe = self._buf[:-i]
                self._thought_buf += safe
                self._buf = self._buf[-i:]
                return None
        # No partial match — accumulate everything as thought
        self._thought_buf += self._buf
        self._buf = ""
        return None

    def flush(self) -> str | None:
        """Flush any remaining buffer at end of stream."""
        if self._inside and self._thought_buf.strip():
            _thought_logger.debug("[stream-incomplete] %s", self._thought_buf.strip())
        remaining = self._buf
        self._buf = ""
        self._thought_buf = ""
        return remaining or None


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
    Routes to cloud engine when in cloud mode, local engine otherwise.
    """
    cloud_engine = get_cloud_engine()
    mode = get_inference_mode()

    # Cloud mode — route to cloud engine
    if mode == InferenceMode.CLOUD and cloud_engine is not None:
        if not cloud_engine.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Cloud engine not loaded. Check provider configuration.",
            )
        return await _cloud_chat(cloud_engine, request)

    engine = get_engine()

    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is in degraded mode.",
        )

    # Convert Pydantic messages to dicts for engine
    messages = []
    for m in request.messages:
        # Flatten multimodal content arrays to plain text for text-only models
        if isinstance(m.content, list):
            text_parts = [p.get("text", "") for p in m.content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts) if text_parts else ""
        else:
            content = m.content
        msg = {"role": m.role, "content": content}
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        messages.append(msg)

    # Estimate prompt tokens and check context budget
    n_ctx = engine.model_config.n_ctx
    estimated_tokens = sum(len(m["content"]) // 3 + 4 for m in messages)  # ~3 chars/token + overhead
    headroom = request.max_tokens + 64  # reserve for generation + template overhead
    if estimated_tokens + headroom > n_ctx:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "context_exceeded",
                "message": (
                    f"Conversation too long: ~{estimated_tokens:,} prompt tokens + "
                    f"{request.max_tokens:,} max_tokens exceeds context window of {n_ctx:,}."
                ),
                "context_max": n_ctx,
                "estimated_tokens": estimated_tokens,
                "suggestion": "Clear the conversation or reduce max_tokens.",
            },
        )

    # Agent mode: route through tool-calling loop
    if request.tools:
        return _agent_response(messages, request)

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
        thinking=request.effective_thinking,
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
        finish_reason = "stop"

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

        # Content chunks — capture finish_reason from generator return
        # The ThinkStreamFilter diverts <think>…</think> tokens to the
        # thought log so the client only receives the answer portion.
        stream = engine.chat_stream(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking=request.effective_thinking,
        )
        think_filter = _ThinkStreamFilter()

        try:
            while True:
                token = next(stream)
                emit = think_filter.feed(token)
                if not emit:
                    continue
                chunk = StreamChunk(
                    id=completion_id,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=StreamDelta(content=emit),
                        )
                    ],
                )
                yield {"data": chunk.model_dump_json()}
        except StopIteration as e:
            if e.value and hasattr(e.value, "finish_reason"):
                finish_reason = e.value.finish_reason

        # Flush any trailing buffer from the filter
        trailing = think_filter.flush()
        if trailing:
            chunk = StreamChunk(
                id=completion_id,
                model=model_name,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=trailing),
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
                    finish_reason=finish_reason,
                )
            ],
        )
        yield {"data": final_chunk.model_dump_json()}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


async def _cloud_chat(cloud_engine, request: ChatRequest):
    """Route chat to cloud engine (non-streaming or streaming).

    When tools are enabled, routes through CloudAgentLoop for
    native cloud function calling.
    """
    # Convert Pydantic messages to dicts
    messages = []
    for m in request.messages:
        if isinstance(m.content, list):
            text_parts = [p.get("text", "") for p in m.content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts) if text_parts else ""
        else:
            content = m.content
        msg = {"role": m.role, "content": content}
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        messages.append(msg)

    # Agent mode: route through cloud tool-calling loop
    if request.tools:
        return _cloud_agent_response(cloud_engine, messages, request)

    if request.stream:
        return _cloud_stream_response(cloud_engine, messages, request)

    return _cloud_sync_response(cloud_engine, messages, request)


def _cloud_sync_response(cloud_engine, messages: list[dict], request: ChatRequest) -> ChatResponse:
    """Handle non-streaming cloud chat completion."""
    response_msg, stats = cloud_engine.chat(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return ChatResponse(
        id=completion_id,
        model=cloud_engine.model_name,
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


def _cloud_stream_response(
    cloud_engine, messages: list[dict], request: ChatRequest
) -> EventSourceResponse:
    """Handle streaming cloud chat completion via SSE."""

    async def event_generator():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = cloud_engine.model_name

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

        # Content chunks via sync generator
        stream = cloud_engine.chat_stream(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        try:
            while True:
                token = next(stream)
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
        except StopIteration:
            pass

        # Final chunk
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


def _cloud_agent_response(cloud_engine, messages: list[dict], request: ChatRequest) -> EventSourceResponse:
    """Handle cloud agent mode with native tool calling via SSE."""
    async def event_generator():
        from src.agent.cloud_loop import CloudAgentLoop
        from src.server.app import get_agent_registry

        registry = get_agent_registry()
        if registry is None:
            yield {"data": json.dumps({"type": "error", "message": "Agent tools not initialized"})}
            yield {"data": "[DONE]"}
            return

        loop = CloudAgentLoop(registry)

        async for event in loop.run(
            cloud_engine,
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            yield {"data": json.dumps(event)}

        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


def _agent_response(messages: list[dict], request: ChatRequest) -> EventSourceResponse:
    """Handle agent mode with tool calling via SSE."""
    async def event_generator():
        from src.agent.loop import AgentLoop
        from src.server.app import get_agent_registry

        registry = get_agent_registry()
        if registry is None:
            yield {"data": json.dumps({"type": "error", "message": "Agent tools not initialized"})}
            yield {"data": "[DONE]"}
            return

        loop = AgentLoop(registry)
        engine = get_engine()

        async for event in loop.run(
            engine,
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking=request.effective_thinking,
        ):
            yield {"data": json.dumps(event)}

        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model and GPU status."""
    try:
        cloud = get_cloud_engine()
        mode = get_inference_mode()

        if mode == InferenceMode.CLOUD and cloud is not None:
            model_loaded = cloud.is_loaded
            stats = cloud.get_stats()
            status = "healthy" if model_loaded else "degraded"
        else:
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

    # GPU/CPU layer distribution report
    layer_report = None
    try:
        from src.utils.gpu_layers import layer_distribution_report

        if model_loaded:
            engine_ref = get_engine()
            layer_report = layer_distribution_report(
                model_path=engine_ref.model_config.model_path,
                n_gpu_layers=engine_ref.model_config.n_gpu_layers,
                n_ctx=engine_ref.model_config.n_ctx,
            )
    except Exception:
        pass

    model_name = stats.get("model_name", "") if stats else ""
    _thinking_prefixes = ("qwen3", "qwq", "deepseek-r1")
    supports_thinking = any(model_name.lower().startswith(p) for p in _thinking_prefixes)

    _tools_prefixes = ("qwen2.5", "qwen3", "qwq", "functionary", "hermes")
    supports_tools = any(model_name.lower().startswith(p) for p in _tools_prefixes)

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_name=model_name,
        inference_mode=get_inference_mode().value,
        loading=is_loading(),
        supports_thinking=supports_thinking,
        supports_tools=supports_tools,
        supports_vision=False,  # GGUF models don't include vision encoder
        context_max=stats.get("context_max", 0) if stats else 0,
        context_used=stats.get("context_used", 0) if stats else 0,
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
        layer_distribution=layer_report,
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    try:
        cloud = get_cloud_engine()
        mode = get_inference_mode()

        if mode == InferenceMode.CLOUD and cloud is not None:
            cloud_models = cloud.list_models()
            models = [ModelInfo(id=m) for m in cloud_models] if cloud_models else [ModelInfo(id=cloud.model_name)]
        else:
            engine = get_engine()
            models = [ModelInfo(id=engine.model_config.model_name)]
    except RuntimeError:
        models = []

    return ModelListResponse(data=models)


@router.post("/v1/switch-mode")
async def switch_mode(request: Request):
    """Switch inference mode at runtime (unloads + reloads engine)."""
    import asyncio

    from src.server.app import switch_inference_mode, InferenceMode as _IM

    body = await request.json()
    mode_str = body.get("mode", "").strip()

    try:
        new_mode = _IM(mode_str)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode: {mode_str}. Valid: {[m.value for m in _IM]}",
        )

    try:
        await asyncio.to_thread(switch_inference_mode, request.app, new_mode)
    except Exception as e:
        logger.exception("Mode switch failed")
        raise HTTPException(status_code=500, detail=f"Mode switch failed: {e}")

    return {"status": "ok", "mode": new_mode.value}


@router.get("/v1/available-models")
async def available_models():
    """List GGUF model files in the models/ directory."""
    from pathlib import Path

    models_dir = Path("models")
    files = []
    if models_dir.is_dir():
        for f in sorted(models_dir.glob("*.gguf")):
            size_gb = f.stat().st_size / (1024**3)
            files.append({
                "filename": f.name,
                "path": str(f),
                "size_gb": round(size_gb, 2),
            })

    # Mark which one is currently loaded
    current_path = ""
    try:
        engine = get_engine()
        current_path = engine.model_config.model_path
    except RuntimeError:
        pass

    for m in files:
        m["loaded"] = (m["path"] == current_path or m["filename"] in current_path)

    return {"models": files, "loading": is_loading()}


@router.post("/v1/switch-model")
async def switch_model_endpoint(request: Request):
    """Switch to a different GGUF model file (unloads + reloads)."""
    import asyncio
    from src.server.app import switch_model

    body = await request.json()
    model_path = body.get("model", "").strip()
    if not model_path:
        raise HTTPException(status_code=422, detail="'model' field is required")

    try:
        result = await asyncio.to_thread(switch_model, request.app, model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Model switch failed")
        raise HTTPException(status_code=500, detail=f"Model switch failed: {e}")

    return result
