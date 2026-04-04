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
from src.agent.approval import get_approval_store
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


@router.post("/v1/agent/approve-tool")
async def approve_tool(request: Request):
    """Allow or deny a pending tool execution request.

    Body: {"approval_id": str, "approved": bool}
    """
    body = await request.json()
    approval_id = body.get("approval_id")
    approved = body.get("approved", False)

    if not approval_id or not isinstance(approval_id, str):
        raise HTTPException(status_code=422, detail="Missing or invalid approval_id")

    store = get_approval_store()
    resolved = store.resolve(approval_id, bool(approved))
    if not resolved:
        raise HTTPException(status_code=404, detail="No pending approval with that ID")

    return {"status": "ok", "approval_id": approval_id, "approved": approved}


@router.get("/v1/agent/tools")
async def list_agent_tools():
    """List all registered agent tools (built-in + MCP)."""
    from src.server.app import get_agent_registry

    registry = get_agent_registry()
    if registry is None:
        return {"tools": []}

    tools = []
    for defn in registry.get_definitions():
        fn = defn["function"]
        tool_obj = registry.get(fn["name"])
        tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "requires_approval": getattr(tool_obj, "requires_approval", False) if tool_obj else False,
            "is_mcp": fn["name"].startswith("mcp_"),
        })
    return {"tools": tools, "count": len(tools)}


@router.post("/v1/agent/mcp/reload")
async def reload_mcp_servers():
    """Disconnect all MCP servers and reconnect from config."""
    from src.agent.mcp_loader import shutdown_mcp_servers, connect_mcp_servers
    from src.server.app import get_agent_registry

    registry = get_agent_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Agent registry not initialized")

    # Remove existing MCP tools from registry
    mcp_tools = [name for name in registry.list_tools() if name.startswith("mcp_")]
    for name in mcp_tools:
        try:
            registry.unregister(name)
        except KeyError:
            pass

    # Disconnect existing servers
    await shutdown_mcp_servers()

    # Reconnect
    count = await connect_mcp_servers(registry)
    return {"status": "ok", "tools_registered": count, "removed": len(mcp_tools)}


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

    return await _cloud_sync_response(cloud_engine, messages, request)


async def _cloud_sync_response(cloud_engine, messages: list[dict], request: ChatRequest) -> ChatResponse:
    """Handle non-streaming cloud chat completion."""
    import asyncio

    # Run the blocking HTTP call in a thread so we don't block the event loop
    response_msg, stats = await asyncio.to_thread(
        cloud_engine.chat,
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
    """Handle streaming cloud chat completion via SSE.

    Uses the provider's native async streaming directly to avoid
    blocking the event loop with sync-to-async bridges.
    """

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

        # Stream content directly from async provider
        think_filter = _ThinkStreamFilter()
        try:
            provider = cloud_engine._provider
            async for token in provider.chat_stream(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                # Filter out <think>…</think> blocks (reasoning tokens)
                filtered = think_filter.feed(token)
                if not filtered:
                    continue
                chunk = StreamChunk(
                    id=completion_id,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=StreamDelta(content=filtered),
                        )
                    ],
                )
                yield {"data": chunk.model_dump_json()}
        except Exception as e:
            logger.exception("Cloud stream error")
            yield {"data": json.dumps({"type": "error", "message": str(e)})}

        # Flush any remaining buffered text from the think filter
        remaining = think_filter.flush()
        if remaining:
            chunk = StreamChunk(
                id=completion_id,
                model=model_name,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=remaining),
                    )
                ],
            )
            yield {"data": chunk.model_dump_json()}

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

    # Cloud reasoning models (GLM-4.5, DeepSeek, o1) have built-in CoT
    if stats and stats.get("supports_reasoning"):
        supports_thinking = True

    _tools_prefixes = ("qwen2.5", "qwen3", "qwq", "functionary", "hermes")
    supports_tools = any(model_name.lower().startswith(p) for p in _tools_prefixes)

    # Cloud models with native tool calling
    if mode == InferenceMode.CLOUD and cloud is not None and cloud.is_loaded:
        _cloud_tool_providers = ("openai", "nvidia", "anthropic", "zhipu", "deepseek")
        if cloud.provider_name in _cloud_tool_providers:
            supports_tools = True

    # Provider name for cloud mode status bar display
    cloud_provider_name = None
    if mode == InferenceMode.CLOUD and cloud is not None and cloud.is_loaded:
        cloud_provider_name = cloud.provider_name

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_name=model_name,
        inference_mode=get_inference_mode().value,
        provider=cloud_provider_name,
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


@router.post("/v1/switch-provider")
async def switch_provider_endpoint(request: Request):
    """Switch cloud provider at runtime (unloads + reloads cloud engine)."""
    import asyncio

    body = await request.json()
    provider_name = body.get("provider", "").strip()
    if not provider_name:
        raise HTTPException(status_code=422, detail="'provider' field is required")

    api_key = body.get("api_key", "").strip()

    try:
        from src.engine.cloud.registry import build_cloud_configs, SUPPORTED_PROVIDERS
        from src.engine.cloud.provider import CloudConfig
        from src.server.app import (
            _create_and_load_engine, _unload_engines, InferenceMode as _IM, _switch_lock,
        )
        import src.server.app as _app_module

        if provider_name not in SUPPORTED_PROVIDERS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown provider: {provider_name}. Valid: {list(SUPPORTED_PROVIDERS.keys())}",
            )

        # Build all cloud configs from existing config
        import os
        cloud_section = getattr(request.app.state, "_cloud_yaml_config", {})
        configs = build_cloud_configs({"cloud": cloud_section})

        # If an api_key was provided, use it; otherwise check existing configs
        if api_key:
            # Build config from the provided key + known defaults
            providers_cfg = cloud_section.get("providers", {}).get(provider_name, {})
            new_config = CloudConfig(
                provider=provider_name,
                api_key=api_key,
                base_url=providers_cfg.get("base_url"),
                model=providers_cfg.get("model", ""),
                max_tokens=providers_cfg.get("max_tokens", 2048),
                temperature=providers_cfg.get("temperature", 0.7),
                top_p=providers_cfg.get("top_p", 0.95),
                timeout=providers_cfg.get("timeout", 120.0),
            )
        elif provider_name in configs:
            new_config = configs[provider_name]
        else:
            # Check env var
            env_key = f"TURBOQUANT_CLOUD_{provider_name.upper()}_API_KEY"
            env_api_key = os.environ.get(env_key, "")
            if not env_api_key:
                raise HTTPException(
                    status_code=400,
                    detail=f"No API key for provider '{provider_name}'. "
                           f"Set {env_key} env var or provide 'api_key' in request.",
                )
            providers_cfg = cloud_section.get("providers", {}).get(provider_name, {})
            new_config = CloudConfig(
                provider=provider_name,
                api_key=env_api_key,
                base_url=providers_cfg.get("base_url"),
                model=providers_cfg.get("model", ""),
                max_tokens=providers_cfg.get("max_tokens", 2048),
                temperature=providers_cfg.get("temperature", 0.7),
                top_p=providers_cfg.get("top_p", 0.95),
                timeout=providers_cfg.get("timeout", 120.0),
            )

        # Update app state and reload cloud engine
        request.app.state.cloud_config = new_config

        with _switch_lock:
            _app_module._loading = True
            try:
                _unload_engines()
                _app_module._inference_mode = _IM.CLOUD
                _create_and_load_engine(request.app, _IM.CLOUD)
            finally:
                _app_module._loading = False

        return {
            "status": "ok",
            "provider": provider_name,
            "model": new_config.model or "(default)",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Provider switch failed")
        raise HTTPException(status_code=500, detail=f"Provider switch failed: {e}")


@router.post("/v1/switch-cloud-model")
async def switch_cloud_model_endpoint(request: Request):
    """Switch to a different model on the active cloud provider (no reload)."""
    body = await request.json()
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=422, detail="'model' field is required")

    cloud = get_cloud_engine()
    if cloud is None or not cloud.is_loaded:
        raise HTTPException(status_code=409, detail="No cloud engine is active")

    try:
        cloud.switch_model(model)
        return {
            "status": "ok",
            "provider": cloud.provider_name,
            "model": model,
        }
    except Exception as e:
        logger.exception("Cloud model switch failed")
        raise HTTPException(status_code=500, detail=f"Cloud model switch failed: {e}")


@router.get("/v1/cloud-providers")
async def list_cloud_providers(request: Request):
    """List available cloud providers and their configuration status."""
    import os
    from src.engine.cloud.registry import SUPPORTED_PROVIDERS, build_cloud_configs

    cloud_section = getattr(request.app.state, "_cloud_yaml_config", {})
    configs = build_cloud_configs({"cloud": cloud_section})

    current_cloud = get_cloud_engine()
    current_provider = current_cloud.provider_name if current_cloud else None

    providers = []
    for name, display in SUPPORTED_PROVIDERS.items():
        env_key = f"TURBOQUANT_CLOUD_{name.upper()}_API_KEY"
        has_key = name in configs or bool(os.environ.get(env_key, ""))
        providers.append({
            "name": name,
            "display_name": display,
            "configured": has_key,
            "active": name == current_provider,
        })

    return {"providers": providers, "active_provider": current_provider}


@router.get("/v1/cloud-providers/{provider_name}/models")
async def list_provider_models(provider_name: str, request: Request):
    """List available models for a specific cloud provider.

    Creates a temporary provider connection to fetch the model list
    from the provider's /models endpoint.
    """
    import asyncio
    import os
    from src.engine.cloud.registry import SUPPORTED_PROVIDERS, build_cloud_configs, create_provider
    from src.engine.cloud.provider import CloudConfig

    if provider_name not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider: {provider_name}. Valid: {list(SUPPORTED_PROVIDERS.keys())}",
        )

    # Check if we already have an active engine for this provider
    cloud = get_cloud_engine()
    if cloud is not None and cloud.is_loaded and cloud.provider_name == provider_name:
        models = await asyncio.to_thread(cloud.list_models)
        return {"provider": provider_name, "models": models}

    # Build a temporary provider to list models
    cloud_section = getattr(request.app.state, "_cloud_yaml_config", {})
    configs = build_cloud_configs({"cloud": cloud_section})

    if provider_name in configs:
        cfg = configs[provider_name]
    else:
        env_key = f"TURBOQUANT_CLOUD_{provider_name.upper()}_API_KEY"
        api_key = os.environ.get(env_key, "")
        if not api_key:
            return {"provider": provider_name, "models": [], "error": "No API key configured"}
        providers_cfg = cloud_section.get("providers", {}).get(provider_name, {})
        cfg = CloudConfig(
            provider=provider_name,
            api_key=api_key,
            base_url=providers_cfg.get("base_url"),
            model=providers_cfg.get("model", ""),
        )

    try:
        provider = create_provider(cfg)
        models = await asyncio.to_thread(provider.list_models)
        return {"provider": provider_name, "models": models}
    except Exception as e:
        logger.warning("Failed to list models for %s: %s", provider_name, e)
        return {"provider": provider_name, "models": [], "error": str(e)}
