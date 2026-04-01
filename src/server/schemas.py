"""Pydantic schemas for TurboQuant chat API — OpenAI-compatible format."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=32_000)


class ChatRequest(BaseModel):
    """Chat completion request — OpenAI-compatible."""
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=100)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False


class UsageStats(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class ChatChoice(BaseModel):
    """A single completion choice."""
    index: int = Field(default=0, ge=0)
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatResponse(BaseModel):
    """Chat completion response — OpenAI-compatible."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    model: str = ""
    choices: list[ChatChoice] = Field(..., min_length=1)
    usage: UsageStats


class StreamDelta(BaseModel):
    """Delta content in a stream chunk."""
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """A single streaming choice."""
    index: int = 0
    delta: StreamDelta
    finish_reason: Literal["stop", "length"] | None = None


class StreamChunk(BaseModel):
    """SSE chunk for streaming responses."""
    id: str
    object: str = "chat.completion.chunk"
    model: str = ""
    choices: list[StreamChoice]


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_name: str = ""
    gpu_memory: dict | None = None
    kv_cache_config: dict | None = None
    uptime_s: float = 0.0


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Response for /v1/models listing."""
    object: str = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response wrapper."""
    error: str
    detail: str | None = None
    status_code: int = 500
