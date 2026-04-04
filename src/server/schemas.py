"""Pydantic schemas for TurboQuant chat API — OpenAI-compatible format."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AttachmentRef(BaseModel):
    """Reference to an uploaded attachment."""
    id: str
    type: Literal['image', 'document'] = 'document'


class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list = Field(default="", description="Text string or OpenAI multimodal content array")
    attachments: list[AttachmentRef] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    """Chat completion request — OpenAI-compatible."""
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=100)
    max_tokens: int = Field(default=2048, ge=1, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    tools: bool = False  # Enable agent mode with tool calling
    thinking: bool = True  # Qwen3: enable/disable chain-of-thought thinking block
    chat_template_kwargs: dict | None = Field(
        default=None,
        description="Extra template vars (e.g. {\"enable_thinking\": false}). "
                    "enable_thinking overrides the 'thinking' field.",
    )

    @field_validator('messages')
    @classmethod
    def validate_attachments(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure total attachments across all messages <= 20."""
        total_attachments = sum(len(msg.attachments) for msg in v)
        if total_attachments > 20:
            raise ValueError(f"Total attachments ({total_attachments}) exceeds maximum of 20")
        return v

    @property
    def effective_thinking(self) -> bool:
        """Resolve the final thinking flag from both fields.

        chat_template_kwargs.enable_thinking takes precedence over thinking.
        """
        if self.chat_template_kwargs is not None:
            v = self.chat_template_kwargs.get("enable_thinking")
            if isinstance(v, bool):
                return v
        return self.thinking


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
    inference_mode: str = "standard"
    provider: str | None = None
    loading: bool = False
    supports_thinking: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_attachments: bool = True  # Always true — documents always work
    context_max: int = 0
    context_used: int = 0
    gpu_memory: dict | None = None
    kv_cache_config: dict | None = None
    turboquant_config: dict | None = None
    layer_distribution: dict | None = None
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


class AttachmentResponse(BaseModel):
    """Response for successful file upload."""
    id: str
    original_name: str
    mime_type: str
    size_bytes: int
    type: str  # 'image' or 'document'
