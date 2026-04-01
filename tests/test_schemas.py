"""Tests for Pydantic API schemas."""

import pytest
from pydantic import ValidationError

from src.server.schemas import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    StreamChunk,
    StreamDelta,
    UsageStats,
)


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------

class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="bot", content="Hi")

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_max_length_content_accepted(self):
        msg = ChatMessage(role="assistant", content="a" * 32_000)
        assert len(msg.content) == 32_000

    def test_over_max_length_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="a" * 32_001)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------

class TestChatRequest:
    @pytest.fixture()
    def _msg(self):
        return ChatMessage(role="user", content="Hi")

    def test_valid_request_with_defaults(self, _msg):
        req = ChatRequest(messages=[_msg])
        assert req.max_tokens == 512
        assert req.temperature == 0.7
        assert req.top_p == 0.95
        assert req.stream is False

    def test_empty_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    @pytest.mark.parametrize("val,ok", [(0, False), (1, True), (4096, True), (4097, False)])
    def test_max_tokens_range(self, _msg, val, ok):
        if ok:
            req = ChatRequest(messages=[_msg], max_tokens=val)
            assert req.max_tokens == val
        else:
            with pytest.raises(ValidationError):
                ChatRequest(messages=[_msg], max_tokens=val)

    @pytest.mark.parametrize("val,ok", [(-0.1, False), (0.0, True), (2.0, True), (2.1, False)])
    def test_temperature_range(self, _msg, val, ok):
        if ok:
            req = ChatRequest(messages=[_msg], temperature=val)
            assert req.temperature == val
        else:
            with pytest.raises(ValidationError):
                ChatRequest(messages=[_msg], temperature=val)

    @pytest.mark.parametrize("val,ok", [(-0.1, False), (0.0, True), (1.0, True), (1.1, False)])
    def test_top_p_range(self, _msg, val, ok):
        if ok:
            req = ChatRequest(messages=[_msg], top_p=val)
            assert req.top_p == val
        else:
            with pytest.raises(ValidationError):
                ChatRequest(messages=[_msg], top_p=val)

    def test_stream_defaults_to_false(self, _msg):
        req = ChatRequest(messages=[_msg])
        assert req.stream is False

    def test_over_100_messages_raises(self):
        msgs = [ChatMessage(role="user", content="Hi")] * 101
        with pytest.raises(ValidationError):
            ChatRequest(messages=msgs)


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------

class TestChatResponse:
    @pytest.fixture()
    def _response(self):
        return ChatResponse(
            choices=[
                ChatChoice(message=ChatMessage(role="assistant", content="Hi"))
            ],
            usage=UsageStats(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )

    def test_valid_response_with_auto_generated_id(self, _response):
        assert _response.id is not None
        assert len(_response.id) > 0

    def test_id_starts_with_chatcmpl(self, _response):
        assert _response.id.startswith("chatcmpl-")

    def test_serialization_includes_all_fields(self, _response):
        data = _response.model_dump()
        assert "id" in data
        assert "object" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data
        assert data["object"] == "chat.completion"


# ---------------------------------------------------------------------------
# UsageStats
# ---------------------------------------------------------------------------

class TestUsageStats:
    def test_valid_stats(self):
        u = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert u.total_tokens == 15

    def test_negative_tokens_raise(self):
        with pytest.raises(ValidationError):
            UsageStats(prompt_tokens=-1, completion_tokens=0, total_tokens=0)


# ---------------------------------------------------------------------------
# Streaming models
# ---------------------------------------------------------------------------

class TestStreamModels:
    def test_stream_chunk_construction(self):
        chunk = StreamChunk(
            id="chatcmpl-abc",
            choices=[StreamChoice(delta=StreamDelta(content="Hi"))],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hi"

    def test_stream_delta_with_content(self):
        d = StreamDelta(content="hello")
        assert d.content == "hello"
        assert d.role is None

    def test_stream_delta_with_role_only(self):
        d = StreamDelta(role="assistant")
        assert d.role == "assistant"
        assert d.content is None

    def test_stream_choice_with_none_finish_reason(self):
        sc = StreamChoice(delta=StreamDelta(content="tok"))
        assert sc.finish_reason is None


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------

class TestHealthResponse:
    def test_healthy_status(self):
        h = HealthResponse(status="healthy", model_loaded=True, model_name="test")
        assert h.status == "healthy"
        assert h.model_loaded is True

    def test_unhealthy_status(self):
        h = HealthResponse(status="unhealthy", model_loaded=False)
        assert h.status == "unhealthy"

    def test_optional_fields_defaulted(self):
        h = HealthResponse(status="degraded", model_loaded=False)
        assert h.model_name == ""
        assert h.gpu_memory is None
        assert h.kv_cache_config is None
        assert h.uptime_s == 0.0


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------

class TestErrorResponse:
    def test_basic_error(self):
        e = ErrorResponse(error="Something went wrong")
        assert e.error == "Something went wrong"

    def test_with_detail(self):
        e = ErrorResponse(error="Bad request", detail="Missing field", status_code=400)
        assert e.detail == "Missing field"
        assert e.status_code == 400

    def test_default_status_code(self):
        e = ErrorResponse(error="fail")
        assert e.status_code == 500


# ---------------------------------------------------------------------------
# ModelInfo / ModelListResponse
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_construction(self):
        m = ModelInfo(id="turboquant-7b")
        assert m.id == "turboquant-7b"

    def test_defaults(self):
        m = ModelInfo(id="x")
        assert m.object == "model"
        assert m.owned_by == "local"


class TestModelListResponse:
    def test_construction(self):
        r = ModelListResponse(data=[ModelInfo(id="m1")])
        assert r.object == "list"
        assert len(r.data) == 1
