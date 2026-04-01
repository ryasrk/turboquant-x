"""Integration tests for the chat API routes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.engine.inference import GenerationStats, InferenceEngine
from src.engine.kv_cache import KVCacheConfig
from src.engine.model_config import ModelConfig
from src.server.app import create_app
import src.server.app as app_module


@pytest.fixture()
def mock_engine():
    """Create a mock inference engine with sensible defaults."""
    engine = MagicMock(spec=InferenceEngine)
    engine.is_loaded = True
    engine.model_config = ModelConfig(
        model_path="/tmp/test.gguf", model_name="test-model"
    )
    engine.get_stats.return_value = {
        "model_name": "test-model",
        "n_ctx": 2048,
        "is_loaded": True,
        "kv_cache_k": "q8_0",
        "kv_cache_v": "turbo4",
        "flash_attention": True,
    }
    engine.chat.return_value = (
        {"role": "assistant", "content": "Hello! How can I help?"},
        GenerationStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            generation_time_s=0.5,
            tokens_per_second=10.0,
        ),
    )
    engine.chat_stream.return_value = iter(["Hello", "!", " How", " can", " I help?"])
    return engine


@pytest.fixture()
def client(mock_engine):
    """Create a test client with mocked engine (skips model loading)."""
    with patch.object(InferenceEngine, "load_model"):
        test_app = create_app(
            model_config=ModelConfig(model_path="/tmp/test.gguf"),
            kv_config=KVCacheConfig(),
        )
        with TestClient(test_app) as c:
            # Override _engine AFTER lifespan has created the real one
            app_module._engine = mock_engine
            yield c

        app_module._engine = None


# ---------------------------------------------------------------------------
# Chat endpoint tests
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    """Tests for POST /v1/chat/completions."""

    def test_valid_request_returns_200(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200

    def test_response_has_openai_compatible_format(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 64,
                "temperature": 0.5,
            },
        )
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert data["model"] == "test-model"

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello! How can I help?"
        assert choice["finish_reason"] == "stop"

        usage = data["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_invalid_request_no_messages_returns_422(self, client):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 422

    def test_invalid_request_empty_messages_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions", json={"messages": []}
        )
        assert resp.status_code == 422

    def test_model_not_loaded_returns_503(self, client, mock_engine):
        mock_engine.is_loaded = False
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 503

    def test_custom_parameters_forwarded(self, client, mock_engine):
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 128,
                "temperature": 1.5,
                "top_p": 0.8,
            },
        )
        mock_engine.chat.assert_called_once()
        kwargs = mock_engine.chat.call_args.kwargs
        assert kwargs["max_tokens"] == 128
        assert kwargs["temperature"] == 1.5
        assert kwargs["top_p"] == 0.8

    def test_streaming_returns_sse(self, client, mock_engine):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events from the response body
        chunks = []
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    chunks.append("[DONE]")
                else:
                    chunks.append(json.loads(payload))

        # First chunk should have role
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Content chunks
        content_parts = []
        for chunk in chunks[1:-2]:  # skip first (role) and last two (finish + DONE)
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                content_parts.append(delta["content"])
        assert "".join(content_parts) == "Hello! How can I help?"

        # Final chunk has finish_reason
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"

        # Last event is [DONE]
        assert chunks[-1] == "[DONE]"


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_healthy_status_when_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "test-model"

    def test_response_includes_kv_cache_config(self, client):
        data = client.get("/health").json()
        kv = data["kv_cache_config"]
        assert kv["cache_type_k"] == "q8_0"
        assert kv["cache_type_v"] == "turbo4"
        assert kv["flash_attention"] is True

    def test_uptime_present(self, client):
        data = client.get("/health").json()
        assert "uptime_s" in data
        assert isinstance(data["uptime_s"], float)

    @patch("src.server.routes.get_gpu_memory")
    def test_gpu_memory_included_when_available(self, mock_gpu, client):
        from src.utils.memory import MemoryInfo

        mock_gpu.return_value = MemoryInfo(
            total=8 * 1024**3, used=4 * 1024**3, free=4 * 1024**3
        )
        data = client.get("/health").json()
        gpu = data["gpu_memory"]
        assert gpu is not None
        assert gpu["total_gb"] == 8.0
        assert gpu["used_gb"] == 4.0

    def test_degraded_when_model_not_loaded(self, client, mock_engine):
        mock_engine.is_loaded = False
        data = client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# Models endpoint tests
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for GET /v1/models."""

    def test_models_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_returns_model_list(self, client):
        data = client.get("/v1/models").json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


class TestMiddleware:
    """Tests for request metadata middleware."""

    def test_x_request_id_header_present(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_x_response_time_ms_header_present(self, client):
        resp = client.get("/health")
        assert "x-response-time-ms" in resp.headers
        # Verify it's a parseable float
        float(resp.headers["x-response-time-ms"])


# ---------------------------------------------------------------------------
# Missing coverage tests
# ---------------------------------------------------------------------------


class TestGetEngineUninitialized:
    """Tests for get_engine() when engine is None."""

    def test_get_engine_raises_when_none(self):
        """Line 30: _engine is None raises RuntimeError."""
        from src.server.app import get_engine

        original = app_module._engine
        try:
            app_module._engine = None
            with pytest.raises(RuntimeError, match="not initialized"):
                get_engine()
        finally:
            app_module._engine = original

    def test_get_uptime_returns_zero_before_start(self):
        """Line 37: _start_time == 0 returns 0."""
        from src.server.app import get_uptime

        original = app_module._start_time
        try:
            app_module._start_time = 0.0
            assert get_uptime() == 0.0
        finally:
            app_module._start_time = original


class TestLifespanDegradedMode:
    """Tests for model load failure in lifespan."""

    def test_server_starts_degraded_when_model_fails(self):
        """Lines 60-61: model load fails, server starts degraded."""
        with patch.object(InferenceEngine, "load_model", side_effect=RuntimeError("no GPU")):
            test_app = create_app(
                model_config=ModelConfig(model_path="/tmp/test.gguf"),
                kv_config=KVCacheConfig(),
            )
            with TestClient(test_app) as c:
                resp = c.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] in ("degraded", "unhealthy")


class TestGlobalExceptionHandler:
    """Tests for the global exception handler (lines 140-146)."""

    def test_unhandled_exception_returns_500(self, mock_engine):
        """Force an unhandled exception in a route."""
        mock_engine.chat.side_effect = Exception("unexpected crash")
        mock_engine.is_loaded = True

        with patch.object(InferenceEngine, "load_model"):
            test_app = create_app(
                model_config=ModelConfig(model_path="/tmp/test.gguf"),
                kv_config=KVCacheConfig(),
            )
            with TestClient(test_app, raise_server_exceptions=False) as c:
                app_module._engine = mock_engine
                resp = c.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )
                assert resp.status_code == 500
                data = resp.json()
                assert "error" in data


class TestRoutesImportFailure:
    """Tests for routes module import failure (lines 156-157)."""

    def test_app_starts_without_routes(self):
        """When routes import fails, app still starts."""
        with patch.dict("sys.modules", {"src.server.routes": None}):
            test_app = create_app(
                model_config=ModelConfig(model_path="/tmp/test.gguf"),
                kv_config=KVCacheConfig(),
            )
            assert test_app is not None


class TestModelsEndpointDegraded:
    """Tests for list_models when engine not initialized (lines 168-171, 206-207)."""

    def test_models_returns_empty_when_no_engine(self):
        """get_engine() raises → empty model list."""
        with patch.object(InferenceEngine, "load_model"):
            test_app = create_app(
                model_config=ModelConfig(model_path="/tmp/test.gguf"),
                kv_config=KVCacheConfig(),
            )
            with TestClient(test_app) as c:
                app_module._engine = None
                resp = c.get("/v1/models")
                assert resp.status_code == 200
                data = resp.json()
                assert data["data"] == []

    def test_health_unhealthy_when_no_engine(self):
        """Lines 168-171: health returns unhealthy when engine is None."""
        with patch.object(InferenceEngine, "load_model"):
            test_app = create_app(
                model_config=ModelConfig(model_path="/tmp/test.gguf"),
                kv_config=KVCacheConfig(),
            )
            with TestClient(test_app) as c:
                app_module._engine = None
                resp = c.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "unhealthy"
                assert data["model_loaded"] is False
