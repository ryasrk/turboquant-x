"""Tests for inference engine (all mocked — no GPU/model required)."""

import pytest
from unittest.mock import MagicMock, patch

from src.engine.inference import InferenceEngine, GenerationStats
from src.engine.kv_cache import CacheType, KVCacheConfig
from src.engine.model_config import ModelConfig


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def model_config():
    return ModelConfig(
        model_path="/tmp/test_model.gguf",
        model_name="test-model",
        n_ctx=2048,
    )


@pytest.fixture
def kv_config():
    return KVCacheConfig()


@pytest.fixture
def turbo_kv_config():
    return KVCacheConfig(
        cache_type_k=CacheType.TURBO4,
        cache_type_v=CacheType.TURBO2,
        flash_attention=True,
    )


@pytest.fixture
def engine(model_config, kv_config):
    return InferenceEngine(model_config, kv_config)


@pytest.fixture
def mock_llama():
    """Return a MagicMock that behaves like a Llama instance."""
    mock = MagicMock()
    # Default generate response
    mock.return_value = {
        "choices": [{"text": "Hello world"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    # Default chat response
    mock.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi there!"},
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        },
    }
    return mock


@pytest.fixture
def loaded_engine(engine, mock_llama):
    """An engine with a mocked model already loaded."""
    engine._model = mock_llama
    engine._loaded = True
    return engine


# ======================================================================
# TestEngineInit
# ======================================================================


class TestEngineInit:
    """Tests for InferenceEngine initialization."""

    def test_initial_state_not_loaded(self, engine):
        assert engine.is_loaded is False

    def test_properties_accessible(self, engine, model_config, kv_config):
        assert engine.model_config is model_config
        assert engine.kv_config is kv_config

    def test_default_kv_config_when_none(self, model_config):
        eng = InferenceEngine(model_config, kv_config=None)
        assert eng.kv_config == KVCacheConfig()
        assert eng.kv_config.cache_type_k is CacheType.Q8_0
        assert eng.kv_config.cache_type_v is CacheType.Q4_0


# ======================================================================
# TestLoadModel
# ======================================================================


class TestLoadModel:
    """Tests for load_model()."""

    @patch("src.engine.inference.to_llama_params")
    def test_load_calls_llama_with_correct_params(
        self, mock_to_llama, engine
    ):
        mock_to_llama.return_value = {"type_k": "f16", "type_v": "f16"}
        mock_llama_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {"llama_cpp": MagicMock(Llama=mock_llama_cls)},
        ):
            engine.load_model()

        mock_llama_cls.assert_called_once_with(
            model_path="/tmp/test_model.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            chat_format="chatml",
            n_threads=14,
            n_threads_batch=28,
            n_batch=512,
            use_mlock=False,
            verbose=False,
            type_k="f16",
            type_v="f16",
        )

    def test_load_sets_is_loaded(self, engine):
        mock_llama_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"llama_cpp": MagicMock(Llama=mock_llama_cls)},
        ):
            engine.load_model()
        assert engine.is_loaded is True

    def test_load_twice_raises_runtime_error(self, engine):
        mock_llama_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"llama_cpp": MagicMock(Llama=mock_llama_cls)},
        ):
            engine.load_model()
            with pytest.raises(RuntimeError, match="already loaded"):
                engine.load_model()

    def test_load_without_llama_cpp_raises_import_error(self, engine):
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                engine.load_model()


# ======================================================================
# TestUnload
# ======================================================================


class TestUnload:
    """Tests for unload()."""

    def test_unload_sets_not_loaded(self, loaded_engine):
        assert loaded_engine.is_loaded is True
        loaded_engine.unload()
        assert loaded_engine.is_loaded is False

    def test_unload_when_not_loaded_is_safe(self, engine):
        # Should not raise
        engine.unload()
        assert engine.is_loaded is False


# ======================================================================
# TestGenerate
# ======================================================================


class TestGenerate:
    """Tests for generate()."""

    def test_generate_returns_text_and_stats(self, loaded_engine):
        text, stats = loaded_engine.generate("Say hello")
        assert text == "Hello world"
        assert isinstance(stats, GenerationStats)

    def test_generate_calls_model_with_correct_params(
        self, loaded_engine, mock_llama
    ):
        loaded_engine.generate(
            "test prompt",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            stop=["END"],
        )
        mock_llama.assert_called_once_with(
            "test prompt",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            stop=["END"],
        )

    def test_generate_when_not_loaded_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.generate("hello")

    def test_stats_fields_populated(self, loaded_engine):
        _, stats = loaded_engine.generate("test")
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 5
        assert stats.total_tokens == 15
        assert stats.generation_time_s > 0
        assert stats.tokens_per_second > 0


# ======================================================================
# TestChat
# ======================================================================


class TestChat:
    """Tests for chat()."""

    def test_chat_returns_message_and_stats(self, loaded_engine):
        messages = [{"role": "user", "content": "Hello"}]
        msg, stats = loaded_engine.chat(messages)
        assert isinstance(msg, dict)
        assert isinstance(stats, GenerationStats)

    def test_message_has_role_and_content(self, loaded_engine):
        messages = [{"role": "user", "content": "Hello"}]
        msg, _ = loaded_engine.chat(messages)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hi there!"

    def test_chat_calls_create_chat_completion(
        self, loaded_engine, mock_llama
    ):
        messages = [{"role": "user", "content": "test"}]
        loaded_engine.chat(
            messages, max_tokens=256, temperature=0.3, top_p=0.8
        )
        mock_llama.create_chat_completion.assert_called_once_with(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
            top_p=0.8,
        )


# ======================================================================
# TestChatStream
# ======================================================================


class TestChatStream:
    """Tests for chat_stream()."""

    def test_yields_string_chunks(self, loaded_engine, mock_llama):
        # Simulate streaming response
        mock_llama.create_chat_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {}}]},  # empty delta
            ]
        )

        chunks = list(
            loaded_engine.chat_stream(
                messages=[{"role": "user", "content": "Hi"}]
            )
        )
        assert chunks == ["Hello", " world"]

    def test_stream_calls_create_chat_completion_with_stream(
        self, loaded_engine, mock_llama
    ):
        mock_llama.create_chat_completion.return_value = iter([])
        messages = [{"role": "user", "content": "test"}]

        # Exhaust the generator
        list(loaded_engine.chat_stream(messages, max_tokens=64))

        mock_llama.create_chat_completion.assert_called_once_with(
            messages=messages,
            max_tokens=64,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        )

    def test_stream_when_not_loaded_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            list(
                engine.chat_stream(
                    messages=[{"role": "user", "content": "Hi"}]
                )
            )


# ======================================================================
# TestGetStats
# ======================================================================


class TestGetStats:
    """Tests for get_stats()."""

    def test_returns_expected_keys(self, engine):
        stats = engine.get_stats()
        expected_keys = {
            "model_name",
            "n_ctx",
            "is_loaded",
            "kv_cache_k",
            "kv_cache_v",
            "flash_attention",
            "context_max",
            "context_used",
        }
        assert set(stats.keys()) == expected_keys

    def test_shows_loaded_state(self, loaded_engine):
        assert loaded_engine.get_stats()["is_loaded"] is True
        loaded_engine.unload()
        assert loaded_engine.get_stats()["is_loaded"] is False

    def test_shows_kv_config_values(self, model_config, turbo_kv_config):
        eng = InferenceEngine(model_config, turbo_kv_config)
        stats = eng.get_stats()
        assert stats["kv_cache_k"] == "turbo4"
        assert stats["kv_cache_v"] == "turbo2"
        assert stats["flash_attention"] is True

    def test_default_kv_values(self, engine):
        stats = engine.get_stats()
        assert stats["kv_cache_k"] == "q8_0"
        assert stats["kv_cache_v"] == "q4_0"
        assert stats["flash_attention"] is True
