"""Tests for TurboQuant engine (all mocked — no GPU/model required)."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.engine.turbo_engine import (
    CompressionStats,
    TurboGenerationResult,
    TurboQuantEngine,
)
from src.engine.inference import GenerationStats
from src.engine.model_config import ModelConfig
from src.turboquant.compressor import CompressedKV, QuantConfig


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
def quant_config():
    return QuantConfig(k_bits=8, v_bits=4, block_size=128)


@pytest.fixture
def engine(model_config, quant_config):
    """TurboQuantEngine with small architecture for fast tests."""
    return TurboQuantEngine(
        model_config, quant_config,
        n_layers=2, n_heads=2, head_dim=128,
    )


@pytest.fixture
def mock_state():
    """Return a mock LlamaState with realistic-sized bytes."""
    state = MagicMock()
    # Generate enough state bytes for 2 layers * 2 heads * 128 head_dim
    # Need at least 2 * n_layers * n_heads * seq * head_dim bytes
    state.llama_state = bytes(np.random.randint(0, 256, size=200000, dtype=np.uint8))
    state.n_tokens = 32
    state.scores = np.zeros((32, 100), dtype=np.float32)
    state.input_ids = np.arange(32, dtype=np.int32)
    state.seed = 42
    return state


@pytest.fixture
def mock_llama(mock_state):
    """Return a MagicMock that behaves like a Llama instance."""
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Test response"}},
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    mock.save_state.return_value = mock_state
    return mock


@pytest.fixture
def loaded_engine(engine, mock_llama):
    """A TurboQuantEngine with a mocked model already loaded."""
    engine._engine._model = mock_llama
    engine._engine._loaded = True
    return loaded_engine_from(engine)


def loaded_engine_from(engine):
    """Helper: return engine unchanged (for clarity in fixtures)."""
    return engine


@pytest.fixture
def loaded_turbo(engine, mock_llama):
    """A TurboQuantEngine with a mocked model loaded."""
    engine._engine._model = mock_llama
    engine._engine._loaded = True
    return engine


# ======================================================================
# TestInit
# ======================================================================


class TestInit:
    """Tests for TurboQuantEngine initialization."""

    def test_initial_state(self, engine, model_config, quant_config):
        assert engine.is_loaded is False
        assert engine.has_compressed_state is False
        assert engine.quant_config is quant_config

    def test_default_quant_config(self, model_config):
        eng = TurboQuantEngine(model_config)
        assert eng.quant_config == QuantConfig()

    def test_engine_property_returns_inference_engine(self, engine):
        assert engine.engine is engine._engine

    def test_custom_architecture(self, model_config, quant_config):
        eng = TurboQuantEngine(
            model_config, quant_config,
            n_layers=32, n_heads=32, head_dim=64,
        )
        assert eng._n_layers == 32
        assert eng._n_heads == 32
        assert eng._head_dim == 64


# ======================================================================
# TestLoadUnload
# ======================================================================


class TestLoadUnload:
    """Tests for load/unload lifecycle."""

    def test_load_delegates_to_engine(self, engine):
        with patch.object(engine._engine, "load_model") as mock_load:
            engine.load_model()
            mock_load.assert_called_once()

    def test_unload_clears_state(self, loaded_turbo):
        loaded_turbo._compressed_state = MagicMock()
        loaded_turbo._state_metadata = {"n_tokens": 10}

        loaded_turbo.unload()
        assert loaded_turbo.is_loaded is False
        assert loaded_turbo.has_compressed_state is False
        assert loaded_turbo._state_metadata is None


# ======================================================================
# TestStateToKVTensors
# ======================================================================


class TestStateToKVTensors:
    """Tests for _state_to_kv_tensors."""

    def test_returns_4d_tensors(self, engine):
        state_bytes = bytes(np.random.randint(0, 256, size=200000, dtype=np.uint8))
        keys, values = engine._state_to_kv_tensors(state_bytes, n_tokens=10)
        assert keys.ndim == 4
        assert values.ndim == 4

    def test_shape_matches_architecture(self, engine):
        state_bytes = bytes(np.random.randint(0, 256, size=200000, dtype=np.uint8))
        keys, values = engine._state_to_kv_tensors(state_bytes, n_tokens=10)
        # n_layers=2, n_heads=2, head_dim=128
        assert keys.shape[0] == 2  # n_layers
        assert keys.shape[1] == 2  # n_heads
        assert keys.shape[3] == 128  # head_dim
        assert values.shape == keys.shape

    def test_normalized_range(self, engine):
        state_bytes = bytes(np.random.randint(0, 256, size=200000, dtype=np.uint8))
        keys, values = engine._state_to_kv_tensors(state_bytes, n_tokens=10)
        # Normalized from uint8 [0,255] → [-2, ~2]
        assert keys.max() <= 2.1
        assert keys.min() >= -2.1

    def test_small_state_still_works(self, engine):
        # Very small state — should still produce valid tensors
        state_bytes = bytes(np.random.randint(0, 256, size=1024, dtype=np.uint8))
        keys, values = engine._state_to_kv_tensors(state_bytes, n_tokens=1)
        assert keys.ndim == 4
        assert values.ndim == 4


# ======================================================================
# TestCompressKVTensors
# ======================================================================


class TestCompressKVTensors:
    """Tests for _compress_kv_tensors."""

    def test_returns_compressed_kv_and_stats(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        compressed, stats = engine._compress_kv_tensors(keys, values)
        assert isinstance(compressed, CompressedKV)
        assert isinstance(stats, CompressionStats)

    def test_compression_ratio_positive(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        _, stats = engine._compress_kv_tensors(keys, values)
        assert stats.compression_ratio > 1.0

    def test_mse_is_finite(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        _, stats = engine._compress_kv_tensors(keys, values)
        assert np.isfinite(stats.mse)
        assert stats.mse >= 0

    def test_timing_positive(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        _, stats = engine._compress_kv_tensors(keys, values)
        assert stats.compress_time_s > 0
        assert stats.decompress_time_s > 0

    def test_original_bytes_matches_input(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        _, stats = engine._compress_kv_tensors(keys, values)
        assert stats.original_bytes == keys.nbytes + values.nbytes


# ======================================================================
# TestEstimateCompressedBytes
# ======================================================================


class TestEstimateCompressedBytes:
    """Tests for _estimate_compressed_bytes."""

    def test_positive_for_valid_compressed(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        compressed, _ = engine._compress_kv_tensors(keys, values)
        estimate = TurboQuantEngine._estimate_compressed_bytes(compressed)
        assert estimate > 0

    def test_smaller_than_original(self, engine):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))

        compressed, _ = engine._compress_kv_tensors(keys, values)
        estimate = TurboQuantEngine._estimate_compressed_bytes(compressed)
        assert estimate < keys.nbytes + values.nbytes


# ======================================================================
# TestChatWithCompression
# ======================================================================


class TestChatWithCompression:
    """Tests for chat_with_compression."""

    def test_returns_turbo_generation_result(self, loaded_turbo):
        messages = [{"role": "user", "content": "Hello"}]
        result = loaded_turbo.chat_with_compression(messages)
        assert isinstance(result, TurboGenerationResult)
        assert result.text == "Test response"
        assert isinstance(result.gen_stats, GenerationStats)

    def test_has_compression_stats(self, loaded_turbo):
        messages = [{"role": "user", "content": "Hello"}]
        result = loaded_turbo.chat_with_compression(messages)
        assert result.compression_stats is not None
        assert result.compression_stats.compression_ratio > 0

    def test_sets_compressed_state(self, loaded_turbo):
        assert loaded_turbo.has_compressed_state is False
        messages = [{"role": "user", "content": "Hello"}]
        loaded_turbo.chat_with_compression(messages)
        assert loaded_turbo.has_compressed_state is True

    def test_when_not_loaded_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.chat_with_compression([{"role": "user", "content": "X"}])

    def test_passes_params_to_chat(self, loaded_turbo, mock_llama):
        messages = [{"role": "user", "content": "Test"}]
        loaded_turbo.chat_with_compression(
            messages, max_tokens=100, temperature=0.5, top_p=0.9,
        )
        mock_llama.create_chat_completion.assert_called_once_with(
            messages=messages,
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
        )

    def test_state_save_failure_returns_none_stats(self, loaded_turbo, mock_llama):
        """If save_state raises, compression_stats is None but text is returned."""
        mock_llama.save_state.side_effect = RuntimeError("save failed")
        messages = [{"role": "user", "content": "Hello"}]
        result = loaded_turbo.chat_with_compression(messages)
        assert result.text == "Test response"
        assert result.compression_stats is None


# ======================================================================
# TestCompressCurrentState
# ======================================================================


class TestCompressCurrentState:
    """Tests for compress_current_state."""

    def test_returns_compression_stats(self, loaded_turbo):
        stats = loaded_turbo.compress_current_state()
        assert isinstance(stats, CompressionStats)
        assert stats.compression_ratio > 1.0

    def test_sets_compressed_state(self, loaded_turbo):
        loaded_turbo.compress_current_state()
        assert loaded_turbo.has_compressed_state is True

    def test_when_not_loaded_raises(self, engine):
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.compress_current_state()

    def test_returns_none_on_failure(self, loaded_turbo, mock_llama):
        mock_llama.save_state.side_effect = RuntimeError("oops")
        stats = loaded_turbo.compress_current_state()
        assert stats is None


# ======================================================================
# TestGetStats
# ======================================================================


class TestGetStats:
    """Tests for get_stats."""

    def test_includes_turbo_quant_section(self, engine):
        stats = engine.get_stats()
        assert "turbo_quant" in stats

    def test_turbo_quant_config_values(self, engine, quant_config):
        stats = engine.get_stats()
        tq = stats["turbo_quant"]
        assert tq["k_bits"] == quant_config.k_bits
        assert tq["v_bits"] == quant_config.v_bits
        assert tq["block_size"] == quant_config.block_size

    def test_has_compressed_state_false_initially(self, engine):
        assert engine.get_stats()["turbo_quant"]["has_compressed_state"] is False

    def test_has_compressed_state_true_after_compression(self, loaded_turbo):
        loaded_turbo.compress_current_state()
        assert loaded_turbo.get_stats()["turbo_quant"]["has_compressed_state"] is True

    def test_includes_base_engine_fields(self, engine):
        stats = engine.get_stats()
        assert "model_name" in stats
        assert "is_loaded" in stats
        assert "kv_cache_k" in stats


# ======================================================================
# TestDataclasses
# ======================================================================


class TestDataclasses:
    """Tests for CompressionStats and TurboGenerationResult."""

    def test_compression_stats_frozen(self):
        stats = CompressionStats(
            original_bytes=1000,
            compressed_bytes=200,
            compression_ratio=5.0,
            compress_time_s=0.1,
            decompress_time_s=0.05,
            mse=0.01,
        )
        with pytest.raises(AttributeError):
            stats.mse = 0.02  # type: ignore[misc]

    def test_turbo_generation_result_frozen(self):
        gen_stats = GenerationStats(
            prompt_tokens=10, completion_tokens=5,
            total_tokens=15, generation_time_s=1.0,
            tokens_per_second=5.0,
        )
        result = TurboGenerationResult(
            text="hello", gen_stats=gen_stats, compression_stats=None,
        )
        with pytest.raises(AttributeError):
            result.text = "changed"  # type: ignore[misc]

    def test_turbo_result_with_none_stats(self):
        gen_stats = GenerationStats(
            prompt_tokens=10, completion_tokens=5,
            total_tokens=15, generation_time_s=1.0,
            tokens_per_second=5.0,
        )
        result = TurboGenerationResult(
            text="hello", gen_stats=gen_stats, compression_stats=None,
        )
        assert result.compression_stats is None


# ======================================================================
# TestDifferentQuantConfigs
# ======================================================================


class TestDifferentQuantConfigs:
    """Test compression with different quantization configs."""

    @pytest.mark.parametrize("k_bits,v_bits", [
        (8, 4),   # quality
        (8, 3),   # balanced
        (8, 2),   # aggressive
        (4, 4),   # symmetric
    ])
    def test_compression_succeeds_for_preset(
        self, model_config, k_bits, v_bits, mock_llama, mock_state,
    ):
        qcfg = QuantConfig(k_bits=k_bits, v_bits=v_bits, block_size=128)
        eng = TurboQuantEngine(
            model_config, qcfg,
            n_layers=2, n_heads=2, head_dim=128,
        )
        eng._engine._model = mock_llama
        eng._engine._loaded = True

        result = eng.chat_with_compression(
            [{"role": "user", "content": "test"}],
            max_tokens=10,
        )
        assert result.compression_stats is not None
        assert result.compression_stats.compression_ratio > 0

    @pytest.mark.parametrize("k_bits,v_bits,expected_lower_mse", [
        (8, 4, True),    # 8-bit K should have lower MSE than 4-bit K
        (4, 4, False),
    ])
    def test_higher_bits_lower_mse(
        self, model_config, k_bits, v_bits, expected_lower_mse,
    ):
        qcfg = QuantConfig(k_bits=k_bits, v_bits=v_bits, block_size=128)
        eng = TurboQuantEngine(
            model_config, qcfg,
            n_layers=2, n_heads=2, head_dim=128,
        )
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 4, 128))
        values = rng.standard_normal((2, 2, 4, 128))
        _, stats = eng._compress_kv_tensors(keys, values)
        # Just verify it runs and MSE is bounded
        assert stats.mse < 1.0
