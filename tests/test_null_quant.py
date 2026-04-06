"""Tests for src.turboquant.null_quant — NullQuant token eviction + zone compression."""

from __future__ import annotations

import numpy as np
import pytest

from src.turboquant.null_quant import (
    NullQuantCompressedKV,
    NullQuantCompressor,
    NullQuantConfig,
    TokenEvictor,
    score_tokens_l2_norm,
    score_tokens_random,
    score_tokens_uniform_stride,
    select_survivors,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> NullQuantConfig:
    return NullQuantConfig()


@pytest.fixture
def small_kv() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic KV: 4 layers, 4 heads, 64 tokens, 32 head_dim."""
    rng = np.random.default_rng(42)
    shape = (4, 4, 64, 32)
    return rng.standard_normal(shape), rng.standard_normal(shape)


@pytest.fixture
def medium_kv() -> tuple[np.ndarray, np.ndarray]:
    """Medium synthetic KV: 4 layers, 4 heads, 512 tokens, 32 head_dim."""
    rng = np.random.default_rng(123)
    shape = (4, 4, 512, 32)
    return rng.standard_normal(shape), rng.standard_normal(shape)


# ── NullQuantConfig Tests ─────────────────────────────────────────────────────


class TestNullQuantConfig:
    """Verify NullQuantConfig defaults, validation, and immutability."""

    def test_default_config(self) -> None:
        cfg = NullQuantConfig()
        assert cfg.eviction_ratio == 0.75
        assert cfg.sink_tokens == 256
        assert cfg.recent_tokens == 256
        assert cfg.scoring_method == "l2_norm"
        assert cfg.block_size == 64

    def test_immutable(self) -> None:
        cfg = NullQuantConfig()
        with pytest.raises(AttributeError):
            cfg.eviction_ratio = 0.5  # type: ignore[misc]

    def test_invalid_eviction_ratio(self) -> None:
        with pytest.raises(ValueError, match="eviction_ratio"):
            NullQuantConfig(eviction_ratio=0.0)
        with pytest.raises(ValueError, match="eviction_ratio"):
            NullQuantConfig(eviction_ratio=1.0)
        with pytest.raises(ValueError, match="eviction_ratio"):
            NullQuantConfig(eviction_ratio=-0.1)

    def test_invalid_sink_tokens(self) -> None:
        with pytest.raises(ValueError, match="sink_tokens"):
            NullQuantConfig(sink_tokens=-1)

    def test_invalid_scoring_method(self) -> None:
        with pytest.raises(ValueError, match="scoring_method"):
            NullQuantConfig(scoring_method="invalid")

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError, match="middle_v_bits"):
            NullQuantConfig(middle_v_bits=5)

    def test_invalid_compress_block_size(self) -> None:
        with pytest.raises(ValueError, match="compress_block_size"):
            NullQuantConfig(compress_block_size=3)
        with pytest.raises(ValueError, match="compress_block_size"):
            NullQuantConfig(compress_block_size=0)

    def test_estimated_reduction(self) -> None:
        cfg = NullQuantConfig()
        est = cfg.estimated_reduction(2048, 28)
        # With 75% eviction (4x) and zone compression (~3-4x), expect > 5x
        assert est > 5.0
        assert est < 200.0

    def test_custom_config(self) -> None:
        cfg = NullQuantConfig(eviction_ratio=0.5, middle_v_bits=4)
        assert cfg.eviction_ratio == 0.5
        assert cfg.middle_v_bits == 4


# ── Token Scoring Tests ───────────────────────────────────────────────────────


class TestTokenScoring:
    """Verify token scoring functions produce correct shapes and properties."""

    def test_l2_norm_shape(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, _ = small_kv
        scores = score_tokens_l2_norm(keys)
        assert scores.shape == (64,)  # seq_len

    def test_l2_norm_positive(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, _ = small_kv
        scores = score_tokens_l2_norm(keys)
        assert np.all(scores >= 0)

    def test_l2_norm_distinguishes_tokens(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, _ = small_kv
        scores = score_tokens_l2_norm(keys)
        # Not all scores should be identical
        assert np.std(scores) > 0

    def test_l2_norm_high_for_large_keys(self) -> None:
        """Tokens with larger key vectors should get higher scores."""
        rng = np.random.default_rng(99)
        keys = rng.standard_normal((2, 2, 10, 8))
        # Make token 5 very large
        keys[:, :, 5, :] *= 100
        scores = score_tokens_l2_norm(keys)
        assert np.argmax(scores) == 5

    def test_random_shape(self) -> None:
        scores = score_tokens_random(100)
        assert scores.shape == (100,)

    def test_random_reproducible(self) -> None:
        s1 = score_tokens_random(50, seed=42)
        s2 = score_tokens_random(50, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_random_different_seeds(self) -> None:
        s1 = score_tokens_random(50, seed=1)
        s2 = score_tokens_random(50, seed=2)
        assert not np.array_equal(s1, s2)

    def test_uniform_stride_shape(self) -> None:
        scores = score_tokens_uniform_stride(100, 0.25)
        assert scores.shape == (100,)

    def test_uniform_stride_quarter(self) -> None:
        scores = score_tokens_uniform_stride(100, 0.25)
        # Every 4th token should have score 1.0
        nonzero = np.nonzero(scores)[0]
        assert len(nonzero) == 25  # 100 / 4

    def test_uniform_stride_zero_keepratio(self) -> None:
        scores = score_tokens_uniform_stride(100, 0.0)
        assert np.all(scores == 0.0)


# ── select_survivors Tests ────────────────────────────────────────────────────


class TestSelectSurvivors:
    """Verify survivor selection with sink/recent protection and eviction ratios."""

    def test_basic_eviction(self) -> None:
        scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        survivors, evicted = select_survivors(scores, 0.4, 0, 0)
        # Keep 60% = 3 tokens (indices with highest scores: 2, 4, 1)
        assert len(survivors) == 3
        assert 2 in survivors  # highest score
        assert 4 in survivors
        assert len(evicted) == 2

    def test_sink_protection(self) -> None:
        scores = np.zeros(10)
        scores[5] = 100.0  # Only token 5 has high score
        survivors, evicted = select_survivors(scores, 0.5, 3, 0)
        # First 3 tokens always survive (sink)
        assert 0 in survivors
        assert 1 in survivors
        assert 2 in survivors
        assert 5 in survivors  # highest score

    def test_recent_protection(self) -> None:
        scores = np.zeros(10)
        scores[3] = 100.0
        survivors, evicted = select_survivors(scores, 0.5, 0, 3)
        # Last 3 tokens always survive (recent)
        assert 7 in survivors
        assert 8 in survivors
        assert 9 in survivors
        assert 3 in survivors  # highest score

    def test_both_protections(self) -> None:
        scores = np.ones(20) * 0.1
        survivors, evicted = select_survivors(scores, 0.75, 4, 4)
        # First 4 sink + last 4 recent = at least 8 protected
        for i in range(4):
            assert i in survivors
        for i in range(16, 20):
            assert i in survivors

    def test_empty_sequence(self) -> None:
        scores = np.array([], dtype=np.float64)
        survivors, evicted = select_survivors(scores, 0.5, 0, 0)
        assert len(survivors) == 0
        assert len(evicted) == 0

    def test_all_protected(self) -> None:
        """If sink+recent >= seq_len, all tokens survive."""
        scores = np.ones(10)
        survivors, evicted = select_survivors(scores, 0.5, 5, 5)
        assert len(survivors) == 10
        assert len(evicted) == 0

    def test_survivors_sorted(self) -> None:
        scores = np.random.default_rng(42).random(100)
        survivors, evicted = select_survivors(scores, 0.5, 10, 10)
        assert np.all(survivors[:-1] <= survivors[1:])  # monotonically increasing
        assert np.all(evicted[:-1] <= evicted[1:])

    def test_survivors_plus_evicted_equals_total(self) -> None:
        scores = np.random.default_rng(42).random(200)
        survivors, evicted = select_survivors(scores, 0.75, 20, 20)
        combined = np.union1d(survivors, evicted)
        np.testing.assert_array_equal(combined, np.arange(200))


# ── TokenEvictor Tests ────────────────────────────────────────────────────────


class TestTokenEvictor:
    """Verify TokenEvictor reduces sequence length while preserving dimensions."""

    def test_evict_reduces_seq_len(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        evictor = TokenEvictor(cfg)
        p_keys, p_values, evicted, t = evictor.evict(keys, values)
        assert p_keys.shape[2] < keys.shape[2]
        assert p_values.shape[2] < values.shape[2]
        assert p_keys.shape[2] == p_values.shape[2]
        assert len(evicted) > 0
        assert t > 0

    def test_evict_preserves_dims(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        evictor = TokenEvictor(cfg)
        p_keys, p_values, _, _ = evictor.evict(keys, values)
        # layers, heads, head_dim should be unchanged
        assert p_keys.shape[0] == keys.shape[0]
        assert p_keys.shape[1] == keys.shape[1]
        assert p_keys.shape[3] == keys.shape[3]

    def test_evict_skips_when_protected_exceeds_seq(self) -> None:
        """If sink+recent >= seq_len, should skip eviction."""
        cfg = NullQuantConfig(eviction_ratio=0.75, sink_tokens=50, recent_tokens=50)
        evictor = TokenEvictor(cfg)
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((2, 2, 20, 8))
        values = rng.standard_normal((2, 2, 20, 8))
        p_keys, p_values, evicted, _ = evictor.evict(keys, values)
        # All tokens should survive
        assert p_keys.shape[2] == 20
        assert len(evicted) == 0

    def test_evict_with_random_scoring(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(
            eviction_ratio=0.5,
            scoring_method="random",
            sink_tokens=4,
            recent_tokens=4,
        )
        evictor = TokenEvictor(cfg)
        p_keys, _, _, _ = evictor.evict(keys, values)
        assert p_keys.shape[2] < keys.shape[2]

    def test_evict_with_stride_scoring(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(
            eviction_ratio=0.75,
            scoring_method="uniform_stride",
            sink_tokens=4,
            recent_tokens=4,
        )
        evictor = TokenEvictor(cfg)
        p_keys, _, _, _ = evictor.evict(keys, values)
        assert p_keys.shape[2] < keys.shape[2]


# ── NullQuantCompressor Tests ─────────────────────────────────────────────────


class TestNullQuantCompressor:
    """Verify compress → decompress roundtrip and compressor properties."""

    def test_compress_decompress_roundtrip(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)

        compressed = compressor.compress_kv(keys, values)
        dec_keys, dec_values = compressor.decompress_kv(compressed)

        # Shape should match surviving count
        assert dec_keys.shape[2] == compressed.n_surviving_tokens
        assert dec_values.shape[2] == compressed.n_surviving_tokens
        assert dec_keys.shape[0] == keys.shape[0]  # layers
        assert dec_keys.shape[1] == keys.shape[1]  # heads
        assert dec_keys.shape[3] == keys.shape[3]  # head_dim

    def test_compression_reduces_memory(self, medium_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = medium_kv
        cfg = NullQuantConfig(eviction_ratio=0.75, sink_tokens=32, recent_tokens=32)
        compressor = NullQuantCompressor(cfg)

        original_bytes = keys.nbytes + values.nbytes
        compressed = compressor.compress_kv(keys, values)
        assert compressed.memory_bytes < original_bytes

    def test_evicted_positions_sorted(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        # Evicted positions should be sorted
        assert np.all(compressed.evicted_positions[:-1] <= compressed.evicted_positions[1:])

    def test_original_tokens_preserved(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        assert compressed.n_original_tokens == 64  # seq_len

    def test_surviving_tokens_correct(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        expected = compressed.n_original_tokens - len(compressed.evicted_positions)
        assert compressed.n_surviving_tokens == expected

    def test_invalid_input_dims(self) -> None:
        cfg = NullQuantConfig()
        compressor = NullQuantCompressor(cfg)
        with pytest.raises(ValueError, match="4D"):
            compressor.compress_kv(np.zeros((10, 10)), np.zeros((10, 10)))

    def test_mismatched_kv_shapes(self) -> None:
        cfg = NullQuantConfig()
        compressor = NullQuantCompressor(cfg)
        with pytest.raises(ValueError, match="same shape"):
            compressor.compress_kv(
                np.zeros((2, 2, 10, 4)),
                np.zeros((2, 2, 8, 4)),
            )

    def test_zone_summary(self) -> None:
        cfg = NullQuantConfig()
        compressor = NullQuantCompressor(cfg)
        summary = compressor.zone_summary(28)
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_mse_reasonable(self, medium_kv: tuple[np.ndarray, np.ndarray]) -> None:
        """MSE should be small for survivors after compress+decompress."""
        keys, values = medium_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=32, recent_tokens=32)
        compressor = NullQuantCompressor(cfg)

        compressed = compressor.compress_kv(keys, values)
        dec_keys, dec_values = compressor.decompress_kv(compressed)

        # Get survivor positions
        all_pos = np.arange(512, dtype=np.int32)
        survivors = np.setdiff1d(all_pos, compressed.evicted_positions)
        orig_k = keys[:, :, survivors, :]
        orig_v = values[:, :, survivors, :]

        mse = float(np.mean((orig_k - dec_keys) ** 2 + (orig_v - dec_values) ** 2)) / 2
        # MSE should be < 1.0 for reasonable quantization
        assert mse < 1.0


# ── NullQuantCompressedKV Properties ──────────────────────────────────────────


class TestNullQuantCompressedKV:
    """Verify compressed KV metadata properties."""

    def test_eviction_ratio_actual(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        ratio = compressed.eviction_ratio_actual
        assert 0.0 < ratio < 1.0

    def test_total_reduction_positive(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        assert compressed.total_reduction > 1.0

    def test_memory_bytes_positive(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        assert compressed.memory_bytes > 0

    def test_timing_positive(self, small_kv: tuple[np.ndarray, np.ndarray]) -> None:
        keys, values = small_kv
        cfg = NullQuantConfig(eviction_ratio=0.5, sink_tokens=4, recent_tokens=4)
        compressor = NullQuantCompressor(cfg)
        compressed = compressor.compress_kv(keys, values)
        assert compressed.eviction_time_s >= 0
        assert compressed.compression_time_s >= 0
