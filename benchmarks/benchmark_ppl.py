#!/usr/bin/env python3
"""Perplexity benchmark for evaluating LLM KV cache quantization configs.

Evaluates perplexity on the WikiText-2 test set across different KV cache
quantization configurations to measure the quality impact of TurboQuant
compression (PolarQuant codebooks for KV cache values).

The f16/f16 baseline is always evaluated first so that subsequent configs
can report their delta.  Results are printed as a table and persisted as JSON.

Usage::

    python -m benchmarks.benchmark_ppl \\
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf

    python -m benchmarks.benchmark_ppl \\
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --n-ctx 4096 --configs f16/f16 q8_0/turbo4

    python -m benchmarks.benchmark_ppl \\
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --max-tokens 2000  # quick sanity run

Requires the ``bench`` extras::

    pip install -e ".[bench,gpu]"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow execution as both ``python benchmarks/benchmark_ppl.py``
# and ``python -m benchmarks.benchmark_ppl`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.kv_cache import (  # noqa: E402
    MODEL_ARCHITECTURES,
    CacheType,
    KVCacheConfig,
    estimate_kv_memory_bytes,
    to_llama_params,
)
from src.engine.model_config import ModelConfig  # noqa: E402
from src.utils.memory import get_gpu_memory  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS: dict[str, KVCacheConfig] = {
    "f16/f16": KVCacheConfig(
        cache_type_k=CacheType.F16,
        cache_type_v=CacheType.F16,
        flash_attention=True,
    ),
    "q8_0/q8_0": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.Q8_0,
        flash_attention=True,
    ),
    "q8_0/turbo4": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO4,
        flash_attention=True,
    ),
    "q8_0/turbo3": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO3,
        flash_attention=True,
    ),
    "q8_0/turbo2": KVCacheConfig(
        cache_type_k=CacheType.Q8_0,
        cache_type_v=CacheType.TURBO2,
        flash_attention=True,
    ),
    "q4_0/q4_0": KVCacheConfig(
        cache_type_k=CacheType.Q4_0,
        cache_type_v=CacheType.Q4_0,
        flash_attention=True,
    ),
}

DEFAULT_ARCH = "qwen2.5-7b"
_EVAL_BATCH_SIZE = 512  # positions per batch for log-prob computation


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PPLResult:
    """Result of a single perplexity evaluation run."""

    config_name: str
    cache_type_k: str
    cache_type_v: str
    perplexity: float
    ppl_delta_vs_f16: float
    ppl_delta_percent: float
    kv_memory_mb: float
    compression_vs_f16: float
    gpu_memory_used_gb: float
    eval_time_s: float
    tokens_evaluated: int


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable log-softmax over the last axis.

    Args:
        logits: Array of shape ``(N, V)`` where *V* is the vocabulary size.

    Returns:
        Log-probabilities with the same shape as *logits*.
    """
    max_vals = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_vals
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_z


def _snapshot_gpu_memory() -> float:
    """Return current GPU memory usage in GB, or ``0.0`` if unavailable."""
    mem = get_gpu_memory()
    return mem.used_gb if mem is not None else 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_wikitext2_text() -> str:
    """Load and concatenate the WikiText-2 *raw* test split.

    Empty / whitespace-only lines are filtered out so that the resulting
    text is a continuous corpus suitable for perplexity evaluation.

    Returns:
        Full text of the WikiText-2 test set.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for this benchmark. "
            "Install with: pip install -e '.[bench]'"
        )

    logger.info("Loading WikiText-2 test set from HuggingFace…")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in ds["text"] if line.strip())
    logger.info("Loaded %d characters of text", len(text))
    return text


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
def _create_llama_model(
    model_config: ModelConfig,
    kv_config: KVCacheConfig,
) -> Any:
    """Create a ``llama_cpp.Llama`` instance with ``logits_all=True``.

    We instantiate the model directly (rather than through
    :class:`InferenceEngine`) because perplexity computation requires
    per-position logits (``logits_all=True``), which ``InferenceEngine``
    does not expose.  All configuration values are still sourced from
    the project's existing ``ModelConfig`` and ``KVCacheConfig`` types.

    Args:
        model_config: Model path, context size, GPU layer configuration.
        kv_config: KV cache quantization parameters.

    Returns:
        A ``llama_cpp.Llama`` instance ready for evaluation.

    Raises:
        ImportError: If ``llama-cpp-python`` is not installed.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is required for this benchmark. "
            "Install with: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
        )

    kv_params = to_llama_params(kv_config)

    logger.info(
        "Loading model %s (n_ctx=%d, K=%s, V=%s, logits_all=True)",
        model_config.model_name,
        model_config.n_ctx,
        kv_config.cache_type_k.value,
        kv_config.cache_type_v.value,
    )

    start = time.monotonic()
    model = Llama(
        model_path=model_config.model_path,
        n_ctx=model_config.n_ctx,
        n_gpu_layers=model_config.n_gpu_layers,
        chat_format=model_config.chat_format,
        verbose=False,
        logits_all=True,
        **kv_params,
    )
    logger.info("Model loaded in %.1fs", time.monotonic() - start)
    return model


def _unload_model(model: Any) -> None:
    """Release model memory and trigger garbage collection."""
    del model
    gc.collect()
    logger.info("Model unloaded")


# ---------------------------------------------------------------------------
# Core PPL computation
# ---------------------------------------------------------------------------
def _compute_window_nll(
    scores: np.ndarray,
    tokens: list[int],
    score_start: int,
    score_end: int,
    batch_size: int = _EVAL_BATCH_SIZE,
) -> tuple[float, int]:
    """Compute negative log-likelihood for a contiguous slice of a window.

    Processes positions in mini-batches to avoid allocating an
    ``(effective_trg, vocab_size)`` array all at once — important when
    the vocabulary is large (e.g. 150 k+ for Qwen-2.5).

    Args:
        scores: Logits array from ``model.scores`` — shape
            ``(n_tokens, vocab_size)``.  ``scores[i]`` contains the logits
            predicting the token at position ``i + 1``.
        tokens: Token IDs for the current chunk.
        score_start: First score index to evaluate (inclusive).
        score_end: Last score index to evaluate (exclusive).
        batch_size: Positions per mini-batch.

    Returns:
        ``(total_nll, n_evaluated)`` — summed NLL and count of positions.
    """
    total_nll = 0.0
    n_evaluated = 0

    for b_start in range(score_start, score_end, batch_size):
        b_end = min(b_start + batch_size, score_end)

        batch_logits = np.asarray(scores[b_start:b_end], dtype=np.float32)
        # scores[i] predicts tokens[i + 1]
        batch_targets = np.array(
            tokens[b_start + 1 : b_end + 1], dtype=np.int64,
        )

        log_probs = _log_softmax(batch_logits)
        token_nlls = -log_probs[np.arange(len(batch_targets)), batch_targets]

        total_nll += float(token_nlls.sum())
        n_evaluated += len(batch_targets)

    return total_nll, n_evaluated


def compute_perplexity(
    model: Any,
    text: str,
    n_ctx: int,
    stride: int | None = None,
    max_tokens: int | None = None,
) -> tuple[float, int, float]:
    """Compute perplexity on *text* using a sliding-window approach.

    For documents longer than ``n_ctx`` the text is processed in
    overlapping windows of size ``n_ctx`` advanced by *stride* tokens.
    Only the **non-overlapping** suffix of each window contributes to the
    loss, so no token is double-counted.

    Formula:  ``PPL = exp(-1/N * Σ log P(token_i | context))``

    Args:
        model: A ``llama_cpp.Llama`` instance created with
            ``logits_all=True``.
        text: The evaluation corpus.
        n_ctx: Context window size in tokens.
        stride: Sliding-window stride.  Defaults to ``n_ctx // 2``.
        max_tokens: Optional hard limit on the number of tokens to
            evaluate (useful for quick sanity runs).

    Returns:
        ``(perplexity, tokens_evaluated, eval_time_seconds)``

    Raises:
        ValueError: If the tokenized text has fewer than 2 tokens.
    """
    if stride is None:
        stride = n_ctx // 2

    tokens: list[int] = model.tokenize(text.encode(), add_bos=False)

    if max_tokens is not None and max_tokens > 0:
        tokens = tokens[:max_tokens]

    seq_len = len(tokens)
    if seq_len < 2:
        raise ValueError(
            f"Text too short for perplexity evaluation: {seq_len} token(s)"
        )

    n_windows = max(1, math.ceil(max(seq_len - n_ctx, 0) / stride) + 1)
    logger.info(
        "Evaluating perplexity: %d tokens, n_ctx=%d, stride=%d, ~%d windows",
        seq_len,
        n_ctx,
        stride,
        n_windows,
    )

    total_nll = 0.0
    total_count = 0
    prev_end = 0
    start_time = time.monotonic()

    for win_idx, begin in enumerate(range(0, seq_len, stride)):
        end = min(begin + n_ctx, seq_len)
        trg_len = end - prev_end
        chunk = tokens[begin:end]
        chunk_len = len(chunk)

        if chunk_len < 2:
            break

        # Number of *new* (non-overlapping) tokens to score this window.
        # Capped at chunk_len - 1 because the first position has no
        # left-context prediction within this window.
        effective_trg = min(trg_len, chunk_len - 1)

        # scores[i] predicts tokens[i+1], so the slice we need is:
        #   scores[score_start : score_end]  →  targets[score_start+1 : score_end+1]
        score_start = chunk_len - effective_trg - 1
        score_end = chunk_len - 1  # exclusive

        # Clear KV cache and evaluate the full chunk
        model.reset()
        model.eval(chunk)

        nll, n_eval = _compute_window_nll(
            model.scores, chunk, score_start, score_end,
        )

        total_nll += nll
        total_count += n_eval

        if (win_idx + 1) % 5 == 0 or win_idx == 0 or end >= seq_len:
            elapsed = time.monotonic() - start_time
            logger.info(
                "  window %d/%d — %d tokens scored, %.1fs elapsed",
                win_idx + 1,
                n_windows,
                total_count,
                elapsed,
            )

        prev_end = end
        if end >= seq_len:
            break

    elapsed = time.monotonic() - start_time

    if total_count == 0:
        raise ValueError("No tokens were evaluated — corpus may be too short")

    ppl = math.exp(total_nll / total_count)
    logger.info(
        "Perplexity = %.4f  (%d tokens, %.1fs)", ppl, total_count, elapsed,
    )
    return ppl, total_count, elapsed


# ---------------------------------------------------------------------------
# Single-config evaluation
# ---------------------------------------------------------------------------
def evaluate_config(
    config_name: str,
    kv_config: KVCacheConfig,
    model_config: ModelConfig,
    text: str,
    arch: dict[str, int],
    baseline_ppl: float | None,
    max_tokens: int | None = None,
) -> PPLResult:
    """Load the model, compute perplexity, and return a frozen result.

    The model is loaded *and unloaded* for every config so that GPU
    memory measurements reflect the config in isolation.

    Args:
        config_name: Human-readable label (e.g. ``"q8_0/turbo4"``).
        kv_config: KV cache quantization configuration.
        model_config: Model path, context size, GPU layer offloading.
        text: Evaluation corpus.
        arch: Model-architecture dict with ``n_layers``, ``n_heads``,
            ``head_dim`` (from :data:`MODEL_ARCHITECTURES`).
        baseline_ppl: f16/f16 perplexity for delta computation;
            ``None`` when evaluating the baseline itself.
        max_tokens: Optional token limit for quick runs.

    Returns:
        A frozen :class:`PPLResult`.
    """
    mem_est = estimate_kv_memory_bytes(
        n_ctx=model_config.n_ctx,
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        head_dim=arch["head_dim"],
        config=kv_config,
    )

    model = _create_llama_model(model_config, kv_config)
    try:
        gpu_mem = _snapshot_gpu_memory()
        ppl, n_tokens, eval_time = compute_perplexity(
            model, text, n_ctx=model_config.n_ctx, max_tokens=max_tokens,
        )
    finally:
        _unload_model(model)

    delta = (ppl - baseline_ppl) if baseline_ppl is not None else 0.0
    delta_pct = (delta / baseline_ppl * 100.0) if baseline_ppl else 0.0

    return PPLResult(
        config_name=config_name,
        cache_type_k=kv_config.cache_type_k.value,
        cache_type_v=kv_config.cache_type_v.value,
        perplexity=round(ppl, 4),
        ppl_delta_vs_f16=round(delta, 4),
        ppl_delta_percent=round(delta_pct, 2),
        kv_memory_mb=round(mem_est["total_mb"], 1),
        compression_vs_f16=round(mem_est["compression_vs_f16"], 2),
        gpu_memory_used_gb=round(gpu_mem, 2),
        eval_time_s=round(eval_time, 1),
        tokens_evaluated=n_tokens,
    )


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    model_config: ModelConfig,
    configs: dict[str, KVCacheConfig],
    arch_name: str = DEFAULT_ARCH,
    max_tokens: int | None = None,
) -> list[PPLResult]:
    """Run the perplexity benchmark across all requested configs.

    The ``f16/f16`` baseline is always evaluated first (if present) so
    that every subsequent config can report its delta.

    Args:
        model_config: Model path and context configuration.
        configs: Ordered mapping of *config_name* → :class:`KVCacheConfig`.
        arch_name: Key into :data:`MODEL_ARCHITECTURES` for KV memory
            estimation.
        max_tokens: Optional per-config token limit for quick runs.

    Returns:
        One :class:`PPLResult` per config, in evaluation order.

    Raises:
        ValueError: If *arch_name* is not found in ``MODEL_ARCHITECTURES``.
    """
    arch = MODEL_ARCHITECTURES.get(arch_name)
    if arch is None:
        available = ", ".join(sorted(MODEL_ARCHITECTURES))
        raise ValueError(
            f"Unknown architecture '{arch_name}'. Available: {available}"
        )

    text = load_wikitext2_text()

    results: list[PPLResult] = []
    baseline_ppl: float | None = None

    for idx, (name, kv_cfg) in enumerate(configs.items(), 1):
        logger.info(
            "=== Config %d/%d: %s ===", idx, len(configs), name,
        )
        try:
            result = evaluate_config(
                config_name=name,
                kv_config=kv_cfg,
                model_config=model_config,
                text=text,
                arch=arch,
                baseline_ppl=baseline_ppl,
                max_tokens=max_tokens,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Skipping config %s: %s", name, exc,
            )
            continue
        results.append(result)

        if baseline_ppl is None:
            baseline_ppl = result.perplexity
            logger.info("Baseline PPL (f16/f16): %.4f", baseline_ppl)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_results_table(results: list[PPLResult]) -> None:
    """Print a human-readable results table to stdout."""
    from tabulate import tabulate

    headers = [
        "Config",
        "PPL",
        "Δ PPL",
        "Δ %",
        "KV MB",
        "Compress",
        "GPU GB",
        "Time (s)",
        "Tokens",
    ]
    rows = [
        [
            r.config_name,
            f"{r.perplexity:.4f}",
            f"{r.ppl_delta_vs_f16:+.4f}" if r.ppl_delta_vs_f16 else "baseline",
            f"{r.ppl_delta_percent:+.2f}%" if r.ppl_delta_percent else "—",
            f"{r.kv_memory_mb:.1f}",
            f"{r.compression_vs_f16:.2f}x",
            f"{r.gpu_memory_used_gb:.2f}",
            f"{r.eval_time_s:.1f}",
            f"{r.tokens_evaluated:,}",
        ]
        for r in results
    ]

    print("\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n")


def save_results(
    results: list[PPLResult],
    model_config: ModelConfig,
    output_dir: Path,
) -> Path:
    """Persist benchmark results as JSON.

    Args:
        results: Evaluation results to serialise.
        model_config: The model configuration used for the run.
        output_dir: Target directory (created if absent).

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model": model_config.model_name,
        "n_ctx": model_config.n_ctx,
        "dataset": "wikitext-2-raw-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(r) for r in results],
    }

    out_path = output_dir / "ppl_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Results saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_configs(
    selected: list[str] | None,
) -> dict[str, KVCacheConfig]:
    """Validate and order the requested config names.

    When ``f16/f16`` is in the selection it is moved to the front so
    it serves as the baseline for delta computation.
    """
    if selected is None:
        return dict(BENCHMARK_CONFIGS)

    resolved: dict[str, KVCacheConfig] = {}
    for name in selected:
        if name not in BENCHMARK_CONFIGS:
            available = ", ".join(BENCHMARK_CONFIGS)
            raise SystemExit(
                f"Error: unknown config '{name}'. Available: {available}"
            )
        resolved[name] = BENCHMARK_CONFIGS[name]

    # Ensure baseline comes first when present
    if "f16/f16" in resolved:
        baseline = resolved.pop("f16/f16")
        resolved = {"f16/f16": baseline, **resolved}

    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perplexity benchmark for KV cache quantization configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf --n-ctx 4096
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf --configs f16/f16 q8_0/turbo4
  %(prog)s --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf --max-tokens 2000
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--model-name",
        default="qwen2.5-7b-instruct",
        help="Model name label for reporting (default: %(default)s)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Context window size in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Layers to offload to GPU; -1 = all (default: %(default)s)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="CFG",
        help=(
            "KV cache configs to benchmark (default: all). "
            f"Choices: {', '.join(BENCHMARK_CONFIGS)}"
        ),
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        choices=sorted(MODEL_ARCHITECTURES),
        help="Model architecture for memory estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for the output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Limit evaluation to N tokens (for quick testing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the perplexity benchmark."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        raise SystemExit(1)

    model_config = ModelConfig(
        model_path=str(model_path),
        model_name=args.model_name,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    configs = _resolve_configs(args.configs)

    logger.info(
        "Starting benchmark: model=%s, n_ctx=%d, configs=%s",
        model_config.model_name,
        model_config.n_ctx,
        list(configs),
    )

    results = run_benchmark(
        model_config=model_config,
        configs=configs,
        arch_name=args.arch,
        max_tokens=args.max_tokens,
    )

    print_results_table(results)

    out_path = save_results(results, model_config, args.output_dir)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
