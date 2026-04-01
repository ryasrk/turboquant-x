#!/usr/bin/env python3
"""Needle-In-A-Haystack (NIAH) benchmark for long-context retrieval accuracy.

Evaluates how well an LLM retrieves a hidden fact ("needle") embedded at
various depth positions within long filler contexts ("haystack"), across
different KV cache quantization configurations.

For each (config, context_length, depth) triple the benchmark:
1. Generates a haystack of filler text sized to the target token count.
2. Inserts a needle (random 8-char code) at the specified depth.
3. Asks the model to retrieve the code.
4. Scores: exact match → 1.0, partial → 0.5, miss → 0.0.
5. Averages over *num_trials* independent codes per triple.

Results are saved as JSON and visualised as per-config seaborn heatmaps.

Usage::

    python -m benchmarks.benchmark_niah \
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf

    python -m benchmarks.benchmark_niah \
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \
        --configs q8_0/turbo4 f16/f16 --n-ctx-max 4096 --num-trials 5

    python -m benchmarks.benchmark_niah \
        --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf \
        --output-dir benchmarks/results

Requires the ``bench`` extras::

    pip install -e ".[bench,gpu]"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow execution as both ``python benchmarks/benchmark_niah.py``
# and ``python -m benchmarks.benchmark_niah`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.engine.inference import GenerationStats, InferenceEngine  # noqa: E402
from src.engine.kv_cache import CacheType, KVCacheConfig  # noqa: E402
from src.engine.model_config import ModelConfig  # noqa: E402
from src.utils.memory import get_gpu_memory  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DEPTHS: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
DEFAULT_CONTEXT_LENGTHS: list[int] = [1024, 2048, 4096, 8192]
DEFAULT_NUM_TRIALS: int = 3
DEFAULT_SEED: int = 42

NEEDLE_TEMPLATE = "The secret code for the vault is: {code}"
RETRIEVAL_QUERY = (
    "What is the secret code for the vault? "
    "Answer with only the code, nothing else."
)

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
    "q4_0/q4_0": KVCacheConfig(
        cache_type_k=CacheType.Q4_0,
        cache_type_v=CacheType.Q4_0,
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
}

# Diverse filler paragraphs unrelated to codes or vaults.
_FILLER_PARAGRAPHS: tuple[str, ...] = (
    "The Amazon rainforest, often referred to as the lungs of the Earth, "
    "spans over five and a half million square kilometers across nine "
    "countries in South America. It is home to an estimated ten percent "
    "of all species on the planet, including jaguars, river dolphins, "
    "and thousands of species of insects yet to be catalogued by science.",

    "Classical music experienced a golden age during the eighteenth "
    "century, with composers such as Mozart, Haydn, and Beethoven pushing "
    "the boundaries of symphonic and chamber music. The development of "
    "the sonata form gave rise to more complex harmonic structures and "
    "emotionally expressive compositions that remain influential today.",

    "The field of oceanography studies the physical, chemical, biological, "
    "and geological aspects of the ocean. Deep-sea hydrothermal vents, "
    "discovered in the late nineteen seventies, host unique ecosystems "
    "that thrive without sunlight, deriving energy from chemosynthesis "
    "rather than photosynthesis.",

    "Ancient Roman engineering achievements, including aqueducts, roads, "
    "and concrete structures, laid the foundation for modern civil "
    "engineering. The Pantheon in Rome, built around 125 AD, features "
    "an unreinforced concrete dome that remains the largest of its kind "
    "nearly two thousand years after construction.",

    "Advances in satellite technology have transformed weather forecasting "
    "from short-range estimates to multi-day predictions with remarkable "
    "accuracy. Geostationary satellites orbit at approximately thirty-six "
    "thousand kilometers above the equator, providing continuous imagery "
    "of cloud patterns and atmospheric conditions.",

    "The history of tea cultivation dates back thousands of years to "
    "ancient China, where it was initially used for medicinal purposes. "
    "Today, tea is the second most consumed beverage worldwide after "
    "water, with major production centers in China, India, Kenya, and "
    "Sri Lanka producing millions of metric tons annually.",

    "Quantum computing leverages the principles of quantum mechanics, "
    "such as superposition and entanglement, to perform certain "
    "calculations exponentially faster than classical computers. While "
    "still in early stages, quantum processors have demonstrated "
    "advantages in optimization problems and molecular simulation.",

    "The Great Barrier Reef off the coast of Australia is the largest "
    "coral reef system in the world, stretching over two thousand three "
    "hundred kilometers. It supports an extraordinary diversity of marine "
    "life and is visible from outer space, yet faces significant threats "
    "from rising ocean temperatures and acidification.",

    "Renaissance art marked a profound shift in European visual culture, "
    "emphasizing perspective, human anatomy, and naturalistic lighting. "
    "Artists like Leonardo da Vinci, Michelangelo, and Raphael produced "
    "works that continue to be studied for their technical mastery and "
    "emotional depth centuries after their creation.",

    "Modern agricultural practices rely heavily on precision farming "
    "techniques including GPS-guided machinery, drone-based crop "
    "monitoring, and sensor-driven irrigation systems. These technologies "
    "help farmers optimize yields while reducing water usage and "
    "minimizing the environmental impact of food production.",
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialResult:
    """Result of a single needle retrieval trial."""

    needle_code: str
    model_response: str
    score: float
    latency_s: float


@dataclass(frozen=True)
class DepthResult:
    """Aggregated scores for one (config, context_length, depth) combination."""

    depth: float
    context_length: int
    trials: tuple[TrialResult, ...]
    mean_score: float
    mean_latency_s: float


@dataclass(frozen=True)
class ConfigResult:
    """Full NIAH results for a single KV cache configuration."""

    config_name: str
    cache_type_k: str
    cache_type_v: str
    scores: dict[str, dict[str, float]]
    average_accuracy: float
    retrieval_latency_s: dict[str, float]
    depth_results: tuple[DepthResult, ...]


# ---------------------------------------------------------------------------
# Needle / haystack generation
# ---------------------------------------------------------------------------
def generate_needle_code(rng: Random) -> str:
    """Generate a random 8-character alphanumeric code.

    Args:
        rng: Seeded Random instance for reproducibility.

    Returns:
        An 8-character string of lowercase letters and digits.
    """
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(8))


def build_haystack(
    model: Any,
    target_tokens: int,
    rng: Random,
) -> str:
    """Generate filler text of exactly *target_tokens* tokens.

    Shuffles filler paragraphs and repeats them until the target is met,
    then tokenizes and truncates to the exact count.

    Args:
        model: A ``llama_cpp.Llama`` instance for tokenization.
        target_tokens: Desired number of tokens in the haystack.
        rng: Seeded Random instance for paragraph ordering.

    Returns:
        A string that tokenizes to exactly *target_tokens* tokens.
    """
    paragraphs = list(_FILLER_PARAGRAPHS)
    rng.shuffle(paragraphs)

    # Build a long-enough raw string by cycling shuffled paragraphs
    repeats = (target_tokens // 40) + 2  # ~40–60 tokens per paragraph
    raw_parts: list[str] = []
    for i in range(repeats):
        raw_parts.append(paragraphs[i % len(paragraphs)])
    raw_text = " ".join(raw_parts)

    # Tokenize, truncate, decode for exact token count
    tokens = model.tokenize(raw_text.encode(), add_bos=False)
    tokens = tokens[:target_tokens]
    text: str = model.detokenize(tokens).decode("utf-8", errors="replace")
    return text


def insert_needle_at_depth(
    model: Any,
    haystack: str,
    needle: str,
    depth: float,
) -> str:
    """Insert a needle sentence into the haystack at the target token depth.

    The depth is measured as a fraction of the haystack token count.
    For depth 0.5 in a 1000-token haystack, the needle is inserted at
    the ~500th token boundary.

    Args:
        model: A ``llama_cpp.Llama`` instance for tokenization.
        haystack: The filler text.
        needle: The needle sentence to insert.
        depth: Fraction (0.0–1.0) indicating insertion position.

    Returns:
        The haystack with the needle inserted at the target depth.
    """
    haystack_tokens = model.tokenize(haystack.encode(), add_bos=False)
    insert_pos = int(len(haystack_tokens) * depth)
    # Clamp to valid range (leave at least 1 token on each side)
    insert_pos = max(1, min(insert_pos, len(haystack_tokens) - 1))

    before_tokens = haystack_tokens[:insert_pos]
    after_tokens = haystack_tokens[insert_pos:]

    before_text: str = model.detokenize(before_tokens).decode(
        "utf-8", errors="replace"
    )
    after_text: str = model.detokenize(after_tokens).decode(
        "utf-8", errors="replace"
    )

    return f"{before_text} {needle} {after_text}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_response(response: str, expected_code: str) -> float:
    """Score a model response against the expected needle code.

    Scoring rules:
    - Exact match (response contains the full code): 1.0
    - Partial match (response contains >= 4 consecutive chars): 0.5
    - No match: 0.0

    Args:
        response: The model's raw response text.
        expected_code: The 8-character code that was inserted.

    Returns:
        Score: 1.0, 0.5, or 0.0.
    """
    cleaned = response.strip().lower()
    code_lower = expected_code.lower()

    if code_lower in cleaned:
        return 1.0

    # Check for partial match — at least 4 consecutive characters
    for length in range(len(code_lower), 3, -1):
        for start in range(len(code_lower) - length + 1):
            substring = code_lower[start : start + length]
            if substring in cleaned:
                return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------
def run_single_trial(
    engine: InferenceEngine,
    model: Any,
    context_length: int,
    depth: float,
    rng: Random,
) -> TrialResult:
    """Run one needle-in-a-haystack trial.

    Generates a fresh needle code, builds the haystack with the needle
    inserted, and asks the model to retrieve it.

    Args:
        engine: Loaded InferenceEngine instance.
        model: The underlying ``llama_cpp.Llama`` for tokenization.
        context_length: Target total context length in tokens.
        depth: Needle insertion depth (0.0–1.0).
        rng: Seeded Random instance.

    Returns:
        A TrialResult with the score and latency.
    """
    code = generate_needle_code(rng)
    needle = NEEDLE_TEMPLATE.format(code=code)

    # Reserve tokens for the needle, system message overhead, and query.
    # Rough estimate: needle ~20 tokens, query ~30 tokens, overhead ~20 tokens.
    needle_tokens = len(model.tokenize(needle.encode(), add_bos=False))
    query_overhead = 70  # conservative estimate for system + query framing
    haystack_budget = max(context_length - needle_tokens - query_overhead, 64)

    haystack = build_haystack(model, haystack_budget, rng)
    full_context = insert_needle_at_depth(model, haystack, needle, depth)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Read the following text carefully "
                "and answer the question about it."
            ),
        },
        {
            "role": "user",
            "content": f"{full_context}\n\n{RETRIEVAL_QUERY}",
        },
    ]

    start = time.monotonic()
    response_msg, stats = engine.chat(
        messages=messages,
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
    )
    latency = time.monotonic() - start

    response_text = response_msg.get("content", "")
    trial_score = score_response(response_text, code)

    logger.debug(
        "  trial: depth=%.2f ctx=%d code=%s response=%r score=%.1f latency=%.2fs",
        depth,
        context_length,
        code,
        response_text[:80],
        trial_score,
        latency,
    )

    return TrialResult(
        needle_code=code,
        model_response=response_text,
        score=trial_score,
        latency_s=round(latency, 3),
    )


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------
def run_niah_benchmark(
    model_config: ModelConfig,
    configs: dict[str, KVCacheConfig],
    context_lengths: list[int],
    depths: list[float],
    num_trials: int,
    seed: int = DEFAULT_SEED,
) -> list[ConfigResult]:
    """Run the NIAH benchmark across all configs, lengths, and depths.

    For each config the engine is loaded once, evaluated across all
    (length, depth, trial) combinations, then unloaded.

    Args:
        model_config: Model path and context configuration.
        configs: Mapping of config_name → KVCacheConfig.
        context_lengths: Token counts to test.
        depths: Needle depth positions (0.0–1.0).
        num_trials: Independent trials per (config, length, depth).
        seed: RNG seed for reproducibility.

    Returns:
        List of ConfigResult, one per KV cache configuration.
    """
    # Filter lengths that exceed the model's n_ctx
    valid_lengths = [c for c in context_lengths if c <= model_config.n_ctx]
    if len(valid_lengths) < len(context_lengths):
        skipped = sorted(set(context_lengths) - set(valid_lengths))
        logger.warning(
            "Skipping context lengths %s (exceeds n_ctx=%d)",
            skipped,
            model_config.n_ctx,
        )

    results: list[ConfigResult] = []

    for cfg_idx, (cfg_name, kv_cfg) in enumerate(configs.items(), 1):
        logger.info(
            "=== Config %d/%d: %s ===", cfg_idx, len(configs), cfg_name,
        )

        engine = InferenceEngine(model_config, kv_cfg)
        try:
            engine.load_model()
        except (ValueError, RuntimeError) as exc:
            logger.warning("Skipping config %s: %s", cfg_name, exc)
            continue

        try:
            # Access the underlying llama model for tokenization
            model = engine._model

            all_depth_results: list[DepthResult] = []
            scores_dict: dict[str, dict[str, float]] = {}
            all_latencies: list[float] = []

            for ctx_len in valid_lengths:
                logger.info("--- Context length: %d tokens ---", ctx_len)
                scores_dict[str(ctx_len)] = {}

                for depth in depths:
                    # Use a deterministic per-combination seed
                    combo_seed = seed + hash((cfg_name, ctx_len, depth)) % (2**31)
                    rng = Random(combo_seed)

                    trials: list[TrialResult] = []
                    for t in range(num_trials):
                        trial = run_single_trial(
                            engine=engine,
                            model=model,
                            context_length=ctx_len,
                            depth=depth,
                            rng=rng,
                        )
                        trials.append(trial)

                    mean_score = round(
                        sum(tr.score for tr in trials) / len(trials), 3
                    )
                    mean_latency = round(
                        sum(tr.latency_s for tr in trials) / len(trials), 3
                    )
                    all_latencies.extend(tr.latency_s for tr in trials)

                    scores_dict[str(ctx_len)][str(depth)] = mean_score

                    depth_result = DepthResult(
                        depth=depth,
                        context_length=ctx_len,
                        trials=tuple(trials),
                        mean_score=mean_score,
                        mean_latency_s=mean_latency,
                    )
                    all_depth_results.append(depth_result)

                    logger.info(
                        "  depth=%.2f → mean_score=%.2f (%d trials)",
                        depth,
                        mean_score,
                        num_trials,
                    )

            # Compute overall stats
            all_scores = [dr.mean_score for dr in all_depth_results]
            avg_accuracy = round(
                sum(all_scores) / len(all_scores), 3
            ) if all_scores else 0.0

            latency_arr = np.array(all_latencies) if all_latencies else np.array([0.0])
            latency_stats = {
                "mean": round(float(np.mean(latency_arr)), 3),
                "std": round(float(np.std(latency_arr)), 3),
            }

            config_result = ConfigResult(
                config_name=cfg_name,
                cache_type_k=kv_cfg.cache_type_k.value,
                cache_type_v=kv_cfg.cache_type_v.value,
                scores=scores_dict,
                average_accuracy=avg_accuracy,
                retrieval_latency_s=latency_stats,
                depth_results=tuple(all_depth_results),
            )
            results.append(config_result)

            logger.info(
                "Config %s: avg_accuracy=%.3f, latency=%.2f±%.2fs",
                cfg_name,
                avg_accuracy,
                latency_stats["mean"],
                latency_stats["std"],
            )

        finally:
            engine.unload()
            gc.collect()

    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def build_report(
    results: list[ConfigResult],
    model_name: str,
    num_trials: int,
    depths: list[float],
    context_lengths: list[int],
) -> dict[str, Any]:
    """Build the JSON-serializable benchmark report.

    Args:
        results: List of ConfigResult from the benchmark run.
        model_name: Human-readable model name.
        num_trials: Number of trials per combination.
        depths: Depth positions tested.
        context_lengths: Context lengths tested.

    Returns:
        A dict matching the output JSON schema.
    """
    gpu_mem = get_gpu_memory()
    gpu_info = f"{gpu_mem.total_gb:.1f}GB VRAM" if gpu_mem else "unknown"

    return {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": gpu_info},
        "settings": {
            "num_trials": num_trials,
            "depths": depths,
            "context_lengths": context_lengths,
        },
        "results": [
            {
                "config_name": cr.config_name,
                "cache_type_k": cr.cache_type_k,
                "cache_type_v": cr.cache_type_v,
                "scores": cr.scores,
                "average_accuracy": cr.average_accuracy,
                "retrieval_latency_s": cr.retrieval_latency_s,
            }
            for cr in results
        ],
    }


def save_results(report: dict[str, Any], output_path: Path) -> None:
    """Write the benchmark report to a JSON file.

    Args:
        report: The serialized report dict.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def generate_heatmap(
    config_result: ConfigResult,
    depths: list[float],
    context_lengths: list[int],
    output_path: Path,
) -> None:
    """Generate and save a seaborn heatmap for one config's NIAH scores.

    X-axis: context length, Y-axis: needle depth.
    Color scale: 0 (red) → 1 (green).

    Args:
        config_result: Results for one KV cache config.
        depths: Depth positions (row labels).
        context_lengths: Context lengths (column labels).
        output_path: Path to save the PNG image.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Build the score matrix: rows = depths (top to bottom), cols = context lengths
    matrix = np.zeros((len(depths), len(context_lengths)))
    for i, depth in enumerate(depths):
        for j, ctx_len in enumerate(context_lengths):
            score = config_result.scores.get(str(ctx_len), {}).get(
                str(depth), float("nan")
            )
            matrix[i, j] = score

    fig, ax = plt.subplots(figsize=(max(8, len(context_lengths) * 1.8), max(5, len(depths) * 1.2)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        xticklabels=[str(c) for c in context_lengths],
        yticklabels=[f"{d:.0%}" for d in depths],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Retrieval Accuracy"},
    )

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Needle Depth")
    ax.set_title(
        f"NIAH — {config_result.config_name} "
        f"(avg: {config_result.average_accuracy:.1%})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", output_path)


def generate_combined_heatmap(
    results: list[ConfigResult],
    depths: list[float],
    context_lengths: list[int],
    output_path: Path,
) -> None:
    """Generate a combined comparison heatmap with subplots for all configs.

    Args:
        results: All ConfigResult objects.
        depths: Depth positions.
        context_lengths: Context lengths.
        output_path: Path to save the combined PNG.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_configs = len(results)
    if n_configs == 0:
        return

    cols = min(3, n_configs)
    rows = (n_configs + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5, rows * 4),
        squeeze=False,
    )

    for idx, cr in enumerate(results):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        matrix = np.zeros((len(depths), len(context_lengths)))
        for i, depth in enumerate(depths):
            for j, ctx_len in enumerate(context_lengths):
                score = cr.scores.get(str(ctx_len), {}).get(
                    str(depth), float("nan")
                )
                matrix[i, j] = score

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            xticklabels=[str(c) for c in context_lengths],
            yticklabels=[f"{d:.0%}" for d in depths],
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar=idx == n_configs - 1,
        )
        ax.set_title(f"{cr.config_name} ({cr.average_accuracy:.1%})")
        ax.set_xlabel("Context Length" if row == rows - 1 else "")
        ax.set_ylabel("Depth" if col == 0 else "")

    # Hide unused subplots
    for idx in range(n_configs, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Needle-In-A-Haystack — Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined heatmap saved to %s", output_path)


# ---------------------------------------------------------------------------
# Reporting (console)
# ---------------------------------------------------------------------------
def print_results_table(results: list[ConfigResult]) -> None:
    """Print a human-readable summary table to stdout.

    Args:
        results: List of ConfigResult from the benchmark.
    """
    from tabulate import tabulate

    headers = [
        "Config",
        "Avg Accuracy",
        "Lat. Mean (s)",
        "Lat. Std (s)",
    ]
    rows = [
        [
            cr.config_name,
            f"{cr.average_accuracy:.1%}",
            f"{cr.retrieval_latency_s['mean']:.2f}",
            f"{cr.retrieval_latency_s['std']:.2f}",
        ]
        for cr in results
    ]

    print("\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n")


def print_detailed_table(
    results: list[ConfigResult],
    depths: list[float],
    context_lengths: list[int],
) -> None:
    """Print a detailed per-config, per-depth score table.

    Args:
        results: List of ConfigResult.
        depths: Depth positions tested.
        context_lengths: Context lengths tested.
    """
    from tabulate import tabulate

    for cr in results:
        print(f"\n{'=' * 60}")
        print(f"Config: {cr.config_name} (K={cr.cache_type_k}, V={cr.cache_type_v})")
        print(f"Average accuracy: {cr.average_accuracy:.1%}")
        print(f"{'=' * 60}")

        headers = ["Depth"] + [str(c) for c in context_lengths]
        rows = []
        for depth in depths:
            row = [f"{depth:.0%}"]
            for ctx_len in context_lengths:
                score = cr.scores.get(str(ctx_len), {}).get(str(depth), float("nan"))
                row.append(f"{score:.2f}")
            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="github"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_configs(config_strs: list[str]) -> dict[str, KVCacheConfig]:
    """Parse config names (e.g., 'q8_0/turbo4') into KVCacheConfig objects.

    Args:
        config_strs: List of config names matching keys in BENCHMARK_CONFIGS.

    Returns:
        Ordered dict of name → KVCacheConfig.

    Raises:
        ValueError: If any config name is unknown.
    """
    selected: dict[str, KVCacheConfig] = {}
    for name in config_strs:
        if name not in BENCHMARK_CONFIGS:
            available = ", ".join(sorted(BENCHMARK_CONFIGS))
            raise ValueError(
                f"Unknown config '{name}'. Available: {available}"
            )
        selected[name] = BENCHMARK_CONFIGS[name]
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Needle-In-A-Haystack benchmark for KV cache quantization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m benchmarks.benchmark_niah --model-path models/qwen2.5-7b-instruct-q4_k_m.gguf\n"
            "  python -m benchmarks.benchmark_niah --model-path models/model.gguf --configs q8_0/turbo4 f16/f16\n"
            "  python -m benchmarks.benchmark_niah --model-path models/model.gguf --num-trials 5 --n-ctx-max 4096\n"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--n-ctx-max",
        type=int,
        default=8192,
        help="Maximum context window size (default: 8192).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(BENCHMARK_CONFIGS.keys()),
        help=(
            "KV cache configs to test. "
            f"Available: {', '.join(BENCHMARK_CONFIGS.keys())} "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results and heatmaps (default: benchmarks/results).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f"Trials per (config, length, depth) triple (default: {DEFAULT_NUM_TRIALS}).",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXT_LENGTHS,
        help=f"Context lengths to test (default: {DEFAULT_CONTEXT_LENGTHS}).",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=float,
        default=DEFAULT_DEPTHS,
        help=f"Needle depth positions (default: {DEFAULT_DEPTHS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (-1 = all, default: -1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    """CLI entry point for the NIAH benchmark."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Parse and validate configs
    configs = parse_configs(args.configs)

    # Filter context lengths to not exceed n_ctx_max
    context_lengths = sorted(c for c in args.context_lengths if c <= args.n_ctx_max)
    if not context_lengths:
        logger.error("No valid context lengths (all exceed --n-ctx-max %d)", args.n_ctx_max)
        sys.exit(1)

    depths = sorted(args.depths)

    # Build model config with max context
    model_config = ModelConfig(
        model_path=str(model_path),
        model_name=model_path.stem,
        n_ctx=args.n_ctx_max,
        n_gpu_layers=args.n_gpu_layers,
    )

    logger.info("NIAH Benchmark Configuration:")
    logger.info("  Model: %s", model_path)
    logger.info("  Max context: %d", args.n_ctx_max)
    logger.info("  Configs: %s", list(configs.keys()))
    logger.info("  Context lengths: %s", context_lengths)
    logger.info("  Depths: %s", depths)
    logger.info("  Trials per combo: %d", args.num_trials)
    logger.info("  Seed: %d", args.seed)

    total_evals = len(configs) * len(context_lengths) * len(depths) * args.num_trials
    logger.info("  Total evaluations: %d", total_evals)

    # Run benchmark
    start_time = time.monotonic()
    results = run_niah_benchmark(
        model_config=model_config,
        configs=configs,
        context_lengths=context_lengths,
        depths=depths,
        num_trials=args.num_trials,
        seed=args.seed,
    )
    elapsed = time.monotonic() - start_time

    # Print summary
    print_results_table(results)
    print_detailed_table(results, depths, context_lengths)
    print(f"\nTotal benchmark time: {elapsed:.1f}s")

    # Build and save report
    report = build_report(
        results=results,
        model_name=model_config.model_name,
        num_trials=args.num_trials,
        depths=depths,
        context_lengths=context_lengths,
    )

    json_path = output_dir / "niah_results.json"
    save_results(report, json_path)

    # Generate heatmaps
    for cr in results:
        safe_name = cr.config_name.replace("/", "_")
        heatmap_path = output_dir / f"niah_heatmap_{safe_name}.png"
        generate_heatmap(cr, depths, context_lengths, heatmap_path)

    combined_path = output_dir / "niah_heatmap_combined.png"
    generate_combined_heatmap(results, depths, context_lengths, combined_path)

    logger.info("Benchmark complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
