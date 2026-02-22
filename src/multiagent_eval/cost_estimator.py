"""
Cost estimation (pre-run / dry-run): estimate cost before running evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CostEstimate:
    """Pre-run cost estimate."""

    estimated_min_usd: float
    estimated_max_usd: float
    n_examples: int
    n_expected_llm_calls_per_example: int
    model: str
    explanation: str


def estimate_eval_cost(
    dataset_path: str,
    model: str = "gpt-4o",
    avg_tokens_per_call: int = 500,
) -> CostEstimate:
    """
    Estimate cost of running evaluation on dataset before executing.

    Uses approximate token counts and model pricing.
    """
    import json
    from pathlib import Path

    path = Path(dataset_path)
    if not path.exists():
        return CostEstimate(
            estimated_min_usd=0.0,
            estimated_max_usd=0.0,
            n_examples=0,
            n_expected_llm_calls_per_example=0,
            model=model,
            explanation="Dataset not found",
        )

    data = json.loads(path.read_text())
    examples = data.get("examples", [])
    n_examples = len(examples)

    # Per example: factual_accuracy (maybe judge), propagation_judge (if flagged), etc.
    # Conservative: 2-5 LLM calls per example
    n_calls_min = n_examples * 2
    n_calls_max = n_examples * 5

    # Rough pricing (per 1K tokens): gpt-4o ~$0.005/1K input, $0.015/1K output
    cost_per_1k = 0.01  # blended approx
    tokens_per_call = avg_tokens_per_call
    cost_min = (n_calls_min * tokens_per_call / 1000) * cost_per_1k
    cost_max = (n_calls_max * tokens_per_call / 1000) * cost_per_1k

    return CostEstimate(
        estimated_min_usd=round(cost_min, 4),
        estimated_max_usd=round(cost_max, 4),
        n_examples=n_examples,
        n_expected_llm_calls_per_example=3,
        model=model,
        explanation=f"~{n_calls_min}-{n_calls_max} LLM calls for {n_examples} examples. Model: {model}",
    )
