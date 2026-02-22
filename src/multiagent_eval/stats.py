"""
Statistical significance: bootstrap CI, p-value, minimum detectable effect.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StatisticalResult:
    """Result of statistical analysis."""

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    p_value: Optional[float]
    n_samples: int


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for mean score.

    Returns (lower, upper) bounds.
    """
    if not scores:
        return (0.0, 0.0)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci_level) / 2
    lo = int(alpha * n_bootstrap)
    hi = int((1 - alpha) * n_bootstrap)
    return (means[lo], means[hi])


def p_value_two_sided(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 1000,
) -> float:
    """
    Permutation test p-value: are the two distributions different?

    H0: no difference. Returns p-value.
    """
    if not scores_a or not scores_b:
        return 1.0
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    observed_diff = abs(mean_a - mean_b)
    combined = scores_a + scores_b
    n_a = len(scores_a)
    count = 0
    for _ in range(n_permutations):
        random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        diff = abs(sum(perm_a) / len(perm_a) - sum(perm_b) / len(perm_b))
        if diff >= observed_diff:
            count += 1
    return count / n_permutations


def analyze(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> StatisticalResult:
    """
    Full statistical analysis of score distribution.
    """
    if not scores:
        return StatisticalResult(0.0, 0.0, 0.0, 0.0, ci_level, None, 0)
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = variance ** 0.5
    ci_lower, ci_upper = bootstrap_ci(scores, n_bootstrap, ci_level)
    return StatisticalResult(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=None,
        n_samples=len(scores),
    )
