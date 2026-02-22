"""
Primacy Bias Detection: order-dependent scoring.

Runs evaluation twice with (A,B) and (B,A) order. Flags if scores differ
beyond threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PrimacyBiasResult:
    """Result of primacy bias check."""

    detected: bool
    score_order_ab: float
    score_order_ba: float
    difference: float
    threshold: float
    explanation: str


class PrimacyBiasDetector:
    """
    Detects primacy bias: when judge scores change based on presentation order.
    """

    def __init__(self, threshold: float = 0.2) -> None:
        """
        Initialize detector.

        Args:
            threshold: Score difference above which to flag bias.
        """
        self.threshold = threshold

    def check(
        self,
        score_fn: Callable[[str, str], float],
        item_a: str,
        item_b: str,
    ) -> PrimacyBiasResult:
        """
        Run evaluation in both orders and compare scores.

        Args:
            score_fn: Function (a, b) -> score.
            item_a: First item.
            item_b: Second item.

        Returns:
            PrimacyBiasResult with both scores and detection result.
        """
        score_ab = score_fn(item_a, item_b)
        score_ba = score_fn(item_b, item_a)
        diff = abs(score_ab - score_ba)
        detected = diff > self.threshold

        return PrimacyBiasResult(
            detected=detected,
            score_order_ab=score_ab,
            score_order_ba=score_ba,
            difference=diff,
            threshold=self.threshold,
            explanation=f"Order (A,B)={score_ab:.3f}, (B,A)={score_ba:.3f}. Diff={diff:.3f}. {'Bias detected' if detected else 'Within threshold'}.",
        )
