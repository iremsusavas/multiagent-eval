"""
Tone Bias Detection: preference for certain writing styles.

Same content in neutral vs apologetic/formal tone. Flags if scores differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ToneBiasResult:
    """Result of tone bias check."""

    detected: bool
    score_neutral: float
    score_formal: float
    difference: float
    threshold: float
    explanation: str


class ToneBiasDetector:
    """
    Detects tone bias: judge scoring same content differently based on tone.
    """

    def __init__(self, threshold: float = 0.2) -> None:
        """
        Initialize detector.

        Args:
            threshold: Score difference above which to flag.
        """
        self.threshold = threshold

    def check(
        self,
        score_fn: Callable[[str], float],
        content: str,
        formal_wrapper: Optional[str] = None,
    ) -> ToneBiasResult:
        """
        Score same content in neutral and formal tone.

        Args:
            score_fn: Function (output) -> score.
            content: The actual content.
            formal_wrapper: Optional wrapper for formal version.

        Returns:
            ToneBiasResult.
        """
        neutral = content
        wrapper = formal_wrapper or "I humbly submit that the following is accurate: "
        formal = wrapper + content + " Thank you for your consideration."

        score_neutral = score_fn(neutral)
        score_formal = score_fn(formal)
        diff = abs(score_neutral - score_formal)
        detected = diff > self.threshold

        return ToneBiasResult(
            detected=detected,
            score_neutral=score_neutral,
            score_formal=score_formal,
            difference=diff,
            threshold=self.threshold,
            explanation=f"Neutral={score_neutral:.3f}, formal={score_formal:.3f}. Diff={diff:.3f}. {'Bias detected' if detected else 'OK'}.",
        )
