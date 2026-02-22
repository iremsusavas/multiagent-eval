"""
Verbosity Bias Detection: preference for longer answers.

Injects a verbose but incorrect answer. Flags if judge prefers it over correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class VerbosityBiasResult:
    """Result of verbosity bias check."""

    detected: bool
    score_correct: float
    score_verbose_wrong: float
    explanation: str


class VerbosityBiasDetector:
    """
    Detects verbosity bias: judge preferring longer answers over correct shorter ones.
    """

    def __init__(self) -> None:
        """Initialize detector."""

    def check(
        self,
        score_fn: Callable[[str], float],
        correct_answer: str,
        verbose_padding: Optional[str] = None,
    ) -> VerbosityBiasResult:
        """
        Compare score of correct answer vs verbose-but-wrong answer.

        Args:
            score_fn: Function (output) -> score.
            correct_answer: The correct answer.
            verbose_padding: Optional padding to make wrong answer longer.

        Returns:
            VerbosityBiasResult.
        """
        padding = verbose_padding or " " + " ".join(["elaborate" * 20, "detailed" * 15, "comprehensive" * 10, "incorrect" * 5])
        verbose_wrong = correct_answer + padding

        score_correct = score_fn(correct_answer)
        score_verbose = score_fn(verbose_wrong)
        detected = score_verbose > score_correct

        return VerbosityBiasResult(
            detected=detected,
            score_correct=score_correct,
            score_verbose_wrong=score_verbose,
            explanation=f"Correct score={score_correct:.3f}, verbose-wrong={score_verbose:.3f}. {'Bias: prefers verbose' if detected else 'OK'}.",
        )
