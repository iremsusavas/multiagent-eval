"""
Base judge interface for all evaluation judges.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""

    score: float
    reasoning: str
    raw_response: Optional[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseJudge(ABC):
    """Abstract base class for all judges."""

    @abstractmethod
    def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """
        Evaluate an output against reference/rubric.

        Args:
            output: The output to evaluate.
            reference: Optional reference/ground truth.
            rubric: Optional scoring rubric.
            **kwargs: Additional judge-specific parameters.

        Returns:
            JudgeResult with score and reasoning.
        """
        pass
