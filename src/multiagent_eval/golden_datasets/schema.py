"""
Schema for golden datasets and human annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HumanLabel:
    """Human annotation for an example."""

    example_id: str
    agent_id: str
    score: float
    rater_id: str
    notes: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenExample:
    """
    A single golden dataset example (test case).

    Attributes:
        example_id: Unique ID.
        pipeline_input: Input to the pipeline.
        expected_outputs_per_agent: Dict agent_id -> expected output.
        final_expected_output: Expected final pipeline output.
        tags: Tags for filtering (e.g., difficulty).
        difficulty: Optional difficulty level.
        human_labels: Human annotations when available.
    """

    example_id: str
    pipeline_input: dict[str, Any]
    expected_outputs_per_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_expected_output: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    human_labels: list[HumanLabel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
