"""
Failure Mode Taxonomy: categorize errors for actionable insights.

Categories: Propagation, Hallucination, Context Loss, Orchestration Break, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from multiagent_eval.core.trace import PipelineTrace


class FailureMode(str, Enum):
    """Standard failure mode taxonomy."""

    PROPAGATION = "propagation"           # Error corrupted downstream agent
    HALLUCINATION = "hallucination"       # Agent output contains fabricated info
    CONTEXT_LOSS = "context_loss"         # Critical info lost between agents
    ORCHESTRATION_BREAK = "orchestration" # Wrong state transition / flow
    CASCADE = "cascade"                   # Single failure caused downstream failures
    TIMEOUT = "timeout"                   # Agent exceeded SLA
    COST_OVERRUN = "cost_overrun"         # Exceeded budget
    CONSISTENCY_DRIFT = "consistency"      # Agents contradict each other
    UNKNOWN = "unknown"


@dataclass
class FailureInstance:
    """A single detected failure with taxonomy."""

    failure_mode: FailureMode
    agent_id: Optional[str]
    explanation: str
    score: float
    raw_data: dict[str, Any] = field(default_factory=dict)


def classify_failures(
    pipeline_trace: PipelineTrace,
    metric_results: list[Any],
) -> list[FailureInstance]:
    """
    Classify all failures in pipeline into taxonomy.

    Returns list of FailureInstance with failure_mode, agent_id, explanation.
    """
    failures: list[FailureInstance] = []

    for m in metric_results:
        if not getattr(m, "flagged", False):
            continue
        score = getattr(m, "score", 0)
        name = getattr(m, "metric_name", "")
        agent_id = getattr(m, "agent_id", None)
        explanation = getattr(m, "explanation", "")

        mode = FailureMode.UNKNOWN
        if "error_propagation" in name or "propagation" in name:
            mode = FailureMode.PROPAGATION
        elif "hallucination" in name:
            mode = FailureMode.HALLUCINATION
        elif "information_retention" in name or "retention" in name:
            mode = FailureMode.CONTEXT_LOSS
        elif "orchestration" in name:
            mode = FailureMode.ORCHESTRATION_BREAK
        elif "cascade" in name:
            mode = FailureMode.CASCADE
        elif "latency" in name:
            mode = FailureMode.TIMEOUT
        elif "cost" in name:
            mode = FailureMode.COST_OVERRUN
        elif "consistency" in name:
            mode = FailureMode.CONSISTENCY_DRIFT

        failures.append(
            FailureInstance(
                failure_mode=mode,
                agent_id=agent_id,
                explanation=explanation,
                score=score,
                raw_data={"metric_name": name},
            )
        )

    # Also check agent errors
    for agent in pipeline_trace.agents:
        if agent.error:
            failures.append(
                FailureInstance(
                    failure_mode=FailureMode.UNKNOWN,
                    agent_id=agent.agent_id,
                    explanation=f"Agent error: {agent.error}",
                    score=0.0,
                    raw_data={"error": agent.error},
                )
            )

    return failures


def failure_mode_summary(failures: list[FailureInstance]) -> dict[str, int]:
    """
    Return count per failure mode.

    Example: {"propagation": 2, "hallucination": 1}
    """
    summary: dict[str, int] = {}
    for f in failures:
        key = f.failure_mode.value
        summary[key] = summary.get(key, 0) + 1
    return summary
