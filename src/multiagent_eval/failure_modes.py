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

    # Per-agent factual_accuracy scores
    agent_fa_scores: dict[str, float] = {}
    for m in metric_results:
        if getattr(m, "metric_name", "") == "factual_accuracy" and getattr(m, "agent_id", None):
            agent_fa_scores[m.agent_id] = getattr(m, "score", 0)

    # Detect propagation: agent_i has very low score, corrupts agent_{i+1}
    agents = pipeline_trace.agents
    for i in range(len(agents) - 1):
        aid = agents[i].agent_id
        next_aid = agents[i + 1].agent_id
        score_i = agent_fa_scores.get(aid, 1.0)
        score_next = agent_fa_scores.get(next_aid, 1.0)
        if score_i < 0.2 and score_next < 0.5:
            failures.append(
                FailureInstance(
                    failure_mode=FailureMode.PROPAGATION,
                    agent_id=aid,
                    explanation=f"Agent {aid} produced incorrect output (score {score_i:.2f}), "
                    f"corrupting downstream agent {next_aid}.",
                    score=score_i,
                    raw_data={"source": aid, "target": next_aid},
                )
            )

    # Cascade: 2+ consecutive agent failures
    failed_agents = [a.agent_id for a in agents if agent_fa_scores.get(a.agent_id, 1.0) < 0.3]
    if len(failed_agents) >= 2:
        failures.append(
            FailureInstance(
                failure_mode=FailureMode.CASCADE,
                agent_id=failed_agents[0],
                explanation=f"Cascade: {len(failed_agents)} agents failed in sequence.",
                score=0.0,
                raw_data={"failed_agents": failed_agents},
            )
        )

    for m in metric_results:
        if not getattr(m, "flagged", False):
            continue
        score = getattr(m, "score", 0)
        name = getattr(m, "metric_name", "")
        agent_id = getattr(m, "agent_id", None)
        explanation = getattr(m, "explanation", "")

        mode = FailureMode.UNKNOWN
        if "error_propagation" in name or ("propagation" in name and "factual" not in name):
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


def failure_modes_display(failures: list[FailureInstance]) -> list[str]:
    """
    Return human-readable failure mode strings for display.

    Example: ["PROPAGATION_ERROR (agent_002 -> agent_003)", "CASCADE_FAILURE"]

    Prioritizes PROPAGATION and CASCADE; skips UNKNOWN when specific modes exist.
    Only shows PROPAGATION_ERROR when a specific agent pair is identified.
    Metric-level propagation flags (no specific agent) show as LOW_PROPAGATION_SCORE.
    """
    has_specific = any(
        f.failure_mode in (FailureMode.PROPAGATION, FailureMode.CASCADE)
        for f in failures
    )
    seen: set[str] = set()
    out: list[str] = []
    for f in failures:
        if has_specific and f.failure_mode == FailureMode.UNKNOWN:
            continue
        if f.failure_mode == FailureMode.PROPAGATION:
            source = f.agent_id
            target = f.raw_data.get("target")
            if source and target:
                # Specific agent-to-agent attribution
                key = f"PROPAGATION_ERROR ({source} -> {target})"
            elif f.raw_data.get("metric_name"):
                # Metric-level flag with no specific pair — score warning only
                key = "LOW_PROPAGATION_SCORE"
            else:
                continue  # Skip unattributed propagation flags
        elif f.failure_mode == FailureMode.CASCADE:
            key = "CASCADE_FAILURE"
        elif f.failure_mode == FailureMode.HALLUCINATION:
            key = f"HALLUCINATION ({f.agent_id})"
        elif f.failure_mode == FailureMode.CONSISTENCY_DRIFT:
            key = "CONSISTENCY_DRIFT"
        else:
            key = f"{f.failure_mode.value.upper()} ({f.agent_id or 'pipeline'})"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


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
