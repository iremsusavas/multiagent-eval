"""
Scoring metrics for multi-agent pipeline evaluation.

All metrics return MetricResult with score 0.0-1.0 and explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from multiagent_eval.core.trace import AgentTrace, PipelineTrace


@dataclass
class MetricResult:
    """
    Result of a metric computation.

    Attributes:
        metric_name: Name of the metric.
        score: Score from 0.0 to 1.0.
        explanation: Human-readable explanation.
        agent_id: Agent this metric applies to (None for pipeline-level).
        flagged: True if score is below threshold.
        threshold: Threshold used for flagging.
        raw_data: Additional data for debugging.
    """

    metric_name: str
    score: float
    explanation: str
    agent_id: Optional[str] = None
    flagged: bool = False
    threshold: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)


# --- Per-agent metrics ---


def factual_accuracy(
    agent_trace: AgentTrace,
    ground_truth: dict[str, Any],
    judge_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.8,
) -> MetricResult:
    """
    Evaluate factual accuracy of agent output against ground truth.

    Uses LLM-as-Judge if judge_fn provided, else simple string matching.
    """
    output_str = str(agent_trace.output_produced)
    gt_str = str(ground_truth)

    if judge_fn:
        score = judge_fn(output_str, gt_str)
    else:
        # Fallback: simple overlap-based heuristic
        output_words = set(output_str.lower().split())
        gt_words = set(gt_str.lower().split())
        if not gt_words:
            score = 1.0
        else:
            overlap = len(output_words & gt_words) / len(gt_words)
            score = min(1.0, overlap * 1.2)

    return MetricResult(
        metric_name="factual_accuracy",
        score=round(score, 4),
        explanation=f"Factual accuracy score: {score:.2%}. Ground truth comparison.",
        agent_id=agent_trace.agent_id,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"ground_truth_keys": list(ground_truth.keys())},
    )


def output_completeness(
    agent_trace: AgentTrace,
    expected_schema: dict[str, Any],
    threshold: float = 0.8,
) -> MetricResult:
    """Check if agent output contains all expected schema keys."""
    output = agent_trace.output_produced

    def check_keys(obj: Any, schema: dict) -> tuple[int, int]:
        if not isinstance(obj, dict) or not isinstance(schema, dict):
            return (1, 1) if obj is not None else (0, 1)
        found = 0
        total = len(schema)
        for key, expected in schema.items():
            if key in obj:
                if isinstance(expected, dict) and isinstance(obj[key], dict):
                    f, t = check_keys(obj[key], expected)
                    found += f / t if t else 1
                else:
                    found += 1
        return (found, total)

    found, total = check_keys(output, expected_schema)
    score = found / total if total > 0 else 1.0

    return MetricResult(
        metric_name="output_completeness",
        score=round(score, 4),
        explanation=f"Output completeness: {found}/{total} expected keys present.",
        agent_id=agent_trace.agent_id,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"expected_keys": list(expected_schema.keys())},
    )


def hallucination_score(
    agent_trace: AgentTrace,
    judge_fn: Optional[Callable[[str], float]] = None,
    threshold: float = 0.2,
) -> MetricResult:
    """
    Score for hallucination (lower is better). 0 = no hallucination, 1 = severe.

    Uses LLM-as-Judge internally if judge_fn provided.
    """
    if judge_fn:
        raw_score = judge_fn(str(agent_trace.output_produced))
        score = 1.0 - raw_score  # Convert to "quality" (higher = less hallucination)
    else:
        # Heuristic: no judge available, assume moderate
        score = 0.7

    return MetricResult(
        metric_name="hallucination_score",
        score=round(score, 4),
        explanation=f"Hallucination inverse score: {score:.2%} (higher = less hallucination).",
        agent_id=agent_trace.agent_id,
        flagged=score < (1.0 - threshold),
        threshold=1.0 - threshold,
        raw_data={},
    )


def latency_compliance(
    agent_trace: AgentTrace,
    sla_ms: float,
    threshold: float = 0.8,
) -> MetricResult:
    """Check if agent met latency SLA."""
    actual = agent_trace.latency_ms
    score = 1.0 if actual <= sla_ms else max(0.0, 1.0 - (actual - sla_ms) / sla_ms)

    return MetricResult(
        metric_name="latency_compliance",
        score=round(score, 4),
        explanation=f"Latency: {actual:.0f}ms vs SLA {sla_ms}ms. Score: {score:.2%}.",
        agent_id=agent_trace.agent_id,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"latency_ms": actual, "sla_ms": sla_ms},
    )


def cost_efficiency(
    agent_trace: AgentTrace,
    budget_usd: float,
    threshold: float = 0.8,
) -> MetricResult:
    """Check if agent stayed within cost budget."""
    actual = agent_trace.total_cost_usd()
    score = 1.0 if actual <= budget_usd else max(0.0, 1.0 - (actual - budget_usd) / budget_usd)

    return MetricResult(
        metric_name="cost_efficiency",
        score=round(score, 4),
        explanation=f"Cost: ${actual:.4f} vs budget ${budget_usd}. Score: {score:.2%}.",
        agent_id=agent_trace.agent_id,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"cost_usd": actual, "budget_usd": budget_usd},
    )


# --- Cross-agent / pipeline-level metrics ---


def error_propagation_score(
    pipeline_trace: PipelineTrace,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.75,
) -> MetricResult:
    """
    Detect if one agent's error corrupted downstream agents.

    Compares each agent's input against previous agent's output; flags semantic divergence.
    """
    if len(pipeline_trace.agents) < 2:
        return MetricResult(
            metric_name="error_propagation_score",
            score=1.0,
            explanation="Single agent pipeline - no propagation to check.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    scores: list[float] = []
    for i in range(1, len(pipeline_trace.agents)):
        prev_output = str(pipeline_trace.agents[i - 1].output_produced)
        curr_input = str(pipeline_trace.agents[i].input_received)

        if similarity_fn:
            sim = similarity_fn(prev_output, curr_input)
        else:
            # Simple Jaccard fallback
            a, b = set(prev_output.lower().split()), set(curr_input.lower().split())
            sim = len(a & b) / len(a | b) if (a or b) else 1.0

        scores.append(sim)

    avg_score = sum(scores) / len(scores) if scores else 1.0

    return MetricResult(
        metric_name="error_propagation_score",
        score=round(avg_score, 4),
        explanation=f"Error propagation fidelity: {avg_score:.2%} across {len(scores)} agent boundaries.",
        agent_id=None,
        flagged=avg_score < threshold,
        threshold=threshold,
        raw_data={"per_edge_scores": scores},
    )


def inter_agent_consistency(
    pipeline_trace: PipelineTrace,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.75,
) -> MetricResult:
    """Check if agents contradict each other via pairwise semantic consistency."""
    agents = pipeline_trace.agents
    if len(agents) < 2:
        return MetricResult(
            metric_name="inter_agent_consistency",
            score=1.0,
            explanation="Single agent - no consistency to check.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    scores: list[float] = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            out_i = str(agents[i].output_produced)
            out_j = str(agents[j].output_produced)
            if similarity_fn:
                sim = similarity_fn(out_i, out_j)
            else:
                a, b = set(out_i.lower().split()), set(out_j.lower().split())
                sim = len(a & b) / len(a | b) if (a or b) else 1.0
            scores.append(sim)

    avg_score = sum(scores) / len(scores) if scores else 1.0

    return MetricResult(
        metric_name="inter_agent_consistency",
        score=round(avg_score, 4),
        explanation=f"Inter-agent consistency: {avg_score:.2%} across {len(scores)} pairs.",
        agent_id=None,
        flagged=avg_score < threshold,
        threshold=threshold,
        raw_data={"pairwise_scores": scores},
    )


def orchestration_adherence(
    pipeline_trace: PipelineTrace,
    expected_state_machine: list[tuple[str, str]],
    threshold: float = 0.8,
) -> MetricResult:
    """Check if pipeline followed expected state transitions."""
    actual = [(s.from_state, s.to_state) for s in pipeline_trace.state_transitions]
    expected = list(expected_state_machine)

    if not expected:
        return MetricResult(
            metric_name="orchestration_adherence",
            score=1.0,
            explanation="No expected state machine defined.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    matches = sum(1 for t in actual if t in expected)
    score = matches / len(expected) if expected else 1.0

    return MetricResult(
        metric_name="orchestration_adherence",
        score=round(score, 4),
        explanation=f"Orchestration: {matches}/{len(expected)} expected transitions followed.",
        agent_id=None,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"actual": actual, "expected": expected},
    )


def cascade_failure_detection(
    pipeline_trace: PipelineTrace,
    threshold: float = 0.8,
) -> MetricResult:
    """
    Detect if a single agent failure caused downstream failures.

    Builds dependency graph and traces failure propagation.
    """
    agents = pipeline_trace.agents
    if not agents:
        return MetricResult(
            metric_name="cascade_failure_detection",
            score=1.0,
            explanation="No agents in pipeline.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    failed = [i for i, a in enumerate(agents) if a.error]
    if not failed:
        return MetricResult(
            metric_name="cascade_failure_detection",
            score=1.0,
            explanation="No agent failures detected.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={"failed_agents": []},
        )

    # Heuristic: consecutive failures suggest cascade
    cascade_count = 0
    for i in range(1, len(failed)):
        if failed[i] == failed[i - 1] + 1:
            cascade_count += 1

    # Score: 1.0 if no cascade, lower if cascade detected
    cascade_ratio = cascade_count / max(1, len(failed) - 1)
    score = 1.0 - cascade_ratio

    return MetricResult(
        metric_name="cascade_failure_detection",
        score=round(score, 4),
        explanation=f"Cascade failure analysis: {cascade_count} potential cascade links among {len(failed)} failed agents.",
        agent_id=None,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"failed_agents": failed, "cascade_links": cascade_count},
    )


def information_retention(
    pipeline_trace: PipelineTrace,
    critical_keys: Optional[list[str]] = None,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.75,
) -> MetricResult:
    """
    Check if critical information from early agents was preserved to final output.
    """
    agents = pipeline_trace.agents
    if len(agents) < 2:
        return MetricResult(
            metric_name="information_retention",
            score=1.0,
            explanation="Single agent - no retention to check.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    first_output = agents[0].output_produced
    final_output = pipeline_trace.final_output

    if critical_keys:
        first_str = " ".join(str(first_output.get(k, "")) for k in critical_keys)
        final_str = " ".join(str(final_output.get(k, "")) for k in critical_keys)
    else:
        first_str = str(first_output)
        final_str = str(final_output)

    if similarity_fn:
        score = similarity_fn(first_str, final_str)
    else:
        a, b = set(first_str.lower().split()), set(final_str.lower().split())
        score = len(a & b) / len(a) if a else 1.0

    return MetricResult(
        metric_name="information_retention",
        score=round(score, 4),
        explanation=f"Information retention from first to final output: {score:.2%}.",
        agent_id=None,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"critical_keys": critical_keys},
    )
