"""
Scoring metrics for multi-agent pipeline evaluation.

All metrics return MetricResult with score 0.0-1.0 and explanation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from multiagent_eval.core.trace import AgentTrace, PipelineTrace

# ---------------------------------------------------------------------------
# Stopword-filtered TF-IDF cosine similarity
# Used by error_propagation_score and inter_agent_consistency as default fallback.
# No external dependencies required.
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "this", "that", "these",
    "those", "it", "its", "and", "or", "but", "not", "no", "so", "if",
    "then", "than", "which", "who", "what", "when", "where", "how",
    "all", "also", "more", "very", "just", "about", "up", "out", "into",
    "through", "between", "each", "other", "both", "few", "most", "such",
    "their", "they", "them", "we", "our", "your", "my", "his", "her",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, return tokens."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using TF * log(IDF) weighting.

    Returns 0.0 (completely different topics) to 1.0 (identical word distribution).
    Handles topic drift detection better than raw Jaccard because it weights
    rare/domain-specific words more heavily than common words.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    # Term frequency
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    # IDF across the 2-document corpus (simple but effective)
    vocab = set(tf_a) | set(tf_b)
    idf: dict[str, float] = {}
    for term in vocab:
        doc_count = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)
        idf[term] = math.log((2 + 1) / (doc_count + 1)) + 1.0  # smoothed

    # Weighted TF vectors
    def vec(tf: Counter) -> dict[str, float]:
        total = sum(tf.values()) or 1
        return {t: (tf[t] / total) * idf[t] for t in tf}

    vec_a = vec(tf_a)
    vec_b = vec(tf_b)

    # Cosine
    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-9
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-9
    return min(1.0, dot / (norm_a * norm_b))


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


def _extract_text_from_output(output: dict[str, Any]) -> str:
    """Extract concatenated text from agent output dict."""
    if not isinstance(output, dict):
        return str(output)
    parts = []
    for v in output.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, dict)):
            parts.append(str(v))
    return " ".join(parts).lower()


def factual_accuracy(
    agent_trace: AgentTrace,
    ground_truth: dict[str, Any],
    judge_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.8,
) -> MetricResult:
    """
    Evaluate factual accuracy of agent output against ground truth.

    Uses LLM-as-Judge if judge_fn provided.
    If ground_truth has "expected_keywords", uses keyword overlap (found/total).
    Otherwise falls back to simple string overlap.
    """
    output = agent_trace.output_produced
    output_text = _extract_text_from_output(output)

    if judge_fn:
        score = judge_fn(output_text, str(ground_truth))
    elif "expected_keywords" in ground_truth:
        # Keyword-based: score = (keywords found in output) / (total keywords)
        keywords = ground_truth["expected_keywords"]
        if isinstance(keywords, str):
            keywords = [w.strip() for w in keywords.split(",")]
        keywords = [k.lower() for k in keywords if k]
        if not keywords:
            score = 1.0
        else:
            found = sum(1 for kw in keywords if kw in output_text)
            score = found / len(keywords)
    else:
        # Fallback: simple word overlap
        output_words = set(output_text.split())
        gt_str = str(ground_truth).lower()
        gt_words = set(gt_str.split())
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

    Requires an LLM-as-Judge function to produce meaningful scores.
    Hallucination cannot be reliably detected without a reference model —
    a hardcoded fallback would silently mask real issues.

    Args:
        agent_trace: The agent trace to evaluate.
        judge_fn: Callable that takes agent output text and returns a float
                  in [0, 1] where 1 = severe hallucination.
        threshold: Flag if score drops below (1 - threshold).

    Raises:
        ValueError: If judge_fn is not provided. Hallucination scoring without
                    a judge produces meaningless results.

    Example:
        from multiagent_eval.judges import LLMJudge
        judge = LLMJudge(primary_model="gpt-4o")
        result = hallucination_score(trace, judge_fn=lambda text: judge.score_hallucination(text))
    """
    if judge_fn is None:
        raise ValueError(
            "hallucination_score requires a judge_fn. "
            "Without an LLM or reference model, hallucination cannot be reliably scored — "
            "a hardcoded value would silently mask real issues.\n\n"
            "Example:\n"
            "  from multiagent_eval.judges import LLMJudge\n"
            "  judge = LLMJudge(primary_model='gpt-4o')\n"
            "  result = hallucination_score(trace, judge_fn=judge.score_hallucination)\n\n"
            "Or exclude 'hallucination_score' from your metrics list and use "
            "'factual_accuracy' with expected_keywords instead."
        )

    raw_score = judge_fn(str(agent_trace.output_produced))
    score = 1.0 - raw_score  # Convert to "quality" (higher = less hallucination)

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
    Detect if agents introduced corruption into the pipeline.

    For each agent, measures semantic similarity between what the agent
    *received* and what it *produced*. A low score means the agent diverged
    from its input topic — the classic hallucination/propagation pattern.

    Two complementary signals are combined:
      - **Drift score**: how much each agent's output diverges from its own input
        (catches the *origin* of corruption — e.g. Agent 2 receives inflation
        content but produces climate content)
      - **Handoff score**: how much each agent's input matches the previous
        agent's output (catches dropped or altered context between agents)

    Uses TF-IDF cosine similarity by default (no external dependencies required).
    Pass a ``similarity_fn`` for embeddings-based comparison (sentence-transformers,
    OpenAI embeddings, etc.) for higher precision on short or paraphrased texts.

    Args:
        pipeline_trace: The pipeline trace to evaluate.
        similarity_fn: Optional callable (text_a, text_b) -> float [0, 1].
                       Defaults to internal TF-IDF cosine.
        threshold: Flag if average score drops below this value.

    Returns:
        MetricResult where score=1.0 means no drift/corruption detected,
        and score near 0.0 signals severe topic corruption at one or more agents.
    """
    agents = pipeline_trace.agents
    if len(agents) < 2:
        return MetricResult(
            metric_name="error_propagation_score",
            score=1.0,
            explanation="Single agent pipeline — no propagation to check.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    sim = similarity_fn if similarity_fn is not None else _tfidf_cosine

    drift_scores: list[float] = []      # per-agent input→output semantic alignment
    handoff_scores: list[float] = []    # cross-agent output[i-1]→input[i] fidelity
    per_agent: list[dict[str, Any]] = []

    for i, agent in enumerate(agents):
        agent_input = str(agent.input_received)
        agent_output = str(agent.output_produced)

        record: dict[str, Any] = {"agent_id": agent.agent_id}

        # Drift: did this agent stay on topic?
        # Skip the first agent — its role is to EXPAND a user query into research;
        # short query vs long output will always have low similarity by design.
        if i > 0:
            drift = sim(agent_input, agent_output)
            drift_scores.append(drift)
            record["drift_score"] = round(drift, 4)

        # Handoff: did this agent receive what the previous one sent?
        if i > 0:
            prev_output = str(agents[i - 1].output_produced)
            handoff = sim(prev_output, agent_input)
            handoff_scores.append(handoff)
            record["handoff_score"] = round(handoff, 4)

        per_agent.append(record)

    # Combine: weight drift 60%, handoff 40%
    avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 1.0
    avg_handoff = sum(handoff_scores) / len(handoff_scores) if handoff_scores else 1.0
    combined = 0.6 * avg_drift + 0.4 * avg_handoff

    # Find worst offender for explanation
    # drift_scores[0] = agents[1], drift_scores[k] = agents[k+1]
    if drift_scores:
        min_drift_idx_in_list = drift_scores.index(min(drift_scores))
        worst_agent = agents[min_drift_idx_in_list + 1].agent_id  # +1 because we skip agents[0]
        worst_score = drift_scores[min_drift_idx_in_list]
    else:
        worst_agent = agents[-1].agent_id if agents else "unknown"
        worst_score = 1.0

    if worst_score < 0.3:
        explanation = (
            f"Likely corruption origin: {worst_agent} "
            f"(drift score {worst_score:.2%} — output is semantically unrelated to its input). "
            f"Combined propagation score: {combined:.2%}."
        )
    else:
        explanation = (
            f"Error propagation score: {combined:.2%} across {len(agents)} agents "
            f"(drift avg: {avg_drift:.2%}, handoff avg: {avg_handoff:.2%})."
        )

    return MetricResult(
        metric_name="error_propagation_score",
        score=round(combined, 4),
        explanation=explanation,
        agent_id=None,
        flagged=combined < threshold,
        threshold=threshold,
        raw_data={
            "per_agent": per_agent,
            "avg_drift": round(avg_drift, 4),
            "avg_handoff": round(avg_handoff, 4),
        },
    )


def inter_agent_consistency(
    pipeline_trace: PipelineTrace,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = 0.4,
) -> MetricResult:
    """
    Check if agents contradict each other via pairwise semantic consistency.

    Measures whether agents are discussing the same general topic — not that
    they use identical words (different roles naturally use different vocabulary).
    A Researcher's factual summary and a Writer's narrative will score ~0.3-0.5
    on topic similarity even when perfectly consistent; scores below 0.3 suggest
    genuine contradiction or topic drift.

    Default threshold is intentionally set at 0.4 to avoid false positives
    in multi-role pipelines where vocabulary naturally diverges by design.
    Use a higher threshold (0.6+) only when agents are expected to produce
    near-identical outputs (e.g. ensemble verification pipelines).

    Args:
        pipeline_trace: The pipeline trace to evaluate.
        similarity_fn: Optional callable (text_a, text_b) -> float [0, 1].
                       Defaults to internal TF-IDF cosine.
        threshold: Flag if average pairwise similarity drops below this value.
    """
    agents = pipeline_trace.agents
    if len(agents) < 2:
        return MetricResult(
            metric_name="inter_agent_consistency",
            score=1.0,
            explanation="Single agent — no consistency to check.",
            agent_id=None,
            flagged=False,
            threshold=threshold,
            raw_data={},
        )

    sim = similarity_fn if similarity_fn is not None else _tfidf_cosine

    scores: list[float] = []
    pairs: list[dict[str, Any]] = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            out_i = str(agents[i].output_produced)
            out_j = str(agents[j].output_produced)
            s = sim(out_i, out_j)
            scores.append(s)
            pairs.append({
                "agent_a": agents[i].agent_id,
                "agent_b": agents[j].agent_id,
                "score": round(s, 4),
            })

    avg_score = sum(scores) / len(scores) if scores else 1.0

    # Find worst pair for actionable explanation
    if pairs:
        worst = min(pairs, key=lambda p: p["score"])
        explanation = (
            f"Inter-agent consistency: {avg_score:.2%} avg across {len(scores)} pairs. "
            f"Lowest: {worst['agent_a']} vs {worst['agent_b']} ({worst['score']:.2%}). "
            f"Note: different agent roles naturally diverge in vocabulary; "
            f"scores below 0.3 indicate genuine topic contradiction."
        )
    else:
        explanation = f"Inter-agent consistency: {avg_score:.2%}."

    return MetricResult(
        metric_name="inter_agent_consistency",
        score=round(avg_score, 4),
        explanation=explanation,
        agent_id=None,
        flagged=avg_score < threshold,
        threshold=threshold,
        raw_data={"pairwise_scores": pairs},
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

    sim = similarity_fn if similarity_fn is not None else _tfidf_cosine

    if similarity_fn:
        score = similarity_fn(first_str, final_str)
    else:
        score = sim(first_str, final_str)

    return MetricResult(
        metric_name="information_retention",
        score=round(score, 4),
        explanation=f"Information retention from first to final output: {score:.2%}.",
        agent_id=None,
        flagged=score < threshold,
        threshold=threshold,
        raw_data={"critical_keys": critical_keys},
    )
