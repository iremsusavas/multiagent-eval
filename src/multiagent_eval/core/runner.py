"""
Evaluation runner: orchestrates metrics, judges, and reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from multiagent_eval.core.metrics import (
    MetricResult,
    factual_accuracy,
    output_completeness,
    hallucination_score,
    latency_compliance,
    cost_efficiency,
)
from multiagent_eval.core.trace import PipelineTrace

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""

    pipeline_name: str
    golden_dataset_path: Optional[str] = None
    metrics: list[str] = field(default_factory=lambda: ["factual_accuracy", "inter_agent_consistency"])
    thresholds: dict[str, float] = field(default_factory=dict)
    judge_primary_model: str = "gpt-4o"
    judge_fallback_model: str = "claude-sonnet-4-6"
    bias_detection: bool = True
    cot_prompting: bool = True
    output_dir: str = "eval_results"
    report_formats: list[str] = field(default_factory=lambda: ["json", "html"])
    dashboard: bool = False


@dataclass
class EvalResult:
    """Result of a full evaluation run."""

    pipeline_trace: PipelineTrace
    metrics: list[MetricResult] = field(default_factory=list)
    config: Optional[EvalConfig] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Average score across all metrics (0–1)."""
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)

    def passed(self, thresholds: Optional[dict[str, float]] = None) -> bool:
        """Check if evaluation passed all thresholds."""
        thresh = thresholds or (self.config.thresholds if self.config else {})
        for m in self.metrics:
            t = thresh.get(m.metric_name, 0.8)
            if m.flagged and m.score < t:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "pipeline_trace": self.pipeline_trace.to_dict(),
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "score": m.score,
                    "explanation": m.explanation,
                    "agent_id": m.agent_id,
                    "flagged": m.flagged,
                    "threshold": m.threshold,
                }
                for m in self.metrics
            ],
            "config": (
                {
                    "pipeline_name": self.config.pipeline_name,
                    "metrics": self.config.metrics,
                    "thresholds": self.config.thresholds,
                }
                if self.config
                else None
            ),
            "metadata": self.metadata,
            "failure_modes": self.metadata.get("failure_modes", {}),
            "stats": self.metadata.get("stats", {}),
        }


class EvaluationRunner:
    """
    Runs evaluations on pipeline traces using configured metrics and judges.
    """

    def __init__(
        self,
        config: Optional[EvalConfig] = None,
        metric_registry: Optional[dict[str, Callable[..., MetricResult]]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        """
        Initialize the evaluation runner.

        Args:
            config: Evaluation configuration.
            metric_registry: Map of metric name -> callable. Uses defaults if not provided.
            similarity_fn: Optional semantic similarity function for cross-agent metrics.
        """
        self.config = config or EvalConfig(pipeline_name="default")
        self.similarity_fn = similarity_fn
        self._metric_registry = metric_registry or self._default_metrics()

    def _default_metrics(self) -> dict[str, Callable[..., MetricResult]]:
        """Default metric implementations."""
        from multiagent_eval.core import metrics as m

        return {
            "factual_accuracy": lambda t, gt, **kw: factual_accuracy(
                t, gt, **{k: v for k, v in kw.items() if k in ["judge_fn", "threshold"]}
            ),
            "output_completeness": lambda t, schema, **kw: output_completeness(
                t, schema, **{k: v for k, v in kw.items() if k == "threshold"}
            ),
            "hallucination_score": lambda t, **kw: hallucination_score(
                t, **{k: v for k, v in kw.items() if k == "threshold"}
            ),
            "latency_compliance": lambda t, sla, **kw: latency_compliance(
                t, sla, **{k: v for k, v in kw.items() if k == "threshold"}
            ),
            "cost_efficiency": lambda t, budget, **kw: cost_efficiency(
                t, budget, **{k: v for k, v in kw.items() if k == "threshold"}
            ),
            "inter_agent_consistency": lambda pt, **kw: m.inter_agent_consistency(
                pt, similarity_fn=self.similarity_fn, **kw
            ),
            "error_propagation_score": lambda pt, **kw: m.error_propagation_score(
                pt, similarity_fn=self.similarity_fn, **kw
            ),
            "orchestration_adherence": m.orchestration_adherence,
            "cascade_failure_detection": m.cascade_failure_detection,
            "information_retention": lambda pt, **kw: m.information_retention(
                pt, similarity_fn=self.similarity_fn, **kw
            ),
            "pii_leakage": self._pii_metric,
            "prompt_injection": self._prompt_injection_metric,
        }

    def _pii_metric(self, pipeline_trace: PipelineTrace, **kw: Any) -> MetricResult:
        """PII leakage detection metric."""
        from multiagent_eval.security import pii_leakage_score
        score = pii_leakage_score(pipeline_trace.agents)
        thresh = kw.get("threshold", 1.0)
        return MetricResult(
            metric_name="pii_leakage",
            score=score,
            explanation="No PII in outputs" if score >= 1 else "PII detected in agent outputs",
            agent_id=None,
            flagged=score < thresh,
            threshold=thresh,
        )

    def _prompt_injection_metric(self, pipeline_trace: PipelineTrace, **kw: Any) -> MetricResult:
        """Prompt injection detection: check if inputs look like injection attempts."""
        from multiagent_eval.security import detect_prompt_injection
        flagged = False
        for agent in pipeline_trace.agents:
            inp = str(agent.input_received)
            if detect_prompt_injection(inp):
                flagged = True
                break
        score = 0.0 if flagged else 1.0
        thresh = kw.get("threshold", 1.0)
        return MetricResult(
            metric_name="prompt_injection",
            score=score,
            explanation="Potential prompt injection in input" if flagged else "No injection patterns detected",
            agent_id=None,
            flagged=flagged,
            threshold=thresh,
        )

    def run(
        self,
        pipeline_trace: PipelineTrace,
        ground_truth: Optional[dict[str, Any]] = None,
        expected_schema: Optional[dict[str, Any]] = None,
        expected_state_machine: Optional[list[tuple[str, str]]] = None,
        sla_ms: float = 30000.0,
        budget_usd: float = 1.0,
    ) -> EvalResult:
        """
        Run all configured metrics on the pipeline trace.

        Args:
            pipeline_trace: The pipeline trace to evaluate.
            ground_truth: Optional ground truth for factual_accuracy.
            expected_schema: Optional schema for output_completeness.
            expected_state_machine: Optional transitions for orchestration_adherence.
            sla_ms: Latency SLA for latency_compliance.
            budget_usd: Cost budget for cost_efficiency.

        Returns:
            EvalResult with all metric results.
        """
        results: list[MetricResult] = []
        metrics_to_run = self.config.metrics
        thresholds = self.config.thresholds

        for metric_name in metrics_to_run:
            fn = self._metric_registry.get(metric_name)
            if not fn:
                logger.warning("Unknown metric: %s", metric_name)
                continue

            try:
                if metric_name in ("factual_accuracy", "output_completeness", "hallucination_score", "latency_compliance", "cost_efficiency"):
                    for agent in pipeline_trace.agents:
                        thresh = thresholds.get(metric_name, 0.8)
                        if metric_name == "factual_accuracy" and ground_truth:
                            r = fn(agent, ground_truth, threshold=thresh)
                        elif metric_name == "output_completeness" and expected_schema:
                            r = fn(agent, expected_schema, threshold=thresh)
                        elif metric_name == "hallucination_score":
                            r = fn(agent, threshold=thresh)
                        elif metric_name == "latency_compliance":
                            r = fn(agent, sla_ms, threshold=thresh)
                        elif metric_name == "cost_efficiency":
                            r = fn(agent, budget_usd, threshold=thresh)
                        else:
                            continue
                        results.append(r)
                else:
                    thresh = thresholds.get(metric_name, 0.75)
                    if metric_name == "orchestration_adherence" and expected_state_machine:
                        r = fn(pipeline_trace, expected_state_machine, threshold=thresh)
                    else:
                        r = fn(pipeline_trace, threshold=thresh)
                    results.append(r)
            except Exception as e:
                logger.exception("Metric %s failed: %s", metric_name, e)
                results.append(
                    MetricResult(
                        metric_name=metric_name,
                        score=0.0,
                        explanation=f"Error: {str(e)}",
                        agent_id=None,
                        flagged=True,
                        threshold=thresholds.get(metric_name, 0.8),
                        raw_data={"error": str(e)},
                    )
                )

        meta = {"ground_truth": ground_truth is not None}
        if ground_truth is not None:
            meta["expected_output"] = ground_truth

        # Failure mode taxonomy
        try:
            from multiagent_eval.failure_modes import classify_failures, failure_mode_summary
            failures = classify_failures(pipeline_trace, results)
            meta["failure_modes"] = failure_mode_summary(failures)
        except Exception:
            pass

        # Statistical significance (bootstrap CI) when we have scores
        try:
            from multiagent_eval.stats import analyze
            scores = [m.score for m in results]
            if scores:
                stat = analyze(scores)
                meta["stats"] = {
                    "mean": stat.mean,
                    "std": stat.std,
                    "ci_95": [stat.ci_lower, stat.ci_upper],
                }
        except Exception:
            pass

        return EvalResult(
            pipeline_trace=pipeline_trace,
            metrics=results,
            config=self.config,
            metadata=meta,
        )
