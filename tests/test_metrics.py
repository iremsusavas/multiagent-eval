"""Tests for core/metrics.py."""

import pytest
from multiagent_eval.core.trace import AgentTrace, PipelineTrace
from multiagent_eval.core.metrics import (
    MetricResult,
    factual_accuracy,
    output_completeness,
    latency_compliance,
    cost_efficiency,
    error_propagation_score,
    inter_agent_consistency,
    cascade_failure_detection,
)


def test_factual_accuracy() -> None:
    """Test factual accuracy metric."""
    trace = AgentTrace(
        agent_id="a",
        agent_role="r",
        output_produced={"answer": "Paris is the capital of France"},
    )
    gt = {"answer": "Paris is the capital of France"}
    r = factual_accuracy(trace, gt, threshold=0.8)
    assert 0 <= r.score <= 1
    assert r.metric_name == "factual_accuracy"
    assert r.agent_id == "a"


def test_output_completeness() -> None:
    """Test output completeness metric."""
    trace = AgentTrace(
        agent_id="a",
        agent_role="r",
        output_produced={"a": 1, "b": 2},
    )
    schema = {"a": {}, "b": {}, "c": {}}
    r = output_completeness(trace, schema, threshold=0.8)
    assert abs(r.score - 2 / 3) < 0.001
    assert r.flagged


def test_latency_compliance() -> None:
    """Test latency compliance metric."""
    trace = AgentTrace(agent_id="a", agent_role="r", latency_ms=100)
    r = latency_compliance(trace, sla_ms=200, threshold=0.8)
    assert r.score == 1.0
    assert not r.flagged


def test_cost_efficiency() -> None:
    """Test cost efficiency metric."""
    trace = AgentTrace(agent_id="a", agent_role="r")
    trace.add_llm_call("", "", "gpt", 0, 0, 0.5, 0)
    r = cost_efficiency(trace, budget_usd=1.0, threshold=0.8)
    assert r.score == 1.0


def test_error_propagation_score() -> None:
    """Test error propagation score."""
    a1 = AgentTrace(agent_id="a1", agent_role="r", output_produced={"x": "same"})
    a2 = AgentTrace(agent_id="a2", agent_role="r", input_received={"x": "same"})
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn", agents=[a1, a2])
    r = error_propagation_score(pt, threshold=0.75)
    assert 0 <= r.score <= 1
    assert r.agent_id is None


def test_inter_agent_consistency() -> None:
    """Test inter-agent consistency."""
    a1 = AgentTrace(agent_id="a1", agent_role="r", output_produced={"x": "hello"})
    a2 = AgentTrace(agent_id="a2", agent_role="r", output_produced={"x": "hello"})
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn", agents=[a1, a2])
    r = inter_agent_consistency(pt, threshold=0.75)
    assert 0 <= r.score <= 1


def test_cascade_failure_detection_no_failures() -> None:
    """Test cascade failure with no failures."""
    a = AgentTrace(agent_id="a", agent_role="r")
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn", agents=[a])
    r = cascade_failure_detection(pt, threshold=0.8)
    assert r.score == 1.0
