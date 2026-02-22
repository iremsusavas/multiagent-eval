"""Tests for core/runner.py."""

import pytest
from multiagent_eval.core.runner import EvalConfig, EvaluationRunner, EvalResult
from multiagent_eval.core.trace import AgentTrace, PipelineTrace


@pytest.fixture
def pipeline_trace() -> PipelineTrace:
    """Create pipeline trace for runner tests."""
    a1 = AgentTrace(
        agent_id="a1",
        agent_role="r",
        input_received={"q": "x"},
        output_produced={"ans": "y"},
    )
    a2 = AgentTrace(
        agent_id="a2",
        agent_role="r",
        input_received=a1.output_produced,
        output_produced={"final": "z"},
    )
    pt = PipelineTrace(
        pipeline_id="p",
        pipeline_name="pn",
        agents=[a1, a2],
        final_output={"final": "z"},
    )
    return pt


def test_runner_run(pipeline_trace: PipelineTrace) -> None:
    """Test evaluation runner."""
    config = EvalConfig(
        pipeline_name="pn",
        metrics=["inter_agent_consistency", "error_propagation_score"],
        thresholds={"inter_agent_consistency": 0.5},
    )
    runner = EvaluationRunner(config=config)
    result = runner.run(pipeline_trace)
    assert len(result.metrics) >= 2
    assert result.pipeline_trace is pipeline_trace


def test_eval_result_passed() -> None:
    """Test EvalResult.passed()."""
    from multiagent_eval.core.metrics import MetricResult
    m1 = MetricResult("m1", 0.9, "ok", flagged=False, threshold=0.8)
    m2 = MetricResult("m2", 0.5, "bad", flagged=True, threshold=0.8)
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn")
    r = EvalResult(pipeline_trace=pt, metrics=[m1, m2])
    assert not r.passed({"m1": 0.8, "m2": 0.8})
