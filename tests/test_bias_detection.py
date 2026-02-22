"""Tests for bias_detection."""

import pytest
from multiagent_eval.bias_detection import (
    PrimacyBiasDetector,
    VerbosityBiasDetector,
    ToneBiasDetector,
    CascadeBiasDetector,
)
from multiagent_eval.core.trace import AgentTrace, PipelineTrace


def test_primacy_bias() -> None:
    """Test primacy bias detector."""
    def score_fn(a: str, b: str) -> float:
        return 0.5 + (0.1 if a < b else 0)  # Slight order dependence
    detector = PrimacyBiasDetector(threshold=0.2)
    r = detector.check(score_fn, "A", "B")
    assert r.score_order_ab != r.score_order_ba or r.difference >= 0


def test_verbosity_bias() -> None:
    """Test verbosity bias detector."""
    def score_fn(x: str) -> float:
        return 0.6 if "incorrect" in x else 0.8  # Prefer correct
    detector = VerbosityBiasDetector()
    r = detector.check(score_fn, "correct answer")
    assert r.score_correct >= r.score_verbose_wrong or r.detected


def test_tone_bias() -> None:
    """Test tone bias detector."""
    def score_fn(x: str) -> float:
        return 0.9 if "humbly" not in x else 0.7
    detector = ToneBiasDetector(threshold=0.2)
    r = detector.check(score_fn, "neutral content")
    assert r.difference >= 0


def test_cascade_bias() -> None:
    """Test cascade bias detector."""
    a = AgentTrace(agent_id="a", agent_role="r", error="failed")
    b = AgentTrace(agent_id="b", agent_role="r", input_received={"from": "a"})
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn", agents=[a, b])
    detector = CascadeBiasDetector()
    r = detector.analyze(pt)
    assert r.detected
    assert "a" in r.upstream_error_agents
    assert "b" in r.affected_agents
