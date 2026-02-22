"""Tests for core/trace.py."""

import pytest
from multiagent_eval.core.trace import (
    AgentTrace,
    PipelineTrace,
    TraceCapture,
    LLMCall,
    ToolCall,
)


def test_agent_trace_add_llm_call() -> None:
    """Test adding LLM call to trace."""
    trace = AgentTrace(agent_id="a", agent_role="r")
    trace.add_llm_call("p", "r", "gpt-4", 10, 20, 0.001, 100)
    assert len(trace.llm_calls) == 1
    assert trace.llm_calls[0].cost_usd == 0.001
    assert trace.total_cost_usd() == 0.001


def test_agent_trace_add_tool_call() -> None:
    """Test adding tool call to trace."""
    trace = AgentTrace(agent_id="a", agent_role="r")
    trace.add_tool_call("search", {"q": "x"}, "result", 50)
    assert len(trace.tools_called) == 1
    assert trace.tools_called[0].tool_name == "search"


def test_agent_trace_to_dict() -> None:
    """Test serialization to dict."""
    trace = AgentTrace(agent_id="a", agent_role="r", output_produced={"x": 1})
    d = trace.to_dict()
    assert d["agent_id"] == "a"
    assert d["output_produced"] == {"x": 1}


def test_trace_capture_context_manager() -> None:
    """Test TraceCapture context manager."""
    with TraceCapture(agent_id="test", agent_role="role") as cap:
        cap.trace.input_received = {"in": 1}
        cap.output_produced = {"out": 2}
    assert cap.trace.agent_id == "test"
    assert cap.trace.output_produced == {"out": 2}
    assert cap.trace.latency_ms >= 0


def test_pipeline_trace_add_agent() -> None:
    """Test adding agent to pipeline trace."""
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn")
    a = AgentTrace(agent_id="a", agent_role="r")
    a.add_llm_call("", "", "gpt", 0, 0, 0.5, 0)
    pt.add_agent(a)
    assert len(pt.agents) == 1
    assert pt.total_cost_usd == 0.5


def test_pipeline_trace_get_agent() -> None:
    """Test getting agent by ID."""
    pt = PipelineTrace(pipeline_id="p", pipeline_name="pn")
    a = AgentTrace(agent_id="x", agent_role="r")
    pt.add_agent(a)
    assert pt.get_agent("x") is a
    assert pt.get_agent("y") is None
