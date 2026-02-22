"""Pytest fixtures for multiagent-eval tests."""

import sys
from pathlib import Path

# Add src to path so multiagent_eval is importable without pip install
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
from multiagent_eval.core.trace import (
    AgentTrace,
    PipelineTrace,
    LLMCall,
    ToolCall,
    TraceCapture,
)


@pytest.fixture
def sample_agent_trace() -> AgentTrace:
    """Sample agent trace for testing."""
    trace = AgentTrace(
        agent_id="researcher",
        agent_role="web_search",
        input_received={"query": "test query"},
        output_produced={"summary": "test summary"},
        start_time=0,
        end_time=1,
        latency_ms=1000,
    )
    trace.add_llm_call(
        prompt="test",
        response="response",
        model="gpt-4",
        tokens_input=10,
        tokens_output=20,
        cost_usd=0.001,
        latency_ms=500,
    )
    return trace


@pytest.fixture
def sample_pipeline_trace(sample_agent_trace: AgentTrace) -> PipelineTrace:
    """Sample pipeline trace for testing."""
    trace2 = AgentTrace(
        agent_id="writer",
        agent_role="writer",
        input_received=sample_agent_trace.output_produced,
        output_produced={"report": "final report"},
        latency_ms=500,
    )
    pt = PipelineTrace(
        pipeline_id="test-123",
        pipeline_name="test_pipeline",
        total_latency_ms=1500,
        total_cost_usd=0.001,
        final_output={"report": "final report"},
        agents=[sample_agent_trace, trace2],
    )
    return pt
