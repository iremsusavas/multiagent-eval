"""
Agent execution trace capture for multi-agent pipeline evaluation.

Provides AgentTrace, TraceCapture, and PipelineTrace for capturing
per-agent and full-pipeline execution data.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMCall:
    """Record of a single LLM invocation within an agent."""

    prompt: str
    response: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Record of a single tool invocation within an agent."""

    tool_name: str
    input_data: dict[str, Any]
    output_data: Any
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransition:
    """Record of a state change in the pipeline."""

    from_state: str
    to_state: str
    timestamp_ms: float
    trigger: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    """
    Per-agent execution trace capturing inputs, outputs, and execution details.

    Attributes:
        agent_id: Unique identifier for the agent.
        agent_role: Role/type of the agent (e.g., web_search, researcher).
        input_received: Exactly what this agent received from upstream.
        output_produced: Exactly what this agent sent downstream.
        llm_calls: List of LLM invocations during execution.
        tools_called: List of tool invocations during execution.
        start_time: Unix timestamp when execution started.
        end_time: Unix timestamp when execution ended.
        latency_ms: Total execution time in milliseconds.
        error: Error message if execution failed.
        metadata: Additional arbitrary data.
    """

    agent_id: str
    agent_role: str
    input_received: dict[str, Any] = field(default_factory=dict)
    output_produced: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[LLMCall] = field(default_factory=list)
    tools_called: list[ToolCall] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Add an LLM call record to this trace."""
        self.llm_calls.append(
            LLMCall(
                prompt=prompt,
                response=response,
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata=kwargs,
            )
        )

    def add_tool_call(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        output_data: Any,
        latency_ms: float,
        **kwargs: Any,
    ) -> None:
        """Add a tool call record to this trace."""
        self.tools_called.append(
            ToolCall(
                tool_name=tool_name,
                input_data=input_data,
                output_data=output_data,
                latency_ms=latency_ms,
                metadata=kwargs,
            )
        )

    def total_cost_usd(self) -> float:
        """Total cost of all LLM calls in this agent."""
        return sum(call.cost_usd for call in self.llm_calls)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "input_received": self.input_received,
            "output_produced": self.output_produced,
            "llm_calls": [
                {
                    "prompt": c.prompt,
                    "response": c.response,
                    "model": c.model,
                    "tokens_input": c.tokens_input,
                    "tokens_output": c.tokens_output,
                    "cost_usd": c.cost_usd,
                    "latency_ms": c.latency_ms,
                    "metadata": c.metadata,
                }
                for c in self.llm_calls
            ],
            "tools_called": [
                {
                    "tool_name": t.tool_name,
                    "input_data": t.input_data,
                    "output_data": t.output_data,
                    "latency_ms": t.latency_ms,
                    "metadata": t.metadata,
                }
                for t in self.tools_called
            ],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class PipelineTrace:
    """
    Full pipeline execution trace containing all agent traces and state transitions.

    Attributes:
        pipeline_id: Unique identifier for this pipeline run.
        pipeline_name: Human-readable pipeline name.
        total_latency_ms: Total execution time in milliseconds.
        total_cost_usd: Total cost across all agents.
        final_output: The final output of the pipeline.
        agents: Ordered list of agent traces.
        state_transitions: List of state changes during execution.
    """

    pipeline_id: str
    pipeline_name: str
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    final_output: dict[str, Any] = field(default_factory=dict)
    agents: list[AgentTrace] = field(default_factory=list)
    state_transitions: list[StateTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_agent(self, trace: AgentTrace) -> None:
        """Add an agent trace to the pipeline."""
        self.agents.append(trace)
        self.total_cost_usd += trace.total_cost_usd()

    def add_state_transition(
        self,
        from_state: str,
        to_state: str,
        timestamp_ms: Optional[float] = None,
        trigger: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a state transition record."""
        import time as t

        self.state_transitions.append(
            StateTransition(
                from_state=from_state,
                to_state=to_state,
                timestamp_ms=timestamp_ms or t.time() * 1000,
                trigger=trigger,
                metadata=kwargs,
            )
        )

    def get_agent(self, agent_id: str) -> Optional[AgentTrace]:
        """Get agent trace by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "final_output": self.final_output,
            "agents": [a.to_dict() for a in self.agents],
            "state_transitions": [
                {
                    "from_state": s.from_state,
                    "to_state": s.to_state,
                    "timestamp_ms": s.timestamp_ms,
                    "trigger": s.trigger,
                    "metadata": s.metadata,
                }
                for s in self.state_transitions
            ],
            "metadata": self.metadata,
        }


class TraceCapture:
    """
    Context manager for capturing agent execution traces with minimal code change.

    Usage:
        with TraceCapture(agent_id="researcher", agent_role="web_search") as trace:
            result = my_researcher_agent.run(input)
        trace.output_produced = result
    """

    def __init__(
        self,
        agent_id: str,
        agent_role: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize trace capture.

        Args:
            agent_id: Unique identifier for the agent.
            agent_role: Role/type of the agent.
            metadata: Optional metadata to attach to the trace.
        """
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.metadata = metadata or {}
        self._trace: Optional[AgentTrace] = None

    @property
    def trace(self) -> AgentTrace:
        """Get the captured trace. Raises if used outside context."""
        if self._trace is None:
            raise RuntimeError("TraceCapture used outside context manager")
        return self._trace

    @property
    def output_produced(self) -> dict[str, Any]:
        """Get the output produced by the agent."""
        return self.trace.output_produced

    @output_produced.setter
    def output_produced(self, value: dict[str, Any] | Any) -> None:
        """Set the output produced by the agent. Accepts dict or converts to dict."""
        if isinstance(value, dict):
            self.trace.output_produced = value
        else:
            self.trace.output_produced = {"result": value}

    def __enter__(self) -> "TraceCapture":
        self._trace = AgentTrace(
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            start_time=time.time(),
            metadata=self.metadata,
        )
        logger.debug("TraceCapture started for agent %s", self.agent_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._trace.end_time = time.time()
        self._trace.latency_ms = (self._trace.end_time - self._trace.start_time) * 1000
        if exc_val is not None:
            self._trace.error = str(exc_val)
            logger.warning("Agent %s failed: %s", self.agent_id, exc_val)
        return False  # Do not suppress exceptions
