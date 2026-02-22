"""
LangGraph adapter: automatically captures traces from compiled LangGraph graphs.

Monkey-patches node execution to inject trace capture. Extracts agent_id from
node names, captures inputs/outputs at each node boundary.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from multiagent_eval.core.trace import AgentTrace, LLMCall, PipelineTrace, ToolCall
from multiagent_eval.integrations.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph compiled graphs.

    Wraps graph invocation and captures traces via LangChain callback integration
    and state inspection. Requires langgraph and langchain packages.
    """

    def __init__(
        self,
        graph: Any,
        pipeline_name: str = "langgraph",
    ) -> None:
        """
        Initialize LangGraph adapter.

        Args:
            graph: Compiled LangGraph graph (CompiledGraph).
            pipeline_name: Name for the pipeline.
        """
        self.graph = graph
        self.pipeline_name = pipeline_name
        self._traces: list[AgentTrace] = []

    def run_and_trace(self, input_data: dict[str, Any], **kwargs: Any) -> PipelineTrace:
        """
        Run the LangGraph and capture trace.

        Uses graph.stream() or graph.invoke() and inspects state updates
        to build agent traces. Falls back to minimal trace if structure differs.
        """
        start = time.time()
        self._traces = []

        try:
            # Try to get graph structure
            nodes = getattr(self.graph, "nodes", {}) or getattr(self.graph, "get_graph", lambda: None)()
            if callable(nodes):
                try:
                    g = nodes()
                    nodes = getattr(g, "nodes", {}) or {}
                except Exception:
                    nodes = {}
        except Exception:
            nodes = {}

        config = kwargs.get("config", {})
        config["callbacks"] = config.get("callbacks", [])
        # Add our callback handler for LLM/tool capture
        handler = _LangGraphTraceHandler(self._traces)
        config["callbacks"].append(handler)

        try:
            if hasattr(self.graph, "stream"):
                final = None
                for event in self.graph.stream(input_data, config=config):
                    for node_name, state in (event.items() if isinstance(event, dict) else [(str(event), event)]):
                        if isinstance(state, dict):
                            self._capture_node(node_name, state, nodes)
                    final = state if isinstance(event, dict) else event
                output = final
            elif hasattr(self.graph, "invoke"):
                output = self.graph.invoke(input_data, config=config)
                if hasattr(output, "keys"):
                    for k in output.keys():
                        self._capture_node(k, output, nodes)
                else:
                    self._capture_node("node", {"output": output}, nodes)
            else:
                output = {"error": "Graph has no stream or invoke"}
        except Exception as e:
            logger.exception("LangGraph execution failed: %s", e)
            output = {"error": str(e)}
            if self._traces:
                self._traces[-1].error = str(e)

        latency_ms = (time.time() - start) * 1000
        total_cost = sum(t.total_cost_usd() for t in self._traces)

        trace = PipelineTrace(
            pipeline_id=str(uuid.uuid4())[:8],
            pipeline_name=self.pipeline_name,
            total_latency_ms=int(latency_ms),
            total_cost_usd=total_cost,
            final_output=output if isinstance(output, dict) else {"result": output},
            agents=self._traces,
            metadata={"adapter": "langgraph"},
        )
        return trace

    def _capture_node(self, node_name: str, state: dict, nodes: dict) -> None:
        """Create or update agent trace for a node."""
        agent_id = node_name
        role = nodes.get(node_name, {}).get("metadata", {}).get("role", node_name) if isinstance(nodes.get(node_name), dict) else node_name

        existing = next((t for t in self._traces if t.agent_id == agent_id), None)
        if existing:
            existing.output_produced = state
            existing.end_time = time.time()
            existing.latency_ms = (existing.end_time - existing.start_time) * 1000
        else:
            trace = AgentTrace(
                agent_id=agent_id,
                agent_role=str(role),
                input_received=state.get("messages", state) if isinstance(state, dict) else {},
                output_produced=state,
                start_time=time.time(),
                end_time=time.time(),
                metadata={"node": node_name},
            )
            self._traces.append(trace)


class _LangGraphTraceHandler:
    """Callback handler to capture LLM and tool calls from LangChain."""

    def __init__(self, traces: list[AgentTrace]) -> None:
        self.traces = traces
        self._current_agent: Optional[AgentTrace] = None

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture LLM completion."""
        if not self.traces:
            return
        agent = self.traces[-1]
        try:
            generations = getattr(response, "generations", [[]])
            for gen_list in generations:
                for g in gen_list:
                    text = getattr(g, "text", str(g))
                    info = getattr(g, "generation_info", {}) or {}
                    agent.add_llm_call(
                        prompt="",
                        response=text,
                        model=info.get("model_name", "unknown"),
                        tokens_input=info.get("input_tokens", 0),
                        tokens_output=info.get("output_tokens", 0),
                        cost_usd=0,
                        latency_ms=0,
                    )
        except Exception:
            pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Capture tool call."""
        if not self.traces:
            return
        agent = self.traces[-1]
        try:
            agent.add_tool_call(
                tool_name=kwargs.get("name", "tool"),
                input_data=kwargs.get("input", {}),
                output_data=output,
                latency_ms=0,
            )
        except Exception:
            pass
