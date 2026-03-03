"""
LangGraph adapter: automatically captures traces from compiled LangGraph graphs.

Captures per-node inputs/outputs via graph.stream() inspection, and captures
per-LLM-call details (tokens, model name) via LangChain's BaseCallbackHandler.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from multiagent_eval.core.trace import AgentTrace, PipelineTrace
from multiagent_eval.integrations.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


def _make_callback_handler(traces: list[AgentTrace]) -> Any:
    """
    Build a LangChain BaseCallbackHandler that captures LLM and tool events.

    Imports BaseCallbackHandler lazily so that multiagent-eval itself does not
    hard-depend on langchain-core at import time. If langchain-core is not
    installed, the adapter still works — node-level traces are captured via
    stream() inspection; only the per-LLM-call detail is skipped.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler  # type: ignore[import-untyped]
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore[import-untyped]
        except ImportError:
            logger.debug(
                "langchain-core not installed; per-LLM-call capture skipped. "
                "Install langchain-core>=0.1 for full call-level tracing."
            )
            return _NoOpHandler(traces)

    class _LangGraphTraceHandler(BaseCallbackHandler):
        """Callback handler that captures LLM and tool calls per LangGraph node."""

        def __init__(self) -> None:
            super().__init__()
            self._traces = traces

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
            if not self._traces:
                return
            agent = self._traces[-1]
            try:
                generations = getattr(response, "generations", [[]])
                llm_output = getattr(response, "llm_output", {}) or {}
                usage = llm_output.get("token_usage", {})
                for gen_list in generations:
                    for g in gen_list:
                        text = getattr(g, "text", str(g))
                        info = getattr(g, "generation_info", {}) or {}
                        model_name = (
                            info.get("model_name")
                            or llm_output.get("model_name")
                            or kwargs.get("invocation_params", {}).get("model_name", "unknown")
                        )
                        agent.add_llm_call(
                            prompt="",
                            response=text,
                            model=model_name,
                            tokens_input=usage.get("prompt_tokens", 0),
                            tokens_output=usage.get("completion_tokens", 0),
                            cost_usd=0.0,
                            latency_ms=0.0,
                        )
            except Exception:
                logger.debug("on_llm_end capture failed", exc_info=True)

        def on_tool_end(self, output: str, **kwargs: Any) -> None:  # type: ignore[override]
            if not self._traces:
                return
            agent = self._traces[-1]
            try:
                agent.add_tool_call(
                    tool_name=kwargs.get("name", "tool"),
                    input_data=kwargs.get("inputs") or kwargs.get("input", {}),
                    output_data=output,
                    latency_ms=0.0,
                )
            except Exception:
                logger.debug("on_tool_end capture failed", exc_info=True)

    return _LangGraphTraceHandler()


class _NoOpHandler:
    """Stub used when langchain-core is not installed."""

    def __init__(self, traces: list[AgentTrace]) -> None:
        self._traces = traces


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph compiled graphs.

    Wraps graph invocation and captures traces via LangChain callback integration
    and state inspection. Requires langgraph; for LLM-call-level detail also
    requires langchain-core>=0.1.

    Node-level input/output traces are always captured via stream() inspection.
    LLM call details (tokens, model name) are captured via BaseCallbackHandler,
    which requires langchain-core to be installed alongside langgraph.
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
        Run the LangGraph and capture a full execution trace.

        Uses graph.stream() to inspect state updates at each node boundary.
        LLM and tool call details are captured via the BaseCallbackHandler.
        Falls back gracefully if graph structure or callbacks differ.
        """
        start = time.time()
        self._traces = []

        # Try to get node metadata for role names
        try:
            nodes: Any = getattr(self.graph, "nodes", {})
            if callable(nodes):
                try:
                    g = nodes()
                    nodes = getattr(g, "nodes", {}) or {}
                except Exception:
                    nodes = {}
        except Exception:
            nodes = {}

        # Merge our callback handler into the run config
        config: dict[str, Any] = dict(kwargs.get("config", {}))
        existing_callbacks = list(config.get("callbacks", []))
        handler = _make_callback_handler(self._traces)
        config["callbacks"] = existing_callbacks + [handler]

        output: Any = None
        try:
            if hasattr(self.graph, "stream"):
                for event in self.graph.stream(input_data, config=config):
                    for node_name, state in (
                        event.items() if isinstance(event, dict) else [(str(event), event)]
                    ):
                        if isinstance(state, dict):
                            self._capture_node(node_name, state, nodes)
                    output = state if isinstance(event, dict) else event
            elif hasattr(self.graph, "invoke"):
                output = self.graph.invoke(input_data, config=config)
                if hasattr(output, "keys"):
                    for k in output.keys():
                        self._capture_node(k, output, nodes)
                else:
                    self._capture_node("node", {"output": output}, nodes)
            else:
                output = {"error": "Graph has no stream or invoke method"}
        except Exception as exc:
            logger.exception("LangGraph execution failed: %s", exc)
            output = {"error": str(exc)}
            if self._traces:
                self._traces[-1].error = str(exc)

        latency_ms = (time.time() - start) * 1000
        total_cost = sum(t.total_cost_usd() for t in self._traces)

        return PipelineTrace(
            pipeline_id=str(uuid.uuid4())[:8],
            pipeline_name=self.pipeline_name,
            total_latency_ms=int(latency_ms),
            total_cost_usd=total_cost,
            final_output=output if isinstance(output, dict) else {"result": output},
            agents=self._traces,
            metadata={"adapter": "langgraph"},
        )

    def _capture_node(self, node_name: str, state: dict[str, Any], nodes: Any) -> None:
        """Create or update agent trace for a node."""
        role = (
            nodes.get(node_name, {}).get("metadata", {}).get("role", node_name)
            if isinstance(nodes.get(node_name), dict)
            else node_name
        )
        existing = next((t for t in self._traces if t.agent_id == node_name), None)
        if existing:
            existing.output_produced = state
            existing.end_time = time.time()
            existing.latency_ms = (existing.end_time - existing.start_time) * 1000
        else:
            self._traces.append(
                AgentTrace(
                    agent_id=node_name,
                    agent_role=str(role),
                    input_received=state.get("messages", state) if isinstance(state, dict) else {},
                    output_produced=state,
                    start_time=time.time(),
                    end_time=time.time(),
                    metadata={"node": node_name},
                )
            )
