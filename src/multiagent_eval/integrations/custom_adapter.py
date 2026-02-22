"""
Custom adapter for pipelines that use TraceCapture manually.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from multiagent_eval.core.trace import AgentTrace, PipelineTrace, TraceCapture
from multiagent_eval.integrations.base_adapter import BaseAdapter


class CustomAdapter(BaseAdapter):
    """
    Adapter for custom pipelines using TraceCapture.

    Wraps a callable that accepts input and returns output, and expects
    the callable to use TraceCapture internally. Collects traces from
    a trace_collector callback.
    """

    def __init__(
        self,
        run_fn: Callable[[dict[str, Any]], Any],
        trace_collector: Callable[[], list[AgentTrace]],
        pipeline_name: str = "custom",
    ) -> None:
        """
        Initialize custom adapter.

        Args:
            run_fn: Function(input) -> output. Should use TraceCapture internally.
            trace_collector: Function that returns list of AgentTrace after run.
            pipeline_name: Name for the pipeline.
        """
        self.run_fn = run_fn
        self.trace_collector = trace_collector
        self.pipeline_name = pipeline_name

    def run_and_trace(self, input_data: dict[str, Any], **kwargs: Any) -> PipelineTrace:
        """Run pipeline and collect traces."""
        start = time.time()
        try:
            output = self.run_fn(input_data)
        except Exception as e:
            output = {"error": str(e)}

        agents = self.trace_collector()
        latency_ms = (time.time() - start) * 1000
        total_cost = sum(a.total_cost_usd() for a in agents)

        trace = PipelineTrace(
            pipeline_id=str(uuid.uuid4())[:8],
            pipeline_name=self.pipeline_name,
            total_latency_ms=int(latency_ms),
            total_cost_usd=total_cost,
            final_output=output if isinstance(output, dict) else {"result": output},
            agents=agents,
            metadata={"adapter": "custom"},
        )
        return trace
