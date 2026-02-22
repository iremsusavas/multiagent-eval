"""
CrewAI adapter: captures traces from CrewAI crew execution.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from multiagent_eval.core.trace import AgentTrace, PipelineTrace
from multiagent_eval.integrations.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class CrewAIAdapter(BaseAdapter):
    """
    Adapter for CrewAI crews.

    Wraps crew.kickoff() and captures traces from task outputs.
    Requires crewai package.
    """

    def __init__(
        self,
        crew: Any,
        pipeline_name: str = "crewai",
    ) -> None:
        """
        Initialize CrewAI adapter.

        Args:
            crew: CrewAI Crew instance.
            pipeline_name: Name for the pipeline.
        """
        self.crew = crew
        self.pipeline_name = pipeline_name

    def run_and_trace(self, input_data: dict[str, Any], **kwargs: Any) -> PipelineTrace:
        """Run crew and capture trace from tasks."""
        start = time.time()
        agents_list: list[AgentTrace] = []

        try:
            result = self.crew.kickoff(inputs=input_data)
            tasks = getattr(self.crew, "tasks", []) or []

            prev_output = input_data
            for i, task in enumerate(tasks):
                agent_id = getattr(task, "agent", None)
                if agent_id is not None:
                    agent_id = getattr(agent_id, "role", str(i))
                else:
                    agent_id = f"task_{i}"

                output = {}
                if hasattr(task, "output"):
                    output = {"result": task.output}
                elif hasattr(result, "raw") and isinstance(result.raw, list) and i < len(result.raw):
                    output = {"result": result.raw[i]}

                trace = AgentTrace(
                    agent_id=str(agent_id),
                    agent_role=getattr(getattr(task, "agent", None), "role", "agent"),
                    input_received=prev_output,
                    output_produced=output,
                    start_time=start,
                    end_time=time.time(),
                    latency_ms=0,
                )
                agents_list.append(trace)
                prev_output = output

            if not agents_list and result is not None:
                agents_list.append(
                    AgentTrace(
                        agent_id="crew",
                        agent_role="crew",
                        input_received=input_data,
                        output_produced={"result": str(result)},
                        start_time=start,
                        end_time=time.time(),
                        latency_ms=(time.time() - start) * 1000,
                    )
                )
        except Exception as e:
            logger.exception("CrewAI execution failed: %s", e)
            agents_list.append(
                AgentTrace(
                    agent_id="crew",
                    agent_role="crew",
                    input_received=input_data,
                    output_produced={},
                    error=str(e),
                    start_time=start,
                    end_time=time.time(),
                    latency_ms=(time.time() - start) * 1000,
                )
            )

        latency_ms = (time.time() - start) * 1000
        total_cost = sum(a.total_cost_usd() for a in agents_list)

        final = agents_list[-1].output_produced if agents_list else {}
        return PipelineTrace(
            pipeline_id=str(uuid.uuid4())[:8],
            pipeline_name=self.pipeline_name,
            total_latency_ms=int(latency_ms),
            total_cost_usd=total_cost,
            final_output=final,
            agents=agents_list,
            metadata={"adapter": "crewai"},
        )
