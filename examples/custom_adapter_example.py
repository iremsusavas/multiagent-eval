"""
Example: CustomAdapter with TraceCapture.

Use when you have a custom pipeline and wrap each agent with TraceCapture.
"""

from multiagent_eval.core.trace import AgentTrace, TraceCapture
from multiagent_eval.integrations import CustomAdapter


def my_pipeline(input_data: dict) -> dict:
    """Your pipeline that uses TraceCapture."""
    collected: list[AgentTrace] = []

    with TraceCapture(agent_id="agent_a", agent_role="analyzer") as cap:
        cap.trace.input_received = input_data
        out_a = {"analysis": str(input_data) + " analyzed"}
        cap.output_produced = out_a
        collected.append(cap.trace)

    with TraceCapture(agent_id="agent_b", agent_role="synthesizer") as cap:
        cap.trace.input_received = out_a
        out_b = {"synthesis": out_a["analysis"] + " synthesized"}
        cap.output_produced = out_b
        collected.append(cap.trace)

    return out_b


def get_traces() -> list[AgentTrace]:
    """Collector - in real usage, use a shared mutable list updated by TraceCapture."""
    # For this example we need to capture from my_pipeline - use a closure
    return get_traces._traces  # type: ignore


# Store traces in module-level list for collector
get_traces._traces: list[AgentTrace] = []


def run_with_adapter() -> None:
    """Run pipeline via CustomAdapter."""
    # We need the pipeline to populate a shared list - refactor to use a class
    class PipelineRunner:
        traces: list[AgentTrace] = []

        def run(self, input_data: dict) -> dict:
            self.traces = []
            with TraceCapture(agent_id="agent_a", agent_role="analyzer") as cap:
                cap.trace.input_received = input_data
                out_a = {"analysis": str(input_data) + " analyzed"}
                cap.output_produced = out_a
                self.traces.append(cap.trace)

            with TraceCapture(agent_id="agent_b", agent_role="synthesizer") as cap:
                cap.trace.input_received = out_a
                out_b = {"synthesis": out_a["analysis"] + " synthesized"}
                cap.output_produced = out_b
                self.traces.append(cap.trace)
            return out_b

        def get_traces(self) -> list[AgentTrace]:
            return self.traces

    runner = PipelineRunner()
    adapter = CustomAdapter(
        run_fn=runner.run,
        trace_collector=runner.get_traces,
        pipeline_name="custom_demo",
    )
    trace = adapter.run_and_trace({"query": "test"})
    print(f"Pipeline: {trace.pipeline_name}, Agents: {[a.agent_id for a in trace.agents]}")


if __name__ == "__main__":
    run_with_adapter()
