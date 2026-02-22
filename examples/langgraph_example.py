"""
End-to-end example: LangGraph integration with multiagent-eval.

Shows how to:
1. Wrap agents with TraceCapture and record real LLM calls
2. Run evaluation
3. Generate HTML report with cost breakdown
"""

import time
import uuid

from multiagent_eval.core.trace import AgentTrace, PipelineTrace, TraceCapture
from multiagent_eval.core.runner import EvalConfig, EvaluationRunner
from multiagent_eval.core.llm_gateway import LLMGateway
from multiagent_eval.reports.html_reporter import HTMLReporter
from multiagent_eval.reports.json_reporter import JSONReporter


def run_researcher_agent(gateway: LLMGateway, input_data: dict, trace: AgentTrace) -> dict:
    """Researcher agent - calls LLM to generate summary."""
    query = input_data.get("query", "")
    prompt = f"Summarize the following query in 1-2 sentences: {query}"
    start = time.time()
    try:
        resp = gateway.complete(messages=[{"role": "user", "content": prompt}], temperature=0.3)
        latency_ms = (time.time() - start) * 1000
        content = resp["content"]
        trace.add_llm_call(
            prompt=prompt,
            response=content,
            model=resp["model"],
            tokens_input=resp["usage"]["prompt_tokens"],
            tokens_output=resp["usage"]["completion_tokens"],
            cost_usd=resp["cost_usd"],
            latency_ms=latency_ms,
        )
        return {"summary": content}
    except Exception as e:
        return {"summary": f"Error: {e}", "error": str(e)}


def run_writer_agent(gateway: LLMGateway, input_data: dict, trace: AgentTrace) -> dict:
    """Writer agent - calls LLM to generate report from summary."""
    summary = input_data.get("summary", "")
    prompt = f"Write a short report (2-3 sentences) based on this research summary: {summary}"
    start = time.time()
    try:
        resp = gateway.complete(messages=[{"role": "user", "content": prompt}], temperature=0.3)
        latency_ms = (time.time() - start) * 1000
        content = resp["content"]
        trace.add_llm_call(
            prompt=prompt,
            response=content,
            model=resp["model"],
            tokens_input=resp["usage"]["prompt_tokens"],
            tokens_output=resp["usage"]["completion_tokens"],
            cost_usd=resp["cost_usd"],
            latency_ms=latency_ms,
        )
        return {"report": content}
    except Exception as e:
        return {"report": f"Error: {e}", "error": str(e)}


def run_pipeline_with_llm(query: str, use_llm: bool = True) -> PipelineTrace:
    """Run 2-agent pipeline with optional real LLM calls."""
    gateway = LLMGateway(primary_model="ollama/mistral") if use_llm else None
    traces: list[AgentTrace] = []

    # Agent 1: Researcher
    with TraceCapture(agent_id="researcher", agent_role="web_search") as cap:
        cap.trace.input_received = {"query": query}
        if gateway:
            result = run_researcher_agent(gateway, {"query": query}, cap.trace)
        else:
            result = {"summary": "Research summary: " + str(query)}
        cap.output_produced = result
        traces.append(cap.trace)

    # Agent 2: Writer
    with TraceCapture(agent_id="writer", agent_role="report_writer") as cap:
        cap.trace.input_received = traces[0].output_produced
        if gateway:
            result = run_writer_agent(gateway, traces[0].output_produced, cap.trace)
        else:
            result = {"report": "Final report based on: " + traces[0].output_produced.get("summary", "")}
        cap.output_produced = result
        traces.append(cap.trace)

    pt = PipelineTrace(
        pipeline_id=str(uuid.uuid4())[:8],
        pipeline_name="research_pipeline",
        agents=traces,
        final_output=traces[-1].output_produced,
    )
    pt.total_latency_ms = int(sum(t.latency_ms for t in traces))
    pt.total_cost_usd = sum(t.total_cost_usd() for t in traces)
    return pt


def main() -> None:
    """Run example evaluation."""
    import sys
    use_llm = "--no-llm" not in sys.argv
    if not use_llm:
        print("Running with mock agents (no LLM calls)...")
    else:
        print("Running with Ollama/Mistral (real LLM calls)...")
        print("Ensure Ollama is running: ollama run mistral")

    # 1. Run pipeline
    trace = run_pipeline_with_llm("What is multi-agent AI?", use_llm=use_llm)

    # 2. Configure and run evaluation
    config = EvalConfig(
        pipeline_name="research_pipeline",
        metrics=["factual_accuracy", "inter_agent_consistency", "error_propagation_score"],
        thresholds={"factual_accuracy": 0.8, "inter_agent_consistency": 0.75},
    )
    runner = EvaluationRunner(config=config)
    ground_truth = {"report": "Multi-agent AI involves multiple AI agents collaborating to solve tasks."}
    result = runner.run(trace, ground_truth=ground_truth)

    # 3. Generate reports
    JSONReporter().generate(result, "eval_results/example_result.json")
    HTMLReporter().generate(result, "eval_results/example_report.html")

    print("Evaluation complete. Check eval_results/")
    print(f"Passed: {result.passed()}")
    print(f"Total cost: ${trace.total_cost_usd:.4f} | LLM calls: {sum(len(a.llm_calls) for a in trace.agents)}")
    for m in result.metrics:
        print(f"  {m.metric_name}: {m.score:.2f} - {m.explanation[:60]}...")


if __name__ == "__main__":
    main()
