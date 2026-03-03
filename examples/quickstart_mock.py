"""
quickstart_mock.py

The fastest way to see multiagent-eval in action.
No API key or external dependencies required.

Scenario: 3-agent research pipeline.
  Agent 1 (Researcher): Gathers information on a topic
  Agent 2 (Analyst):    Analyzes the findings          <- we inject a fault here
  Agent 3 (Writer):     Writes the final report

multiagent-eval's propagation detection identifies Agent 2 as the fault origin
by measuring semantic drift between each agent's input and output — not just
by checking the final output score.
"""

import time
from multiagent_eval.core.trace import AgentTrace, PipelineTrace, LLMCall
from multiagent_eval import evaluate


def build_healthy_pipeline() -> PipelineTrace:
    """3-agent pipeline where all agents stay on topic."""

    t0 = time.time()
    researcher_trace = AgentTrace(
        agent_id="agent_001",
        agent_role="researcher",
        input_received={"query": "What are the main causes of inflation?"},
        output_produced={
            "findings": "Inflation is caused by: (1) demand-pull factors where "
            "consumer demand exceeds supply, (2) cost-push factors "
            "from rising production costs, (3) monetary expansion "
            "when money supply grows faster than output.",
        },
        llm_calls=[
            LLMCall(
                prompt="Research the causes of inflation",
                response="Inflation is caused by...",
                model="gpt-4o",
                tokens_input=400,
                tokens_output=50,
                cost_usd=0.0054,
                latency_ms=1200,
            )
        ],
        tools_called=[],
        start_time=t0,
        end_time=t0 + 1.2,
        latency_ms=1200,
        error=None,
        metadata={},
    )

    analyst_trace = AgentTrace(
        agent_id="agent_002",
        agent_role="analyst",
        input_received=researcher_trace.output_produced,
        output_produced={
            "analysis": "Based on the research findings, demand-pull inflation "
            "is currently dominant due to post-pandemic consumer spending. "
            "Cost-push pressures from energy prices are secondary contributors.",
        },
        llm_calls=[
            LLMCall(
                prompt="Analyze these inflation findings: ...",
                response="Based on the research findings...",
                model="gpt-4o",
                tokens_input=350,
                tokens_output=30,
                cost_usd=0.0046,
                latency_ms=980,
            )
        ],
        tools_called=[],
        start_time=t0 + 1.2,
        end_time=t0 + 2.18,
        latency_ms=980,
        error=None,
        metadata={},
    )

    writer_trace = AgentTrace(
        agent_id="agent_003",
        agent_role="writer",
        input_received=analyst_trace.output_produced,
        output_produced={
            "report": "## Inflation Analysis Report\n\n"
            "Current inflationary pressures are primarily driven by "
            "demand-pull factors, with energy costs as a secondary contributor. "
            "Policy recommendations include gradual interest rate adjustments."
        },
        llm_calls=[
            LLMCall(
                prompt="Write a report based on this analysis: ...",
                response="## Inflation Analysis Report...",
                model="gpt-4o",
                tokens_input=400,
                tokens_output=120,
                cost_usd=0.0062,
                latency_ms=1400,
            )
        ],
        tools_called=[],
        start_time=t0 + 2.18,
        end_time=t0 + 3.58,
        latency_ms=1400,
        error=None,
        metadata={},
    )

    return PipelineTrace(
        pipeline_id="demo_healthy_001",
        pipeline_name="inflation_research_pipeline",
        total_latency_ms=3580,
        total_cost_usd=0.0162,
        final_output=writer_trace.output_produced,
        agents=[researcher_trace, analyst_trace, writer_trace],
        state_transitions=[],
    )


def build_corrupted_pipeline() -> PipelineTrace:
    """Pipeline where Agent 2 produces a semantically unrelated output.

    Agent 2 receives inflation research but produces output about climate change —
    a classic cascading hallucination pattern. The Propagation Judge detects
    the topic drift at Agent 2 (low input→output semantic similarity) rather
    than blaming Agent 3, which merely forwarded corrupted content.
    """

    t0 = time.time()
    researcher_trace = AgentTrace(
        agent_id="agent_001",
        agent_role="researcher",
        input_received={"query": "What are the main causes of inflation?"},
        output_produced={
            "findings": "Inflation is caused by: (1) demand-pull factors, "
            "(2) cost-push factors, (3) monetary expansion."
        },
        llm_calls=[
            LLMCall(
                prompt="Research the causes of inflation",
                response="Inflation is caused by...",
                model="gpt-4o",
                tokens_input=400,
                tokens_output=50,
                cost_usd=0.0054,
                latency_ms=1200,
            )
        ],
        tools_called=[],
        start_time=t0,
        end_time=t0 + 1.2,
        latency_ms=1200,
        error=None,
        metadata={},
    )

    # FAULT ORIGIN: Agent 2 received inflation research but produced
    # climate change content — a topic switch that corrupts all downstream agents.
    analyst_trace = AgentTrace(
        agent_id="agent_002",
        agent_role="analyst",
        input_received=researcher_trace.output_produced,
        output_produced={
            "analysis": "Climate change is accelerating due to greenhouse gas emissions. "
            "Carbon dioxide levels have risen 50% since pre-industrial times. "
            "Renewable energy adoption is the primary solution."
        },
        llm_calls=[
            LLMCall(
                prompt="Analyze these inflation findings: ...",
                response="Climate change is accelerating...",  # hallucination: wrong topic
                model="gpt-4o",
                tokens_input=350,
                tokens_output=30,
                cost_usd=0.0046,
                latency_ms=980,
            )
        ],
        tools_called=[],
        start_time=t0 + 1.2,
        end_time=t0 + 2.18,
        latency_ms=980,
        error=None,
        metadata={},
    )

    # Agent 3 receives corrupted input and produces a plausible-looking but
    # completely off-topic report. It is a victim, not the fault origin.
    writer_trace = AgentTrace(
        agent_id="agent_003",
        agent_role="writer",
        input_received=analyst_trace.output_produced,
        output_produced={
            "report": "## Climate Report\n\n"
            "Renewable energy is the key to addressing current economic challenges. "
            "Carbon emission reductions will stabilize prices long-term."
        },
        llm_calls=[
            LLMCall(
                prompt="Write a report based on this analysis: ...",
                response="## Climate Report...",
                model="gpt-4o",
                tokens_input=400,
                tokens_output=120,
                cost_usd=0.0062,
                latency_ms=1400,
            )
        ],
        tools_called=[],
        start_time=t0 + 2.18,
        end_time=t0 + 3.58,
        latency_ms=1400,
        error=None,
        metadata={},
    )

    return PipelineTrace(
        pipeline_id="demo_corrupted_001",
        pipeline_name="inflation_research_pipeline",
        total_latency_ms=3580,
        total_cost_usd=0.0162,
        final_output=writer_trace.output_produced,
        agents=[researcher_trace, analyst_trace, writer_trace],
        state_transitions=[],
    )


def print_result(label: str, result) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Overall Score : {result.overall_score:.2f}")
    print(f"  Passed        : {result.passed()}")
    print(f"\n  Per-Agent Scores:")
    agent_ids = list(result.agent_scores.keys())
    last_agent = agent_ids[-1] if agent_ids else None
    for agent_id, score in result.agent_scores.items():
        marker = "  <- fault origin" if score == 0.0 and agent_id != last_agent else ""
        print(f"    {agent_id}: {score:.2f}{marker}")
    print(f"\n  Failure Modes:")
    if result.failure_modes:
        for fm in result.failure_modes:
            print(f"    ⚠  {fm}")
    else:
        print(f"    ✓  None detected")
    print()


if __name__ == "__main__":
    # Ground truth: keywords expected in a correct inflation research report.
    # All 3 agents in the healthy pipeline produce output containing these terms.
    ground_truth = {
        "expected_topic": "inflation",
        "expected_keywords": [
            "inflation",
            "demand",
            "cost",
            "demand-pull",
            "cost-push",
            "factors",
        ],
        "expected": "A report covering the causes of inflation including demand-pull, "
        "cost-push, and monetary factors.",
    }

    print("\nmultiagent-eval quickstart demo")
    print("Scenario: 3-agent research pipeline (Researcher -> Analyst -> Writer)")

    print("\n[1/2] Running evaluation on HEALTHY pipeline...")
    healthy_trace = build_healthy_pipeline()
    healthy_result = evaluate(
        pipeline=healthy_trace,
        ground_truth=ground_truth,
        metrics=["factual_accuracy", "error_propagation_score"],
        thresholds={"factual_accuracy": 0.75, "error_propagation_score": 0.5},
    )
    print_result("HEALTHY PIPELINE", healthy_result)

    print("[2/2] Running evaluation on CORRUPTED pipeline...")
    print("      (Agent 2 hallucinates — switches topic from inflation to climate change)")
    corrupted_trace = build_corrupted_pipeline()
    corrupted_result = evaluate(
        pipeline=corrupted_trace,
        ground_truth=ground_truth,
        metrics=["factual_accuracy", "error_propagation_score"],
        thresholds={"factual_accuracy": 0.75, "error_propagation_score": 0.5},
    )
    print_result("CORRUPTED PIPELINE", corrupted_result)

    print("Key insight:")
    print("  Standard eval checks only the final output — it might miss where the fault began.")
    print("  multiagent-eval's error_propagation_score measures semantic drift at each agent,")
    print("  pinpointing agent_002 as the corruption origin (its output is unrelated to its input),")
    print("  rather than blaming agent_003 which merely propagated the corrupted content.")
    print()
