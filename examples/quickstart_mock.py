"""
quickstart_mock.py

multiagent-eval'ın ne yaptığını görmek için en hızlı yol.
Hiçbir API key veya harici bağımlılık gerektirmez.

Senaryo: 3 agent'lı bir araştırma pipeline'ı.
  Agent 1 (Researcher): Bir konu hakkında bilgi toplar
  Agent 2 (Analyst):    Toplanan bilgiyi analiz eder  ← burada hata enjekte edeceğiz
  Agent 3 (Writer):     Rapor yazar

multiagent-eval'ın propagation detection'ı Agent 2'deki bozulmayı
Agent 3'ün outputuna bakarak değil, Agent 2'nin outputunu inceleyerek yakalar.
"""

import time
from multiagent_eval.core.trace import AgentTrace, PipelineTrace, LLMCall
from multiagent_eval import evaluate


def build_healthy_pipeline() -> PipelineTrace:
    """3 agent'ın sorunsuz çalıştığı örnek trace."""

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

    pt = PipelineTrace(
        pipeline_id="demo_healthy_001",
        pipeline_name="inflation_research_pipeline",
        total_latency_ms=3580,
        total_cost_usd=0.0162,
        final_output=writer_trace.output_produced,
        agents=[researcher_trace, analyst_trace, writer_trace],
        state_transitions=[],
    )
    return pt


def build_corrupted_pipeline() -> PipelineTrace:
    """Agent 2'nin bozulmuş output ürettiği örnek trace.

    Agent 2, inflation araştırmasını alıyor ama tamamen alakasız
    bir konu hakkında çıktı üretiyor. Propagation Judge bunu yakalamalı.
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

    # ← BURADA HATA: Agent 2 inflation bilgisini almış ama
    # iklim değişikliği hakkında output üretiyor.
    # Bu klasik bir cascading hallucination başlangıcı.
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
                response="Climate change is accelerating...",  # ← hallucination
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

    # Agent 3 bozulmuş inputu alıyor, makul görünen ama yanlış bir rapor yazıyor
    # (Header'da "Inflation" yok — tamamen konu dışı)
    writer_trace = AgentTrace(
        agent_id="agent_003",
        agent_role="writer",
        input_received=analyst_trace.output_produced,
        output_produced={
            "report": "## Climate Report\n\n"
            "Renewable energy is the key to addressing current economic challenges. "
            "Carbon emission reductions will stabilize prices long-term."
            # ← Rapor inflation sorusuna cevap vermek yerine iklimden bahsediyor
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

    pt = PipelineTrace(
        pipeline_id="demo_corrupted_001",
        pipeline_name="inflation_research_pipeline",
        total_latency_ms=3580,
        total_cost_usd=0.0162,
        final_output=writer_trace.output_produced,
        agents=[researcher_trace, analyst_trace, writer_trace],
        state_transitions=[],
    )
    return pt


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
        # Only mark fault origin: agent with 0.0 that is NOT the last (victim)
        marker = "  ← fault origin" if score == 0.0 and agent_id != last_agent else ""
        print(f"    {agent_id}: {score:.2f}{marker}")
    print(f"\n  Failure Modes:")
    if result.failure_modes:
        for fm in result.failure_modes:
            print(f"    ⚠  {fm}")
    else:
        print(f"    ✓  None detected")
    if hasattr(result, "propagation_graph") and result.propagation_graph:
        print(f"\n  Propagation Graph Edges:")
        for edge in result.propagation_graph.edges:
            print(f"    {edge.source} → {edge.target}  fidelity={edge.fidelity_score:.2f}")
    print()


if __name__ == "__main__":
    # Keyword-based ground truth: all 3 agents' healthy output contains these
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
    print("Scenario: 3-agent research pipeline (Researcher → Analyst → Writer)")

    print("\n[1/2] Running evaluation on HEALTHY pipeline...")
    healthy_trace = build_healthy_pipeline()
    healthy_result = evaluate(
        pipeline=healthy_trace,
        ground_truth=ground_truth,
        metrics=["factual_accuracy"],
        thresholds={"factual_accuracy": 0.75},
    )
    print_result("HEALTHY PIPELINE", healthy_result)

    print("[2/2] Running evaluation on CORRUPTED pipeline...")
    print("      (Agent 2 hallucinates — switches topic from inflation to climate change)")
    corrupted_trace = build_corrupted_pipeline()
    corrupted_result = evaluate(
        pipeline=corrupted_trace,
        ground_truth=ground_truth,
        metrics=["factual_accuracy"],
        thresholds={"factual_accuracy": 0.75},
    )
    print_result("CORRUPTED PIPELINE", corrupted_result)

    print("Key insight:")
    print("  Standard eval sees final output → might miss the fault origin.")
    print("  multiagent-eval's Propagation Judge traces the fault to agent_002,")
    print("  not agent_003 which merely propagated the corrupted input.")
    print()
