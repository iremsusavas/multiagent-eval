"""
quickstart_langgraph.py

Gerçek bir LangGraph pipeline'ını multiagent-eval ile değerlendirme.

Kurulum:
    pip install multiagent-eval langgraph openai

Çalıştırma:
    export OPENAI_API_KEY=sk-...
    python examples/quickstart_langgraph.py
"""

import os
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from multiagent_eval.integrations import LangGraphAdapter
from multiagent_eval import evaluate


# ── 1. LangGraph State tanımı ──────────────────────────────────────────

class ResearchState(TypedDict):
    query: str
    research_findings: str
    analysis: str
    final_report: str
    messages: Annotated[list, operator.add]


# ── 2. Agent node fonksiyonları ────────────────────────────────────────

def researcher_node(state: ResearchState) -> ResearchState:
    """Agent 1: Araştırma yapar."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research agent. Be concise."},
            {"role": "user", "content": f"Research this topic briefly: {state['query']}"},
        ],
        max_tokens=200,
    )
    findings = response.choices[0].message.content
    return {**state, "research_findings": findings}


def analyst_node(state: ResearchState) -> ResearchState:
    """Agent 2: Araştırma bulgularını analiz eder."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an analyst agent. Be concise."},
            {"role": "user", "content": f"Analyze these findings: {state['research_findings']}"},
        ],
        max_tokens=200,
    )
    analysis = response.choices[0].message.content
    return {**state, "analysis": analysis}


def writer_node(state: ResearchState) -> ResearchState:
    """Agent 3: Rapor yazar."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a writer agent. Be concise."},
            {"role": "user", "content": f"Write a brief report based on: {state['analysis']}"},
        ],
        max_tokens=300,
    )
    report = response.choices[0].message.content
    return {**state, "final_report": report}


# ── 3. Graph oluştur ───────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


# ── 4. Evaluate ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Run: export OPENAI_API_KEY=sk-..."
        )

    query = "What are the main causes of inflation?"
    print(f"\nRunning LangGraph pipeline for query: '{query}'")

    graph = build_graph()

    # LangGraphAdapter pipeline'ı wrap eder ve trace'i otomatik yakalar
    adapter = LangGraphAdapter(
        graph=graph,
        pipeline_name="research_pipeline",
    )

    input_data = {
        "query": query,
        "research_findings": "",
        "analysis": "",
        "final_report": "",
        "messages": [],
    }

    print("Evaluating...")
    result = evaluate(
        pipeline=adapter,
        input_data=input_data,
        ground_truth={
            "expected": "A report covering the causes of inflation.",
            "expected_keywords": ["inflation", "demand", "monetary", "cost"],
        },
    )

    print(f"\n{'='*50}")
    print(f"  Overall Score : {result.overall_score:.2f}")
    print(f"  Passed        : {result.passed()}")
    print(f"\n  Per-Agent Scores:")
    for agent_id, score in result.agent_scores.items():
        print(f"    {agent_id}: {score:.2f}")
    if result.failure_modes:
        print(f"\n  Failure Modes:")
        for fm in result.failure_modes:
            print(f"    ⚠  {fm}")
    print()
