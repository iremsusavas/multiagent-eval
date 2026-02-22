"""
Streamlit dashboard for evaluation results.

Loads results from JSON, shows sortable table, side-by-side comparison,
agent drill-down, propagation graph, golden dataset management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from multiagent_eval.golden_datasets.manager import GoldenDatasetManager
from multiagent_eval.reports.html_reporter import HTMLReporter
from multiagent_eval.core.runner import EvalResult
from multiagent_eval.core.trace import PipelineTrace
from multiagent_eval.judges.propagation_judge import PropagationJudge


def _load_results(results_dir: str) -> list[dict[str, Any]]:
    """Load all JSON result files from directory."""
    path = Path(results_dir)
    if not path.exists():
        return []
    results = []
    for f in path.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            data["_file"] = str(f)
            results.append(data)
        except Exception:
            pass
    return results


def run_dashboard(
    results_dir: str = "eval_results",
    datasets_dir: str = "datasets",
) -> None:
    """
    Launch Streamlit dashboard.

    Args:
        results_dir: Directory containing evaluation JSON results.
        datasets_dir: Directory for golden datasets.
    """
    st.set_page_config(page_title="MultiAgent-Eval Dashboard", layout="wide")

    st.title("MultiAgent-Eval Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Compare Runs", "Golden Datasets", "Export"])

    with tab1:
        results = _load_results(results_dir)
        if not results:
            st.info(f"No results found in {results_dir}. Run evaluations first.")
        else:
            # Sortable table
            scores = []
            for r in results:
                pt = r.get("pipeline_trace", {})
                metrics = r.get("metrics", [])
                avg = sum(m["score"] for m in metrics) / len(metrics) if metrics else 0
                scores.append({
                    "Pipeline": pt.get("pipeline_name", "?"),
                    "ID": pt.get("pipeline_id", "?"),
                    "Avg Score": f"{avg:.2%}",
                    "Cost": f"${pt.get('total_cost_usd', 0):.4f}",
                    "Latency": f"{pt.get('total_latency_ms', 0)}ms",
                    "File": r.get("_file", ""),
                    "_data": r,
                })

            st.dataframe(
                [{"Pipeline": s["Pipeline"], "ID": s["ID"], "Avg Score": s["Avg Score"], "Cost": s["Cost"], "Latency": s["Latency"]} for s in scores],
                use_container_width=True,
                hide_index=True,
            )

            selected_idx = st.selectbox("Select run for drill-down", range(len(scores)), format_func=lambda i: f"{scores[i]['Pipeline']} - {scores[i]['ID']}")
            if selected_idx is not None:
                r = scores[selected_idx]["_data"]
                pt_data = r.get("pipeline_trace", {})
                agents = pt_data.get("agents", [])
                agent_id = st.selectbox("Select agent", [a["agent_id"] for a in agents], key="agent_select")
                if agent_id:
                    agent = next((a for a in agents if a["agent_id"] == agent_id), None)
                    if agent:
                        st.subheader(f"Agent: {agent_id}")
                        st.json(agent)
                        st.subheader("LLM Calls")
                        for call in agent.get("llm_calls", []):
                            with st.expander(f"{call.get('model', '?')} - ${call.get('cost_usd', 0):.4f}"):
                                st.text(call.get("prompt", "")[:500] + "...")
                                st.text(call.get("response", "")[:500] + "...")

    with tab2:
        results = _load_results(results_dir)
        if len(results) < 2:
            st.info("Need at least 2 runs to compare.")
        else:
            opts = [f"{r.get('pipeline_trace', {}).get('pipeline_name', '?')} - {r.get('pipeline_trace', {}).get('pipeline_id', '?')}" for r in results]
            a_idx = st.selectbox("Run A", range(len(results)), key="comp_a")
            b_idx = st.selectbox("Run B", range(len(results)), key="comp_b")
            if a_idx != b_idx:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Run A")
                    st.json(results[a_idx].get("metrics", []))
                with col2:
                    st.subheader("Run B")
                    st.json(results[b_idx].get("metrics", []))

    with tab3:
        manager = GoldenDatasetManager(base_path=datasets_dir)
        ds_name = st.text_input("Dataset name", "my_dataset")
        if st.button("Create Dataset"):
            manager.create_dataset(ds_name, "Created via dashboard")
            st.success(f"Created {ds_name}")

        st.subheader("Add Example")
        pipeline_input = st.text_area("Pipeline input (JSON)", "{}")
        if st.button("Add Example"):
            try:
                inp = json.loads(pipeline_input)
                eid = manager.add_example(ds_name, inp)
                st.success(f"Added example {eid}")
            except Exception as e:
                st.error(str(e))

        st.subheader("Human Labels")
        ex_id = st.text_input("Example ID")
        agent_id = st.text_input("Agent ID")
        score = st.number_input("Score", 0.0, 1.0, 0.5)
        rater_id = st.text_input("Rater ID")
        if st.button("Add Label"):
            manager.add_human_label(ds_name, ex_id, agent_id, score, rater_id)
            st.success("Label added")

        st.subheader("Inter-Annotator Agreement")
        kappa = manager.compute_inter_annotator_agreement(ds_name)
        st.metric("Cohen's Kappa", f"{kappa:.3f}")

    with tab4:
        results = _load_results(results_dir)
        if results:
            idx = st.selectbox("Select run to export", range(len(results)), key="export_select")
            if st.button("Generate HTML Report"):
                r = results[idx]
                pt = PipelineTrace(
                    pipeline_id=r["pipeline_trace"]["pipeline_id"],
                    pipeline_name=r["pipeline_trace"]["pipeline_name"],
                    total_latency_ms=r["pipeline_trace"].get("total_latency_ms", 0),
                    total_cost_usd=r["pipeline_trace"].get("total_cost_usd", 0),
                    final_output=r["pipeline_trace"].get("final_output", {}),
                    metadata=r["pipeline_trace"].get("metadata", {}),
                )
                from multiagent_eval.core.trace import AgentTrace, LLMCall, ToolCall
                for a in r["pipeline_trace"].get("agents", []):
                    agent = AgentTrace(agent_id=a["agent_id"], agent_role=a["agent_role"])
                    agent.input_received = a.get("input_received", {})
                    agent.output_produced = a.get("output_produced", {})
                    agent.latency_ms = a.get("latency_ms", 0)
                    agent.error = a.get("error")
                    for c in a.get("llm_calls", []):
                        agent.llm_calls.append(LLMCall(
                            prompt=c.get("prompt", ""),
                            response=c.get("response", ""),
                            model=c.get("model", ""),
                            tokens_input=c.get("tokens_input", 0),
                            tokens_output=c.get("tokens_output", 0),
                            cost_usd=c.get("cost_usd", 0),
                            latency_ms=c.get("latency_ms", 0),
                            metadata=c.get("metadata", {}),
                        ))
                    pt.agents.append(agent)

                from multiagent_eval.core.metrics import MetricResult
                metrics = [
                    MetricResult(
                        metric_name=m["metric_name"],
                        score=m["score"],
                        explanation=m["explanation"],
                        agent_id=m.get("agent_id"),
                        flagged=m.get("flagged", False),
                        threshold=m.get("threshold", 0),
                        raw_data=m.get("raw_data", {}),
                    )
                    for m in r.get("metrics", [])
                ]
                eval_result = EvalResult(pipeline_trace=pt, metrics=metrics)
                out_path = Path(results_dir) / "report.html"
                HTMLReporter().generate(eval_result, str(out_path))
                st.success(f"Report saved to {out_path}")
        else:
            st.info("No results to export.")


def main() -> None:
    """Entry point for streamlit run."""
    run_dashboard()


if __name__ == "__main__":
    main()
