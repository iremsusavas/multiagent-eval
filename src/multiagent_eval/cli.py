"""
Command-line interface for multiagent-eval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from multiagent_eval.config import EvalConfigYAML
from multiagent_eval.core.runner import EvalConfig, EvaluationRunner, EvalResult
from multiagent_eval.core.trace import PipelineTrace
from multiagent_eval.golden_datasets.manager import GoldenDatasetManager
from multiagent_eval.reports.html_reporter import HTMLReporter
from multiagent_eval.reports.json_reporter import JSONReporter
from multiagent_eval.reports.dashboard import run_dashboard

app = typer.Typer(help="MultiAgent-Eval: Multi-Agent System Evaluation Framework")
logger = logging.getLogger(__name__)


def _get_similarity_fn():
    """Lazy load sentence-transformers for semantic similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _model = SentenceTransformer("all-MiniLM-L6-v2")

        def _sim(a: str, b: str) -> float:
            emb = _model.encode([a, b])
            return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9))

        return _sim
    except ImportError:
        return None


@app.command()
def run(
    config_path: str = typer.Option("eval_config.yaml", "--config", "-c", help="Path to eval config YAML"),
    example_index: Optional[int] = typer.Option(None, "--example", "-e", help="Example index (0-based). Default: 0"),
    all_examples: bool = typer.Option(False, "--all", "-a", help="Run on all examples in golden dataset"),
) -> None:
    """Run evaluation from config file."""
    path = Path(config_path)
    if not path.exists():
        typer.echo(f"Config not found: {config_path}")
        raise typer.Exit(1)

    cfg = EvalConfigYAML.from_yaml(path)
    eval_cfg = EvalConfig(
        pipeline_name=cfg.pipeline.name,
        golden_dataset_path=cfg.evaluation.golden_dataset,
        metrics=cfg.evaluation.metrics,
        thresholds=cfg.evaluation.thresholds,
        judge_primary_model=cfg.judge.primary_model,
        judge_fallback_model=cfg.judge.fallback_model,
        bias_detection=cfg.judge.bias_detection,
        cot_prompting=cfg.judge.cot_prompting,
        output_dir=cfg.reporting.output_dir,
        report_formats=cfg.reporting.formats,
        dashboard=cfg.reporting.dashboard,
    )

    # If golden dataset provided, load and run
    golden_path = Path(cfg.evaluation.golden_dataset) if cfg.evaluation.golden_dataset else None
    if golden_path and golden_path.exists():
        data = json.loads(golden_path.read_text())
        examples = data.get("examples", [])
        if examples:
            from multiagent_eval.core.trace import AgentTrace

            indices = range(len(examples)) if all_examples else [example_index if example_index is not None else 0]
            out_dir = Path(cfg.reporting.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            results_all: list[dict] = []

            for idx in indices:
                if idx < 0 or idx >= len(examples):
                    continue
                ex = examples[idx]
                ex_id = ex.get("example_id", f"ex{idx}")
                pt = PipelineTrace(
                    pipeline_id=f"cli_run_{ex_id}",
                    pipeline_name=cfg.pipeline.name,
                )
                agent = AgentTrace(
                    agent_id="example",
                    agent_role="agent",
                    input_received=ex.get("pipeline_input", {}),
                    output_produced=ex.get("final_expected_output", {}),
                )
                pt.add_agent(agent)
                pt.final_output = ex.get("final_expected_output", {})

                runner = EvaluationRunner(config=eval_cfg, similarity_fn=_get_similarity_fn())
                result = runner.run(pt, ground_truth=ex.get("final_expected_output"))
                result.metadata["example_id"] = ex_id
                result.metadata["example_index"] = idx

                if "json" in cfg.reporting.formats:
                    if all_examples:
                        results_all.append(result.to_dict())
                    else:
                        JSONReporter().generate(result, str(out_dir / "result.json"))
                if "html" in cfg.reporting.formats and not all_examples:
                    HTMLReporter().generate(result, str(out_dir / "report.html"))

            if all_examples:
                (out_dir / "results_batch.json").write_text(json.dumps({"results": results_all, "count": len(results_all)}, indent=2))
                avg_score = sum(sum(m["score"] for m in r["metrics"]) / len(r["metrics"]) for r in results_all) / len(results_all) if results_all else 0
                typer.echo(f"Ran {len(results_all)} examples. Avg score: {avg_score:.2%}")
            typer.echo(f"Results saved to {out_dir}")
            typer.echo(f"  JSON: {out_dir / 'result.json'}")
            typer.echo(f"  HTML: {out_dir / 'report.html'}")
    else:
        typer.echo("Golden dataset not found. Example usage:")
        typer.echo("  python examples/langgraph_example.py")
        typer.echo("  multiagent-eval report --input eval_results/example_result.json --format html")


@app.command()
def report(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to results JSON"),
    format: str = typer.Option("html", "--format", "-f", help="Output format: json, html"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Generate report from JSON results."""
    path = Path(input_path)
    if not path.exists():
        typer.echo(f"File not found: {input_path}")
        raise typer.Exit(1)

    data = json.loads(path.read_text())
    pt_data = data.get("pipeline_trace", data)
    pt = _dict_to_pipeline_trace(pt_data)
    metrics = [
        __import__("multiagent_eval.core.metrics", fromlist=["MetricResult"]).MetricResult(
            metric_name=m.get("metric_name", "?"),
            score=m.get("score", 0),
            explanation=m.get("explanation", ""),
            agent_id=m.get("agent_id"),
            flagged=m.get("flagged", False),
            threshold=m.get("threshold", 0),
            raw_data=m.get("raw_data", {}),
        )
        for m in data.get("metrics", [])
    ]
    result = EvalResult(
        pipeline_trace=pt,
        metrics=metrics,
        config=None,
        metadata=data.get("metadata", {}) or {},
    )

    out = output or (path.stem + "_report." + ("html" if format == "html" else "json"))
    if format == "html":
        HTMLReporter().generate(result, out)
    else:
        JSONReporter().generate(result, out)
    typer.echo(f"Report saved to {out}")


def _dict_to_pipeline_trace(d: dict) -> PipelineTrace:
    """Convert dict to PipelineTrace."""
    from multiagent_eval.core.trace import AgentTrace, LLMCall, ToolCall
    pt = PipelineTrace(
        pipeline_id=d.get("pipeline_id", "?"),
        pipeline_name=d.get("pipeline_name", "?"),
        total_latency_ms=d.get("total_latency_ms", 0),
        total_cost_usd=d.get("total_cost_usd", 0),
        final_output=d.get("final_output", {}),
        metadata=d.get("metadata", {}),
    )
    for a in d.get("agents", []):
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
        for t in a.get("tools_called", []):
            agent.tools_called.append(ToolCall(
                tool_name=t.get("tool_name", ""),
                input_data=t.get("input_data", {}),
                output_data=t.get("output_data"),
                latency_ms=t.get("latency_ms", 0),
                metadata=t.get("metadata", {}),
            ))
        pt.agents.append(agent)
    return pt


@app.command()
def dataset(
    action: str = typer.Argument(..., help="add, list, export, import"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    description: Optional[str] = typer.Option("", "--description"),
) -> None:
    """Manage golden datasets."""
    mgr = GoldenDatasetManager()
    if action == "add":
        n = name or typer.prompt("Dataset name")
        mgr.create_dataset(n, description or "")
        typer.echo(f"Created dataset: {n}")
    elif action == "list":
        for f in Path(mgr.base_path).glob("*.json"):
            typer.echo(f"  {f.stem}")
    else:
        typer.echo(f"Unknown action: {action}. Use: add, list, export, import")


@app.command()
def dashboard(
    results_dir: str = typer.Option("eval_results", "--results-dir"),
    datasets_dir: str = typer.Option("datasets", "--datasets-dir"),
) -> None:
    """Launch Streamlit dashboard."""
    typer.echo("Launching dashboard... Run: streamlit run multiagent_eval/reports/dashboard.py")
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "reports" / "dashboard.py"),
        "--server.headless", "true",
    ], env={**__import__("os").environ, "RESULTS_DIR": results_dir, "DATASETS_DIR": datasets_dir})


@app.command()
def estimate_cost(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Path to golden dataset JSON"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model for cost estimation"),
) -> None:
    """Pre-run cost estimation (dry-run): estimate $ before running evaluation."""
    from multiagent_eval.cost_estimator import estimate_eval_cost
    est = estimate_eval_cost(dataset_path, model=model)
    typer.echo(f"Cost estimate for {est.n_examples} examples ({model}):")
    typer.echo(f"  Min: ${est.estimated_min_usd:.4f}")
    typer.echo(f"  Max: ${est.estimated_max_usd:.4f}")
    typer.echo(f"  {est.explanation}")


@app.command()
def regression_diff(
    result_a: str = typer.Option(..., "--from", "-a", help="Path to first result JSON"),
    result_b: str = typer.Option(..., "--to", "-b", help="Path to second result JSON"),
    version_a: str = typer.Option("v1", "--version-from"),
    version_b: str = typer.Option("v2", "--version-to"),
) -> None:
    """Compare two eval results: which examples worsened/improved?"""
    from multiagent_eval.regression import diff_results
    diff = diff_results(result_a, result_b, version_a, version_b)
    typer.echo(diff.summary)
    typer.echo("Metric changes: " + str(diff.metric_changes))
    if diff.worsened_examples:
        typer.echo("Worsened: " + ", ".join(diff.worsened_examples))
    if diff.improved_examples:
        typer.echo("Improved: " + ", ".join(diff.improved_examples))


@app.command()
def bias_check(
    trace_path: str = typer.Option(..., "--trace", "-t", help="Path to trace JSON"),
) -> None:
    """Run bias detection on a trace."""
    path = Path(trace_path)
    if not path.exists():
        typer.echo(f"Trace not found: {trace_path}")
        raise typer.Exit(1)
    data = json.loads(path.read_text())
    pt = _dict_to_pipeline_trace(data.get("pipeline_trace", data))
    from multiagent_eval.bias_detection import PrimacyBiasDetector, VerbosityBiasDetector, ToneBiasDetector
    p = PrimacyBiasDetector()
    v = VerbosityBiasDetector()
    t = ToneBiasDetector()
    typer.echo("Bias detection (requires judge for full check). Use LLMJudge with bias_detection=True for full analysis.")
    typer.echo("Trace loaded: %d agents" % len(pt.agents))


if __name__ == "__main__":
    app()
