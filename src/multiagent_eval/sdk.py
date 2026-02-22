"""
SDK mode: library-first one-liner for evaluation.

Usage:
    from multiagent_eval import evaluate
    result = evaluate(pipeline=my_graph, dataset="my_dataset")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from multiagent_eval.core.runner import EvalConfig, EvaluationRunner, EvalResult
from multiagent_eval.core.trace import PipelineTrace
from multiagent_eval.golden_datasets.manager import GoldenDatasetManager
from multiagent_eval.reports.html_reporter import HTMLReporter
from multiagent_eval.reports.json_reporter import JSONReporter


def evaluate(
    pipeline: Union[PipelineTrace, Any],
    dataset: Optional[str] = None,
    input_data: Optional[dict[str, Any]] = None,
    ground_truth: Optional[dict[str, Any]] = None,
    metrics: Optional[list[str]] = None,
    thresholds: Optional[dict[str, float]] = None,
    output_dir: Optional[str] = None,
    report_formats: Optional[list[str]] = None,
        **kwargs: Any,
) -> EvalResult:
    """
    One-liner evaluation. Run evaluation on pipeline or adapter.

    Args:
        pipeline: PipelineTrace, or adapter (LangGraphAdapter, etc.) with run_and_trace().
        dataset: Path to golden dataset JSON. If set, loads examples for evaluation.
        input_data: Input for pipeline (if pipeline is adapter).
        ground_truth: Expected output for comparison.
        metrics: List of metric names.
        thresholds: Metric thresholds.
        output_dir: Where to save reports.
        **kwargs: Passed to adapter.run_and_trace() or config.

    Returns:
        EvalResult with metrics and pipeline trace.

    Example:
        from multiagent_eval import evaluate
        from multiagent_eval.integrations import LangGraphAdapter

        adapter = LangGraphAdapter(graph=my_graph)
        result = evaluate(pipeline=adapter, input_data={"query": "..."}, dataset="datasets/qa.json")
    """
    # Load dataset for ground_truth and/or input_data
    dataset_example = None
    if dataset:
        try:
            path = Path(dataset)
            if path.exists():
                data = json.loads(path.read_text())
            else:
                mgr = GoldenDatasetManager(base_path=".")
                data = mgr._load_dataset(dataset.replace(".json", "").split("/")[-1])
            examples = data.get("examples", [])
            if examples:
                dataset_example = examples[0]
        except Exception:
            pass

    # Resolve pipeline trace
    if isinstance(pipeline, PipelineTrace):
        trace = pipeline
    elif hasattr(pipeline, "run_and_trace"):
        inp = input_data or (dataset_example.get("pipeline_input", {}) if dataset_example else {})
        trace = pipeline.run_and_trace(inp, **kwargs)
    else:
        raise TypeError("pipeline must be PipelineTrace or adapter with run_and_trace()")

    # Ground truth from dataset if needed
    if dataset_example and not ground_truth:
        ground_truth = dataset_example.get("final_expected_output", {})

    config = EvalConfig(
        pipeline_name=trace.pipeline_name,
        metrics=metrics or ["factual_accuracy", "inter_agent_consistency", "error_propagation_score"],
        thresholds=thresholds or {},
    )
    runner = EvaluationRunner(config=config)
    result = runner.run(trace, ground_truth=ground_truth)

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        formats = report_formats or ["json", "html"]
        if "json" in formats:
            JSONReporter().generate(result, f"{output_dir}/result.json")
        if "html" in formats:
            HTMLReporter().generate(result, f"{output_dir}/report.html")

    return result
