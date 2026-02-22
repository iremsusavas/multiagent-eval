"""
Regression testing: version diff, commit-based eval history, impact table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RegressionDiff:
    """Diff between two eval runs."""

    version_from: str
    version_to: str
    worsened_examples: list[str] = field(default_factory=list)
    improved_examples: list[str] = field(default_factory=list)
    metric_changes: dict[str, float] = field(default_factory=dict)
    summary: str = ""


def load_result(path: str) -> dict[str, Any]:
    """Load eval result JSON."""
    return json.loads(Path(path).read_text())


def diff_results(
    result_a_path: str,
    result_b_path: str,
    version_a: str = "v1",
    version_b: str = "v2",
) -> RegressionDiff:
    """
    Compare two eval results. Which examples worsened/improved?
    """
    a = load_result(result_a_path)
    b = load_result(result_b_path)

    metrics_a = {m.get("metric_name", ""): m.get("score", 0) for m in a.get("metrics", [])}
    metrics_b = {m.get("metric_name", ""): m.get("score", 0) for m in b.get("metrics", [])}

    metric_changes: dict[str, float] = {}
    for name in set(metrics_a) | set(metrics_b):
        va = metrics_a.get(name, 0)
        vb = metrics_b.get(name, 0)
        metric_changes[name] = vb - va

    ex_id_a = a.get("metadata", {}).get("example_id", "?")
    ex_id_b = b.get("metadata", {}).get("example_id", "?")

    worsened: list[str] = []
    improved: list[str] = []
    avg_a = sum(metrics_a.values()) / len(metrics_a) if metrics_a else 0
    avg_b = sum(metrics_b.values()) / len(metrics_b) if metrics_b else 0
    if avg_b < avg_a:
        worsened.append(ex_id_b or ex_id_a)
    elif avg_b > avg_a:
        improved.append(ex_id_b or ex_id_a)

    summary = f"{version_a} → {version_b}: avg {avg_a:.3f} → {avg_b:.3f}"
    return RegressionDiff(
        version_from=version_a,
        version_to=version_b,
        worsened_examples=worsened,
        improved_examples=improved,
        metric_changes=metric_changes,
        summary=summary,
    )
