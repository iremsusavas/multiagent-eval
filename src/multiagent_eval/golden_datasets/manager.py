"""
Golden dataset manager: create, add examples, human labels, inter-annotator agreement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from multiagent_eval.golden_datasets.schema import GoldenExample, HumanLabel

logger = logging.getLogger(__name__)


def _cohens_kappa(ratings: list[list[float]]) -> float:
    """Compute Cohen's Kappa for inter-rater agreement. Simplified for 2+ raters."""
    if not ratings or len(ratings) < 2:
        return 0.0
    n = len(ratings[0])
    if n == 0:
        return 0.0
    # Simplified: average pairwise correlation as proxy for kappa
    total = 0.0
    count = 0
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            r1, r2 = ratings[i], ratings[j]
            if len(r1) != len(r2):
                continue
            # Pearson-like correlation
            m1 = sum(r1) / len(r1)
            m2 = sum(r2) / len(r2)
            num = sum((a - m1) * (b - m2) for a, b in zip(r1, r2))
            den1 = sum((a - m1) ** 2 for a in r1) ** 0.5
            den2 = sum((b - m2) ** 2 for b in r2) ** 0.5
            if den1 and den2:
                total += num / (den1 * den2)
                count += 1
    return total / count if count else 0.0


class GoldenDatasetManager:
    """
    Manages golden datasets: create, add examples, human labels, export/import.
    """

    def __init__(
        self,
        base_path: str = "datasets",
        use_postgres: bool = False,
        connection_string: Optional[str] = None,
    ) -> None:
        """
        Initialize manager.

        Args:
            base_path: Directory for JSON storage.
            use_postgres: Use PostgreSQL backend (requires optional deps).
            connection_string: DB connection string if use_postgres.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.use_postgres = use_postgres
        self.connection_string = connection_string
        self._datasets: dict[str, dict[str, Any]] = {}

    def create_dataset(self, name: str, description: str = "") -> str:
        """Create a new dataset. Returns dataset path."""
        path = self.base_path / f"{name}.json"
        data = {
            "name": name,
            "description": description,
            "examples": [],
            "version": 1,
            "snapshots": {},
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Created dataset %s at %s", name, path)
        return str(path)

    def _load_dataset(self, name: str) -> dict[str, Any]:
        """Load dataset from file."""
        path = self.base_path / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Dataset {name} not found at {path}")
        return json.loads(path.read_text())

    def _save_dataset(self, name: str, data: dict[str, Any]) -> None:
        """Save dataset to file."""
        path = self.base_path / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))

    def add_example(
        self,
        dataset_name: str,
        pipeline_input: dict[str, Any],
        expected_outputs_per_agent: Optional[dict[str, dict[str, Any]]] = None,
        final_expected_output: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
    ) -> str:
        """
        Add a test case to the dataset.

        Returns:
            example_id
        """
        import uuid

        data = self._load_dataset(dataset_name)
        example_id = str(uuid.uuid4())[:8]
        example = {
            "example_id": example_id,
            "pipeline_input": pipeline_input,
            "expected_outputs_per_agent": expected_outputs_per_agent or {},
            "final_expected_output": final_expected_output or {},
            "tags": tags or [],
            "difficulty": difficulty,
            "human_labels": [],
            "metadata": {},
        }
        data["examples"].append(example)
        self._save_dataset(dataset_name, data)
        return example_id

    def add_human_label(
        self,
        dataset_name: str,
        example_id: str,
        agent_id: str,
        score: float,
        rater_id: str,
        notes: Optional[str] = None,
    ) -> None:
        """Record human annotation for an example."""
        data = self._load_dataset(dataset_name)
        for ex in data["examples"]:
            if ex["example_id"] == example_id:
                ex["human_labels"].append({
                    "example_id": example_id,
                    "agent_id": agent_id,
                    "score": score,
                    "rater_id": rater_id,
                    "notes": notes,
                })
                break
        self._save_dataset(dataset_name, data)

    def compute_inter_annotator_agreement(
        self,
        dataset_name: str,
        example_id: Optional[str] = None,
    ) -> float:
        """
        Compute Cohen's Kappa across raters.

        If example_id given, for that example. Else average across all.
        """
        data = self._load_dataset(dataset_name)
        examples = [e for e in data["examples"] if example_id is None or e["example_id"] == example_id]

        kappas: list[float] = []
        for ex in examples:
            labels = ex.get("human_labels", [])
            if len(labels) < 2:
                continue
            by_rater: dict[str, list[float]] = {}
            for l in labels:
                rid = l["rater_id"]
                if rid not in by_rater:
                    by_rater[rid] = []
                by_rater[rid].append(l["score"])
            if len(by_rater) >= 2:
                ratings = list(by_rater.values())
                kappas.append(_cohens_kappa(ratings))

        return sum(kappas) / len(kappas) if kappas else 0.0

    def get_hard_examples(
        self,
        dataset_name: str,
        agreement_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Return examples where raters disagree (low agreement = ambiguous = hard)."""
        data = self._load_dataset(dataset_name)
        hard = []
        for ex in data["examples"]:
            kappa = self.compute_inter_annotator_agreement(dataset_name, ex["example_id"])
            if kappa < agreement_threshold and ex.get("human_labels"):
                hard.append({**ex, "agreement": kappa})
        return hard

    def export_json(self, dataset_name: str, path: str) -> None:
        """Export dataset to JSON file."""
        data = self._load_dataset(dataset_name)
        Path(path).write_text(json.dumps(data, indent=2))

    def import_json(self, path: str, dataset_name: Optional[str] = None) -> str:
        """Import dataset from JSON. Returns dataset name."""
        data = json.loads(Path(path).read_text())
        name = dataset_name or data.get("name", Path(path).stem)
        self._save_dataset(name, data)
        return name

    def version_snapshot(self, dataset_name: str, tag: str) -> None:
        """Snapshot dataset at current state for regression testing."""
        data = self._load_dataset(dataset_name)
        if "snapshots" not in data:
            data["snapshots"] = {}
        data["snapshots"][tag] = json.loads(json.dumps(data["examples"]))
        self._save_dataset(dataset_name, data)
