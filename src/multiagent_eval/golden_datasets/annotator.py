"""
Annotator utilities for golden dataset labeling.
"""

from __future__ import annotations

from typing import Any, Optional

from multiagent_eval.golden_datasets.manager import GoldenDatasetManager
from multiagent_eval.golden_datasets.schema import GoldenExample, HumanLabel


class DatasetAnnotator:
    """
    Helper for adding and managing human annotations on golden datasets.
    """

    def __init__(self, manager: GoldenDatasetManager) -> None:
        """Initialize with a dataset manager."""
        self.manager = manager

    def record_label(
        self,
        dataset_name: str,
        example_id: str,
        agent_id: str,
        score: float,
        rater_id: str,
        notes: Optional[str] = None,
    ) -> None:
        """Record a human label."""
        self.manager.add_human_label(
            dataset_name=dataset_name,
            example_id=example_id,
            agent_id=agent_id,
            score=score,
            rater_id=rater_id,
            notes=notes,
        )

    def get_examples_needing_labels(
        self,
        dataset_name: str,
        min_labels_per_example: int = 2,
    ) -> list[dict[str, Any]]:
        """Get examples that need more human labels for agreement computation."""
        data = self.manager._load_dataset(dataset_name)
        needing = []
        for ex in data.get("examples", []):
            labels = ex.get("human_labels", [])
            if len(labels) < min_labels_per_example:
                needing.append(ex)
        return needing
