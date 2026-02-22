"""
JSON report generation for evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from multiagent_eval.core.runner import EvalResult


class JSONReporter:
    """Generates JSON reports from evaluation results."""

    def generate(
        self,
        result: EvalResult,
        output_path: str,
        indent: int = 2,
    ) -> str:
        """
        Generate JSON report file.

        Args:
            result: Evaluation result.
            output_path: Path to write JSON.
            indent: JSON indent level.

        Returns:
            Path to generated file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = result.to_dict()
        path.write_text(json.dumps(data, indent=indent))
        return str(path)
