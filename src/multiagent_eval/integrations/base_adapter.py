"""
Base adapter interface for pipeline integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from multiagent_eval.core.trace import PipelineTrace


class BaseAdapter(ABC):
    """Abstract base for pipeline adapters."""

    @abstractmethod
    def run_and_trace(self, input_data: dict[str, Any], **kwargs: Any) -> PipelineTrace:
        """
        Run the pipeline with given input and capture full trace.

        Args:
            input_data: Pipeline input (e.g., {"query": "..."}).
            **kwargs: Adapter-specific options.

        Returns:
            PipelineTrace with all agent traces.
        """
        pass
