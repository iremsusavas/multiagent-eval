"""
Pydantic Settings + YAML config for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    name: str = "default"
    adapter: str = "custom"


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    golden_dataset: Optional[str] = None
    metrics: list[str] = Field(default_factory=lambda: ["factual_accuracy", "inter_agent_consistency"])
    thresholds: dict[str, float] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    """Judge configuration."""

    primary_model: str = "gpt-4o"
    fallback_model: str = "claude-sonnet-4-6"
    api_base: Optional[str] = None  # e.g. http://localhost:11434 for Ollama
    bias_detection: bool = True
    cot_prompting: bool = True


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    output_dir: str = "eval_results"
    formats: list[str] = Field(default_factory=lambda: ["json", "html"])
    dashboard: bool = False


class EvalConfigYAML(BaseModel):
    """Full eval config matching eval_config.yaml schema."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfigYAML":
        """Load config from YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data or {})
