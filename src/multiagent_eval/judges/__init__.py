"""Judges for LLM-as-Judge and specialized evaluation."""

from multiagent_eval.judges.base_judge import BaseJudge
from multiagent_eval.judges.llm_judge import LLMJudge

__all__ = ["BaseJudge", "LLMJudge"]
