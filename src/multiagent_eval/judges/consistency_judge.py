"""
Consistency Judge: evaluates inter-agent consistency via pairwise semantic checks.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from multiagent_eval.core.trace import PipelineTrace
from multiagent_eval.core.metrics import MetricResult
from multiagent_eval.judges.base_judge import BaseJudge, JudgeResult


class ConsistencyJudge(BaseJudge):
    """
    Judge for inter-agent consistency. Uses semantic similarity and optional LLM
    to resolve contradictions.
    """

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        """
        Initialize consistency judge.

        Args:
            similarity_fn: Function (str, str) -> float for semantic similarity.
        """
        self.similarity_fn = similarity_fn

    def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """
        Evaluate consistency between output and reference (another agent's output).

        Returns score 0-1 where 1 = fully consistent.
        """
        if not reference:
            return JudgeResult(score=1.0, reasoning="No reference for consistency check.")

        if self.similarity_fn:
            sim = self.similarity_fn(output, reference)
        else:
            a, b = set(output.lower().split()), set(reference.lower().split())
            sim = len(a & b) / len(a | b) if (a or b) else 1.0

        return JudgeResult(
            score=sim,
            reasoning=f"Semantic consistency score: {sim:.2%}",
            metadata={"similarity": sim},
        )

    def evaluate_pipeline(self, pipeline_trace: PipelineTrace) -> list[tuple[str, str, float]]:
        """
        Pairwise consistency across all agents. Returns list of (agent_i, agent_j, score).
        """
        agents = pipeline_trace.agents
        results: list[tuple[str, str, float]] = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                out_i = str(agents[i].output_produced)
                out_j = str(agents[j].output_produced)
                r = self.evaluate(output=out_i, reference=out_j)
                results.append((agents[i].agent_id, agents[j].agent_id, r.score))
        return results
