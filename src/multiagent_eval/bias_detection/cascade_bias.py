"""
Cascade Bias Detection: bias introduced by upstream agent errors.

When an upstream agent produces low-quality output, downstream agents
may be penalized for faithfully propagating that error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from multiagent_eval.core.trace import PipelineTrace


@dataclass
class CascadeBiasResult:
    """Result of cascade bias analysis."""

    detected: bool
    affected_agents: list[str]
    upstream_error_agents: list[str]
    explanation: str
    raw_data: dict[str, Any]


class CascadeBiasDetector:
    """
    Detects cascade bias: downstream agents penalized for upstream errors.

    Identifies agents whose input quality is low due to upstream failure,
    and flags if evaluation doesn't account for this (treating propagation
    of bad input as agent failure).
    """

    def __init__(self, quality_threshold: float = 0.5) -> None:
        """
        Initialize detector.

        Args:
            quality_threshold: Below this input quality = potential cascade victim.
        """
        self.quality_threshold = quality_threshold

    def analyze(
        self,
        pipeline_trace: PipelineTrace,
        agent_quality_scores: Optional[dict[str, float]] = None,
    ) -> CascadeBiasResult:
        """
        Analyze pipeline for cascade bias.

        Args:
            pipeline_trace: The pipeline trace.
            agent_quality_scores: Optional per-agent quality (output) scores.

        Returns:
            CascadeBiasResult with affected agents and upstream error sources.
        """
        agents = pipeline_trace.agents
        upstream_errors: list[str] = []
        affected: list[str] = []

        for i, agent in enumerate(agents):
            if agent.error:
                upstream_errors.append(agent.agent_id)
                for j in range(i + 1, len(agents)):
                    affected.append(agents[j].agent_id)

        if agent_quality_scores:
            for aid, score in agent_quality_scores.items():
                if score < self.quality_threshold and aid not in upstream_errors:
                    agent = pipeline_trace.get_agent(aid)
                    if agent and agent.input_received:
                        idx = next((j for j, a in enumerate(agents) if a.agent_id == aid), -1)
                        if idx > 0 and agents[idx - 1].agent_id in upstream_errors:
                            affected.append(aid)

        detected = len(affected) > 0

        return CascadeBiasResult(
            detected=detected,
            affected_agents=list(set(affected)),
            upstream_error_agents=upstream_errors,
            explanation=f"Upstream errors in {upstream_errors}. Downstream agents {affected} may be cascade victims.",
            raw_data={"agent_count": len(agents)},
        )
