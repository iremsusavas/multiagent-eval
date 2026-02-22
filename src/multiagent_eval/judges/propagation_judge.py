"""
Error Propagation Judge: detects information corruption across agent boundaries.

Novel component - builds PropagationGraph, identifies corruption origins,
and uses LLM to distinguish legitimate transformation from error propagation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from multiagent_eval.core.trace import AgentTrace, PipelineTrace
from multiagent_eval.core.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


@dataclass
class EdgeFidelity:
    """Fidelity score for an edge in the propagation graph."""

    from_agent: str
    to_agent: str
    similarity: float
    is_legitimate_transformation: Optional[bool] = None
    llm_verdict: Optional[str] = None


@dataclass
class PropagationReport:
    """Report from propagation analysis."""

    pipeline_trace: PipelineTrace
    fidelity_scores: list[EdgeFidelity] = field(default_factory=list)
    corruption_origins: list[str] = field(default_factory=list)
    graph_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PropagationJudge:
    """
    Analyzes error propagation across agent boundaries.

    Algorithm:
    1. For each consecutive agent pair, compute semantic similarity
    2. Flag pairs below threshold as potential corruption
    3. Use LLM to determine legitimate transformation vs error
    4. Build PropagationGraph, identify corruption origins
    """

    def __init__(
        self,
        similarity_fn: Optional[Any] = None,
        gateway: Optional[LLMGateway] = None,
        fidelity_threshold: float = 0.7,
        judge_model: Optional[str] = None,
    ) -> None:
        """
        Initialize propagation judge.

        Args:
            similarity_fn: (str, str) -> float for semantic similarity.
            gateway: LLM gateway for verdict on flagged pairs.
            fidelity_threshold: Below this = potential corruption.
            judge_model: Model for LLM verdict (e.g. ollama/mistral). Used if gateway not provided.
        """
        self.similarity_fn = similarity_fn
        self.gateway = gateway or LLMGateway(primary_model=judge_model or "ollama/mistral")
        self.fidelity_threshold = fidelity_threshold

    def analyze(self, pipeline_trace: PipelineTrace) -> PropagationReport:
        """
        Analyze pipeline for error propagation.

        Returns PropagationReport with fidelity scores, corruption origins,
        and graph data for visualization.
        """
        agents = pipeline_trace.agents
        fidelity_scores: list[EdgeFidelity] = []
        corruption_origins: list[str] = []

        for i in range(len(agents) - 1):
            prev = agents[i]
            curr = agents[i + 1]
            prev_out = str(prev.output_produced)
            curr_in = str(curr.input_received)

            sim = self._similarity(prev_out, curr_in)
            edge = EdgeFidelity(from_agent=prev.agent_id, to_agent=curr.agent_id, similarity=sim)

            if sim < self.fidelity_threshold:
                verdict, is_legit = self._llm_verdict(prev_out, curr_in)
                edge.is_legitimate_transformation = is_legit
                edge.llm_verdict = verdict
                if not is_legit:
                    if prev.agent_id not in corruption_origins:
                        corruption_origins.append(prev.agent_id)

            fidelity_scores.append(edge)

        # Net value destroyers: output quality < input quality
        for i, agent in enumerate(agents):
            if agent.error:
                if agent.agent_id not in corruption_origins:
                    corruption_origins.append(agent.agent_id)

        graph_data = self._build_graph_data(agents, fidelity_scores)

        return PropagationReport(
            pipeline_trace=pipeline_trace,
            fidelity_scores=fidelity_scores,
            corruption_origins=corruption_origins,
            graph_data=graph_data,
            metadata={"threshold": self.fidelity_threshold},
        )

    def _similarity(self, a: str, b: str) -> float:
        """Compute semantic similarity."""
        if self.similarity_fn:
            return self.similarity_fn(a, b)
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0

    def _llm_verdict(self, prev_out: str, curr_in: str) -> tuple[str, bool]:
        """Use LLM to determine if divergence is legitimate or error propagation."""
        try:
            prompt = f"""Two consecutive agents in a pipeline. Agent A produced:
---
{prev_out[:1500]}
---

Agent B received as input:
---
{curr_in[:1500]}
---

Is the transformation from A's output to B's input a LEGITIMATE transformation (e.g., summarization, reformatting) or ERROR PROPAGATION (information lost/corrupted)?

Answer JSON: {{"verdict": "legitimate" or "error_propagation", "reason": "brief explanation"}}"""
            r = self.gateway.complete(messages=[{"role": "user", "content": prompt}], temperature=0.0)
            content = r["content"]
            if "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                obj = __import__("json").loads(content[start:end])
                verdict = obj.get("verdict", "error_propagation").lower()
                reason = obj.get("reason", "")
                is_legit = "legitimate" in verdict
                return (reason, is_legit)
        except Exception as e:
            logger.warning("LLM verdict failed: %s", e)
        return ("Unable to determine", False)

    def _build_graph_data(self, agents: list[AgentTrace], edges: list[EdgeFidelity]) -> dict[str, Any]:
        """Build graph structure for visualization (networkx/D3)."""
        nodes = [{"id": a.agent_id, "label": a.agent_role, "error": a.error or ""} for a in agents]
        links = [
            {
                "source": e.from_agent,
                "target": e.to_agent,
                "weight": e.similarity,
                "fidelity": e.similarity,
                "legitimate": e.is_legitimate_transformation,
            }
            for e in edges
        ]
        return {"nodes": nodes, "links": links}
