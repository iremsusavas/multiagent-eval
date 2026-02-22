"""
LLM-as-Judge with built-in bias detection.

Implements Chain-of-Thought prompting, partial correctness scoring,
and automatic bias checks (primacy, verbosity, tone).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from multiagent_eval.judges.base_judge import BaseJudge, JudgeResult
from multiagent_eval.core.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


DEFAULT_RUBRIC = """
Score the output from 1 (worst) to 5 (best). Use partial correctness - never binary.
1: Completely wrong or irrelevant
2: Mostly wrong, minimal correctness
3: Partially correct, significant issues
4: Mostly correct, minor issues
5: Fully correct and complete

Always follow: Compare → Analyze → Decide → Score
"""


@dataclass
class BiasCheckResult:
    """Result of a bias detection check."""

    bias_type: str
    detected: bool
    score_a: float
    score_b: float
    threshold: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationEntry:
    """Entry in judge calibration log (judge score vs human label)."""

    output_id: str
    judge_score: float
    human_score: Optional[float] = None
    human_rater_id: Optional[str] = None
    notes: Optional[str] = None


class LLMJudge(BaseJudge):
    """
    LLM-as-Judge with Chain-of-Thought, partial correctness, and bias detection.

    Supports primacy, verbosity, and tone bias checks. Tracks calibration
    for agreement with human labels.
    """

    def __init__(
        self,
        gateway: Optional[LLMGateway] = None,
        primary_model: str = "gpt-4o",
        fallback_models: Optional[list[str]] = None,
        bias_detection: bool = True,
        cot_prompting: bool = True,
        primacy_threshold: float = 0.2,
        verbosity_threshold: float = 0.2,
        tone_threshold: float = 0.2,
    ) -> None:
        """
        Initialize LLM Judge.

        Args:
            gateway: LLM gateway (creates default if not provided).
            primary_model: Primary judge model.
            fallback_models: Fallback models on failure.
            bias_detection: Whether to run bias checks automatically.
            cot_prompting: Use Chain-of-Thought prompting.
            primacy_threshold: Score difference threshold for primacy bias.
            verbosity_threshold: Threshold for verbosity bias.
            tone_threshold: Threshold for tone bias.
        """
        self.gateway = gateway or LLMGateway(
            primary_model=primary_model,
            fallback_models=fallback_models or [],
        )
        self.bias_detection = bias_detection
        self.cot_prompting = cot_prompting
        self.primacy_threshold = primacy_threshold
        self.verbosity_threshold = verbosity_threshold
        self.tone_threshold = tone_threshold
        self.judge_calibration_log: list[CalibrationEntry] = []
        self._bias_results: list[BiasCheckResult] = []

    def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        rubric: Optional[str] = None,
        output_id: Optional[str] = None,
        **kwargs: Any,
    ) -> JudgeResult:
        """
        Evaluate output with CoT prompting and optional bias checks.

        Args:
            output: Output to evaluate.
            reference: Reference/ground truth.
            rubric: Scoring rubric with examples.
            output_id: Optional ID for calibration tracking.
            **kwargs: Additional params (e.g., human_score for calibration).

        Returns:
            JudgeResult with score (1-5 normalized to 0-1), reasoning, and bias info.
        """
        rubric_text = rubric or DEFAULT_RUBRIC
        if self.cot_prompting:
            rubric_text += "\n\nYou MUST reason step by step: Compare → Analyze → Decide → Score. Output JSON: {\"reasoning\": \"...\", \"score\": N}"

        prompt = self._build_prompt(output, reference, rubric_text)
        messages = [{"role": "user", "content": prompt}]

        response = self.gateway.complete(messages=messages, temperature=0.0)
        content = response["content"]

        score, reasoning = self._parse_response(content)
        result = JudgeResult(
            score=score,
            reasoning=reasoning,
            raw_response=content,
            metadata={"model": response["model"], "cost_usd": response["cost_usd"]},
        )

        if self.bias_detection and reference:
            bias_results = self._run_bias_checks(output, reference, rubric_text)
            result.metadata["bias_checks"] = [
                {
                    "bias_type": b.bias_type,
                    "detected": b.detected,
                    "explanation": b.explanation,
                }
                for b in bias_results
            ]
            self._bias_results.extend(bias_results)

        if output_id and "human_score" in kwargs:
            self.judge_calibration_log.append(
                CalibrationEntry(
                    output_id=output_id,
                    judge_score=score,
                    human_score=kwargs.get("human_score"),
                    human_rater_id=kwargs.get("human_rater_id"),
                    notes=kwargs.get("notes"),
                )
            )

        return result

    def _build_prompt(self, output: str, reference: Optional[str], rubric: str) -> str:
        """Build the evaluation prompt."""
        parts = [f"## Rubric\n{rubric}", "\n## Output to Evaluate\n" + output]
        if reference:
            parts.append("\n## Reference\n" + reference)
        parts.append("\n## Task\nEvaluate the output. Output JSON: {\"reasoning\": \"...\", \"score\": N} where N is 1-5.")
        return "\n".join(parts)

    def _parse_response(self, content: str) -> tuple[float, str]:
        """Parse judge response to extract score and reasoning."""
        reasoning = ""
        score = 0.5

        try:
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                obj = json.loads(content[start:end])
                score = float(obj.get("score", 0.5))
                reasoning = obj.get("reasoning", content)
            else:
                reasoning = content
                for line in content.split("\n"):
                    if "score" in line.lower():
                        for word in line.replace(":", " ").split():
                            if word.replace(".", "").isdigit():
                                score = float(word)
                                break
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse judge response: %s", e)
            reasoning = content

        score = max(1, min(5, score))
        normalized = (score - 1) / 4  # 1-5 -> 0-1
        return normalized, reasoning

    def _run_bias_checks(self, output: str, reference: str, rubric: str) -> list[BiasCheckResult]:
        """Run primacy, verbosity, and tone bias checks."""
        results: list[BiasCheckResult] = []

        # Primacy: (A,B) vs (B,A) order
        score_ab = self._quick_score(output, reference, rubric)
        score_ba = self._quick_score(reference, output, rubric)
        diff = abs(score_ab - score_ba)
        results.append(
            BiasCheckResult(
                bias_type="primacy",
                detected=diff > self.primacy_threshold,
                score_a=score_ab,
                score_b=score_ba,
                threshold=self.primacy_threshold,
                explanation=f"Order (output, ref) score={score_ab:.2f}, (ref, output)={score_ba:.2f}. Diff={diff:.2f}.",
            )
        )

        # Verbosity: inject verbose but wrong answer
        verbose_wrong = reference + " " + " ".join(["very" * 20, "detailed" * 10, "but" * 5, "incorrect" * 5])
        score_correct = self._quick_score(output, reference, rubric)
        score_verbose = self._quick_score(verbose_wrong, reference, rubric)
        results.append(
            BiasCheckResult(
                bias_type="verbosity",
                detected=score_verbose > score_correct,
                score_a=score_correct,
                score_b=score_verbose,
                threshold=self.verbosity_threshold,
                explanation=f"Correct score={score_correct:.2f}, verbose-wrong score={score_verbose:.2f}.",
            )
        )

        # Tone: same content, different tone
        neutral = output
        formal = "I humbly submit that " + output + " Thank you for your consideration."
        score_neutral = self._quick_score(neutral, reference, rubric)
        score_formal = self._quick_score(formal, reference, rubric)
        tone_diff = abs(score_neutral - score_formal)
        results.append(
            BiasCheckResult(
                bias_type="tone",
                detected=tone_diff > self.tone_threshold,
                score_a=score_neutral,
                score_b=score_formal,
                threshold=self.tone_threshold,
                explanation=f"Neutral score={score_neutral:.2f}, formal score={score_formal:.2f}. Diff={tone_diff:.2f}.",
            )
        )

        return results

    def _quick_score(self, output: str, reference: str, rubric: str) -> float:
        """Quick score without full CoT (for bias checks)."""
        prompt = f"Rubric: {rubric[:200]}...\nOutput: {output[:500]}\nReference: {reference[:500]}\nScore 1-5 as JSON: {{\"score\": N}}"
        try:
            r = self.gateway.complete(messages=[{"role": "user", "content": prompt}], temperature=0.0)
            content = r["content"]
            if "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                obj = json.loads(content[start:end])
                return float(obj.get("score", 0.5))
        except Exception:
            pass
        return 0.5

    def get_agreement_percentage(self) -> float:
        """Compute agreement between judge and human labels (when both present)."""
        entries = [e for e in self.judge_calibration_log if e.human_score is not None]
        if not entries:
            return 0.0
        # Consider "agreement" if within 0.2 (1 point on 1-5 scale)
        agreed = sum(1 for e in entries if abs(e.judge_score - (e.human_score or 0)) <= 0.2)
        return agreed / len(entries)

    def calibrate(self) -> dict[str, Any]:
        """
        Analyze where judge and humans disagree. Suggests rubric improvements.
        """
        entries = [e for e in self.judge_calibration_log if e.human_score is not None]
        if not entries:
            return {"message": "No human labels to calibrate against", "suggestions": []}

        disagreements = [e for e in entries if abs(e.judge_score - (e.human_score or 0)) > 0.2]
        avg_judge_over = sum(e.judge_score - (e.human_score or 0) for e in disagreements) / max(1, len(disagreements))

        suggestions = []
        if avg_judge_over > 0.1:
            suggestions.append("Judge tends to overscore. Add stricter criteria to rubric.")
        elif avg_judge_over < -0.1:
            suggestions.append("Judge tends to underscore. Consider more lenient partial credit.")

        return {
            "total_calibration_entries": len(entries),
            "disagreements": len(disagreements),
            "agreement_pct": self.get_agreement_percentage(),
            "avg_judge_human_diff": avg_judge_over,
            "suggestions": suggestions,
        }
