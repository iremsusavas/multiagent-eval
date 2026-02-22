"""
MultiAgent-Eval: Open Source Multi-Agent System Evaluation Framework.

Enables systematic evaluation of multi-agent AI systems with detection of:
- Agent-to-agent error propagation
- Cascading hallucinations
- Orchestration breakdowns
- Inter-agent consistency drift
"""

__version__ = "0.1.0"

from multiagent_eval.core.trace import AgentTrace, PipelineTrace, TraceCapture, LLMCall, ToolCall, StateTransition
from multiagent_eval.core.metrics import MetricResult
from multiagent_eval.core.runner import EvalResult, EvalConfig
from multiagent_eval.sdk import evaluate

__all__ = [
    "__version__",
    "AgentTrace",
    "PipelineTrace",
    "TraceCapture",
    "LLMCall",
    "ToolCall",
    "StateTransition",
    "MetricResult",
    "EvalResult",
    "EvalConfig",
    "evaluate",
]
