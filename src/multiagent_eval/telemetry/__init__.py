"""OpenTelemetry integration for real-time monitoring."""

from multiagent_eval.telemetry.otel_tracer import (
    get_tracer,
    emit_agent_span,
    emit_pipeline_span,
    configure_otel,
)

__all__ = [
    "get_tracer",
    "emit_agent_span",
    "emit_pipeline_span",
    "configure_otel",
]
