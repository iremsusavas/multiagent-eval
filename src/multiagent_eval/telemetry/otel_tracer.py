"""
OpenTelemetry integration for real-time monitoring.

Emits agent traces as OTLP spans for Datadog, Grafana, Jaeger integration.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Optional

from multiagent_eval.core.trace import AgentTrace, PipelineTrace

logger = logging.getLogger(__name__)

_tracer = None
_otel_enabled = False


def configure_otel(
    service_name: str = "multiagent-eval",
    endpoint: Optional[str] = None,
    enabled: bool = True,
) -> None:
    """
    Configure OpenTelemetry. Call before running pipelines.

    Args:
        service_name: Service name for traces.
        endpoint: OTLP endpoint (e.g. http://localhost:4317).
        enabled: Whether to emit spans.
    """
    global _tracer, _otel_enabled
    _otel_enabled = enabled
    if not enabled:
        _tracer = None
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("multiagent-eval", "0.1.0")
    except ImportError:
        logger.warning("OpenTelemetry not installed. pip install multiagent-eval[otel]")
        _tracer = None
        _otel_enabled = False


def get_tracer():
    """Get the OTL tracer or None if not configured."""
    return _tracer


@contextmanager
def agent_span(agent_id: str, agent_role: str):
    """
    Context manager that creates an OTL span for agent execution.
    Use as drop-in around agent execution for real-time monitoring.
    """
    tracer = get_tracer()
    if tracer and _otel_enabled:
        with tracer.start_as_current_span(
            f"agent.{agent_id}",
            attributes={
                "agent.id": agent_id,
                "agent.role": agent_role,
            },
        ) as span:
            yield span
    else:
        yield None


def emit_agent_span(agent_trace: AgentTrace, pipeline_id: str) -> None:
    """
    Emit a completed agent trace as an OTL span (for post-hoc export).
    """
    tracer = get_tracer()
    if not tracer or not _otel_enabled:
        return
    with tracer.start_as_current_span(
        f"agent.{agent_trace.agent_id}",
        attributes={
            "agent.id": agent_trace.agent_id,
            "agent.role": agent_trace.agent_role,
            "pipeline.id": pipeline_id,
            "agent.latency_ms": agent_trace.latency_ms,
            "agent.llm_calls": len(agent_trace.llm_calls),
            "agent.cost_usd": agent_trace.total_cost_usd(),
            "agent.error": agent_trace.error or "",
        },
    ) as span:
        if agent_trace.error and span:
            span.set_status(span.Status(span.StatusCode.ERROR, agent_trace.error))


def emit_pipeline_span(pipeline_trace: PipelineTrace) -> None:
    """
    Emit pipeline trace as OTL span.
    """
    tracer = get_tracer()
    if not tracer or not _otel_enabled:
        return
    with tracer.start_as_current_span(
        f"pipeline.{pipeline_trace.pipeline_name}",
        attributes={
            "pipeline.id": pipeline_trace.pipeline_id,
            "pipeline.name": pipeline_trace.pipeline_name,
            "pipeline.latency_ms": pipeline_trace.total_latency_ms,
            "pipeline.cost_usd": pipeline_trace.total_cost_usd,
            "pipeline.agents": len(pipeline_trace.agents),
        },
    ):
        for agent in pipeline_trace.agents:
            emit_agent_span(agent, pipeline_trace.pipeline_id)
