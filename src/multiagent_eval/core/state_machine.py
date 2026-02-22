"""
Pipeline state tracker for orchestration adherence evaluation.

Tracks expected and actual state transitions in multi-agent pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from multiagent_eval.core.trace import PipelineTrace, StateTransition


@dataclass
class StateDefinition:
    """Definition of a valid state in the pipeline."""

    state_id: str
    description: str
    allowed_transitions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStateMachine:
    """
    Tracks pipeline state transitions and validates against expected flow.

    Used for orchestration_adherence metric.
    """

    def __init__(
        self,
        states: Optional[list[StateDefinition]] = None,
        initial_state: str = "start",
        expected_transitions: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        """
        Initialize state machine.

        Args:
            states: List of valid state definitions.
            initial_state: Starting state.
            expected_transitions: List of (from_state, to_state) tuples for validation.
        """
        self.states = {s.state_id: s for s in (states or [])}
        self.initial_state = initial_state
        self.expected_transitions = expected_transitions or []
        self._current_state = initial_state
        self._transition_history: list[StateTransition] = []
        self._start_time_ms = time.time() * 1000

    @property
    def current_state(self) -> str:
        """Current state of the pipeline."""
        return self._current_state

    def transition(
        self,
        to_state: str,
        trigger: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Attempt a state transition.

        Args:
            to_state: Target state.
            trigger: What triggered the transition.
            metadata: Additional metadata.

        Returns:
            True if transition is valid, False otherwise.
        """
        from_state = self._current_state

        if self.states:
            state_def = self.states.get(from_state)
            if state_def and state_def.allowed_transitions and to_state not in state_def.allowed_transitions:
                return False

        self._transition_history.append(
            StateTransition(
                from_state=from_state,
                to_state=to_state,
                timestamp_ms=time.time() * 1000,
                trigger=trigger,
                metadata=metadata or {},
            )
        )
        self._current_state = to_state
        return True

    def get_transitions(self) -> list[StateTransition]:
        """Get all recorded transitions."""
        return list(self._transition_history)

    def get_expected_transitions(self) -> list[tuple[str, str]]:
        """Get expected transition sequence for validation."""
        return list(self.expected_transitions)

    def apply_to_trace(self, pipeline_trace: PipelineTrace) -> None:
        """Apply recorded transitions to a pipeline trace."""
        for t in self._transition_history:
            pipeline_trace.add_state_transition(
                from_state=t.from_state,
                to_state=t.to_state,
                timestamp_ms=t.timestamp_ms,
                trigger=t.trigger,
                **t.metadata,
            )
