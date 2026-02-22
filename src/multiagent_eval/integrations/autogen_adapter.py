"""
AutoGen adapter: captures traces from AutoGen multi-agent conversations.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from multiagent_eval.core.trace import AgentTrace, PipelineTrace
from multiagent_eval.integrations.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class AutoGenAdapter(BaseAdapter):
    """
    Adapter for AutoGen GroupChat or similar.

    Wraps conversation execution and captures traces from message history.
    Requires pyautogen or autogen package.
    """

    def __init__(
        self,
        group_chat: Any = None,
        run_fn: Any = None,
        pipeline_name: str = "autogen",
    ) -> None:
        """
        Initialize AutoGen adapter.

        Args:
            group_chat: AutoGen GroupChat or ConversableAgent.
            run_fn: Optional custom run function (chat_initiator, message) -> result.
            pipeline_name: Name for the pipeline.
        """
        self.group_chat = group_chat
        self.run_fn = run_fn
        self.pipeline_name = pipeline_name

    def run_and_trace(self, input_data: dict[str, Any], **kwargs: Any) -> PipelineTrace:
        """Run AutoGen conversation and capture trace."""
        start = time.time()
        agents_list: list[AgentTrace] = []

        try:
            if self.run_fn:
                result = self.run_fn(input_data)
            elif self.group_chat:
                initiator = kwargs.get("initiator")
                message = input_data.get("message", input_data.get("query", str(input_data)))
                if hasattr(self.group_chat, "initiate_chat"):
                    result = self.group_chat.initiate_chat(initiator or list(self.group_chat.agents)[0], message=message)
                else:
                    result = {"error": "No initiate_chat"}
            else:
                result = {"error": "No group_chat or run_fn provided"}

            # Extract from chat history
            history = getattr(result, "chat_history", getattr(self.group_chat, "chat_history", [])) if result else []
            if not history and hasattr(result, "messages"):
                history = result.messages

            prev_content = str(input_data)
            seen_senders: set[str] = set()
            for msg in (history or []):
                sender = getattr(msg, "name", getattr(msg, "sender", "unknown"))
                content = getattr(msg, "content", str(msg))
                if sender and sender not in seen_senders:
                    trace = AgentTrace(
                        agent_id=sender,
                        agent_role=sender,
                        input_received={"message": prev_content},
                        output_produced={"message": content},
                        start_time=start,
                        end_time=time.time(),
                        latency_ms=0,
                    )
                    agents_list.append(trace)
                    seen_senders.add(sender)
                    prev_content = content

            if not agents_list:
                agents_list.append(
                    AgentTrace(
                        agent_id="autogen",
                        agent_role="autogen",
                        input_received=input_data,
                        output_produced={"result": str(result)},
                        start_time=start,
                        end_time=time.time(),
                        latency_ms=(time.time() - start) * 1000,
                    )
                )
        except Exception as e:
            logger.exception("AutoGen execution failed: %s", e)
            agents_list.append(
                AgentTrace(
                    agent_id="autogen",
                    agent_role="autogen",
                    input_received=input_data,
                    output_produced={},
                    error=str(e),
                    start_time=start,
                    end_time=time.time(),
                    latency_ms=(time.time() - start) * 1000,
                )
            )

        latency_ms = (time.time() - start) * 1000
        total_cost = sum(a.total_cost_usd() for a in agents_list)
        final = agents_list[-1].output_produced if agents_list else {}

        return PipelineTrace(
            pipeline_id=str(uuid.uuid4())[:8],
            pipeline_name=self.pipeline_name,
            total_latency_ms=int(latency_ms),
            total_cost_usd=total_cost,
            final_output=final,
            agents=agents_list,
            metadata={"adapter": "autogen"},
        )
