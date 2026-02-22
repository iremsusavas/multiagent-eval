"""
LLM Gateway: provider-agnostic interface for LLM calls with retry, timeout, and cost tracking.

All LLM calls in the framework go through this gateway.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMGateway:
    """
    Provider-agnostic LLM gateway with retry, timeout, and cost tracking.

    Uses litellm for unified API access to OpenAI, Anthropic, and OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        primary_model: str = "gpt-4o",
        fallback_models: Optional[list[str]] = None,
        max_retries: int = 3,
        timeout_seconds: float = 120.0,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM gateway.

        Args:
            primary_model: Primary model to use (e.g., gpt-4o, claude-3-5-sonnet-20241022).
            fallback_models: Models to try if primary fails.
            max_retries: Number of retries with exponential backoff.
            timeout_seconds: Request timeout.
            api_base: Optional custom API base URL.
            api_key: Optional API key (uses env vars if not set).
        """
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.api_base = api_base
        self.api_key = api_key
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Send a completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Override model for this call.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            Dict with keys: content, model, usage, cost_usd, latency_ms, raw_response.
        """
        import litellm

        models_to_try = [model or self.primary_model] + [
            m for m in self.fallback_models if m != (model or self.primary_model)
        ]
        last_error: Optional[Exception] = None

        for attempt, try_model in enumerate(models_to_try):
            try:
                start = time.time()
                kwargs: dict[str, Any] = {
                    "model": try_model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                if self.api_key:
                    kwargs["api_key"] = self.api_key

                response = litellm.completion(**kwargs)
                latency_ms = (time.time() - start) * 1000

                content = response.choices[0].message.content or ""
                usage = response.usage or type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0})()
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

                cost_usd = self._estimate_cost(try_model, prompt_tokens, completion_tokens)
                self._total_cost_usd += cost_usd
                self._call_count += 1

                return {
                    "content": content,
                    "model": try_model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                    "raw_response": response,
                }
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM call failed (attempt %d, model %s): %s",
                    attempt + 1,
                    try_model,
                    str(e),
                )
                if attempt < self.max_retries - 1:
                    import random

                    backoff = (2**attempt) + random.uniform(0, 1)
                    time.sleep(backoff)

        raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}") from last_error

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD based on model and token counts."""
        try:
            import litellm

            cost = litellm.completion_cost(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return cost or 0.0
        except Exception:
            return 0.0

    def get_total_cost(self) -> float:
        """Get total cost across all calls in this session."""
        return self._total_cost_usd

    def get_call_count(self) -> int:
        """Get number of LLM calls made."""
        return self._call_count

    def reset_stats(self) -> None:
        """Reset cost and call count."""
        self._total_cost_usd = 0.0
        self._call_count = 0
