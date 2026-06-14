"""
NoOp adapter — no-AI fallback.

Used when no LLM provider is configured, or all configured
providers are unavailable.  Always returns a "skipped" state
so the core engine can proceed with non-LLM phases gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import LLMAdapter

log = logging.getLogger("lcortex.intelligence.noop")


class NoOpAdapter(LLMAdapter):
    """No-operation adapter — skips all LLM calls.

    ``is_available()`` always returns ``False``.  ``complete()`` returns
    a fixed dict that signals "no LLM — skipped" so the workflow engine
    can handle the absence of AI responses without crashing.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

    def is_available(self) -> bool:
        return False

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict | None = None,
    ) -> dict:
        return {
            "mode": "no_llm",
            "skipped": True,
        }
