"""Intelligence module — LLM adapter abstraction + auto-detection factory.

Adapters: DeepSeek, OpenAI, Ollama, Claude, NoOp (fallback).
Auto-detection from environment variables.
"""

from lcortex.intelligence.factory import get_adapter
from lcortex.intelligence.base import LLMAdapter

__all__ = [
    "get_adapter",
    "LLMAdapter",
]
