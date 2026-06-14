"""
Adapter factory — resolves the configured LLM provider to an adapter instance.

Usage::

    from lcortex.intelligence import get_adapter

    adapter = get_adapter(config)
    result = adapter.complete(system_prompt, user_message, output_schema)
"""

from __future__ import annotations

import logging
from typing import Any

from .base import LLMAdapter

log = logging.getLogger("lcortex.intelligence.factory")

# Provider name → fully-qualified class path (lazy import)
_PROVIDER_CLASSES: dict[str, str] = {
    "deepseek": "lcortex.intelligence.adapters.deepseek.DeepSeekAdapter",
    "openai": "lcortex.intelligence.adapters.openai.OpenAIAdapter",
    "claude": "lcortex.intelligence.adapters.claude.ClaudeAdapter",
    "ollama": "lcortex.intelligence.adapters.ollama.OllamaAdapter",
    "none": "lcortex.intelligence.adapters.noop.NoOpAdapter",
    "noop": "lcortex.intelligence.adapters.noop.NoOpAdapter",
}


def get_adapter(config: Any = None) -> LLMAdapter:
    """Return an :class:`LLMAdapter` instance based on configuration.

    Resolution order (first match wins):
      1. ``config.llm.provider`` — settings object / dict / env
      2. Environment variable ``LCORTEX_LLM_PROVIDER``
      3. Auto-detect from available API keys (DEEPSEEK_API_KEY → deepseek, etc.)
      4. Fallback to :class:`NoOpAdapter`

    Parameters
    ----------
    config:
        An object or dict with ``llm.provider`` and optionally
        ``llm.api_key``, ``llm.model``, etc.

    Returns
    -------
    LLMAdapter
    """
    import os

    provider = _resolve_provider(config)

    # Auto-detect provider from environment API keys
    if provider is None:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key and _is_valid_deepseek_key(deepseek_key):
            provider = "deepseek"
        elif deepseek_key:
            log.warning(
                "DEEPSEEK_API_KEY is set but appears invalid (len=%d, starts_with_sk=%s) — "
                "skipping deepseek auto-detect",
                len(deepseek_key),
                deepseek_key.startswith("sk-"),
            )
        if provider is None and os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        if provider is None and os.environ.get("ANTHROPIC_API_KEY"):
            provider = "claude"

    if provider is None:
        log.info("No LLM provider configured — using NoOpAdapter")
        return _make_adapter("noop", config)

    class_path = _PROVIDER_CLASSES.get(provider.lower())
    if class_path is None:
        log.warning(
            "Unknown LLM provider '%s' — falling back to NoOpAdapter. "
            "Known providers: %s",
            provider,
            ", ".join(_PROVIDER_CLASSES.keys()),
        )
        return _make_adapter("noop", config)

    # Try the primary adapter
    adapter = _make_adapter(class_path, config)
    if adapter.is_available():
        log.info("LLM provider: %s", provider)
        return adapter

    # Primary unavailable — attempt fallback
    fallback = _resolve_fallback(config)
    if fallback:
        log.warning(
            "LLM provider '%s' is unavailable — falling back to '%s'",
            provider,
            fallback,
        )
        fb_class = _PROVIDER_CLASSES.get(fallback.lower())
        if fb_class:
            fb_adapter = _make_adapter(fb_class, config)
            if fb_adapter.is_available():
                return fb_adapter

    # Everything failed — fall back to NoOp
    log.warning(
        "LLM provider '%s' (and fallback '%s') unavailable — using NoOpAdapter",
        provider,
        fallback,
    )
    return _make_adapter("noop", config)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_valid_deepseek_key(key: str) -> bool:
    """Validate that a DeepSeek API key looks legitimate.

    DeepSeek keys start with 'sk-' and are >30 chars.
    Rejects garbage values that would produce API errors.
    """
    if not key or not isinstance(key, str):
        return False
    return key.startswith("sk-") and len(key) >= 30


def _resolve_provider(config: Any) -> str | None:
    """Extract provider name from config or env."""
    import os

    # config.llm.provider
    try:
        if isinstance(config, dict):
            p = config.get("llm", {}).get("provider")
        else:
            p = getattr(config, "llm", None)
            if p is not None:
                p = getattr(p, "provider", None)
        if p and str(p).lower() not in ("", "none"):
            return str(p)
    except Exception:
        pass

    # env override
    env = os.environ.get("LCORTEX_LLM_PROVIDER", "").strip()
    if env and env.lower() != "none":
        return env

    return None


def _resolve_fallback(config: Any) -> str | None:
    """Extract fallback provider name from config."""
    try:
        if isinstance(config, dict):
            return config.get("llm", {}).get("fallback")
        else:
            llm = getattr(config, "llm", None)
            if llm is not None:
                return getattr(llm, "fallback", None)
    except Exception:
        pass
    return None


def _make_adapter(class_path_or_key: str, config: Any) -> LLMAdapter:
    """Instantiate an adapter from a dotted class path *or* known key."""
    import importlib

    # Resolve shorthand key to full path
    class_path = _PROVIDER_CLASSES.get(class_path_or_key, class_path_or_key)

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)

    try:
        return adapter_cls(config)
    except Exception as exc:
        log.exception("Failed to instantiate %s: %s", class_name, exc)
        # Return NoOp on construction failure
        from .adapters.noop import NoOpAdapter
        return NoOpAdapter(config)
