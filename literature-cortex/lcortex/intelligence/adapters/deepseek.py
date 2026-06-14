"""
DeepSeek adapter — OpenAI-compatible chat completions.

Primary LLM backend for Literature Cortex.  DeepSeek's API is
compatible with the OpenAI ``/v1/chat/completions`` wire format.

Configuration
-------------
Environment variables:
  ``DEEPSEEK_API_KEY``   — required
  ``DEEPSEEK_BASE_URL``  — optional (default: https://api.deepseek.com/v1)

Config object keys (``config.llm.*``):
  ``api_key``, ``base_url``, ``model``, ``max_tokens``, ``temperature``
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from ..base import LLMAdapter, extract_json

log = logging.getLogger("lcortex.intelligence.deepseek")

_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.1


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek chat-completions adapter (OpenAI-compatible protocol)."""

    def __init__(self, config: Any = None):
        super().__init__(config)
        # Resolve settings: config.llm.* → env vars → defaults
        cfg = self._config
        self._api_key = (
            _get_cfg(cfg, "llm", "api_key")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self._base_url = (
            _get_cfg(cfg, "llm", "base_url")
            or os.environ.get("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._model = (
            _get_cfg(cfg, "llm", "model")
            or os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODEL)
        )
        self._max_tokens = int(
            _get_cfg(cfg, "llm", "max_tokens")
            or os.environ.get("DEEPSEEK_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS))
        )
        self._temperature = float(
            _get_cfg(cfg, "llm", "temperature")
            or os.environ.get("DEEPSEEK_TEMPERATURE", str(_DEFAULT_TEMPERATURE))
        )

    # ------------------------------------------------------------------
    # LLMAdapter interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check that the adapter is configured with a valid API key.

        DeepSeek keys must start with 'sk-' and be at least 30 chars.
        Short/garbage values (like truncated env vars) are rejected.
        """
        if not self._api_key:
            return False
        if not self._api_key.startswith("sk-"):
            log.warning(
                "DeepSeek API key does not start with 'sk-' (len=%d, prefix=%s...)",
                len(self._api_key),
                self._api_key[:8],
            )
            return False
        if len(self._api_key) < 30:
            log.warning(
                "DeepSeek API key too short (len=%d, need ≥30)",
                len(self._api_key),
            )
            return False
        return True

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict | None = None,
    ) -> dict:
        if not self.is_available():
            return {
                "error": "DeepSeek adapter not configured (missing DEEPSEEK_API_KEY)",
            }

        messages = self._build_messages(system_prompt, user_message, output_schema)
        return self._retry_complete(messages, self._call_api)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict[str, str]]) -> dict:
        """Single API call.  Returns parsed dict or error dict.

        Does NOT retry — retry logic is in :meth:`_retry_complete`.
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(10, 120),  # (connect, read)
            )
        except requests.exceptions.ConnectionError as exc:
            log.error("DeepSeek connection error: %s", exc)
            return {"error": f"Connection error: {exc}"}
        except requests.exceptions.Timeout as exc:
            log.error("DeepSeek timeout: %s", exc)
            return {"error": f"Request timeout: {exc}"}
        except requests.exceptions.RequestException as exc:
            log.error("DeepSeek request error: %s", exc)
            return {"error": f"Request error: {exc}"}

        if not resp.ok:
            log.error("DeepSeek API error %d: %s", resp.status_code, resp.text[:500])
            return {
                "error": f"API returned {resp.status_code}",
                "detail": resp.text[:1000],
            }

        try:
            body = resp.json()
        except ValueError:
            return {
                "error": "Invalid JSON in API response",
                "raw": resp.text[:2000],
            }

        # Extract the assistant message text
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            usage = body.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
        except (KeyError, IndexError, TypeError) as exc:
            log.error("Unexpected DeepSeek response shape: %s", exc)
            return {
                "error": f"Unexpected response shape: {exc}",
                "raw": json.dumps(body)[:2000],
            }

        # Try to parse as JSON directly
        parsed = extract_json(content)
        if parsed is not None:
            if isinstance(parsed, dict):
                parsed["tokens_in"] = tokens_in
                parsed["tokens_out"] = tokens_out
            return parsed

        # Return raw text for retry handling (also include token info)
        return {"_raw": content, "tokens_in": tokens_in, "tokens_out": tokens_out}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get_cfg(cfg: Any, *path: str) -> str | None:
    """Walk nested config object *path*, returning a str or None."""
    try:
        for key in path:
            if isinstance(cfg, dict):
                cfg = cfg.get(key)
            else:
                cfg = getattr(cfg, key, None)
            if cfg is None:
                return None
        return str(cfg) if cfg is not None else None
    except Exception:
        return None
