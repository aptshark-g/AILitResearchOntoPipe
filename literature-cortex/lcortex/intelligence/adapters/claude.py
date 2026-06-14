"""
Anthropic Claude adapter — Messages API.

Uses Anthropic's native Messages endpoint (not OpenAI-compatible).

Configuration
-------------
Environment variables:
  ``ANTHROPIC_API_KEY``  — required
  ``ANTHROPIC_BASE_URL`` — optional (default: https://api.anthropic.com)

Config object keys (``config.llm.*``):
  ``api_key``, ``base_url``, ``model``, ``max_tokens``, ``temperature``
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from ..base import LLMAdapter, extract_json

log = logging.getLogger("lcortex.intelligence.claude")

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.1
_ANTHROPIC_VERSION = "2023-06-01"


class ClaudeAdapter(LLMAdapter):
    """Anthropic Claude Messages API adapter."""

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = self._config

        self._api_key = (
            _get_cfg(cfg, "llm", "api_key")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._base_url = (
            _get_cfg(cfg, "llm", "base_url")
            or os.environ.get("ANTHROPIC_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._model = (
            _get_cfg(cfg, "llm", "model")
            or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        )
        self._max_tokens = int(
            _get_cfg(cfg, "llm", "max_tokens")
            or os.environ.get("ANTHROPIC_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS))
        )
        self._temperature = float(
            _get_cfg(cfg, "llm", "temperature")
            or os.environ.get("ANTHROPIC_TEMPERATURE", str(_DEFAULT_TEMPERATURE))
        )

    # ------------------------------------------------------------------
    # LLMAdapter interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self._api_key)

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict | None = None,
    ) -> dict:
        if not self.is_available():
            return {
                "error": "Claude adapter not configured (missing ANTHROPIC_API_KEY)",
            }

        messages = self._build_messages(system_prompt, user_message, output_schema)
        return self._retry_complete(messages, self._call_api)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict[str, str]]) -> dict:
        """Convert OpenAI-style message list to Anthropic format and call API."""
        # Anthropic uses separate "system" and "messages" fields
        system_content = ""
        anthropic_msgs: list[dict[str, Any]] = []

        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                anthropic_msgs.append({"role": "user", "content": m["content"]})

        # Ensure alternating user/assistant — if first is assistant, insert
        # a dummy user message so the API doesn't reject it
        if anthropic_msgs and anthropic_msgs[0]["role"] == "assistant":
            anthropic_msgs.insert(0, {"role": "user", "content": "."})

        url = f"{self._base_url}/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "system": system_content,
            "messages": anthropic_msgs,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(10, 120),
            )
        except requests.exceptions.ConnectionError as exc:
            log.error("Claude connection error: %s", exc)
            return {"error": f"Connection error: {exc}"}
        except requests.exceptions.Timeout as exc:
            log.error("Claude timeout: %s", exc)
            return {"error": f"Request timeout: {exc}"}
        except requests.exceptions.RequestException as exc:
            log.error("Claude request error: %s", exc)
            return {"error": f"Request error: {exc}"}

        if not resp.ok:
            log.error("Claude API error %d: %s", resp.status_code, resp.text[:500])
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

        # Anthropic response shape: content[0].text
        try:
            content = body["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            return {
                "error": f"Unexpected response shape: {exc}",
                "raw": json.dumps(body)[:2000],
            }

        parsed = extract_json(content)
        if parsed is not None:
            return parsed
        return content


def _get_cfg(cfg: Any, *path: str) -> str | None:
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
