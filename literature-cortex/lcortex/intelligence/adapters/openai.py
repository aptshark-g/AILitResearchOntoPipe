"""
OpenAI adapter — standard OpenAI chat completions API.

Configuration
-------------
Environment variables:
  ``OPENAI_API_KEY``   — required
  ``OPENAI_BASE_URL``  — optional (default: https://api.openai.com/v1)

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

log = logging.getLogger("lcortex.intelligence.openai")

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.1


class OpenAIAdapter(LLMAdapter):
    """Standard OpenAI chat completions adapter."""

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = self._config

        self._api_key = (
            _get_cfg(cfg, "llm", "api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._base_url = (
            _get_cfg(cfg, "llm", "base_url")
            or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self._model = (
            _get_cfg(cfg, "llm", "model")
            or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
        )
        self._max_tokens = int(
            _get_cfg(cfg, "llm", "max_tokens")
            or os.environ.get("OPENAI_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS))
        )
        self._temperature = float(
            _get_cfg(cfg, "llm", "temperature")
            or os.environ.get("OPENAI_TEMPERATURE", str(_DEFAULT_TEMPERATURE))
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
                "error": "OpenAI adapter not configured (missing OPENAI_API_KEY)",
            }

        messages = self._build_messages(system_prompt, user_message, output_schema)
        return self._retry_complete(messages, self._call_api)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict[str, str]]) -> dict:
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
                timeout=(10, 120),
            )
        except requests.exceptions.ConnectionError as exc:
            log.error("OpenAI connection error: %s", exc)
            return {"error": f"Connection error: {exc}"}
        except requests.exceptions.Timeout as exc:
            log.error("OpenAI timeout: %s", exc)
            return {"error": f"Request timeout: {exc}"}
        except requests.exceptions.RequestException as exc:
            log.error("OpenAI request error: %s", exc)
            return {"error": f"Request error: {exc}"}

        if not resp.ok:
            log.error("OpenAI API error %d: %s", resp.status_code, resp.text[:500])
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

        try:
            content = body["choices"][0]["message"]["content"]
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
