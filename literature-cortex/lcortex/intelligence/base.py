"""
LLMAdapter — abstract base class for pluggable LLM backends.

All intelligence-layer operations (Phase B scoring, Phase E synthesis,
Phase F structure extraction, Phase F-2 conflict detection) go through
a single abstract interface so the core engine never depends on a
specific provider.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger("lcortex.intelligence")

# ---------------------------------------------------------------------------
# JSON extraction helpers (shared across adapters)
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_JSON_BRACE_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(raw: str) -> dict | None:
    """Best-effort JSON extraction from an LLM text response.

    Tries, in order:
      1. `` ```json ... ``` `` or `` ``` ... ``` `` fences
      2. First outermost ``{...}`` brace pair
      3. Raw string parse

    Returns the parsed dict, or *None* if every attempt fails.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # 1) Markdown code fences
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass  # fall through to other strategies

    # 2) Outermost brace pair
    m = _JSON_BRACE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            # Try to fix common issues: trailing commas, single quotes
            pass

    # 3) Raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def _build_json_prompt_suffix(schema: dict | None) -> str:
    """Return a suffix instruction asking the model to output JSON."""
    suffix = "\n\nReturn ONLY a valid JSON object. No markdown, no explanation."
    if schema:
        suffix += f"\nExpected shape:\n```json\n{json.dumps(schema, indent=2)}\n```"
    return suffix


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMAdapter(ABC):
    """Pluggable LLM backend.

    Each adapter wraps one provider (DeepSeek, OpenAI, Claude, local Ollama,
    or a no-op fallback) and exposes a single :meth:`complete` method that
    always returns a ``dict``.
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._max_retries = 3
        self._retry_delay_s = 2.0

    # -- mandatory overrides -------------------------------------------------

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict | None = None,
    ) -> dict:
        """Send prompts to the LLM and return a structured JSON dict.

        Parameters
        ----------
        system_prompt:
            System-level instruction describing the task and tone.
        user_message:
            The concrete question / content to evaluate.
        output_schema:
            Optional expected JSON shape (used only as a prompt hint).

        Returns
        -------
        dict
            Parsed JSON response.  Implementations should never raise;
            errors are surfaced as ``{"error": …, "raw": …}``.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this adapter's backend is reachable and configured."""
        ...

    # -- shared retry wrapper ------------------------------------------------

    def _retry_complete(
        self,
        messages: list[dict[str, str]],
        callable_fn,
    ) -> dict:
        """Call *callable_fn(messages)* with JSON extraction and retries.

        On JSON parse failure the prompt is simplified (system prompt only)
        and retried up to ``_max_retries`` times.
        """
        last_raw = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                result = callable_fn(messages)
                if isinstance(result, dict):
                    # Already a dict (e.g. error dict returned by adapter)
                    if "error" in result:
                        return result
                    return result

                raw_text = str(result) if result else ""
                parsed = extract_json(raw_text)

                if parsed is not None:
                    return parsed

                # JSON parse failed — retry with a simpler prompt
                log.warning(
                    "JSON parse failed on attempt %d/%d. Raw (first 300): %s",
                    attempt,
                    self._max_retries,
                    raw_text[:300],
                )
                last_raw = raw_text

                if attempt < self._max_retries:
                    # Simplify: re-send with a stronger JSON-only instruction
                    fallback_msg = (
                        "Your previous response was not valid JSON. "
                        "Please respond with ONLY a JSON object (no markdown, "
                        "no explanation, no code fences).\n\n"
                        "Repeat your answer as pure JSON:"
                    )
                    messages = [
                        {"role": "system", "content": messages[0]["content"]},
                        {"role": "user", "content": fallback_msg},
                    ]
                    time.sleep(self._retry_delay_s)

            except Exception as exc:
                log.exception("Adapter call failed on attempt %d: %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay_s)
                last_raw = str(exc)

        return {
            "error": f"Failed to parse JSON after {self._max_retries} attempts",
            "raw": last_raw[:2000],
        }

    # -- helpers for subclasses ----------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict | None,
    ) -> list[dict[str, str]]:
        """Build the standard message list with JSON-output instruction."""
        suffix = _build_json_prompt_suffix(output_schema)
        return [
            {"role": "system", "content": system_prompt + suffix},
            {"role": "user", "content": user_message},
        ]
