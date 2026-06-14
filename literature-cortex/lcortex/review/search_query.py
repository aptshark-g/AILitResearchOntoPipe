"""Phase C/D: Generate limitation and extension search queries.

Given a paper's Phase B analysis (keywords, self_limitations,
extension_directions), produces targeted arXiv search queries for
finding critique papers (Phase C) and advancement papers (Phase D).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..intelligence.base import LLMAdapter, extract_json
from ..prompts.loader import render_prompt

log = logging.getLogger("lcortex.review.search_query")


def _build_variables(paper: dict[str, Any]) -> dict[str, Any]:
    """Build template variables from a paper dict (usually from Phase B output)."""
    keywords = paper.get("keywords", [])
    self_limitations = paper.get("self_limitations", [])
    extension_directions = paper.get("extension_directions", [])

    return {
        "title": paper.get("title", paper.get("paper_id", "")),
        "keywords": json.dumps(keywords, ensure_ascii=False),
        "self_limitations": json.dumps(self_limitations, ensure_ascii=False),
        "extension_directions": json.dumps(extension_directions, ensure_ascii=False),
        "method_description": ", ".join(keywords) if keywords else "unknown method",
    }


def _extract_from_llm(raw: str) -> dict | None:
    """Best-effort JSON extraction."""
    parsed = extract_json(raw)
    if parsed is not None:
        return parsed

    # Extra heuristic
    cleaned = raw.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def generate_queries(
    paper: dict[str, Any],
    adapter: LLMAdapter,
    max_retries: int = 3,
    retry_delay_s: float = 2.0,
) -> dict[str, Any]:
    """Generate limitation (Phase C) and extension (Phase D) search queries for a paper.

    Args:
        paper: Paper dict from Phase B analysis.  Must include ``keywords``,
               ``self_limitations``, ``extension_directions``, and optionally
               ``title`` and ``paper_id``.
        adapter: LLM adapter instance.
        max_retries: Max retries on JSON parse failure.
        retry_delay_s: Delay between retries.

    Returns:
        Dict with keys ``phase_c_queries`` and ``phase_d_queries``, each a list
        of ``{"query": ..., "rationale": ..., "expected_focus": ...}`` objects.
        On failure returns ``{"error": ..., "paper_id": ...}``.
    """
    system_prompt, user_message = render_prompt(
        "stage_c_d_search", _build_variables(paper)
    )

    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = adapter.complete(system_prompt, user_message)

            if isinstance(result, dict) and "error" in result:
                log.warning(
                    "LLM error generating queries (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    result.get("error", ""),
                )
                if attempt < max_retries:
                    time.sleep(retry_delay_s)
                continue

            if isinstance(result, dict):
                # Already parsed by adapter
                return result

            raw_text = str(result) if result else ""
            parsed = _extract_from_llm(raw_text)

            if parsed is not None:
                # Ensure both keys exist
                parsed.setdefault("phase_c_queries", [])
                parsed.setdefault("phase_d_queries", [])
                return parsed

            log.warning(
                "JSON parse failed on query generation (attempt %d/%d). Raw (first 300): %s",
                attempt,
                max_retries,
                raw_text[:300],
            )
            last_raw = raw_text

            if attempt < max_retries:
                user_message = (
                    "Your previous response was not valid JSON. "
                    "Respond with ONLY a JSON object. No markdown. No code fences."
                )
                time.sleep(retry_delay_s)

        except Exception as exc:
            log.exception(
                "Exception generating queries (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            last_raw = str(exc)
            if attempt < max_retries:
                time.sleep(retry_delay_s)

    return {
        "error": f"Failed to generate queries after {max_retries} attempts",
        "paper_id": paper.get("paper_id", paper.get("id", "")),
        "raw": last_raw[:2000],
    }
