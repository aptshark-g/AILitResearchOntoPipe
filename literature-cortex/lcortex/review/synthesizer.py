"""Phase E: Synthesize a structured literature review.

Combines core papers (Phase A-B), limitation papers (Phase C), and extension
papers (Phase D) into a comprehensive markdown review document.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..intelligence.base import LLMAdapter
from ..prompts.loader import render_prompt

log = logging.getLogger("lcortex.review.synthesizer")


def _build_variables(
    topic: str,
    core_papers: list[dict[str, Any]],
    lim_papers: list[dict[str, Any]],
    ext_papers: list[dict[str, Any]],
    version: str = "v3.5",
) -> dict[str, Any]:
    """Build template variables for the stage_e_synthesis prompt."""
    return {
        "topic": topic,
        "version": version,
        "core_papers_json": json.dumps(core_papers, ensure_ascii=False, indent=2),
        "limitation_papers_json": json.dumps(lim_papers, ensure_ascii=False, indent=2),
        "extension_papers_json": json.dumps(ext_papers, ensure_ascii=False, indent=2),
    }


def synthesize_review(
    topic: str,
    core_papers: list[dict[str, Any]],
    lim_papers: list[dict[str, Any]],
    ext_papers: list[dict[str, Any]],
    adapter: LLMAdapter,
    version: str = "v3.5",
    max_retries: int = 3,
    retry_delay_s: float = 2.0,
) -> str:
    """Synthesize a complete markdown literature review.

    Args:
        topic: Research topic string (e.g. "Active Vibration Control with FxLMS").
        core_papers: List of paper dicts that passed the Phase B quality gate.
        lim_papers: List of limitation/critique paper dicts from Phase C search.
        ext_papers: List of extension/advancement paper dicts from Phase D search.
        adapter: LLM adapter instance.
        version: System version string, included in the review metadata.
        max_retries: Max retries on failure.
        retry_delay_s: Delay between retries.

    Returns:
        Complete markdown review document as a string.

    Notes:
        Because Phase E output is markdown (not JSON), the retry logic is
        simpler — we retry on LLM errors but not on format validation.
    """
    system_prompt, user_message = render_prompt(
        "stage_e_synthesis",
        _build_variables(topic, core_papers, lim_papers, ext_papers, version),
    )

    last_result = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = adapter.complete(system_prompt, user_message)

            if isinstance(result, dict):
                # Check for _raw wrapper (LLM returned raw text after JSON parse failure)
                if "_raw" in result:
                    review_text = result["_raw"].strip()
                    if review_text:
                        return review_text
                if "error" in result:
                    log.warning(
                        "LLM error synthesizing review (attempt %d/%d): %s",
                        attempt,
                        max_retries,
                        result.get("error", ""),
                    )
                    last_result = result.get("raw", str(result))
                    if attempt < max_retries:
                        time.sleep(retry_delay_s)
                    continue

                # Unexpected dict without error key — try to stringify
                raw_text = json.dumps(result, ensure_ascii=False)
                if len(raw_text) > 100:
                    return raw_text
                # Fall through to get raw from adapter
                log.warning(
                    "Adapter returned short dict instead of review text (attempt %d/%d)",
                    attempt,
                    max_retries,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay_s)
                continue

            review_text = str(result).strip()
            if review_text:
                return review_text

            log.warning(
                "Empty review text (attempt %d/%d)", attempt, max_retries
            )
            if attempt < max_retries:
                time.sleep(retry_delay_s)

        except Exception as exc:
            log.exception(
                "Exception synthesizing review (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay_s)

    # Return whatever we got (or error)
    if not last_result:
        return (
            "# Review Synthesis Failed\n\n"
            f"The review could not be synthesized after {max_retries} attempts.\n"
            f"Topic: {topic}\n"
            f"Core papers: {len(core_papers)}\n"
            f"Limitation papers: {len(lim_papers)}\n"
            f"Extension papers: {len(ext_papers)}\n"
        )

    return last_result
