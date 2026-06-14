"""Phase B: 4C+L scoring for candidate papers.

Uses the ``stage_b_scoring`` prompt template to evaluate each paper's
contribution, correctness, clarity, connectedness, and likelihood/relevance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from ..intelligence.base import LLMAdapter, extract_json
from ..prompts.loader import render_prompt

log = logging.getLogger("lcortex.analysis.scorer")


def _extract_json_from_llm(raw: str) -> dict | None:
    """Best-effort JSON extraction from LLM response.

    Uses the shared ``extract_json`` from intelligence.base, then adds
    an extra heuristic to strip markdown code-fence remnants that the
    shared extractor might miss on the second pass.
    """
    parsed = extract_json(raw)
    if parsed is not None:
        return parsed

    # Extra heuristic: strip ```json or ``` that appear as standalone lines
    # but weren't caught by the regex (e.g. bracketed by whitespace edge cases)
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


def _build_variables(paper: dict[str, Any]) -> dict[str, Any]:
    """Build the template variable dict for stage_b_scoring."""
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "source_keywords": ", ".join(paper.get("keywords", paper.get("source_keywords", []))),
        "year": str(paper.get("year", "")),
        "source": paper.get("source", "arXiv"),
        "citation_count": str(paper.get("citation_count", "unknown")),
        "paper_id": paper.get("id", paper.get("paper_id", "")),
    }


def score_paper(
    paper: dict[str, Any],
    adapter: LLMAdapter,
    max_retries: int = 3,
    retry_delay_s: float = 2.0,
) -> dict[str, Any]:
    """Score a single paper using the 4C+L framework.

    Args:
        paper: Paper dict with keys ``title``, ``abstract``, ``keywords``,
               ``source``, ``year``, ``citation_count``, and ``id``.
        adapter: LLM adapter instance.
        max_retries: Maximum number of retries on JSON parse failure.
        retry_delay_s: Delay in seconds between retries.

    Returns:
        Parsed scoring dict with keys: ``scores``, ``mean_score``, ``passed``,
        ``triggers``, ``keywords``, ``self_limitations``,
        ``extension_directions``, ``rationale``.  On total failure returns
        ``{"error": ..., "paper_id": ...}``.
    """
    system_prompt, user_message = render_prompt("stage_b_scoring", _build_variables(paper))

    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = adapter.complete(system_prompt, user_message)

            if isinstance(result, dict) and "error" in result:
                # Adapter itself returned an error
                log.warning(
                    "LLM error on paper %s (attempt %d/%d): %s",
                    paper.get("id", "?"),
                    attempt,
                    max_retries,
                    result.get("error", ""),
                )
                if attempt < max_retries:
                    time.sleep(retry_delay_s)
                continue

            # result might be the parsed dict (from adapter) or raw string
            if isinstance(result, dict) and "_raw" not in result:
                # Already parsed JSON from adapter — normalize and return
                result = _normalize_scoring(result)
                log.info(
                    "Scorer (adapter-parsed): keys=%s, has_scores=%s, mean_score=%s",
                    list(result.keys())[:8],
                    bool(result.get("scores")),
                    result.get("mean_score"),
                )
                return result

            if isinstance(result, dict) and "_raw" in result:
                raw_text = result["_raw"]
                tokens_in = result.get("tokens_in", 0)
                tokens_out = result.get("tokens_out", 0)
            else:
                raw_text = str(result) if result else ""
                tokens_in = 0
                tokens_out = 0

            parsed = _extract_json_from_llm(raw_text)

            if parsed is not None:
                # Normalize scores (handle abbreviated keys, compute mean_score)
                parsed = _normalize_scoring(parsed)
                # Ensure paper_id and token counts are set
                parsed.setdefault("paper_id", paper.get("id", paper.get("paper_id", "")))
                parsed["tokens_in"] = tokens_in
                parsed["tokens_out"] = tokens_out
                # Validate required fields
                if "scores" in parsed and parsed["scores"]:
                    log.info(
                        "Scorer returned: keys=%s, has_scores=%s, mean_score=%s, scores=%s",
                        list(parsed.keys())[:8],
                        bool(parsed.get("scores")),
                        parsed.get("mean_score"),
                        parsed.get("scores"),
                    )
                    return parsed

                log.warning(
                    "Parsed JSON missing 'scores' on paper %s (attempt %d/%d)",
                    paper.get("id", "?"),
                    attempt,
                    max_retries,
                )

            log.warning(
                "JSON parse failed on paper %s (attempt %d/%d). Raw (first 300): %s",
                paper.get("id", "?"),
                attempt,
                max_retries,
                raw_text[:300],
            )
            last_raw = raw_text

            if attempt < max_retries:
                # Strengthen the user message to insist on pure JSON
                user_message = (
                    "Your previous response was not valid JSON. "
                    "Please respond with ONLY a JSON object (no markdown, "
                    "no explanation, no code fences).\n\n"
                    "Repeat your answer as pure JSON:"
                )
                time.sleep(retry_delay_s)

        except Exception as exc:
            log.exception(
                "Exception scoring paper %s (attempt %d/%d): %s",
                paper.get("id", "?"),
                attempt,
                max_retries,
                exc,
            )
            last_raw = str(exc)
            if attempt < max_retries:
                time.sleep(retry_delay_s)

    return {
        "error": f"Failed to score after {max_retries} attempts",
        "paper_id": paper.get("id", paper.get("paper_id", "")),
        "raw": last_raw[:2000],
        "scores": {},
        "mean_score": 0,
    }


# ─── Score key normalization ─────────────────────────────────────────────

# Mapping from abbreviated C1/C2/C3/C4/L keys to full 4C+L names
_SCORE_KEY_MAP: dict[str, str] = {
    "C1": "contribution",
    "C2": "correctness",
    "C3": "clarity",
    "C4": "connectedness",
    "L": "likelihood",
    # Also handle lowercase variants
    "c1": "contribution",
    "c2": "correctness",
    "c3": "clarity",
    "c4": "connectedness",
    "l": "likelihood",
    # Already full names (pass through)
    "contribution": "contribution",
    "correctness": "correctness",
    "clarity": "clarity",
    "connectedness": "connectedness",
    "likelihood": "likelihood",
}


def _normalize_scoring(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a scoring result dict.

    - Handles flat C1/C2/C3/C4/L top-level keys (no 'scores' wrapper)
    - Maps abbreviated score keys (C1/C2/C3/C4/L) to full names.
    - Computes ``mean_score`` from scores if missing.
    - Sets ``passed`` if missing.
    - Returns the same dict object (mutated in place).
    """
    # Handle flat abbreviated keys: {"C1":4, "C2":3, ...} → wrap into scores (包括全称)
    flat_keys = {"C1", "C2", "C3", "C4", "L"}
    full_flat_keys = {"contribution", "correctness", "clarity", "connectedness", "likelihood"}
    if flat_keys & set(result.keys()) or full_flat_keys & set(result.keys()):
        flat_scores = {}
        for k in (flat_keys | full_flat_keys):
            if k in result:
                v = result.pop(k)
                if isinstance(v, str):
                    try: v = float(v)
                    except (ValueError, TypeError): v = 0
                if isinstance(v, (int, float)) and v > 0:
                    flat_scores[k] = v
        if flat_scores:
            result["scores"] = flat_scores
    
    if "scores" in result and isinstance(result["scores"], dict):
        raw_scores = result["scores"]
        normalized: dict[str, int | float] = {}
        for key, value in raw_scores.items():
            mapped = _SCORE_KEY_MAP.get(key, key)
            # Ensure numeric value (LLM may return string)
            if isinstance(value, str):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0
            normalized[mapped] = value
        result["scores"] = normalized

    # Auto-compute mean_score if missing but scores exist
    if "scores" in result and result["scores"]:
        score_vals = [v for v in result["scores"].values() if isinstance(v, (int, float)) and v > 0]
        if score_vals:
            computed_mean = sum(score_vals) / len(score_vals)
            if "mean_score" not in result or result.get("mean_score", 0) == 0:
                # Use computed mean if original is missing or zero
                if "mean_score" not in result:
                    result["mean_score"] = round(computed_mean, 2)
                    log.debug(
                        "Computed mean_score=%.2f from scores (was missing)",
                        computed_mean,
                    )
                elif result["mean_score"] == 0 and computed_mean > 0:
                    result["mean_score"] = round(computed_mean, 2)
                    log.debug(
                        "Replaced zero mean_score with computed=%.2f",
                        computed_mean,
                    )

    # Set passed if missing
    if "passed" not in result:
        mean = result.get("mean_score", 0)
        result["passed"] = mean >= 3.0

    return result


def score_batch(
    papers: list[dict[str, Any]],
    adapter: LLMAdapter,
    output_dir: str | None = None,
    output_filename: str = "analysis.jsonl",
    max_retries: int = 3,
    retry_delay_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Score multiple papers, writing results incrementally to a JSONL file.

    Each scored paper is appended to the JSONL file immediately after scoring
    so that partial progress is preserved in case of a crash.  On the next run
    the caller can check which paper IDs already exist in the JSONL and skip
    them.

    Args:
        papers: List of paper dicts.
        adapter: LLM adapter instance.
        output_dir: Directory for the output JSONL file.  Defaults to the
                    current working directory.
        output_filename: Name of the JSONL file (default ``"analysis.jsonl"``).
        max_retries: Retries per paper on JSON parse failure.
        retry_delay_s: Delay between retries.

    Returns:
        List of scoring result dicts in the same order as ``papers``.
    """
    results: list[dict[str, Any]] = []

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, output_filename)

    for i, paper in enumerate(papers):
        paper_id = paper.get("id", paper.get("paper_id", f"paper-{i}"))
        log.info(
            "Scoring paper %d/%d: %s",
            i + 1,
            len(papers),
            paper.get("title", paper_id)[:80],
        )

        result = score_paper(
            paper,
            adapter,
            max_retries=max_retries,
            retry_delay_s=retry_delay_s,
        )
        results.append(result)

        # Write incrementally
        try:
            with open(jsonl_path, "a") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.error("Failed to write JSONL line for paper %s: %s", paper_id, exc)

    log.info("Batch scoring complete. %d papers → %s", len(papers), jsonl_path)
    return results
