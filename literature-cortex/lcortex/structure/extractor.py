"""
Phase F: Structure template extractor.

Extracts knowledge_level and structure_template from a paper using the
Stage F prompt template.  Uses an LLMAdapter for intelligence and
returns a fully-parsed dict matching the prompt's output schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lcortex.intelligence.base import LLMAdapter

log = logging.getLogger("lcortex.structure.extractor")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase F: Structure template extraction
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA_F = {
    "knowledge_level": ["string"],
    "knowledge_level_confidence": 0.0,
    "structure_template": {
        "signal_chain": ["string"],
        "control_architecture": "string",
        "optimization_target": "string",
        "constraint_type": ["string"],
        "abstract_pattern": "string",
        "mathematical_core": "string",
        "domain_abstraction": "string",
    },
}


def extract_structure(paper: dict, adapter: LLMAdapter) -> dict:
    """Extract knowledge level and structure template from a paper.

    Phase F of the Literature Cortex pipeline: takes a paper dict from
    Phase B (with scores) and returns a structure + knowledge level
    classification.

    Args:
        paper: Paper dict with keys:
            title, abstract, keywords, and optionally scores (c1-c4, l).
        adapter: Configured LLM adapter for inference.

    Returns:
        Parsed dict matching the stage_f_structure output schema:

        .. code-block:: json

            {
                "knowledge_level": ["L3-Algorithm"],
                "knowledge_level_confidence": 0.85,
                "structure_template": {
                    "signal_chain": ["error_measurement", ...],
                    "control_architecture": "feedforward_adaptive",
                    "optimization_target": "minimize residual signal...",
                    "constraint_type": ["convergence_rate", ...],
                    "abstract_pattern": "perceive → adapt → constrain → actuate",
                    "mathematical_core": "gradient_descent_on_manifold",
                    "domain_abstraction": "adaptive filtering with..."
                }
            }

        On error, returns ``{"error": "...", "raw": "..."}``.
    """
    # Load and render prompt via unified loader
    from lcortex.prompts.loader import render_prompt
    title = paper.get("title", "")
    abstract = paper.get("abstract", paper.get("abstract_text", ""))
    keywords = paper.get("keywords", paper.get("keyword_list", []))
    if isinstance(keywords, list):
        keywords = ", ".join(str(k) for k in keywords)
    scores = paper.get("scores", {})

    system_prompt, user_message = render_prompt("stage_f_structure", {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "c1": scores.get("contribution", scores.get("c1", "N/A")),
        "c2": scores.get("correctness", scores.get("c2", "N/A")),
        "c3": scores.get("clarity", scores.get("c3", "N/A")),
        "c4": scores.get("connectedness", scores.get("c4", "N/A")),
        "l": scores.get("likelihood", scores.get("l", "N/A")),
    })

    log.info(
        "Phase F: Extracting structure template for paper '%s'",
        title[:80],
    )

    # Call LLM
    result = adapter.complete(system_prompt, user_message, _OUTPUT_SCHEMA_F)

    # Validate structure_template sub-object
    if "structure_template" in result and isinstance(result["structure_template"], dict):
        _validate_structure_template(result["structure_template"])

    # Validate knowledge_level
    if "knowledge_level" in result:
        kl = result["knowledge_level"]
        if isinstance(kl, str):
            result["knowledge_level"] = [kl]
        elif not isinstance(kl, list):
            result["knowledge_level"] = [str(kl)]

    return result


def _validate_structure_template(st: dict) -> None:
    """Coerce structure_template fields to expected types."""
    # Ensure signal_chain is a list
    if "signal_chain" in st and isinstance(st["signal_chain"], str):
        # Try to split on → or ,
        if "→" in st["signal_chain"]:
            st["signal_chain"] = [s.strip() for s in st["signal_chain"].split("→")]
        else:
            st["signal_chain"] = [s.strip() for s in st["signal_chain"].split(",")]

    # Ensure constraint_type is a list
    if "constraint_type" in st and isinstance(st["constraint_type"], str):
        st["constraint_type"] = [c.strip() for c in st["constraint_type"].split(",")]

    # Ensure abstract_pattern is a string (it already should be)
    if "abstract_pattern" in st and isinstance(st["abstract_pattern"], list):
        st["abstract_pattern"] = " → ".join(st["abstract_pattern"])


# ---------------------------------------------------------------------------
# Convenience: compute similarity between structure templates
# ---------------------------------------------------------------------------

def structure_similarity(
    template_a: dict | None,
    template_b: dict | None,
) -> float:
    """Compute a rough similarity score between two structure templates.

    Compares: control_architecture match, abstract_pattern overlap,
    mathematical_core overlap.  Returns 0.0–1.0.

    Args:
        template_a: First structure template dict.
        template_b: Second structure template dict.

    Returns:
        Similarity score.
    """
    if not template_a or not template_b:
        return 0.0

    score = 0.0
    weight_total = 0.0

    # 1) Control architecture exact match (weight: 0.35)
    arch_a = template_a.get("control_architecture", "")
    arch_b = template_b.get("control_architecture", "")
    if arch_a and arch_b:
        weight_total += 0.35
        if arch_a == arch_b:
            score += 0.35

    # 2) Abstract pattern word overlap (weight: 0.25)
    pat_a = _tokenize(template_a.get("abstract_pattern", ""))
    pat_b = _tokenize(template_b.get("abstract_pattern", ""))
    if pat_a and pat_b:
        weight_total += 0.25
        score += 0.25 * _jaccard(pat_a, pat_b)

    # 3) Mathematical core overlap (weight: 0.20)
    math_a = _tokenize(template_a.get("mathematical_core", ""))
    math_b = _tokenize(template_b.get("mathematical_core", ""))
    if math_a and math_b:
        weight_total += 0.20
        score += 0.20 * _jaccard(math_a, math_b)

    # 4) Signal chain overlap (weight: 0.20)
    chain_a = set(template_a.get("signal_chain", []))
    chain_b = set(template_b.get("signal_chain", []))
    if chain_a and chain_b:
        weight_total += 0.20
        score += 0.20 * _jaccard(chain_a, chain_b)

    if weight_total == 0:
        return 0.0
    return min(1.0, score / weight_total)


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on whitespace/punctuation, return set of tokens."""
    import re
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
