"""
Phase F-2: Deconstructor — Divergence / Double-Loop conflict detection.

Implements the full 5-step pipeline:
  1. Deconstruction → indivisible atoms
  2. Retrieval → match atoms to existing knowledge graph
  3. Reconstruction → attempt to explain paper with K-only concepts
  4. Conflict Assessment → evaluate Posner conditions C1-C4
  5. Decision → single_loop / double_loop / seed_anchored / degraded_by_meta

Only triggered when: structure similarity < 0.3, knowledge confidence < 0.5,
T1/T4 trigger, or explicit --mode double-loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lcortex.graph.store import GraphStore
from lcortex.intelligence.base import LLMAdapter

log = logging.getLogger("lcortex.structure.deconstructor")

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA_F2 = {
    "deconstruction": {
        "assumptions": ["string"],
        "axioms_or_theorems": ["string"],
        "methodology_atoms": ["string"],
        "emergent_properties": ["string"],
    },
    "retrieval": [
        {
            "atom": "string",
            "best_match_in_K": "string",
            "match_score": 0.0,
            "reason": "string",
        }
    ],
    "reconstruction": {
        "attempt": "string",
        "reconstructable_pct": 0.0,
        "unexplainable_pct": 0.0,
        "unexplainable_core": [
            {"concept": "string", "why_unexplainable": "string"}
        ],
    },
    "conflict_assessment": {
        "unexplainability_score": 0.0,
        "threshold_used": 0.5,
        "triggers_double_loop": False,
        "downgraded_by_meta": False,
        "downgrade_reason": "string",
        "conceptual_change_conditions": {
            "dissatisfaction": {"value": False, "evidence": "string"},
            "intelligibility": {"value": False, "evidence": "string"},
            "plausibility": {"value": False, "evidence": "string"},
            "fruitfulness": {"value": False, "evidence": "string"},
        },
        "conditions_true_count": 0,
        "passed_two_condition_guard": False,
        "recommended_action": "single_loop",
    },
    "changes": [{"type": "string", "detail": "string"}],
    "single_loop_output": {
        "link_to": ["string"],
        "link_type": "string",
        "knowledge_level": "string",
    },
}




def _load_meta_policy(meta_path: str | None = None) -> dict:
    """Load the meta control policy from JSON.

    Args:
        meta_path: Optional path to meta policy. Uses default seeds/meta/control_policy.json.

    Returns:
        Parsed policy dict.
    """
    path = Path(meta_path) if meta_path else _DEFAULT_META_PATH
    if not path.exists():
        log.warning("Meta policy not found at %s, using hardcoded defaults", path)
        return _DEFAULT_META_FALLBACK
    return json.loads(path.read_text(encoding="utf-8"))


_DEFAULT_META_FALLBACK = {
    "unexplainability": {
        "threshold_default": 0.5,
        "threshold_cold_start": 0.6,
        "cold_start_node_count": 10,
        "l1_l4_weight": 0.3,
        "l5_l6_weight": 0.7,
    },
    "double_loop_guard": {
        "max_impact_ratio": 0.3,
        "consecutive_trigger_limit": 2,
        "require_two_conditions": True,
        "impact_estimation_window": 5,
    },
    "retrieval": {
        "max_retrieval_candidates": 20,
    },
}


# ---------------------------------------------------------------------------
# Knowledge graph helpers
# ---------------------------------------------------------------------------


def _get_seed_nodes(graph_store: GraphStore, levels: list[str] | None = None) -> list[dict]:
    """Retrieve seed nodes from the graph store, filtering by level prefix.

    Args:
        graph_store: GraphStore instance.
        levels: List of level prefixes to include (e.g., ["L1", "L2"]).
                If None, returns L1-L4 seeds.

    Returns:
        List of node dicts with id, title, type, knowledge_level.
    """
    if levels is None:
        levels = ["L1", "L2", "L3", "L4"]

    all_nodes = graph_store.get_all_nodes()
    seeds = []
    for node in all_nodes:
        if node.get("type") != "seed":
            continue
        kl = node.get("knowledge_level", [])
        if isinstance(kl, str):
            try:
                kl = json.loads(kl)
            except (json.JSONDecodeError, TypeError):
                kl = [kl]
        # Check if any knowledge level matches the requested levels
        for lvl in kl:
            for prefix in levels:
                if str(lvl).startswith(prefix):
                    seeds.append(node)
                    break
            else:
                continue
            break
    return seeds


def _get_relevant_existing_papers(
    graph_store: GraphStore,
    paper: dict,
    structure_template: dict | None = None,
    max_candidates: int = 15,
) -> list[dict]:
    """Retrieve existing paper nodes most relevant to the new paper.

    Uses a simple keyword + knowledge-level matching strategy. In production
    this would use semantic embeddings, but we provide a deterministic
    fallback based on title/keyword overlap and knowledge-level proximity.

    Args:
        graph_store: GraphStore instance.
        paper: New paper dict with title, abstract, keywords.
        structure_template: Optional structure template for similarity scoring.
        max_candidates: Maximum number of existing papers to return.

    Returns:
        List of existing paper node dicts, sorted by relevance.
    """
    all_nodes = graph_store.get_all_nodes()

    # Filter to paper-type nodes only
    existing_papers = [n for n in all_nodes if n.get("type") == "paper"]
    if not existing_papers:
        return []

    # Build keyword set from the new paper
    new_keywords = set()
    if paper.get("keywords"):
        kws = paper["keywords"]
        if isinstance(kws, list):
            new_keywords = {str(k).lower().strip() for k in kws}
        elif isinstance(kws, str):
            new_keywords = {k.lower().strip() for k in kws.split(",")}
    new_keywords.update(
        w.lower()
        for w in str(paper.get("title", "")).split()
        if len(w) > 3
    )

    # Score each existing paper
    scored = []
    for p in existing_papers:
        p_keywords = set()
        if p.get("title"):
            p_keywords.update(
                w.lower() for w in p["title"].split() if len(w) > 3
            )
        keyword_overlap = len(new_keywords & p_keywords) / max(
            len(new_keywords | p_keywords), 1
        )

        # Knowledge level proximity bonus
        kl_bonus = 0.0
        new_kl = paper.get("knowledge_level", [])
        p_kl = p.get("knowledge_level", [])
        if isinstance(p_kl, str):
            try:
                p_kl = json.loads(p_kl)
            except (json.JSONDecodeError, TypeError):
                p_kl = [p_kl]
        if new_kl and p_kl:
            for a in new_kl:
                for b in p_kl:
                    a_num = _level_to_number(a)
                    b_num = _level_to_number(b)
                    if a_num is not None and b_num is not None:
                        kl_bonus = max(kl_bonus, 1.0 - abs(a_num - b_num) * 0.15)

        score = keyword_overlap * 0.5 + kl_bonus * 0.5
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:max_candidates]]


def _level_to_number(level_str) -> int | None:
    """Convert a knowledge level string to a numeric index."""
    s = str(level_str).strip().upper()
    for i in range(1, 7):
        if f"L{i}" in s:
            return i
    return None


# ---------------------------------------------------------------------------
# Main pipeline: detect_conflict
# ---------------------------------------------------------------------------


def detect_conflict(
    paper: dict,
    structure: dict,
    graph_store: GraphStore,
    adapter: LLMAdapter,
    meta_path: str | None = None,
) -> dict:
    """Run the full Phase F-2 divergence detection pipeline.

    Implements the 5-step pipeline:
      1. Deconstruction — break paper into indivisible atoms via LLM
      2. Retrieval — match atoms against existing knowledge graph K
      3. Reconstruction — attempt to explain paper using only K concepts
      4. Conflict Assessment — evaluate C1-C4 Posner conditions
      5. Decision — determine single_loop/double_loop/seed_anchored/degraded

    Args:
        paper: Paper dict with title, abstract, keywords, knowledge_level.
        structure: Structure template dict from Phase F extractor.
        graph_store: GraphStore instance with existing knowledge graph.
        adapter: LLM adapter for intelligence calls.
        meta_path: Optional path to meta control_policy.json.

    Returns:
        conflict_report dict matching the stage_f2_divergence output schema:

        .. code-block:: json

            {
                "deconstruction": {...},
                "retrieval": [...],
                "reconstruction": {...},
                "conflict_assessment": {...},
                "changes": [...],
                "single_loop_output": {...}
            }
    """
    # Load prompts
    raw = _load_prompt("stage_f2_divergence.md")
    system_prompt, user_template = _parse_prompt_sections(raw)

    if not system_prompt or not user_template:
        return {
            "error": "Failed to parse stage_f2_divergence prompt template",
            "raw": raw[:500],
        }

    # Load meta policy
    meta = _load_meta_policy(meta_path)
    unexplain_cfg = meta.get("unexplainability", _DEFAULT_META_FALLBACK["unexplainability"])
    guard_cfg = meta.get("double_loop_guard", _DEFAULT_META_FALLBACK["double_loop_guard"])
    retrieval_cfg = meta.get("retrieval", _DEFAULT_META_FALLBACK["retrieval"])

    threshold = unexplain_cfg.get("threshold_default", 0.5)
    l1_l4_weight = unexplain_cfg.get("l1_l4_weight", 0.3)
    l5_l6_weight = unexplain_cfg.get("l5_l6_weight", 0.7)
    max_impact_ratio = guard_cfg.get("max_impact_ratio", 0.3)
    require_two = guard_cfg.get("require_two_conditions", True)
    cold_start_count = unexplain_cfg.get("cold_start_node_count", 10)
    cold_start_threshold = unexplain_cfg.get("threshold_cold_start", 0.6)
    max_candidates = retrieval_cfg.get("max_retrieval_candidates", 20)

    # Gather knowledge graph context
    k_node_count = graph_store.node_count()
    seed_nodes = _get_seed_nodes(graph_store, ["L1", "L2", "L3", "L4"])
    relevant_papers = _get_relevant_existing_papers(
        graph_store, paper,
        structure_template=structure.get("structure_template"),
        max_candidates=max_candidates,
    )

    # Prepare seed node summary for the prompt
    seed_node_titles = [n.get("title", n.get("id", "?")) for n in seed_nodes[:20]]
    seed_nodes_text = "\n".join(f"- {t}" for t in seed_node_titles) if seed_node_titles else "(none)"

    # Prepare relevant papers summary
    relevant_text_parts = []
    for p in relevant_papers[:10]:
        title = p.get("title", p.get("id", "?"))
        kl = p.get("knowledge_level", [])
        if isinstance(kl, str):
            try:
                kl = json.loads(kl)
            except (json.JSONDecodeError, TypeError):
                kl = [kl]
        st = p.get("structure_template", {})
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except (json.JSONDecodeError, TypeError):
                st = {}
        abstract_pattern = st.get("abstract_pattern", "") if isinstance(st, dict) else ""
        relevant_text_parts.append(
            f"- {title} (KL: {kl}, Pattern: {abstract_pattern})"
        )
    relevant_text = "\n".join(relevant_text_parts) if relevant_text_parts else "(none)"

    # Apply cold-start threshold adjustment
    effective_threshold = threshold
    if k_node_count < cold_start_count:
        effective_threshold = cold_start_threshold

    # ── Load and render prompt via unified loader ───────────────────
    from lcortex.prompts.loader import render_prompt

    title = paper.get("title", "")
    abstract = paper.get("abstract", paper.get("abstract_text", ""))
    keywords = paper.get("keywords", paper.get("keyword_list", []))
    if isinstance(keywords, list):
        keywords = ", ".join(str(k) for k in keywords)
    knowledge_level = structure.get("knowledge_level", paper.get("knowledge_level", []))
    knowledge_level_str = ", ".join(str(k) for k in knowledge_level) if isinstance(knowledge_level, list) else str(knowledge_level)
    structure_template_json = json.dumps(structure.get("structure_template", {}), indent=2)

    system_prompt, user_message = render_prompt("stage_f2_divergence", {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "knowledge_level": knowledge_level_str,
        "structure_template_json": structure_template_json,
        "k_node_count": k_node_count,
        "seed_nodes_l1_l4": seed_nodes_text,
        "relevant_existing_papers": relevant_text,
        "threshold": effective_threshold,
        "l1_l4_weight": l1_l4_weight,
        "l5_l6_weight": l5_l6_weight,
        "max_impact_ratio": max_impact_ratio,
        "require_two_conditions": require_two,
    })

    log.info(
        "Phase F-2: Running divergence detection for '%s' (K=%d nodes)",
        title[:80],
        k_node_count,
    )

    # Call LLM for the full pipeline
    result = adapter.complete(system_prompt, user_message, _OUTPUT_SCHEMA_F2)

    if "error" in result:
        return result

    # ── Normalize numeric fields (LLM may return strings) ─────────────
    recon = result.get("reconstruction", {})
    for key in ("reconstructable_pct", "unexplainable_pct"):
        if key in recon:
            try:
                recon[key] = float(recon[key])
                # Clamp to [0, 1] — LLM sometimes returns 0.05 as 0.05% (5.0)
                if recon[key] > 1.0:
                    recon[key] = recon[key] / 100.0
                recon[key] = max(0.0, min(1.0, recon[key]))
            except (ValueError, TypeError):
                recon[key] = 0.0

    # Normalize retrieval match_scores
    for item in result.get("retrieval", []):
        if "match_score" in item:
            try:
                item["match_score"] = float(item["match_score"])
            except (ValueError, TypeError):
                item["match_score"] = 0.0

    # Post-process: apply meta policy overrides to the LLM's decision
    result = _apply_meta_policy(
        result,
        k_node_count=k_node_count,
        cold_start_count=cold_start_count,
        effective_threshold=effective_threshold,
        max_impact_ratio=max_impact_ratio,
        require_two=require_two,
        consecutive_triggers=_count_consecutive_triggers(graph_store, guard_cfg),
        consecutive_limit=guard_cfg.get("consecutive_trigger_limit", 2),
    )

    return result


# ---------------------------------------------------------------------------
# Meta policy post-processing
# ---------------------------------------------------------------------------


def _apply_meta_policy(
    report: dict,
    *,
    k_node_count: int,
    cold_start_count: int,
    effective_threshold: float,
    max_impact_ratio: float,
    require_two: bool,
    consecutive_triggers: int,
    consecutive_limit: int,
) -> dict:
    """Apply meta policy rules to the raw LLM conflict report.

    This is the safety layer — it can override the LLM's recommended action
    based on the meta control policy thresholds.
    """
    assessment = report.get("conflict_assessment", {})
    if not assessment:
        return report

    unexplain_score = assessment.get("unexplainability_score", 0.0)

    # Decision tree from stage_f2_divergence.md:
    # 1. K < cold_start_count → seed_anchored
    # 2. unexplainability_score < threshold → single_loop
    # 3. unexplainability_score >= threshold AND N < 2 → single_loop
    # 4. unexplainability_score >= threshold AND N >= 2 → double_loop
    # 4a. If estimated impact > max_impact_ratio → degraded_by_meta

    if k_node_count < cold_start_count:
        assessment["recommended_action"] = "seed_anchored"
        assessment["triggers_double_loop"] = False
        assessment["threshold_used"] = effective_threshold
        return report

    # Count true conditions
    conds = assessment.get("conceptual_change_conditions", {})
    true_count = sum(
        1
        for c in ("dissatisfaction", "intelligibility", "plausibility", "fruitfulness")
        if conds.get(c, {}).get("value", False) is True
    )
    assessment["conditions_true_count"] = true_count

    # Two-condition guard
    if require_two:
        assessment["passed_two_condition_guard"] = true_count >= 2
    else:
        assessment["passed_two_condition_guard"] = True

    # Consecutive trigger guard
    if consecutive_triggers >= consecutive_limit:
        log.warning(
            "Double-loop blocked by consecutive trigger limit (%d >= %d)",
            consecutive_triggers,
            consecutive_limit,
        )
        assessment["recommended_action"] = "degraded_by_meta"
        assessment["downgraded_by_meta"] = True
        assessment["downgrade_reason"] = (
            f"Consecutive Double-loop limit ({consecutive_limit}) reached. "
            f"Wait for human review."
        )
        assessment["triggers_double_loop"] = False
        return report

    # Core decision
    if unexplain_score < effective_threshold:
        assessment["recommended_action"] = "single_loop"
        assessment["triggers_double_loop"] = False
    elif not assessment.get("passed_two_condition_guard", False):
        assessment["recommended_action"] = "single_loop"
        assessment["triggers_double_loop"] = False
    else:
        # Double-loop candidate — check impact
        changes = report.get("changes", [])
        if _estimate_impact(changes, k_node_count) > max_impact_ratio:
            assessment["recommended_action"] = "degraded_by_meta"
            assessment["downgraded_by_meta"] = True
            assessment["downgrade_reason"] = (
                f"Estimated impact > max_impact_ratio ({max_impact_ratio})"
            )
            assessment["triggers_double_loop"] = False
        else:
            assessment["recommended_action"] = "double_loop"
            assessment["triggers_double_loop"] = True
            assessment["downgraded_by_meta"] = False

    assessment["threshold_used"] = effective_threshold
    return report


def _estimate_impact(changes: list[dict], total_nodes: int) -> float:
    """Estimate what fraction of nodes would be affected by proposed changes.

    Uses a simple heuristic: each change type has an estimated node impact.
    """
    if not changes or total_nodes <= 1:
        return 0.0

    estimated_affected = 0
    for c in changes:
        ctype = c.get("type", "")
        if ctype == "new_level":
            estimated_affected += min(3, total_nodes * 0.1)
        elif ctype == "reparent":
            estimated_affected += 2
        elif ctype == "merge":
            estimated_affected += 2
        elif ctype == "split":
            estimated_affected += 3

    return min(1.0, estimated_affected / total_nodes)


def _count_consecutive_triggers(
    graph_store: GraphStore,
    guard_cfg: dict,
) -> int:
    """Count consecutive recent double_loop triggers in ontology history.

    Looks at the last N evolution events and counts how many in a row
    were Double-loop triggers.
    """
    window = guard_cfg.get("impact_estimation_window", 5)
    history = graph_store.get_ontology_history(limit=window)
    count = 0
    for event in history:
        details = event.get("details", {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except (json.JSONDecodeError, TypeError):
                details = {}
        if details.get("action") in ("double_loop", "ontology_change_applied"):
            count += 1
        else:
            break  # Stop at first non-double-loop event
    return count
