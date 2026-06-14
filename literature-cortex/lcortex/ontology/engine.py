"""
Ontology Engine — Applies Double-loop ontology changes to the knowledge graph.

The engine receives a conflict_report from Phase F-2 and executes the
recommended changes, subject to Meta policy double_loop_guard checks.

Handles 4 change types: new_level, reparent, merge, split.

Flow:
  1. Check action type from conflict_report.conflict_assessment
  2. If double_loop → verify Meta guardrails
  3. Execute changes via evolution operations
  4. Log to ontology_evolution table
  5. Return success/blocked status
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lcortex.graph.store import GraphStore
from lcortex.ontology.evolution import (
    insert_level,
    merge_nodes,
    reparent_node,
    split_node,
)

log = logging.getLogger("lcortex.ontology.engine")

# ---------------------------------------------------------------------------
# Meta policy loading
# ---------------------------------------------------------------------------

_DEFAULT_META_PATH = (
    Path(__file__).resolve().parents[1] / "seeds" / "meta" / "control_policy.json"
)


def _load_meta(meta_path: str | None = None) -> dict:
    try:
        path = Path(meta_path) if meta_path else _DEFAULT_META_PATH
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "double_loop_guard": {
            "max_impact_ratio": 0.3,
            "consecutive_trigger_limit": 2,
            "require_two_conditions": True,
            "impact_estimation_window": 5,
        },
    }


def _count_consecutive_triggers(graph_store: GraphStore, window: int = 5) -> int:
    """Count consecutive double_loop triggers in recent history."""
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
            break
    return count


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def apply_ontology_change(
    graph_store: GraphStore,
    conflict_report: dict,
    trigger_paper: str | None = None,
    meta_path: str | None = None,
) -> bool:
    """Apply ontology changes from a conflict report to the knowledge graph.

    Checks the Meta policy double_loop_guard before applying changes.
    Only executes if the conflict_assessment.recommended_action is
    "double_loop" (LLM recommended) AND the Meta guard says it's safe.

    Args:
        graph_store: GraphStore instance with the knowledge graph.
        conflict_report: Full conflict report dict from Phase F-2
            detect_conflict(). Must contain:
            - conflict_assessment.recommended_action
            - conflict_assessment.triggers_double_loop
            - changes (list of change dicts)
        trigger_paper: Paper id that triggered this ontology change.
        meta_path: Optional path to control_policy.json (loads default
            seeds/meta/control_policy.json if not provided).

    Returns:
        True if changes were successfully applied, False if blocked by
        Meta policy or if the recommended action is not double_loop.
    """
    meta = _load_meta(meta_path)
    guard_cfg = meta.get("double_loop_guard", {})

    assessment = conflict_report.get("conflict_assessment", {})
    recommended = assessment.get("recommended_action", "single_loop")

    if recommended not in ("double_loop", "seed_anchored"):
        log.info(
            "Ontology change skipped: recommended_action=%s",
            recommended,
        )
        return False

    # seed_anchored bypasses all Meta guard checks (K too small for divergence)
    if recommended == "seed_anchored":
        return _apply_changes(graph_store, conflict_report, trigger_paper)

    # ── Meta guard: consecutive trigger limit ────────────────────────────
    limit = guard_cfg.get("consecutive_trigger_limit", 2)
    window = guard_cfg.get("impact_estimation_window", 5)
    consecutive = _count_consecutive_triggers(graph_store, window)

    if consecutive >= limit:
        log.warning(
            "Ontology change BLOCKED by Meta: %d consecutive triggers >= limit %d",
            consecutive,
            limit,
        )
        graph_store.log_ontology_evolution(
            trigger_paper=trigger_paper or "system",
            change_type="double_loop_blocked",
            description=f"Blocked by Meta: consecutive triggers ({consecutive} >= {limit})",
            details={
                "action": "double_loop_blocked",
                "reason": "consecutive_trigger_limit",
                "consecutive_triggers": consecutive,
                "limit": limit,
            },
        )
        return False

    # ── Meta guard: require_two_conditions ────────────────────────────────
    if guard_cfg.get("require_two_conditions", True):
        conds = assessment.get("conceptual_change_conditions", {})
        true_count = sum(
            1
            for c in ("dissatisfaction", "intelligibility", "plausibility", "fruitfulness")
            if conds.get(c, {}).get("value", False) is True
        )
        if true_count < 2:
            log.warning(
                "Ontology change BLOCKED by Meta: only %d conditions true (need >= 2)",
                true_count,
            )
            graph_store.log_ontology_evolution(
                trigger_paper=trigger_paper or "system",
                change_type="double_loop_blocked",
                description=(
                    f"Blocked by Meta: only {true_count} conditions true"
                ),
                details={
                    "action": "double_loop_blocked",
                    "reason": "insufficient_conditions",
                    "true_count": true_count,
                },
            )
            return False

    # ── Meta guard: impact ratio ─────────────────────────────────────────
    max_impact = guard_cfg.get("max_impact_ratio", 0.3)
    changes = conflict_report.get("changes", [])
    k_node_count = graph_store.node_count()
    estimated_impact = _estimate_impact(changes, k_node_count)

    if estimated_impact > max_impact:
        log.warning(
            "Ontology change BLOCKED by Meta: impact %.2f > max %.2f",
            estimated_impact,
            max_impact,
        )
        graph_store.log_ontology_evolution(
            trigger_paper=trigger_paper or "system",
            change_type="double_loop_blocked",
            description=(
                f"Blocked by Meta: impact ratio {estimated_impact:.2f} > "
                f"max {max_impact}"
            ),
            details={
                "action": "double_loop_blocked",
                "reason": "impact_ratio_exceeded",
                "estimated_impact": estimated_impact,
                "max_impact_ratio": max_impact,
            },
        )
        return False

    return _apply_changes(graph_store, conflict_report, trigger_paper)


def _apply_changes(
    graph_store: GraphStore,
    conflict_report: dict,
    trigger_paper: str | None = None,
) -> bool:
    """Execute the changes from a conflict report."""
    changes = conflict_report.get("changes", [])
    assessment = conflict_report.get("conflict_assessment", {})

    if not changes:
        log.info("No concrete changes specified in conflict report, nothing to apply")
        return False

    applied = 0
    errors = 0
    results = []

    for change in changes:
        ctype = change.get("type", "")
        detail = change.get("detail", "")

        try:
            if ctype == "new_level":
                level_name = detail or change.get("level_name", "new_level")
                parent_level = change.get("parent_level")
                result = insert_level(
                    graph_store,
                    level_name,
                    parent_level=parent_level,
                    trigger_paper=trigger_paper,
                )

            elif ctype == "reparent":
                node_id = change.get("node_id", "")
                new_parent = change.get("new_parent", detail)
                result = reparent_node(
                    graph_store,
                    node_id,
                    new_parent,
                    trigger_paper=trigger_paper,
                )

            elif ctype == "merge":
                node_a = change.get("node_a", "")
                node_b = change.get("node_b", detail)
                result = merge_nodes(
                    graph_store,
                    node_a,
                    node_b,
                    trigger_paper=trigger_paper,
                )

            elif ctype == "split":
                node_id = change.get("node_id", "")
                children = change.get("new_children", [])
                if not children and detail:
                    children = [s.strip() for s in detail.split(",")]
                result = split_node(
                    graph_store,
                    node_id,
                    children,
                    trigger_paper=trigger_paper,
                )

            else:
                log.warning("Unknown change type '%s', skipping", ctype)
                errors += 1
                results.append({"type": ctype, "status": "unknown_type"})
                continue

            if result.get("status") == "ok":
                applied += 1
            else:
                errors += 1
            results.append(result)

        except Exception as exc:
            log.exception("Error applying change '%s': %s", ctype, exc)
            errors += 1
            results.append({"type": ctype, "status": "error", "reason": str(exc)})

    # Log the ontology change application
    graph_store.log_ontology_evolution(
        trigger_paper=trigger_paper or "system",
        change_type="ontology_change_applied",
        description=(
            f"Applied {applied} ontology changes "
            f"{'(+ ' + str(errors) + ' errors)' if errors else ''}"
        ),
        details={
            "action": "double_loop",
            "changes_applied": applied,
            "changes_errors": errors,
            "results": results,
            "conflict_assessment": assessment,
        },
    )

    log.info(
        "Ontology engine: applied %d/%d changes (errors=%d)",
        applied,
        len(changes),
        errors,
    )

    return applied > 0


def _estimate_impact(changes: list[dict], total_nodes: int) -> float:
    """Estimate fraction of nodes affected by proposed changes."""
    if not changes or total_nodes <= 1:
        return 0.0
    estimated = 0
    for c in changes:
        ctype = c.get("type", "")
        if ctype == "new_level":
            estimated += min(3, total_nodes * 0.1)
        elif ctype in ("reparent", "merge"):
            estimated += 2
        elif ctype == "split":
            estimated += 3
    return min(1.0, estimated / total_nodes)
