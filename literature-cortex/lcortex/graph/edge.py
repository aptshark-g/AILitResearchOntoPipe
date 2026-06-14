"""Edge management for Literature Cortex knowledge graph.

Handles creation, promotion, and downgrade of edges between correlation
and causation based on the four causal constraint types:
  C1 - temporal_precedence
  C2 - mechanistic_chain
  C3 - intervention_test
  C4 - boundary_conditions
"""

import json
import time
import uuid
from typing import Any


CAUSAL_CONSTRAINTS = [
    "temporal_precedence",
    "mechanistic_chain",
    "intervention_test",
    "boundary_conditions",
]

PROMOTION_FIELDS = [
    "is_causal",
    "conditions_met",
    "confidence",
    "last_evaluated",
    "promotion_path",
]


def _make_edge_id(source: str, target: str) -> str:
    """Generate a deterministic edge ID from source and target."""
    return f"edge::{source}::{target}"


def _now_iso() -> str:
    """Current time as ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _history_entry(action: str, details: dict[str, Any]) -> dict[str, Any]:
    """Build a standard history entry."""
    return {
        "action": action,
        "timestamp": _now_iso(),
        **details,
    }


def create_correlation_edge(
    source: str,
    target: str,
    score: float = 0.0,
    mechanism_desc: str = "",
) -> dict[str, Any]:
    """Create a new correlation edge (not yet causal).

    Args:
        source: Source node ID.
        target: Target node ID.
        score: Base score for the correlation (0.0-1.0).
        mechanism_desc: Description of the mechanism or relationship.

    Returns:
        Edge data dictionary ready for insertion into the store.
    """
    edge_id = _make_edge_id(source, target)
    now = _now_iso()

    causal_promotion = {
        "is_causal": False,
        "conditions_met": [],
        "confidence": 0.0,
        "last_evaluated": None,
        "promotion_path": {},
    }

    history_entry = _history_entry("created", {"initial_score": score})

    return {
        "id": edge_id,
        "source_id": source,
        "target_id": target,
        "base_type": "correlation",
        "base_score": score,
        "causal_promotion": json.dumps(causal_promotion),
        "mechanism_description": mechanism_desc,
        "history": json.dumps([history_entry]),
        "created_at": now,
    }


def promote_to_causal(
    edge: dict[str, Any],
    conditions_met: list[str],
    confidence: float,
) -> dict[str, Any]:
    """Promote an existing edge from correlation to causation.

    Updates the edge's base_type, causal_promotion, and appends to history.

    Args:
        edge: Existing edge dict (as returned from store).
        conditions_met: List of condition labels met (C1-C4 names).
        confidence: Confidence score for the causal claim (0.0-1.0).

    Returns:
        Updated edge dict with causation fields set.

    Raises:
        ValueError: If edge is already causal or conditions are invalid.
    """
    causal_promo = _parse_json_field(edge, "causal_promotion")
    if causal_promo.get("is_causal"):
        raise ValueError(f"Edge {edge['id']} is already causal")

    valid = [c for c in conditions_met if c in CAUSAL_CONSTRAINTS]
    if not valid:
        raise ValueError(f"No valid conditions in {conditions_met}. Must be from: {CAUSAL_CONSTRAINTS}")

    new_causal = {
        "is_causal": True,
        "conditions_met": valid,
        "confidence": confidence,
        "last_evaluated": _now_iso(),
        "promotion_path": causal_promo.get("promotion_path", {}),
        "promoted_from_correlation": True,
    }

    history = _parse_json_field(edge, "history")
    history.append(_history_entry("promoted_to_causal", {
        "conditions_met": valid,
        "confidence": confidence,
    }))

    edge["base_type"] = "causation"
    edge["causal_promotion"] = json.dumps(new_causal)
    edge["history"] = json.dumps(history)
    return edge


def downgrade_to_correlation(
    edge: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Demote a causal edge back to correlation.

    Args:
        edge: Existing edge dict.
        reason: Reason for downgrading (e.g., 'counter-evidence found').

    Returns:
        Updated edge dict.

    Raises:
        ValueError: If edge is not currently causal.
    """
    causal_promo = _parse_json_field(edge, "causal_promotion")
    if not causal_promo.get("is_causal"):
        raise ValueError(f"Edge {edge['id']} is not causal")

    new_causal = {
        "is_causal": False,
        "conditions_met": [],
        "confidence": 0.0,
        "last_evaluated": _now_iso(),
        "promotion_path": causal_promo.get("promotion_path", {}),
        "downgrade_reason": reason,
    }

    history = _parse_json_field(edge, "history")
    history.append(_history_entry("downgraded_to_correlation", {
        "reason": reason,
        "previous_conditions": causal_promo.get("conditions_met", []),
        "previous_confidence": causal_promo.get("confidence", 0.0),
    }))

    edge["base_type"] = "correlation"
    edge["causal_promotion"] = json.dumps(new_causal)
    edge["history"] = json.dumps(history)
    return edge


def check_promotion_conditions(edge: dict[str, Any]) -> dict[str, Any]:
    """Check which of the four causal constraint types are met for an edge.

    Evaluates C1-C4 based on available evidence in the edge's promotion_path,
    mechanism_description, and any associated metadata.

    C1 - temporal_precedence: source must precede target in time
    C2 - mechanistic_chain: plausible mechanistic link described
    C3 - intervention_test: evidence of experimental manipulation
    C4 - boundary_conditions: scope/limits of the causal claim identified

    Args:
        edge: Edge dict (as returned from store).

    Returns:
        Dict with keys:
            status: 'ready' | 'not_ready' | 'already_causal'
            met: list of condition labels currently met
            unmet: list of condition labels not yet met
            confidence: estimated confidence if promoted (0.0-1.0)
            details: per-condition assessment notes
    """
    causal_promo = _parse_json_field(edge, "causal_promotion")
    if causal_promo.get("is_causal"):
        return {
            "status": "already_causal",
            "met": causal_promo.get("conditions_met", []),
            "unmet": [],
            "confidence": causal_promo.get("confidence", 0.0),
            "details": {c: "previously met" for c in causal_promo.get("conditions_met", [])},
        }

    promotion_path = causal_promo.get("promotion_path", {})
    mech_desc = edge.get("mechanism_description", "") or ""

    assessments = {
        "temporal_precedence": _check_temporal_precedence(edge, promotion_path),
        "mechanistic_chain": _check_mechanistic_chain(mech_desc, promotion_path),
        "intervention_test": _check_intervention_test(promotion_path),
        "boundary_conditions": _check_boundary_conditions(promotion_path),
    }

    met = [c for c, a in assessments.items() if a["met"]]
    unmet = [c for c, a in assessments.items() if not a["met"]]
    details = {c: a["note"] for c, a in assessments.items()}

    confidence = len(met) / len(CAUSAL_CONSTRAINTS) if CAUSAL_CONSTRAINTS else 0.0
    ready = len(met) >= 2  # At least 2 conditions needed

    return {
        "status": "ready" if ready else "not_ready",
        "met": met,
        "unmet": unmet,
        "confidence": confidence,
        "details": details,
    }


def _check_temporal_precedence(edge: dict[str, Any], path: dict[str, Any]) -> dict[str, Any]:
    """Check if temporal precedence evidence exists."""
    if path.get("temporal_precedence") is not None:
        return {"met": bool(path["temporal_precedence"]), "note": "from promotion_path"}
    if edge.get("source_id") and edge.get("target_id"):
        return {
            "met": False,
            "note": "no temporal evidence recorded; node years can be compared if available",
        }
    return {"met": False, "note": "insufficient node metadata"}


def _check_mechanistic_chain(mech_desc: str, path: dict[str, Any]) -> dict[str, Any]:
    """Check if a mechanistic chain is described."""
    if path.get("mechanistic_chain") is not None:
        return {"met": bool(path["mechanistic_chain"]), "note": "from promotion_path"}
    if mech_desc and len(mech_desc.strip()) > 20:
        return {"met": True, "note": "mechanism description present"}
    return {"met": False, "note": "no mechanism description or evidence"}


def _check_intervention_test(path: dict[str, Any]) -> dict[str, Any]:
    """Check if experimental intervention evidence exists."""
    if path.get("intervention_test") is not None:
        return {"met": bool(path["intervention_test"]), "note": "from promotion_path"}
    return {"met": False, "note": "no intervention evidence recorded"}


def _check_boundary_conditions(path: dict[str, Any]) -> dict[str, Any]:
    """Check if boundary conditions are identified."""
    if path.get("boundary_conditions") is not None:
        return {"met": bool(path["boundary_conditions"]), "note": "from promotion_path"}
    return {"met": False, "note": "no boundary conditions recorded"}


def _parse_json_field(edge: dict[str, Any], field: str) -> Any:
    """Parse a JSON text field from an edge dict, handling both str and dict."""
    val = edge.get(field, "{}")
    if isinstance(val, dict) or isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {} if field != "history" else []
