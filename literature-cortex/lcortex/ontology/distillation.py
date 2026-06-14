"""
Ontology Distillation — Self-compression of the knowledge ontology.

Periodically compresses the ontology to prevent fragmentation:
  1. Merge siblings with Jaccard similarity >= meta.distillation.merge_jaccard_threshold
  2. Collapse levels with fewer than meta.distillation.stable_after_n_no_updates papers
  3. Return summary of changes applied
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from lcortex.graph.store import GraphStore
from lcortex.ontology.evolution import merge_nodes, reparent_node

log = logging.getLogger("lcortex.ontology.distillation")

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
        "distillation": {
            "merge_jaccard_threshold": 0.7,
            "stable_after_n_no_updates": 3,
            "decay_rate_l5_l6": 0.1,
            "l1_l4_decay": False,
            "min_nodes_per_level": 2,
        }
    }


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9_]{2,}", str(text).lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _keyword_similarity(node_a: dict, node_b: dict) -> float:
    """Compute keyword-based similarity between two ontology nodes.

    Uses title and structure_template.abstract_pattern.
    """
    a_words = _tokenize(node_a.get("title", ""))
    b_words = _tokenize(node_b.get("title", ""))

    # Add abstract pattern tokens
    st_a = node_a.get("structure_template", {})
    st_b = node_b.get("structure_template", {})
    if isinstance(st_a, str):
        try:
            st_a = json.loads(st_a)
        except (json.JSONDecodeError, TypeError):
            st_a = {}
    if isinstance(st_b, str):
        try:
            st_b = json.loads(st_b)
        except (json.JSONDecodeError, TypeError):
            st_b = {}

    a_words.update(_tokenize(st_a.get("abstract_pattern", "")))
    b_words.update(_tokenize(st_b.get("abstract_pattern", "")))
    a_words.update(_tokenize(st_a.get("mathematical_core", "")))
    b_words.update(_tokenize(st_b.get("mathematical_core", "")))

    return _jaccard(a_words, b_words)


# ---------------------------------------------------------------------------
# Self-distillation
# ---------------------------------------------------------------------------


def self_distill(
    graph_store: GraphStore,
    meta_path: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run self-distillation on the knowledge ontology.

    Two compression passes:
      1. **Sibling merging**: for each knowledge level, find sibling pairs
         with Jaccard similarity >= threshold and merge them.
      2. **Level collapsing**: merge levels that have fewer than
         stable_after_n_no_updates papers with their parent level.

    Args:
        graph_store: GraphStore instance.
        meta_path: Optional path to control_policy.json.
        dry_run: If True, report what would be done without applying changes.

    Returns:
        Summary dict:
        .. code-block:: json

            {
                "sibling_merges": [...],
                "level_collapses": [...],
                "total_changes": 0,
                "dry_run": false
            }
    """
    meta = _load_meta(meta_path)
    dist_cfg = meta.get("distillation", _load_meta()["distillation"])

    merge_threshold = dist_cfg.get("merge_jaccard_threshold", 0.7)
    min_nodes = dist_cfg.get("min_nodes_per_level", 2)
    stable_after = dist_cfg.get("stable_after_n_no_updates", 3)

    summary = {
        "sibling_merges": [],
        "level_collapses": [],
        "total_changes": 0,
        "dry_run": dry_run,
    }

    # Get all nodes
    all_nodes = graph_store.get_all_nodes()
    if len(all_nodes) < 2:
        return summary

    # ── Pass 1: Sibling merging ──────────────────────────────────────────
    # Group nodes by their knowledge_level
    level_groups: dict[str, list[dict]] = defaultdict(list)
    for node in all_nodes:
        kl = node.get("knowledge_level", [])
        if isinstance(kl, str):
            try:
                kl = json.loads(kl)
            except (json.JSONDecodeError, TypeError):
                kl = [kl]
        if not kl:
            continue
        for lvl in kl:
            level_groups[str(lvl).strip()].append(node)

    # Within each level, find similar sibling pairs
    sibling_merges = []
    for level, nodes in level_groups.items():
        if len(nodes) < 2:
            continue
        # Check all pairs
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                sim = _keyword_similarity(nodes[i], nodes[j])
                if sim >= merge_threshold:
                    sibling_merges.append({
                        "level": level,
                        "node_a": nodes[i]["id"],
                        "node_a_title": nodes[i].get("title", "?"),
                        "node_b": nodes[j]["id"],
                        "node_b_title": nodes[j].get("title", "?"),
                        "jaccard": round(sim, 3),
                    })

    # Apply sibling merges (deduplicate)
    seen_ids = set()
    applied_merges = []
    for m in sibling_merges:
        if m["node_a"] in seen_ids or m["node_b"] in seen_ids:
            continue
        if not dry_run:
            result = merge_nodes(
                graph_store,
                m["node_a"],
                m["node_b"],
                trigger_paper="self_distill",
            )
            if result.get("status") == "ok":
                applied_merges.append(m)
                seen_ids.add(m["node_a"])
                seen_ids.add(m["node_b"])
        else:
            applied_merges.append(m)
            seen_ids.add(m["node_a"])
            seen_ids.add(m["node_b"])

    summary["sibling_merges"] = applied_merges

    # ── Pass 2: Level collapsing ─────────────────────────────────────────
    # Count papers per level
    level_paper_counts: dict[str, int] = defaultdict(int)
    for node in all_nodes:
        kl = node.get("knowledge_level", [])
        if isinstance(kl, str):
            try:
                kl = json.loads(kl)
            except (json.JSONDecodeError, TypeError):
                kl = [kl]
        for lvl in kl:
            level_paper_counts[str(lvl).strip()] += 1

    # Find levels with too few papers
    level_collapses = []
    for level, count in level_paper_counts.items():
        if count < min_nodes:
            # Find a parent level (strip numeric suffix)
            parent = _find_parent_level(level)
            if parent and level_paper_counts.get(parent, 0) > 0:
                level_collapses.append({
                    "level": level,
                    "paper_count": count,
                    "parent_level": parent,
                    "reason": f"Only {count} papers (< {min_nodes} min_nodes_per_level)",
                })

    # Apply collapses
    applied_collapses = []
    for c in level_collapses:
        if not dry_run:
            # Reparent all nodes in this level to the parent
            nodes_in_level = level_groups.get(c["level"], [])
            for node in nodes_in_level:
                reparent_node(
                    graph_store,
                    node["id"],
                    c["parent_level"],
                    trigger_paper="self_distill",
                )
            applied_collapses.append(c)
        else:
            applied_collapses.append(c)

    summary["level_collapses"] = applied_collapses
    summary["total_changes"] = len(applied_merges) + len(applied_collapses)

    log.info(
        "Self-distillation complete: %d merges, %d collapses (dry_run=%s)",
        len(applied_merges),
        len(applied_collapses),
        dry_run,
    )

    return summary


def _find_parent_level(level_name: str) -> str | None:
    """Heuristically find the parent level of a given knowledge level.

    Examples:
        "L3-Algorithm" → "L2-Math"
        "L5.5-Hybrid" → "L4-Physical"
    """
    import re
    # Try to extract a numeric prefix
    m = re.match(r"L(\d+)", str(level_name))
    if not m:
        return None
    num = int(m.group(1))
    if num <= 1:
        return None
    return f"L{num - 1}"
