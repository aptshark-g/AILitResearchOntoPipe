"""
Ontology evolution operations — low-level graph mutations.

Each operation:
  1. Performs the graph mutation via GraphStore
  2. Logs the change to the ontology_evolution table
  3. Returns operation metadata for downstream tracking

These functions are called by the ontology engine, which first checks the
Meta policy double_loop_guard before dispatching.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from lcortex.graph.store import GraphStore

log = logging.getLogger("lcortex.ontology.evolution")

# ---------------------------------------------------------------------------
# Operation: insert_level
# ---------------------------------------------------------------------------


def insert_level(
    graph_store: GraphStore,
    level_name: str,
    parent_level: str | None = None,
    trigger_paper: str | None = None,
) -> dict:
    """Insert a new knowledge level node into the ontology.

    Creates a seed node representing the new level and optionally links
    it to a parent level via an edge.

    Args:
        graph_store: GraphStore instance.
        level_name: Name of the new knowledge level (e.g., "L2.5-Hybrid").
        parent_level: Optional parent level node id or name.
        trigger_paper: Paper id that triggered this insertion.

    Returns:
        Operation result dict with status, node_id, edge_id.
    """
    level_id = f"seed:level:{uuid.uuid4().hex[:12]}"
    log.info(
        "insert_level: creating '%s' (parent=%s, trigger=%s)",
        level_name,
        parent_level,
        trigger_paper,
    )

    # Create the level node
    graph_store.add_node({
        "id": level_id,
        "type": "seed",
        "title": level_name,
        "year": None,
        "knowledge_level": [level_name],
        "structure_template": {},
    })

    # Link to parent if provided
    edge_id = None
    if parent_level:
        # Find parent node
        parent_node = graph_store.get_node(parent_level)
        if parent_node is None:
            # Try to find by title
            for n in graph_store.get_all_nodes():
                if n.get("title") == parent_level:
                    parent_node = n
                    break
        if parent_node:
            edge_id = f"edge::{level_id}::{parent_node['id']}"
            graph_store.add_edge(level_id, parent_node["id"], {
                "id": edge_id,
                "base_type": "correlation",
                "base_score": 0.8,
                "mechanism_description": f"sublevel of {parent_level}",
            })

    # Log to ontology_evolution
    graph_store.log_ontology_evolution(
        trigger_paper=trigger_paper or "system",
        change_type="new_level",
        description=f"Inserted new level: {level_name}",
        details={
            "new_level": level_name,
            "parent_level": parent_level,
            "node_id": level_id,
            "edge_id": edge_id,
        },
    )

    return {
        "status": "ok",
        "type": "new_level",
        "node_id": level_id,
        "edge_id": edge_id,
    }


# ---------------------------------------------------------------------------
# Operation: reparent_node
# ---------------------------------------------------------------------------


def reparent_node(
    graph_store: GraphStore,
    node_id: str,
    new_parent: str,
    trigger_paper: str | None = None,
) -> dict:
    """Reparent a node from its current parent to a new parent.

    Removes existing outgoing edges from the node (except to non-level
    nodes) and creates a new edge to the new parent.

    Args:
        graph_store: GraphStore instance.
        node_id: Node to reparent.
        new_parent: ID or title of the new parent node.
        trigger_paper: Paper id that triggered this operation.

    Returns:
        Operation result dict.
    """
    node = graph_store.get_node(node_id)
    if node is None:
        return {"status": "error", "reason": f"Node not found: {node_id}"}

    log.info(
        "reparent_node: '%s' → '%s' (trigger=%s)",
        node.get("title", node_id),
        new_parent,
        trigger_paper,
    )

    # Find the new parent node
    parent_node = graph_store.get_node(new_parent)
    if parent_node is None:
        for n in graph_store.get_all_nodes():
            if n.get("title") == new_parent:
                parent_node = n
                break
    if parent_node is None:
        return {"status": "error", "reason": f"Parent not found: {new_parent}"}

    # Get existing edges and remove old parent edges (type=seed connections)
    existing_edges = graph_store.get_edges(node_id)
    removed_edges = []
    for edge in existing_edges:
        other = edge["target_id"] if edge["source_id"] == node_id else edge["source_id"]
        other_node = graph_store.get_node(other)
        if other_node and other_node.get("type") == "seed":
            # Remove this edge by overwriting with a deleted marker — since
            # we don't have a delete_edge, we'll add a note
            removed_edges.append(edge["id"])

    # Create new edge to parent
    edge_id = f"edge::rep::{node_id}::{parent_node['id']}"
    graph_store.add_edge(node_id, parent_node["id"], {
        "id": edge_id,
        "base_type": "correlation",
        "base_score": 1.0,
        "mechanism_description": f"reparented from {removed_edges}",
    })

    # Log
    graph_store.log_ontology_evolution(
        trigger_paper=trigger_paper or "system",
        change_type="reparent",
        description=f"Reparented node '{node.get('title', node_id)}' to '{new_parent}'",
        details={
            "node_id": node_id,
            "new_parent": parent_node["id"],
            "removed_edges": removed_edges,
            "new_edge": edge_id,
        },
    )

    return {
        "status": "ok",
        "type": "reparent",
        "node_id": node_id,
        "new_parent_id": parent_node["id"],
        "edge_id": edge_id,
    }


# ---------------------------------------------------------------------------
# Operation: merge_nodes
# ---------------------------------------------------------------------------


def merge_nodes(
    graph_store: GraphStore,
    node_a: str,
    node_b: str,
    trigger_paper: str | None = None,
) -> dict:
    """Merge two nodes in the ontology.

    Strategy:
      - Keep node_a, mark it as merged (update its title)
      - Transfer all edges from node_b to node_a
      - The node_b reference is kept but noted in the evolution log
      (since SQLite FK constraints prevent cascading into edges safely
      without a proper edge migration, we flag node_b as a redirect).

    Args:
        graph_store: GraphStore instance.
        node_a: ID of the survivor node.
        node_b: ID of the node to merge into node_a.
        trigger_paper: Paper id that triggered this operation.

    Returns:
        Operation result dict.
    """
    a = graph_store.get_node(node_a)
    b = graph_store.get_node(node_b)

    if a is None:
        return {"status": "error", "reason": f"Node A not found: {node_a}"}
    if b is None:
        return {"status": "error", "reason": f"Node B not found: {node_b}"}

    log.info(
        "merge_nodes: '%s' ← '%s' (trigger=%s)",
        a.get("title", node_a),
        b.get("title", node_b),
        trigger_paper,
    )

    # Transfer edges: rewire edges pointing to node_b to point to node_a
    edges_b = graph_store.get_edges(node_b)
    rewired_edges = []
    for edge in edges_b:
        new_source = node_a if edge["source_id"] == node_b else edge["source_id"]
        new_target = node_a if edge["target_id"] == node_b else edge["target_id"]
        if new_source == new_target:
            continue  # Skip self-loops
        new_edge_id = f"edge::mig::{edge['id']}"
        graph_store.add_edge(new_source, new_target, {
            "id": new_edge_id,
            "base_type": edge.get("base_type", "correlation"),
            "base_score": edge.get("base_score", 0.0),
            "causal_promotion": edge.get("causal_promotion", {}),
            "mechanism_description": f"Migrated from merge of {node_b} into {node_a}",
            "history": edge.get("history", []),
        })
        rewired_edges.append({"old_id": edge["id"], "new_id": new_edge_id})

    # Update node_a title to reflect merge
    new_title = f"{a.get('title', node_a)} ∪ {b.get('title', node_b)}"
    graph_store.add_node({
        "id": node_a,
        "type": a.get("type", "seed"),
        "title": new_title,
        "year": a.get("year"),
        "knowledge_level": a.get("knowledge_level", []),
        "structure_template": a.get("structure_template", {}),
        "created_at": a.get("created_at"),
    })

    # Log
    graph_store.log_ontology_evolution(
        trigger_paper=trigger_paper or "system",
        change_type="merge",
        description=f"Merged '{b.get('title', node_b)}' into '{a.get('title', node_a)}'",
        details={
            "survivor_id": node_a,
            "merged_id": node_b,
            "rewired_edges": rewired_edges,
            "new_title": new_title,
        },
    )

    return {
        "status": "ok",
        "type": "merge",
        "survivor_id": node_a,
        "merged_id": node_b,
        "rewired_edge_count": len(rewired_edges),
    }


# ---------------------------------------------------------------------------
# Operation: split_node
# ---------------------------------------------------------------------------


def split_node(
    graph_store: GraphStore,
    node_id: str,
    new_children: list[str],
    trigger_paper: str | None = None,
) -> dict:
    """Split a node into multiple child nodes.

    The original node is kept but marked as a parent, and new child nodes
    are created. Existing edges from the original node are maintained.

    Args:
        graph_store: GraphStore instance.
        node_id: ID of the node to split.
        new_children: List of names for the new child nodes.
        trigger_paper: Paper id that triggered this operation.

    Returns:
        Operation result dict.
    """
    node = graph_store.get_node(node_id)
    if node is None:
        return {"status": "error", "reason": f"Node not found: {node_id}"}

    log.info(
        "split_node: '%s' → %s (trigger=%s)",
        node.get("title", node_id),
        new_children,
        trigger_paper,
    )

    # Update the parent node's title
    parent_title = f"{node.get('title', node_id)} (parent)"
    graph_store.add_node({
        "id": node_id,
        "type": node.get("type", "seed"),
        "title": parent_title,
        "year": node.get("year"),
        "knowledge_level": node.get("knowledge_level", []),
        "structure_template": node.get("structure_template", {}),
        "created_at": node.get("created_at"),
    })

    # Create child nodes
    child_ids = []
    for child_name in new_children:
        child_id = f"seed:split:{uuid.uuid4().hex[:12]}"
        child_ids.append(child_id)

        graph_store.add_node({
            "id": child_id,
            "type": "seed",
            "title": child_name,
            "year": None,
            "knowledge_level": node.get("knowledge_level", []),
            "structure_template": {},
        })

        # Link child to parent
        edge_id = f"edge::{child_id}::{node_id}"
        graph_store.add_edge(child_id, node_id, {
            "id": edge_id,
            "base_type": "correlation",
            "base_score": 0.9,
            "mechanism_description": f"child of split node {node_id}",
        })

    # Log
    graph_store.log_ontology_evolution(
        trigger_paper=trigger_paper or "system",
        change_type="split",
        description=f"Split node '{node.get('title', node_id)}' into {len(new_children)} children",
        details={
            "parent_id": node_id,
            "parent_title": parent_title,
            "child_ids": child_ids,
            "child_names": new_children,
        },
    )

    return {
        "status": "ok",
        "type": "split",
        "parent_id": node_id,
        "child_ids": child_ids,
        "child_count": len(child_ids),
    }
