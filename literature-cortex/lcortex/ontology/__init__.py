"""
Ontology module — Knowledge graph evolution and compression.

Exports:
    apply_ontology_change  — Apply Double-loop ontology changes from a conflict report
    insert_level           — Insert a new knowledge level node
    reparent_node          — Reparent a node to a new parent
    merge_nodes            — Merge two ontology nodes
    split_node             — Split a node into child nodes
    self_distill           — Self-distillation: compress ontology
"""

from lcortex.ontology.engine import apply_ontology_change
from lcortex.ontology.evolution import (
    insert_level,
    merge_nodes,
    reparent_node,
    split_node,
)
from lcortex.ontology.distillation import self_distill

__all__ = [
    "apply_ontology_change",
    "insert_level",
    "reparent_node",
    "merge_nodes",
    "split_node",
    "self_distill",
]
