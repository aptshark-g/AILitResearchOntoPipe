"""
Structure module — Phase F (extraction) and Phase F-2 (divergence).

Exports:
    extract_structure  — Phase F: extract knowledge_level + structure_template
    detect_conflict    — Phase F-2: divergence detection / Double-loop trigger
    structure_similarity — Compute similarity between two structure templates
"""

from lcortex.structure.extractor import extract_structure, structure_similarity
from lcortex.structure.deconstructor import detect_conflict

__all__ = [
    "extract_structure",
    "detect_conflict",
    "structure_similarity",
]
