"""Literature Cortex — AI-Powered Literature Research Workflow.

A standalone literature cognition system with knowledge graph, structural analogy,
and double-loop learning. LLM is pluggable, domain-agnostic.

Core concepts:
- Phase A: Multi-source paper search (arXiv, OpenAlex, Semantic Scholar)
- Phase B: 4C+L scoring via LLM (Completeness/Correctness/Clarity/Comparison + Limitation)
- Phase C/D: Limitation/Extension paper search
- Phase E: Synthesis (LLM-generated structured review)
- Phase F: Structure template extraction + knowledge level inference
- Phase F-2: Deconstruct-reconstruct-conflict detection (Double-loop)
- Phase G: Dual-linkage knowledge graph generation + Obsidian export + Ontology evolution
"""

__version__ = "0.1.0"
__author__ = "Literature Cortex Contributors"

__all__ = [
    "__version__",
]
