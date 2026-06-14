"""Seeds package — built-in knowledge foundations.

Provides the initial concepts that bootstrap the knowledge graph:
- L0 Meta (6): strategy, epistemology, meta-learning policies
- L1 Axioms (12): mathematical foundations (ZFC, Peano, Gödel, etc.)
- L2 Math (10): mathematical frameworks and tools
- L3 Methods (13): algorithm core paradigms
- L4 Physics (8): physical constraints and reality layer

Total: 49 pre-built ontology nodes as JSON seed files.
"""

from lcortex.seeds.loader import SeedLoader, OntologyNode, auto_initialize, demo

__all__ = [
    "SeedLoader",
    "OntologyNode",
    "auto_initialize",
    "demo",
]
