"""Literature Cortex — Graph Storage Engine.

Provides:
  - GraphStore: SQLite-backed CRUD for nodes, edges, ontology evolution,
    and meta-policy history.
  - Edge management: correlation creation, causal promotion/downgrade,
    and constraint checking.
"""

from lcortex.graph.store import GraphStore
from lcortex.graph import edge

__all__ = ["GraphStore", "edge"]
