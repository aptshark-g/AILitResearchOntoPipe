"""Seed Loader — Load L0-L4 pre-built ontology nodes into the graph store.

All seeds are JSON files under lcortex/seeds/. L5-L6 are generated dynamically
from paper content during Phase F extraction.
"""

import json
import glob
import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class OntologyNode:
    node_id: str
    name: str
    level: int
    category: str
    description: str
    aliases: List[str]
    keywords: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OntologyNode":
        return cls(
            node_id=d["node_id"],
            name=d["name"],
            level=d["level"],
            category=d.get("category", "general"),
            description=d.get("description", ""),
            aliases=d.get("aliases", []),
            keywords=d.get("keywords", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "level": self.level,
            "category": self.category,
            "description": self.description,
            "aliases": self.aliases,
            "keywords": self.keywords,
        }


class SeedLoader:
    """Load all seed JSON files from a directory."""

    SEED_PATTERN = "seed_L*.json"

    @classmethod
    def load_all(cls, seeds_dir: str) -> List[OntologyNode]:
        """Load all L0-L4 seed files. Returns ordered list by level, then node_id."""
        nodes: List[OntologyNode] = []
        pattern = os.path.join(seeds_dir, cls.SEED_PATTERN)

        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    try:
                        nodes.append(OntologyNode.from_dict(item))
                    except KeyError as e:
                        raise ValueError(f"Invalid seed in {path}: missing {e}")

        nodes.sort(key=lambda n: (n.level, n.node_id))
        return nodes

    @classmethod
    def count_by_level(cls, nodes: List[OntologyNode]) -> Dict[int, int]:
        """Return {level: count} for logging."""
        counts: Dict[int, int] = {}
        for n in nodes:
            counts[n.level] = counts.get(n.level, 0) + 1
        return counts


def auto_initialize(graph_store, seeds_dir: str) -> Dict[str, Any]:
    """
    Initialize graph store with seed nodes if empty.
    Returns stats dict: {inserted, skipped, by_level}.
    """
    # Check if graph already has nodes
    existing = graph_store.node_count()
    if existing > 0:
        return {"inserted": 0, "skipped": existing, "reason": "graph_not_empty"}

    nodes = SeedLoader.load_all(seeds_dir)
    stats = {"inserted": 0, "skipped": 0, "by_level": SeedLoader.count_by_level(nodes)}

    for node in nodes:
        graph_store.add_node({
            "id": node.node_id,
            "type": "seed",
            "title": node.name,
            "year": None,
            "knowledge_level": {
                "level": node.level,
                "category": node.category,
                "description": node.description,
                "keywords": node.keywords,
            },
            "structure_template": {},
        })
        stats["inserted"] += 1

    return stats


def demo():
    """Self-test: load seeds and print summary."""
    # Find seeds relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = this_dir  # this file is in lcortex/seeds/ directly

    nodes = SeedLoader.load_all(seeds_dir)
    counts = SeedLoader.count_by_level(nodes)

    print(f"Loaded {len(nodes)} seed nodes")
    print(f"By level: {counts}")
    print()

    # Print first 3 per level
    for level in sorted(counts.keys()):
        level_nodes = [n for n in nodes if n.level == level]
        print(f"L{level} ({counts[level]} nodes):")
        for n in level_nodes[:3]:
            print(f"  {n.node_id}: {n.name}")
        if counts[level] > 3:
            print(f"  ... and {counts[level] - 3} more")
        print()


if __name__ == "__main__":
    demo()
