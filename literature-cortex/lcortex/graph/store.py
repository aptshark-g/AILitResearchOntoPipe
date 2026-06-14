"""GraphStore — SQLite-backed knowledge graph for Literature Cortex.

Thread-safe, stdlib-only, single-file database. Stores nodes, edges,
ontology evolution events, and meta-policy snapshots.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any


_SCHEMA_SQL = os.path.join(os.path.dirname(__file__), "schema.sql")


class GraphStore:
    """SQLite-backed storage for the Literature Cortex knowledge graph.

    Provides CRUD for nodes and edges, evolution logging, meta-policy
    history tracking, and basic statistics.

    All JSON fields are stored as TEXT and parsed on read.  Connections
    are per-call for thread safety, with WAL mode enabled at init time.
    """

    def __init__(self, db_path: str):
        """Connect to SQLite database, auto-creating tables from schema.sql.

        Args:
            db_path: Path to the SQLite database file. Parent directories
                     are created if they do not exist.
        """
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)

        self._db_path = os.path.abspath(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a new per-call connection in WAL mode."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Read schema.sql and execute it against a fresh connection."""
        with open(_SCHEMA_SQL, "r") as fh:
            schema = fh.read()
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(schema)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, paper_dict: dict[str, Any]) -> str:
        """Insert or update a node.

        If a node with the same id already exists it is replaced
        (INSERT OR REPLACE).  The caller should pass a dict with at
        least an 'id' key.

        Args:
            paper_dict: Dict with keys matching nodes columns:
                id, type, title, year, knowledge_level,
                structure_template.

        Returns:
            The node id.
        """
        node_id = paper_dict.get("id")
        if not node_id:
            node_id = f"node::{uuid.uuid4().hex[:12]}"

        knowledge_level = paper_dict.get("knowledge_level", [])
        if not isinstance(knowledge_level, str):
            knowledge_level = json.dumps(knowledge_level)

        structure_template = paper_dict.get("structure_template", {})
        if not isinstance(structure_template, str):
            structure_template = json.dumps(structure_template)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO nodes
                       (id, type, title, year, knowledge_level,
                        structure_template, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        node_id,
                        paper_dict.get("type", "paper"),
                        paper_dict.get("title", ""),
                        paper_dict.get("year"),
                        knowledge_level,
                        structure_template,
                        paper_dict.get("created_at", _now()),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return node_id

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve a node by id.

        Returns:
            Node dict with parsed JSON fields, or None if not found.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                if row is None:
                    return None
                return _row_to_dict(row, json_fields=["knowledge_level", "structure_template"])
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str, edge_data: dict[str, Any]) -> str:
        """Insert or replace an edge.

        Args:
            source: Source node id.
            target: Target node id.
            edge_data: Dict with keys matching edges columns (base_type,
                       base_score, causal_promotion, mechanism_description,
                       history).  The 'causal_promotion' and 'history'
                       fields may be passed as dicts or JSON strings.

        Returns:
            The edge id.
        """
        edge_id = edge_data.get("id", f"edge::{source}::{target}")

        causal_promotion = edge_data.get("causal_promotion", {})
        if not isinstance(causal_promotion, str):
            causal_promotion = json.dumps(causal_promotion)

        history = edge_data.get("history", [])
        if not isinstance(history, str):
            history = json.dumps(history)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO edges
                       (id, source_id, target_id, base_type, base_score,
                        causal_promotion, mechanism_description, history,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        edge_id,
                        source,
                        target,
                        edge_data.get("base_type", "correlation"),
                        edge_data.get("base_score", 0.0),
                        causal_promotion,
                        edge_data.get("mechanism_description", ""),
                        history,
                        edge_data.get("created_at", _now()),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return edge_id

    def get_edges(self, node_id: str) -> list[dict[str, Any]]:
        """Retrieve all edges connected to a node (as source or target).

        Args:
            node_id: Node id.

        Returns:
            List of edge dicts with parsed JSON fields.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM edges
                       WHERE source_id = ? OR target_id = ?
                       ORDER BY created_at DESC""",
                    (node_id, node_id),
                ).fetchall()
                return [
                    _row_to_dict(r, json_fields=["causal_promotion", "history"])
                    for r in rows
                ]
            finally:
                conn.close()

    def get_edge_by_id(self, edge_id: str) -> dict[str, Any] | None:
        """Retrieve a single edge by its id."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM edges WHERE id = ?", (edge_id,)
                ).fetchone()
                if row is None:
                    return None
                return _row_to_dict(row, json_fields=["causal_promotion", "history"])
            finally:
                conn.close()

    def update_edge(self, edge_id: str, edge_data: dict[str, Any]) -> bool:
        """Update an existing edge's fields.

        Args:
            edge_id: Edge id.
            edge_data: Dict of fields to update. JSON fields will be
                       serialized if passed as non-string types.

        Returns:
            True if edge was found and updated, False otherwise.
        """
        existing = self.get_edge_by_id(edge_id)
        if existing is None:
            return False

        causal_promotion = edge_data.get("causal_promotion", existing["causal_promotion"])
        if not isinstance(causal_promotion, str):
            causal_promotion = json.dumps(causal_promotion)

        history = edge_data.get("history", existing["history"])
        if not isinstance(history, str):
            history = json.dumps(history)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE edges SET
                       base_type = ?, base_score = ?, causal_promotion = ?,
                       mechanism_description = ?, history = ?
                       WHERE id = ?""",
                    (
                        edge_data.get("base_type", existing["base_type"]),
                        edge_data.get("base_score", existing["base_score"]),
                        causal_promotion,
                        edge_data.get("mechanism_description", existing.get("mechanism_description", "")),
                        history,
                        edge_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return True

    def get_all_causal_edges(self) -> list[dict[str, Any]]:
        """Get all edges where causal_promotion.is_causal is true.

        Since SQLite stores causal_promotion as TEXT, we retrieve all
        edges and filter in Python.

        Returns:
            List of edge dicts that are causal.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE base_type = 'causation' ORDER BY created_at DESC"
                ).fetchall()
                results = []
                for r in rows:
                    edge = _row_to_dict(r, json_fields=["causal_promotion", "history"])
                    # Double-check the JSON flag to be safe
                    cp = edge.get("causal_promotion", {})
                    if isinstance(cp, dict) and cp.get("is_causal"):
                        results.append(edge)
                    elif isinstance(cp, str):
                        try:
                            cp_parsed = json.loads(cp)
                            if cp_parsed.get("is_causal"):
                                edge["causal_promotion"] = cp_parsed
                                results.append(edge)
                        except json.JSONDecodeError:
                            pass
                return results
            finally:
                conn.close()

    def get_all_edges(self) -> list[dict[str, Any]]:
        """Retrieve all edges in the graph."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM edges ORDER BY created_at DESC"
                ).fetchall()
                return [
                    _row_to_dict(r, json_fields=["causal_promotion", "history"])
                    for r in rows
                ]
            finally:
                conn.close()

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Retrieve all nodes in the graph."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM nodes ORDER BY created_at DESC"
                ).fetchall()
                return [
                    _row_to_dict(r, json_fields=["knowledge_level", "structure_template"])
                    for r in rows
                ]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Ontology evolution
    # ------------------------------------------------------------------

    def log_ontology_evolution(
        self,
        trigger_paper: str,
        change_type: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Record an ontology evolution event.

        Args:
            trigger_paper: Paper id that triggered the change.
            change_type: Type of change (reparent | merge | split | rename).
            description: Human-readable description.
            details: Optional JSON-serializable metadata.

        Returns:
            The auto-incremented id of the new record.
        """
        details_json = json.dumps(details if details is not None else {})

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO ontology_evolution
                       (trigger_paper, change_type, description, details)
                       VALUES (?, ?, ?, ?)""",
                    (trigger_paper, change_type, description, details_json),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_ontology_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent ontology evolution events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of evolution event dicts, most recent first.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM ontology_evolution
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                return [_row_to_dict(r, json_fields=["details"]) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Meta policy
    # ------------------------------------------------------------------

    def update_meta_policy(
        self,
        old_policy_json: dict[str, Any],
        new_policy_json: dict[str, Any],
        change_description: str,
    ) -> int:
        """Record a meta-policy change with before/after snapshots.

        Args:
            old_policy_json: Previous policy state.
            new_policy_json: New policy state.
            change_description: Human-readable description of the change.

        Returns:
            The auto-incremented id of the new history record.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO meta_policy_history
                       (change_description, old_policy, new_policy)
                       VALUES (?, ?, ?)""",
                    (
                        change_description,
                        json.dumps(old_policy_json),
                        json.dumps(new_policy_json),
                    ),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def get_meta_policy_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent meta-policy history entries."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM meta_policy_history
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                return [_row_to_dict(r, json_fields=["old_policy", "new_policy"]) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Return the total number of nodes."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
                return row[0]
            finally:
                conn.close()

    def edge_count(self) -> int:
        """Return the total number of edges."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
                return row[0]
            finally:
                conn.close()

    def causal_edge_count(self) -> int:
        """Return the number of causal edges."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE base_type = 'causation'"
                ).fetchone()
                return row[0]
            finally:
                conn.close()

    def stats(self) -> dict[str, int]:
        """Return a summary dict with node_count, edge_count, causal_edges."""
        return {
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "causal_edges": self.causal_edge_count(),
        }

    # ------------------------------------------------------------------
    # Bulk / convenience
    # ------------------------------------------------------------------

    def add_paper(self, paper_id: str, title: str, year: int | None = None) -> str:
        """Convenience: add a paper node with minimal fields."""
        return self.add_node({
            "id": paper_id,
            "type": "paper",
            "title": title,
            "year": year,
        })

    def add_seed(self, seed_id: str, title: str) -> str:
        """Convenience: add a seed node."""
        return self.add_node({
            "id": seed_id,
            "type": "seed",
            "title": title,
        })

    def add_meta_control(self, control_id: str, title: str) -> str:
        """Convenience: add a meta_control node."""
        return self.add_node({
            "id": control_id,
            "type": "meta_control",
            "title": title,
        })


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _now() -> str:
    """Current UTC timestamp as ISO 8601 with TZ."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_to_dict(
    row: sqlite3.Row,
    json_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, parsing named JSON fields.

    Args:
        row: sqlite3.Row object.
        json_fields: List of column names to parse as JSON.

    Returns:
        Dict with all columns, JSON fields parsed.
    """
    d = dict(row)
    for field in (json_fields or []):
        val = d.get(field)
        if isinstance(val, str):
            try:
                d[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass  # Keep as string if not valid JSON
    return d
