"""
Obsidian Vault Exporter — Export the Literature Cortex knowledge graph
as an Obsidian-compatible vault.

Generates:
  - 00-meta/knowledge-tree.md    — Hierarchical ontology with [[wikilinks]]
  - 00-meta/analogy-index.md     — Cross-domain analogy index
  - 00-meta/causal-map.md        — Mermaid flowchart of causal relationships
  - papers/*.md                  — Individual paper notes with YAML frontmatter

Each paper .md has YAML frontmatter with metadata followed by structured
sections (abstract, knowledge level, structure template, related papers).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from lcortex.graph.store import GraphStore

log = logging.getLogger("lcortex.export.obsidian")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VAULT_README = """# Literature Cortex — Obsidian Vault

This vault is auto-generated from the Literature Cortex knowledge graph.
Do not edit files in `00-meta/` or `papers/` directly — they will be
overwritten on the next export.

## Structure

- `00-meta/` — Ontology overview, analogy index, causal map
- `papers/` — Individual paper notes with metadata and connections
- `seeds/` — Foundation knowledge nodes (axioms, mathematics, algorithms)

## Navigation

Open `00-meta/knowledge-tree.md` for the hierarchical ontology view.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_filename(title: str, max_len: int = 80) -> str:
    """Convert a title to a safe filename."""
    # Remove chars that are unsafe for filenames
    safe = re.sub(r'[<>:"/\\|?*\'\.]', "", title)
    safe = re.sub(r"\s+", " ", safe).strip()
    safe = safe.replace(" ", "-")
    safe = re.sub(r"-+", "-", safe)
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("-")
    return safe or "untitled"


def _format_knowledge_level(kl) -> str:
    """Format knowledge level for display."""
    if isinstance(kl, str):
        return kl
    if isinstance(kl, list):
        return ", ".join(str(k) for k in kl)
    return str(kl)


def _yaml_frontmatter(fields: dict) -> str:
    """Generate YAML frontmatter block."""
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f'  - "{item}"')
            else:
                lines.append(f"{key}: {json.dumps(value)}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                if isinstance(v, list):
                    lines.append(f'  {k}: [{", ".join(str(x) for x in v)}]')
                else:
                    lines.append(f"  {k}: {v}")
        elif isinstance(value, str):
            # Escape colons in values
            escaped = value.replace("\n", "\\n")
            if ":" in escaped and not escaped.startswith('"'):
                escaped = f'"{escaped}"'
            lines.append(f"{key}: {escaped}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def generate_vault_from_b1(
    b1_jsonl_path: str | Path,
    output_dir: str | Path,
    query: str = "",
) -> dict:
    """Generate Obsidian vault from Phase B1 dry-scoring data (analysis_b1.jsonl).

    Reads analysis_b1.jsonl and produces real paper notes with scoring info,
    an index page, and summary statistics. Does NOT require a GraphStore.

    Args:
        b1_jsonl_path: Path to analysis_b1.jsonl from Phase B1.
        output_dir: Path to the output vault directory (created if missing).
        query: Original search query for context.

    Returns:
        Export summary dict with vault_path, papers_exported, meta_files, etc.
    """
    import datetime

    vault = Path(output_dir).resolve()
    vault.mkdir(parents=True, exist_ok=True)

    summary = {
        "vault_path": str(vault),
        "papers_exported": 0,
        "meta_files": 0,
        "total_files": 0,
        "errors": [],
        "query": query,
    }

    b1_path = Path(b1_jsonl_path)
    if not b1_path.exists():
        log.warning("B1 JSONL not found: %s — exporting empty vault skeleton", b1_path)
        _write_empty_vault(vault)
        summary["meta_files"] = 3
        summary["total_files"] = 4
        return summary

    # Load B1 scoring data
    scored_papers: list[dict] = []
    with open(b1_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                scored_papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not scored_papers:
        log.info("B1 JSONL is empty — exporting empty vault skeleton")
        _write_empty_vault(vault)
        summary["meta_files"] = 3
        summary["total_files"] = 4
        return summary

    # Sort by dry_score descending
    scored_papers.sort(key=lambda p: p.get("dry_score", 0.0), reverse=True)

    # ── Write vault README ───────────────────────────────────────────
    readme = _VAULT_README + f"\n\n**Query**: {query}\n**Generated**: {datetime.datetime.now().isoformat()}\n"
    try:
        (vault / "README.md").write_text(readme, encoding="utf-8")
    except Exception as exc:
        summary["errors"].append(f"README.md: {exc}")

    # ── 00-meta/ directory ───────────────────────────────────────────
    meta_dir = vault / "00-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Write dry-summary.md
    try:
        _write_dry_summary(meta_dir, scored_papers, query)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"dry-summary.md: {exc}")
        log.warning("Failed to write dry-summary.md: %s", exc)

    # Write knowledge-tree.md (simple version from B1 data)
    try:
        _write_dry_knowledge_tree(meta_dir, scored_papers)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"knowledge-tree.md: {exc}")
        log.warning("Failed to write knowledge-tree.md: %s", exc)

    # ── papers/ directory ────────────────────────────────────────────
    papers_dir = vault / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for paper in scored_papers:
        try:
            _write_paper_from_b1(papers_dir, paper, query)
            summary["papers_exported"] += 1
        except Exception as exc:
            paper_id = paper.get("paper_id", "?")
            summary["errors"].append(f"papers/{paper_id}.md: {exc}")
            log.warning("Failed to write paper note for %s: %s", paper_id, exc)

    # ── Write index.md ───────────────────────────────────────────────
    try:
        _write_papers_index(papers_dir, scored_papers)
    except Exception as exc:
        summary["errors"].append(f"papers/index.md: {exc}")
        log.warning("Failed to write index.md: %s", exc)

    summary["total_files"] = (
        2  # README + index.md
        + summary["meta_files"]
        + summary["papers_exported"]
    )

    log.info(
        "Vault export from B1: %d papers, %d meta, %d total files",
        summary["papers_exported"],
        summary["meta_files"],
        summary["total_files"],
    )

    return summary


def export_vault(
    graph_store: GraphStore,
    output_dir: str,
) -> dict:
    """Export the entire knowledge graph as an Obsidian vault.

    Creates all directories and files. Does NOT overwrite user-editable
    content outside of 00-meta/ and papers/.

    Args:
        graph_store: GraphStore instance with populated knowledge graph.
        output_dir: Path to the output vault directory (created if missing).

    Returns:
        Export summary dict:
        .. code-block:: json

            {
                "vault_path": "/path/to/vault",
                "papers_exported": 15,
                "meta_files": 3,
                "total_files": 18,
                "errors": []
            }
    """
    vault = Path(output_dir).resolve()
    vault.mkdir(parents=True, exist_ok=True)

    summary = {
        "vault_path": str(vault),
        "papers_exported": 0,
        "meta_files": 0,
        "total_files": 0,
        "errors": [],
    }

    # Get all graph data
    all_nodes = graph_store.get_all_nodes()
    all_edges = graph_store.get_all_edges()

    if not all_nodes:
        log.info("Graph store is empty — exporting empty vault skeleton")
        _write_empty_vault(vault)
        summary["meta_files"] = 3
        summary["total_files"] = 4  # README + 3 meta
        return summary

    # Separate nodes by type
    seeds = [n for n in all_nodes if n.get("type") == "seed"]
    papers = [n for n in all_nodes if n.get("type") == "paper"]
    meta_nodes = [n for n in all_nodes if n.get("type") == "meta_control"]

    # ── Write vault README ───────────────────────────────────────────────
    try:
        (vault / "README.md").write_text(_VAULT_README, encoding="utf-8")
    except Exception as exc:
        summary["errors"].append(f"README.md: {exc}")

    # ── 00-meta/ directory ───────────────────────────────────────────────
    meta_dir = vault / "00-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. knowledge-tree.md
    try:
        _write_knowledge_tree(meta_dir, seeds, papers, all_edges)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"knowledge-tree.md: {exc}")

    # 2. analogy-index.md
    try:
        _write_analogy_index(meta_dir, papers, all_edges)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"analogy-index.md: {exc}")

    # 3. causal-map.md
    try:
        _write_causal_map(meta_dir, all_nodes, all_edges)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"causal-map.md: {exc}")

    # ── papers/ directory ────────────────────────────────────────────────
    papers_dir = vault / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        try:
            _write_paper_note(papers_dir, paper, all_nodes, all_edges)
            summary["papers_exported"] += 1
        except Exception as exc:
            summary["errors"].append(
                f"papers/{_sanitize_filename(paper.get('title', '?'))}.md: {exc}"
            )

    summary["total_files"] = (
        1  # README
        + summary["meta_files"]
        + summary["papers_exported"]
    )

    log.info(
        "Vault export complete: %d papers, %d meta, %d total files",
        summary["papers_exported"],
        summary["meta_files"],
        summary["total_files"],
    )

    return summary


# ---------------------------------------------------------------------------
# Empty vault skeleton
# ---------------------------------------------------------------------------


def _write_empty_vault(vault: Path) -> None:
    """Write minimal files for an empty vault."""
    (vault / "README.md").write_text(_VAULT_README, encoding="utf-8")
    meta_dir = vault / "00-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "knowledge-tree.md").write_text(
        "# Knowledge Tree\n\n*No knowledge graph data available.*\n", encoding="utf-8"
    )
    (meta_dir / "analogy-index.md").write_text(
        "# Analogy Index\n\n*No cross-domain analogies available.*\n", encoding="utf-8"
    )
    (meta_dir / "causal-map.md").write_text(
        "# Causal Map\n\n```mermaid\nflowchart TD\n    A[Empty Graph]\n```\n", encoding="utf-8"
    )
    (vault / "papers").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Knowledge tree writer
# ---------------------------------------------------------------------------


def _write_knowledge_tree(
    meta_dir: Path,
    seeds: list[dict],
    papers: list[dict],
    edges: list[dict],
) -> None:
    """Generate 00-meta/knowledge-tree.md with hierarchical ontology.

    Seeds form the backbone structure (L1-L6 levels), papers are linked
    under their knowledge levels with [[wikilinks]].
    """
    lines = [
        "# Knowledge Tree",
        "",
        "> Auto-generated from Literature Cortex knowledge graph.",
        "",
        "## Ontology Levels",
        "",
    ]

    def _parse_levels(kl: any) -> list[str]:
        """Normalize knowledge_level to a list of label strings."""
        if isinstance(kl, str):
            try:
                kl = json.loads(kl)
            except (json.JSONDecodeError, TypeError):
                return [kl]
        if isinstance(kl, dict):
            # Seed format: {"level": 1, "category": "axiom", ...}
            lvl = kl.get("level", "?")
            cat = kl.get("category", "")
            return [f"L{lvl}" if cat else str(lvl)]
        if isinstance(kl, list):
            return [str(x).strip() for x in kl]
        return [str(kl)]

    # Group seeds by knowledge level
    level_seeds: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        kl = seed.get("knowledge_level", [])
        for lvl in _parse_levels(kl):
            level_seeds[lvl].append(seed)

    # Group papers by knowledge level
    level_papers: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        kl = paper.get("knowledge_level", [])
        for lvl in _parse_levels(kl):
            level_papers[lvl].append(paper)

    # Build edge adjacency for links
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.get("source_id", "")].add(edge.get("target_id", ""))
        adjacency[edge.get("target_id", "")].add(edge.get("source_id", ""))

    # Write all levels from L1 to L6 (plus any extras)
    all_level_names = sorted(set(list(level_seeds.keys()) + list(level_papers.keys())))

    if not all_level_names:
        lines.append("*No levels found in the knowledge graph.*")
    else:
        for level_name in all_level_names:
            lines.append(f"### {level_name}")
            lines.append("")

            # Seed nodes
            for seed in sorted(
                level_seeds.get(level_name, []), key=lambda s: s.get("title", "")
            ):
                title = seed.get("title", seed.get("id", "?"))
                seed_id = seed["id"]
                lines.append(f"- 🌱 `{seed_id}` {title}")

            # Paper nodes
            for paper in sorted(
                level_papers.get(level_name, []), key=lambda p: p.get("title", "")
            ):
                title = paper.get("title", paper.get("id", "?"))
                paper_id = paper["id"]
                filename = _sanitize_filename(title)
                wikilink = f"[[{filename}]]"
                lines.append(f"- 📄 {wikilink} ({paper.get('year', '?')})")

            lines.append("")

    # Connectedness overview
    lines.append("## Connectedness")
    lines.append("")
    lines.append(f"- **Total nodes:** {len(seeds) + len(papers)}")
    lines.append(f"- **Total edges:** {len(edges)}")
    lines.append(f"- **Seed nodes:** {len(seeds)}")
    lines.append(f"- **Paper nodes:** {len(papers)}")
    lines.append("")

    (meta_dir / "knowledge-tree.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Analogy index writer
# ---------------------------------------------------------------------------


def _write_analogy_index(
    meta_dir: Path,
    papers: list[dict],
    edges: list[dict],
) -> None:
    """Generate 00-meta/analogy-index.md with cross-domain analogies.

    Groups papers by their domain_abstraction from structure_template
    and creates cross-reference tables.
    """
    lines = [
        "# Analogy Index",
        "",
        "> Cross-domain analogies detected across the literature corpus.",
        "",
        "## Domain Abstractions",
        "",
        "| Domain Abstraction | Papers | Levels |",
        "|-------------------|--------|--------|",
    ]

    # Group by domain_abstraction
    domain_groups: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        st = paper.get("structure_template", {})
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except (json.JSONDecodeError, TypeError):
                st = {}
        da = st.get("domain_abstraction", "") if isinstance(st, dict) else ""
        if da:
            domain_groups[da].append(paper)
        else:
            domain_groups["(unspecified)"].append(paper)

    for domain, group in sorted(domain_groups.items()):
        paper_links = []
        levels = set()
        for p in group[:5]:  # Show up to 5 papers per domain
            title = p.get("title", p.get("id", "?"))
            filename = _sanitize_filename(title)
            paper_links.append(f"[[{filename}]]")
            kl = p.get("knowledge_level", [])
            if isinstance(kl, str):
                try:
                    kl = json.loads(kl)
                except (json.JSONDecodeError, TypeError):
                    kl = [kl]
            for lvl in kl:
                levels.add(str(lvl))

        lines.append(
            f"| {domain} | {', '.join(paper_links)} | {', '.join(sorted(levels)) or '—'} |"
        )

    # If empty
    if len(domain_groups) <= 1 and "(unspecified)" in domain_groups:
        lines.append("| *(none)* | — | — |")

    lines.append("")
    lines.append("## Structural Patterns")
    lines.append("")

    # Group by abstract_pattern
    pattern_groups: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        st = paper.get("structure_template", {})
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except (json.JSONDecodeError, TypeError):
                st = {}
        ap = st.get("abstract_pattern", "") if isinstance(st, dict) else ""
        if ap:
            pattern_groups[ap].append(paper)

    if pattern_groups:
        lines.append("| Abstract Pattern | Papers |")
        lines.append("|-----------------|--------|")
        for pattern, group in sorted(pattern_groups.items(), key=lambda x: -len(x[1])):
            paper_links = []
            for p in group[:5]:
                title = p.get("title", p.get("id", "?"))
                filename = _sanitize_filename(title)
                paper_links.append(f"[[{filename}]]")
            lines.append(f"| {pattern} | {', '.join(paper_links)} |")
    else:
        lines.append("*No structural patterns available.*")

    lines.append("")

    (meta_dir / "analogy-index.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Causal map writer
# ---------------------------------------------------------------------------


def _write_causal_map(
    meta_dir: Path,
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Generate 00-meta/causal-map.md with a Mermaid flowchart.

    Shows causal relationships between nodes as a directed graph.
    """
    lines = [
        "# Causal Map",
        "",
        "> Mermaid flowchart of causal relationships in the knowledge graph.",
        "",
    ]

    # Build node label map
    node_labels: dict[str, str] = {}
    for node in all_nodes:
        node_id = node["id"]
        title = node.get("title", node_id)
        # Truncate long titles for Mermaid
        if len(title) > 40:
            title = title[:37] + "..."
        # Sanitize Mermaid node id (some chars break Mermaid)
        safe_id = _mermaid_safe_id(node_id)
        node_labels[node_id] = (safe_id, title)

    if not all_edges:
        lines.append("```mermaid")
        lines.append("flowchart TD")
        lines.append("    A[No causal edges found]")
        lines.append("```")
        lines.append("")
        (meta_dir / "causal-map.md").write_text("\n".join(lines), encoding="utf-8")
        return

    # Separate causal and correlation edges
    causal_edges = []
    correlation_edges = []
    for edge in all_edges:
        cp = edge.get("causal_promotion", {})
        if isinstance(cp, str):
            try:
                cp = json.loads(cp)
            except (json.JSONDecodeError, TypeError):
                cp = {}
        if cp.get("is_causal") or edge.get("base_type") == "causation":
            causal_edges.append(edge)
        else:
            correlation_edges.append(edge)

    # Write Mermaid flowchart
    lines.append("```mermaid")
    lines.append("flowchart TD")

    # Draw all edges
    drawn_nodes = set()
    for edge in causal_edges + correlation_edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")

        if src not in node_labels or tgt not in node_labels:
            continue

        src_id, src_label = node_labels[src]
        tgt_id, tgt_label = node_labels[tgt]

        # Only define node once
        if src not in drawn_nodes:
            lines.append(f'    {src_id}["{_escape_mermaid(src_label)}"]')
            drawn_nodes.add(src)
        if tgt not in drawn_nodes:
            lines.append(f'    {tgt_id}["{_escape_mermaid(tgt_label)}"]')
            drawn_nodes.add(tgt)

        cp = edge.get("causal_promotion", {})
        if isinstance(cp, str):
            try:
                cp = json.loads(cp)
            except (json.JSONDecodeError, TypeError):
                cp = {}
        is_causal = cp.get("is_causal") or edge.get("base_type") == "causation"

        # Edge style based on type
        base_type = edge.get("base_type", "correlation")
        edge_label = edge.get("mechanism_description", base_type)
        if len(edge_label) > 30:
            edge_label = edge_label[:27] + "..."

        if is_causal:
            lines.append(
                f'    {src_id} -->|"{_escape_mermaid(edge_label)}"| {tgt_id}'
            )
        else:
            lines.append(
                f'    {src_id} -.->|"{_escape_mermaid(edge_label)}"| {tgt_id}'
            )

    # Also draw orphan nodes (no edges)
    for node in all_nodes:
        if node["id"] not in drawn_nodes:
            safe_id, label = node_labels[node["id"]]
            lines.append(f'    {safe_id}["{_escape_mermaid(label)}"]')

    lines.append("```")
    lines.append("")

    # Legend
    lines.append("### Legend")
    lines.append("- **Solid arrow (→)**: Causation edge")
    lines.append("- **Dashed arrow (-·→)**: Correlation edge")
    lines.append("")
    lines.append(f"- Causal edges: {len(causal_edges)}")
    lines.append(f"- Correlation edges: {len(correlation_edges)}")
    lines.append(f"- Total nodes: {len(all_nodes)}")
    lines.append("")

    (meta_dir / "causal-map.md").write_text("\n".join(lines), encoding="utf-8")


def _mermaid_safe_id(node_id: str) -> str:
    """Convert a node ID to a Mermaid-safe identifier."""
    # Replace non-alphanumeric chars with underscores
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    # Ensure it starts with a letter
    if not safe or safe[0].isdigit():
        safe = "n" + safe
    return safe


def _escape_mermaid(text: str) -> str:
    """Escape special characters for Mermaid labels."""
    return text.replace('"', "'").replace("\n", " ").replace("[", "(").replace("]", ")")


# ---------------------------------------------------------------------------
# Paper note writer
# ---------------------------------------------------------------------------


def _write_paper_note(
    papers_dir: Path,
    paper: dict,
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Generate a single paper note with YAML frontmatter and sections.

    Format:
    ---
    id: paper_id
    title: Paper Title
    year: 2024
    type: paper
    knowledge_level: [L3-Algorithm, L4-Physical]
    tags: [keyword1, keyword2]
    ---

    # Paper Title

    ## Abstract
    ...

    ## Knowledge Level
    ...

    ## Structure Template
    ...
    """

    paper_id = paper.get("id", "unknown")
    title = paper.get("title", "Untitled")
    filename = _sanitize_filename(title)

    # Build YAML frontmatter
    kl = paper.get("knowledge_level", [])
    if isinstance(kl, str):
        try:
            kl = json.loads(kl)
        except (json.JSONDecodeError, TypeError):
            kl = [kl]

    st = paper.get("structure_template", {})
    if isinstance(st, str):
        try:
            st = json.loads(st)
        except (json.JSONDecodeError, TypeError):
            st = {}

    # Build tags from structure
    tags = []
    if kl:
        for lvl in kl:
            tags.append(str(lvl))
    arch = st.get("control_architecture", "") if isinstance(st, dict) else ""
    if arch:
        tags.append(arch)

    frontmatter = _yaml_frontmatter({
        "id": paper_id,
        "title": title,
        "year": paper.get("year"),
        "type": paper.get("type", "paper"),
        "knowledge_level": kl,
        "tags": tags,
    })

    # ── Body sections ────────────────────────────────────────────────────
    sections = []
    sections.append(frontmatter)
    sections.append("")
    sections.append(f"# {title}")
    sections.append("")

    # Abstract
    abstract = paper.get("abstract", paper.get("abstract_text", ""))
    sections.append("## Abstract")
    sections.append("")
    sections.append(abstract if abstract else "*No abstract available.*")
    sections.append("")

    # Knowledge Level
    sections.append("## Knowledge Level")
    sections.append("")
    if kl:
        sections.append(f"- {', '.join(kl)}")
    else:
        sections.append("*Not classified.*")
    sections.append("")

    # Structure Template
    sections.append("## Structure Template")
    sections.append("")

    if isinstance(st, dict) and st:
        signal_chain = st.get("signal_chain", [])
        if signal_chain:
            sections.append("**Signal Chain:**")
            sections.append("")
            for step in signal_chain:
                sections.append(f"- {step}")
            sections.append("")

        arch = st.get("control_architecture", "")
        if arch:
            sections.append(f"**Control Architecture:** {arch}")
            sections.append("")

        opt_target = st.get("optimization_target", "")
        if opt_target:
            sections.append(f"**Optimization Target:** {opt_target}")
            sections.append("")

        constraints = st.get("constraint_type", [])
        if constraints:
            sections.append(f"**Constraints:** {', '.join(constraints)}")
            sections.append("")

        abstract_pattern = st.get("abstract_pattern", "")
        if abstract_pattern:
            sections.append(f"**Abstract Pattern:** {abstract_pattern}")
            sections.append("")

        math_core = st.get("mathematical_core", "")
        if math_core:
            sections.append(f"**Mathematical Core:** {math_core}")
            sections.append("")

        domain_abstraction = st.get("domain_abstraction", "")
        if domain_abstraction:
            sections.append(f"**Domain Abstraction:** {domain_abstraction}")
            sections.append("")
    else:
        sections.append("*No structure template extracted.*")
        sections.append("")

    # Related papers (from edges)
    related = []
    for edge in all_edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        related_id = None
        if src == paper_id:
            related_id = tgt
        elif tgt == paper_id:
            related_id = src
        if related_id:
            for node in all_nodes:
                if node["id"] == related_id:
                    edge_type = edge.get("base_type", "correlation")
                    mechanism = edge.get("mechanism_description", "")
                    related.append((node, edge_type, mechanism))
                    break

    if related:
        sections.append("## Related Papers")
        sections.append("")
        for rel_node, etype, mech in related[:15]:
            rel_title = rel_node.get("title", rel_node.get("id", "?"))
            rel_filename = _sanitize_filename(rel_title)
            extra = f" ({mech})" if mech else ""
            sections.append(f"- [[{rel_filename}]] — `{etype}`{extra}")
        sections.append("")

    # Write the file
    content = "\n".join(sections)
    (papers_dir / f"{filename}.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# B1-based vault generators (real content from analysis_b1.jsonl)
# ---------------------------------------------------------------------------


def _write_paper_from_b1(
    papers_dir: Path,
    paper: dict,
    query: str = "",
) -> None:
    """Generate a single paper .md from B1 dry-scoring data.

    Frontmatter includes dry_score, intent_words, domain.
    Body includes abstract, B1 analysis scores, and links.
    """
    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    title = paper.get("title", "Untitled")
    year = paper.get("year", 0)
    dry_score = paper.get("dry_score", 0)
    dry_detail = paper.get("dry_detail", {})
    reason = paper.get("reason", "")
    abstract = paper.get("abstract", "")
    passed = paper.get("passed", False)

    # Extract intent words from reason field
    intent_words: list[str] = []
    if reason:
        for part in reason.split(";"):
            part = part.strip()
            if part.startswith("intent_in_title:"):
                val = part.split(":", 1)[1] if ":" in part else ""
                if val and val != "0":
                    intent_words.append(f"intent_title_count={val}")
            if part.startswith("intent_in_abstract:"):
                val = part.split(":", 1)[1] if ":" in part else ""
                if val and val != "0":
                    intent_words.append(f"intent_abstract_count={val}")

    # Extract domain from reason
    domain = "unknown"
    domain_reason = ""
    if reason:
        for part in reason.split(";"):
            part = part.strip()
            if part.startswith("neighbor_domain"):
                domain = "neighbor_domain"
                domain_reason = part
            elif part.startswith("no_domain_word"):
                domain = "no_domain"
                domain_reason = part
            elif part.startswith("exclude_words:"):
                domain_reason = part

    bm25 = dry_detail.get("bm25", 0)
    intent = dry_detail.get("intent", 0)
    bg_match = dry_detail.get("bg_match", 0)
    recency = dry_detail.get("recency", 0)
    impact = dry_detail.get("impact", 0)

    # Determine filename: use paper_id for uniqueness
    filename = paper_id.replace("/", "-").replace("\\", "-")

    # Build frontmatter
    fm_fields = {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "dry_score": dry_score,
        "passed": passed,
    }
    if intent_words:
        fm_fields["intent_words"] = intent_words
    if domain:
        fm_fields["domain"] = domain
    fm_fields["source"] = paper.get("source", "arxiv")
    fm_fields["citations"] = paper.get("citations", 0)

    frontmatter = _yaml_frontmatter(fm_fields)

    # ── Body ──
    escaped_title = title.replace("`", "'")
    sections = [
        frontmatter,
        "",
        f"# {escaped_title}",
        "",
        f"**Year**: {year} | **Dry Score**: {dry_score:.3f}"
        f" | **BM25**: {bm25:.3f}"
        f" | **Intent**: {intent:.3f}",
        f"**Verdict**: {'✅ Passed' if passed else '⚠️ Below Threshold'}",
        "",
        "## Abstract",
        "",
        abstract if abstract else "*No abstract available.*",
        "",
        "## B1 Analysis",
        "",
        f"- **BM25 Relevance**: {bm25:.3f}",
        f"- **Intent Match**: {intent:.3f}",
        f"- **Background Match**: {bg_match:.3f}",
        f"- **Domain**: {domain_reason if domain_reason else 'standard'}",
        f"- **Recency Boost**: {recency:.3f}",
        f"- **Citation Impact**: {impact:.3f}",
        "",
        f"**Raw Reason**: {reason}",
        "",
        "## Links",
        "",
        f"- [[../00-meta/dry-summary|Dry Score Summary]]",
        f"- [[index|Paper Index]]",
        "",
    ]

    # ArXiv link if available
    arxiv_id = paper.get("arxiv_id", "")
    if arxiv_id:
        sections.insert(-2, f"- [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")

    content = "\n".join(sections)
    (papers_dir / f"{filename}.md").write_text(content, encoding="utf-8")


def _write_dry_summary(
    meta_dir: Path,
    scored_papers: list[dict],
    query: str = "",
) -> None:
    """Generate 00-meta/dry-summary.md with statistics and top-10 ranking."""
    total = len(scored_papers)
    passed_count = sum(1 for p in scored_papers if p.get("passed", False))
    pass_rate = passed_count / max(total, 1)
    avg_score = sum(p.get("dry_score", 0) for p in scored_papers) / max(total, 1)

    # Score distribution
    score_bins = {"0.0-0.2": 0, "0.2-0.3": 0, "0.3-0.4": 0, "0.4-0.5": 0,
                   "0.5-0.6": 0, "0.6-0.7": 0, "0.7-1.0": 0}
    for p in scored_papers:
        s = p.get("dry_score", 0)
        if s < 0.2:
            score_bins["0.0-0.2"] += 1
        elif s < 0.3:
            score_bins["0.2-0.3"] += 1
        elif s < 0.4:
            score_bins["0.3-0.4"] += 1
        elif s < 0.5:
            score_bins["0.4-0.5"] += 1
        elif s < 0.6:
            score_bins["0.5-0.6"] += 1
        elif s < 0.7:
            score_bins["0.6-0.7"] += 1
        else:
            score_bins["0.7-1.0"] += 1

    lines = [
        "# Dry Score Summary",
        "",
        f"> Auto-generated from B1 NLP dry scoring.  Query: `{query}`",
        "",
        "## Statistics",
        "",
        f"- **Total Papers**: {total}",
        f"- **Passed**: {passed_count} ({pass_rate:.1%})",
        f"- **Average Score**: {avg_score:.3f}",
        f"- **Threshold**: 0.25",
        "",
        "## Score Distribution",
        "",
        "| Range | Count | Percent |",
        "|-------|-------|---------|",
    ]

    for range_label, count in score_bins.items():
        pct = count / max(total, 1) * 100
        bar = "█" * max(1, int(pct / 2))
        lines.append(f"| {range_label} | {count} ({bar}) | {pct:.1f}% |")

    lines.extend([
        "",
        "## Top 10 Papers",
        "",
        "| Rank | Score | Year | Title |",
        "|------|-------|------|-------|",
    ])

    for i, p in enumerate(scored_papers[:10], 1):
        score = p.get("dry_score", 0)
        year = p.get("year", "?")
        title = (p.get("title", "?") or "?")[:60]
        pid = p.get("paper_id", "?")
        escaped_title = title.replace("|", "\\|")
        lines.append(f"| {i} | {score:.3f} | {year} | [[../papers/{pid}|{escaped_title}]] |")

    lines.append("")

    (meta_dir / "dry-summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_dry_knowledge_tree(
    meta_dir: Path,
    scored_papers: list[dict],
) -> None:
    """Generate 00-meta/knowledge-tree.md from B1 data (simplified)."""
    lines = [
        "# Knowledge Tree",
        "",
        "> Simplified from Phase B1 dry scoring. Full ontology requires Phase F extraction.",
        "",
        "## Scored Papers",
        "",
    ]

    # Group by domain (from reason field)
    passed = [p for p in scored_papers if p.get("passed", False)]
    below = [p for p in scored_papers if not p.get("passed", False)]

    lines.append(f"### Passed ({len(passed)} papers)")
    lines.append("")
    for p in passed:
        pid = p.get("paper_id", "?")
        title = p.get("title", pid)
        score = p.get("dry_score", 0)
        year = p.get("year", "?")
        lines.append(f"- 📄 [[../papers/{pid}|{title}]] ({year}) — {score:.3f}")

    lines.append("")
    lines.append(f"### Below Threshold ({len(below)} papers)")
    lines.append("")
    for p in below[:10]:  # Show max 10 below-threshold
        pid = p.get("paper_id", "?")
        title = p.get("title", pid)
        score = p.get("dry_score", 0)
        reason = p.get("reason", "no_match")
        lines.append(f"- 📄 [[../papers/{pid}|{title}]] — {score:.3f} ({reason[:50]})")

    lines.append("")
    (meta_dir / "knowledge-tree.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# GraphStore → graph.json export
# ---------------------------------------------------------------------------


def export_graph_json(store: GraphStore, output_path: Path) -> dict:
    """Export GraphStore contents to a single graph.json file.

    Reads ALL nodes and edges from a GraphStore instance, parses JSON fields
    (causal_promotion, history, knowledge_level, structure_template) from
    their TEXT storage, and writes a structured JSON file.

    Args:
        store: Initialized GraphStore instance with data.
        output_path: Path to write graph.json.

    Returns:
        The graph dict that was written to disk.
    """
    all_nodes = store.get_all_nodes()
    all_edges = store.get_all_edges()
    ontology_evolution = store.get_ontology_history(limit=100)

    # Count stats
    papers = [n for n in all_nodes if n.get("type") == "paper"]
    causal_count = sum(
        1 for e in all_edges
        if e.get("base_type") == "causation"
        or (isinstance(e.get("causal_promotion"), dict) and e["causal_promotion"].get("is_causal"))
    )

    stats = {
        "nodes": len(all_nodes),
        "edges": len(all_edges),
        "causal_edges": causal_count,
        "papers": len(papers),
    }

    tz = datetime.timezone(datetime.timedelta(hours=8))
    exported_at = datetime.datetime.now(tz).isoformat()

    graph = {
        "nodes": all_nodes,
        "edges": all_edges,
        "ontology_evolution": ontology_evolution,
        "exported_at": exported_at,
        "stats": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("graph.json exported: %d nodes, %d edges → %s", stats["nodes"], stats["edges"], output_path)
    return graph


# ---------------------------------------------------------------------------
# GraphStore → Obsidian vault export
# ---------------------------------------------------------------------------


def generate_vault_from_graphstore(
    store: GraphStore,
    output_dir: Path,
    query: str = "",
) -> dict:
    """Generate Obsidian vault from GraphStore (SQLite).

    For each paper node (type='paper'):
      - Generates vault/papers/{paper_id}.md with YAML frontmatter,
        B1/B2 scores (from connected score nodes), abstract, and
        wikilinks to related papers.
      - Marks causal edges with "🔗 Causation".

    Generates meta files:
      - vault/papers/index.md — ranked table of all papers with scores
      - vault/00-meta/knowledge-tree.md — papers grouped by year/domain
      - vault/00-meta/causal-map.md — Mermaid flowchart of all edges
      - vault/graph.json — complete serialized graph (via export_graph_json)

    Args:
        store: GraphStore instance with populated knowledge graph.
        output_dir: Path to output vault directory (created if missing).
        query: Original search query for context.

    Returns:
        Export summary dict with vault_path, papers_exported, meta_files, etc.
    """
    vault = Path(output_dir).resolve()
    vault.mkdir(parents=True, exist_ok=True)

    summary = {
        "vault_path": str(vault),
        "papers_exported": 0,
        "meta_files": 0,
        "total_files": 0,
        "errors": [],
        "query": query,
    }

    # Get all data from store
    all_nodes = store.get_all_nodes()
    all_edges = store.get_all_edges()

    if not all_nodes:
        log.info("Graph store is empty — exporting empty vault skeleton")
        _write_empty_vault(vault)
        summary["meta_files"] = 3
        summary["total_files"] = 4
        return summary

    # Separate by type
    papers = [n for n in all_nodes if n.get("type") == "paper"]
    score_nodes = {n["id"]: n for n in all_nodes if n.get("type") == "score"}

    # Build edge index: paper_id → list of (connected_node, edge)
    paper_score_edges: dict[str, dict[str, dict]] = {}  # paper_id → {score_node_id: edge}
    paper_paper_edges: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

    for edge in all_edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        # Paper → score edges
        if tgt in score_nodes and src not in score_nodes:
            paper_score_edges.setdefault(src, {})[tgt] = edge
        elif src in score_nodes and tgt not in score_nodes:
            paper_score_edges.setdefault(tgt, {})[src] = edge
        # Paper ↔ paper edges (both directions)
        if src not in score_nodes and tgt not in score_nodes:
            # Find the other paper
            for node in all_nodes:
                if node["id"] == tgt and node.get("type") == "paper":
                    paper_paper_edges[src].append((node, edge))
                    break
            for node in all_nodes:
                if node["id"] == src and node.get("type") == "paper":
                    paper_paper_edges[tgt].append((node, edge))
                    break

    # ── Write vault README ───────────────────────────────────────────
    tz = datetime.timezone(datetime.timedelta(hours=8))
    now_str = datetime.datetime.now(tz).isoformat()
    readme = _VAULT_README + f"\n\n**Query**: {query}\n**Generated**: {now_str}\n"
    try:
        (vault / "README.md").write_text(readme, encoding="utf-8")
    except Exception as exc:
        summary["errors"].append(f"README.md: {exc}")

    # ── papers/ directory ────────────────────────────────────────────
    papers_dir = vault / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        try:
            _write_paper_from_graphstore(
                papers_dir, paper, score_nodes, paper_score_edges,
                paper_paper_edges, all_nodes
            )
            summary["papers_exported"] += 1
        except Exception as exc:
            paper_id = paper.get("id", "?")
            summary["errors"].append(f"papers/{paper_id}.md: {exc}")
            log.warning("Failed to write paper note for %s: %s", paper_id, exc)

    # ── papers/index.md ──────────────────────────────────────────────
    try:
        _write_graphstore_papers_index(papers_dir, papers, paper_score_edges, score_nodes)
    except Exception as exc:
        summary["errors"].append(f"papers/index.md: {exc}")
        log.warning("Failed to write index.md: %s", exc)

    # ── 00-meta/ directory ───────────────────────────────────────────
    meta_dir = vault / "00-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    try:
        seeds = [n for n in all_nodes if n.get("type") == "seed"]
        summary["seed_nodes"] = len(seeds)
        _write_graphstore_knowledge_tree(meta_dir, papers, paper_score_edges, score_nodes, seeds)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"knowledge-tree.md: {exc}")
        log.warning("Failed to write knowledge-tree.md: %s", exc)

    try:
        _write_graphstore_causal_map(meta_dir, all_nodes, all_edges)
        summary["meta_files"] += 1
    except Exception as exc:
        summary["errors"].append(f"causal-map.md: {exc}")
        log.warning("Failed to write causal-map.md: %s", exc)

    # ── Dual-linkage transfer indices ────────────────────────────
    try:
        from lcortex.export.transfer_index import write_transfer_indices
        tx_stats = write_transfer_indices(meta_dir, papers, all_nodes, all_edges)
        summary["meta_files"] += 2
        summary["transfer"] = tx_stats
        log.info("Transfer indices: near=%d clusters, far=%d pattern/%d math/%d arch, cross-domain=%d",
                 tx_stats["near_clusters"], tx_stats["far_pattern_clusters"],
                 tx_stats["far_math_clusters"], tx_stats["far_arch_clusters"],
                 tx_stats["cross_domain_groups"])
    except Exception as exc:
        summary["errors"].append(f"transfer-indices: {exc}")
        log.warning("Failed to write transfer indices: %s", exc)

    # ── vault/graph.json ─────────────────────────────────────────────
    try:
        export_graph_json(store, vault / "graph.json")
    except Exception as exc:
        summary["errors"].append(f"graph.json: {exc}")
        log.warning("Failed to write graph.json: %s", exc)

    summary["total_files"] = (
        2  # README + graph.json
        + summary["meta_files"]
        + summary["papers_exported"]
        + 1  # index.md
    )

    log.info(
        "GraphStore vault export: %d papers, %d meta, %d total files",
        summary["papers_exported"],
        summary["meta_files"],
        summary["total_files"],
    )

    return summary


def _write_paper_from_graphstore(
    papers_dir: Path,
    paper: dict,
    score_nodes: dict[str, dict],
    paper_score_edges: dict[str, dict[str, dict]],
    paper_paper_edges: dict[str, list[tuple[dict, dict]]],
    all_nodes: list[dict],
) -> None:
    """Generate a single paper .md from GraphStore data."""
    paper_id = paper.get("id", "unknown")
    title = paper.get("title", "Untitled")
    year = paper.get("year", 0)

    # Parse JSON fields from store
    kl = paper.get("knowledge_level", [])
    if isinstance(kl, str):
        try:
            kl = json.loads(kl)
        except (json.JSONDecodeError, TypeError):
            kl = []

    st = paper.get("structure_template", {})
    if isinstance(st, str):
        try:
            st = json.loads(st)
        except (json.JSONDecodeError, TypeError):
            st = {}

    # Get score edges for this paper
    scores = paper_score_edges.get(paper_id, {})
    b1_score = None
    b2_score = None
    for sid, edge in scores.items():
        sn = score_nodes.get(sid, {})
        stitle = sn.get("title", "")
        score_val = edge.get("base_score", 0)
        if "B2" in stitle or "b2" in sid.lower():
            b2_score = (score_val, edge.get("mechanism_description", ""))
        else:
            b1_score = (score_val, edge.get("mechanism_description", ""))

    # Build tags
    tags = []
    if kl:
        for lvl in kl:
            tags.append(str(lvl))
    arch = st.get("control_architecture", "") if isinstance(st, dict) else ""
    if arch:
        tags.append(arch)

    # YAML frontmatter
    fm_fields: dict[str, Any] = {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "type": paper.get("type", "paper"),
        "knowledge_level": kl,
        "tags": tags,
    }
    if b1_score:
        fm_fields["b1_score"] = round(b1_score[0], 4)
    if b2_score:
        fm_fields["b2_score"] = round(b2_score[0], 4)

    frontmatter = _yaml_frontmatter(fm_fields)

    # ── Body ──
    sections = [
        frontmatter,
        "",
        f"# {title}",
        "",
        f"**Year**: {year}",
    ]
    if b1_score:
        sections.append(f"**B1 Score**: {b1_score[0]:.4f} ({b1_score[1]})")
    if b2_score:
        sections.append(f"**B2 Score**: {b2_score[0]:.4f} ({b2_score[1]})")
    sections.append("")

    # Abstract
    abstract = paper.get("abstract", "")
    sections.append("## Abstract")
    sections.append("")
    sections.append(abstract if abstract else "*No abstract available.*")
    sections.append("")

    # Knowledge Level
    sections.append("## Knowledge Level")
    sections.append("")
    if kl:
        sections.append(f"- {', '.join(kl)}")
    else:
        sections.append("*Not classified.*")
    sections.append("")

    # Structure Template
    sections.append("## Structure Template")
    sections.append("")
    if isinstance(st, dict) and st:
        signal_chain = st.get("signal_chain", [])
        if signal_chain:
            sections.append("**Signal Chain:**")
            for step in signal_chain:
                sections.append(f"- {step}")
            sections.append("")
        arch_val = st.get("control_architecture", "")
        if arch_val:
            sections.append(f"**Control Architecture:** {arch_val}")
            sections.append("")
        opt_target = st.get("optimization_target", "")
        if opt_target:
            sections.append(f"**Optimization Target:** {opt_target}")
            sections.append("")
        constraints = st.get("constraint_type", [])
        if constraints:
            sections.append(f"**Constraints:** {', '.join(constraints)}")
            sections.append("")
        abstract_pattern = st.get("abstract_pattern", "")
        if abstract_pattern:
            sections.append(f"**Abstract Pattern:** {abstract_pattern}")
            sections.append("")
        math_core = st.get("mathematical_core", "")
        if math_core:
            sections.append(f"**Mathematical Core:** {math_core}")
            sections.append("")
        domain_abstraction = st.get("domain_abstraction", "")
        if domain_abstraction:
            sections.append(f"**Domain Abstraction:** {domain_abstraction}")
            sections.append("")
    else:
        sections.append("*No structure template extracted.*")
        sections.append("")

    # Related papers (wikilinks)
    related = paper_paper_edges.get(paper_id, [])
    if related:
        sections.append("## Related Papers")
        sections.append("")
        for rel_node, edge in related:
            rel_title = rel_node.get("title", rel_node.get("id", "?"))
            rel_id = rel_node["id"]
            etype = edge.get("base_type", "correlation")
            mech = edge.get("mechanism_description", "")
            cp = edge.get("causal_promotion", {})
            if isinstance(cp, str):
                try:
                    cp = json.loads(cp)
                except (json.JSONDecodeError, TypeError):
                    cp = {}
            is_causal = cp.get("is_causal") or etype == "causation"
            prefix = "🔗 Causation" if is_causal else f"`{etype}`"
            extra = f" ({mech})" if mech else ""
            sections.append(f"- [[{rel_id}]] — {prefix}{extra}")
        sections.append("")

    # Score nodes detail
    if scores:
        sections.append("## Scoring")
        sections.append("")
        for sid, edge in sorted(scores.items(), key=lambda x: -x[1].get("base_score", 0)):
            sn = score_nodes.get(sid, {})
            stitle = sn.get("title", sid)
            score = edge.get("base_score", 0)
            mech = edge.get("mechanism_description", "")
            sections.append(f"- **{stitle}**: {score:.4f} ({mech})")
        sections.append("")

    content = "\n".join(sections)
    (papers_dir / f"{paper_id}.md").write_text(content, encoding="utf-8")


def _write_graphstore_papers_index(
    papers_dir: Path,
    papers: list[dict],
    paper_score_edges: dict[str, dict[str, dict]],
    score_nodes: dict[str, dict],
) -> None:
    """Generate vault/papers/index.md — ranked table of all papers with scores."""
    # Compute ranking: primary by B2 max then B1 max
    ranked: list[tuple[dict, float, float]] = []
    for paper in papers:
        pid = paper["id"]
        scores = paper_score_edges.get(pid, {})
        b1 = 0.0
        b2 = 0.0
        for sid, edge in scores.items():
            sn = score_nodes.get(sid, {})
            stitle = sn.get("title", "")
            sv = edge.get("base_score", 0)
            if "B2" in stitle or "b2" in sid.lower():
                b2 = max(b2, sv)
            else:
                b1 = max(b1, sv)
        ranked.append((paper, b1, b2))

    ranked.sort(key=lambda x: (-x[2], -x[1], x[0].get("title", "")))

    lines = [
        "# Paper Index",
        "",
        "> Ranked list of all papers with scores from GraphStore.",
        "",
        "| Rank | Paper ID | Year | B1 Score | B2 Score | Title |",
        "|------|----------|------|----------|----------|-------|",
    ]

    for i, (paper, b1, b2) in enumerate(ranked, 1):
        pid = paper["id"]
        year = paper.get("year", "?")
        title = (paper.get("title", "?") or "?")[:50]
        escaped_title = title.replace("|", "\\|")
        b1_str = f"{b1:.4f}" if b1 else "—"
        b2_str = f"{b2:.4f}" if b2 else "—"
        lines.append(f"| {i} | [[{pid}]] | {year} | {b1_str} | {b2_str} | {escaped_title} |")

    lines.append("")
    lines.append(f"*{len(papers)} papers total*")
    lines.append("")

    (papers_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _write_graphstore_knowledge_tree(
    meta_dir: Path,
    papers: list[dict],
    paper_score_edges: dict[str, dict[str, dict]],
    score_nodes: dict[str, dict],
    seeds: list[dict] = None,
) -> None:
    """Generate 00-meta/knowledge-tree.md from GraphStore data."""
    seeds2 = seeds or []
    lines = [
        "# Knowledge Tree",
        "",
        "> Auto-generated from Literature Cortex GraphStore.",
        "",
    ]

    # ── Ontology seeds ───────────────────────────────────────────
    if seeds2:
        lines.append("## Ontology Levels")
        lines.append("")
        by_level: defaultdict[int, list[dict]] = defaultdict(list)
        for seed in seeds2:
            kl = seed.get("knowledge_level", {})
            if isinstance(kl, str):
                try:
                    kl = json.loads(kl)
                except (json.JSONDecodeError, TypeError):
                    kl = {}
            lvl = kl.get("level", "?") if isinstance(kl, dict) else "?"
            by_level[lvl].append(seed)

        for lvl in sorted(by_level.keys()):
            level_seeds = sorted(by_level[lvl], key=lambda s: s.get("title", ""))
            cat = level_seeds[0].get("knowledge_level", {})
            if isinstance(cat, str):
                try:
                    cat = json.loads(cat)
                except (json.JSONDecodeError, TypeError):
                    cat = {}
            cat_name = cat.get("category", "") if isinstance(cat, dict) else ""
            label = f"L{lvl}" if isinstance(lvl, int) else str(lvl)
            if cat_name:
                label += f" — {cat_name}"
            lines.append(f"### {label} ({len(level_seeds)} nodes)")
            lines.append("")
            for s in level_seeds[:8]:
                title = s.get("title", s.get("id", "?"))
                lines.append(f"- 🌱 `{s['id']}` {title}")
            if len(level_seeds) > 8:
                lines.append(f"- ... and {len(level_seeds) - 8} more")
            lines.append("")

    # ── Papers by Year ───────────────────────────────────────────
    lines.append("## Papers by Year")
    lines.append("")

    # Group by year
    by_year: dict[int, list[dict]] = defaultdict(list)
    for paper in papers:
        year = paper.get("year", 0) or 0
        by_year[year].append(paper)

    for year in sorted(by_year.keys(), reverse=True):
        year_papers = by_year[year]
        lines.append(f"### {year} ({len(year_papers)} papers)")
        lines.append("")
        for paper in sorted(year_papers, key=lambda p: p.get("title", "")):
            pid = paper["id"]
            title = paper.get("title", pid)
            scores = paper_score_edges.get(pid, {})
            b1 = 0.0
            for sid, edge in scores.items():
                sn = score_nodes.get(sid, {})
                if "B2" not in sn.get("title", "") and "b2" not in sid.lower():
                    b1 = max(b1, edge.get("base_score", 0))
            b1_str = f" — B1:{b1:.3f}" if b1 else ""
            lines.append(f"- 📄 [[../papers/{pid}|{title}]]{b1_str}")
        lines.append("")

    # Group by domain from knowledge_level
    by_kl: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        kl = paper.get("knowledge_level", [])
        if isinstance(kl, str):
            try:
                kl = json.loads(kl)
            except (json.JSONDecodeError, TypeError):
                kl = []
        if kl:
            for lvl in kl:
                by_kl[str(lvl)].append(paper)
        else:
            by_kl["(unclassified)"].append(paper)

    if len(by_kl) > 1 or "(unclassified)" not in by_kl:
        lines.append("## By Knowledge Level")
        lines.append("")
        for lvl in sorted(by_kl.keys()):
            lvl_papers = by_kl[lvl]
            lines.append(f"### {lvl} ({len(lvl_papers)} papers)")
            lines.append("")
            for paper in sorted(lvl_papers, key=lambda p: p.get("title", "")):
                pid = paper["id"]
                title = paper.get("title", pid)
                lines.append(f"- 📄 [[../papers/{pid}|{title}]]")
            lines.append("")

    lines.append("## Stats")
    lines.append("")
    lines.append(f"- **Total papers:** {len(papers)}")
    lines.append(f"- **Seed nodes:** {len(seeds2)}")
    lines.append(f"- **Total nodes:** {len(papers) + len(seeds2)}")
    lines.append("")

    (meta_dir / "knowledge-tree.md").write_text("\n".join(lines), encoding="utf-8")


def _write_graphstore_causal_map(
    meta_dir: Path,
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Generate 00-meta/causal-map.md with Mermaid flowchart from GraphStore.

    Causal edges: solid arrow with label.
    Correlation edges with score > 0.3: dashed arrow.
    Nodes grouped by type (paper vs score).
    """
    lines = [
        "# Causal Map",
        "",
        "> Mermaid flowchart of edges in the knowledge graph.",
        "",
    ]

    # Build node label map
    node_labels: dict[str, tuple[str, str, str]] = {}  # id → (safe_id, label, type)
    for node in all_nodes:
        node_id = node["id"]
        node_type = node.get("type", "paper")
        title = node.get("title", node_id)
        if len(title) > 40:
            title = title[:37] + "..."
        safe_id = _mermaid_safe_id(node_id)
        node_labels[node_id] = (safe_id, title, node_type)

    if not all_edges:
        lines.append("```mermaid")
        lines.append("flowchart TD")
        lines.append("    A[No edges found]")
        lines.append("```")
        lines.append("")
        (meta_dir / "causal-map.md").write_text("\n".join(lines), encoding="utf-8")
        return

    # Separate edges
    causal_edges = []
    strong_correlation = []
    weak_correlation = []
    for edge in all_edges:
        cp = edge.get("causal_promotion", {})
        if isinstance(cp, str):
            try:
                cp = json.loads(cp)
            except (json.JSONDecodeError, TypeError):
                cp = {}
        is_causal = cp.get("is_causal") or edge.get("base_type") == "causation"
        if is_causal:
            causal_edges.append(edge)
        elif edge.get("base_score", 0) > 0.3:
            strong_correlation.append(edge)
        else:
            weak_correlation.append(edge)

    # Write Mermaid
    lines.append("```mermaid")
    lines.append("flowchart TD")

    # Group nodes by type
    lines.append("    subgraph papers [Papers]")
    drawn_nodes = set()
    for node in all_nodes:
        if node.get("type") == "paper":
            safe_id, label, _ = node_labels[node["id"]]
            lines.append(f'        {safe_id}["{_escape_mermaid(label)}"]')
            drawn_nodes.add(node["id"])
    lines.append("    end")
    lines.append("")
    lines.append("    subgraph scores [Scores]")
    for node in all_nodes:
        if node.get("type") == "score":
            safe_id, label, _ = node_labels[node["id"]]
            score_label = label[:30]
            lines.append(f'        {safe_id}("{_escape_mermaid(score_label)}")')
            drawn_nodes.add(node["id"])
    lines.append("    end")
    lines.append("")

    # Draw causal edges (solid)
    for edge in causal_edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        if src not in node_labels or tgt not in node_labels:
            continue
        src_id, _, _ = node_labels[src]
        tgt_id, _, _ = node_labels[tgt]
        edge_label = edge.get("mechanism_description", "causal")
        if len(edge_label) > 25:
            edge_label = edge_label[:22] + "..."
        lines.append(f'    {src_id} ==>|\"{_escape_mermaid(edge_label)}\"| {tgt_id}')

    # Draw strong correlation edges (dashed)
    for edge in strong_correlation:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        if src not in node_labels or tgt not in node_labels:
            continue
        src_id, _, _ = node_labels[src]
        tgt_id, _, _ = node_labels[tgt]
        score = edge.get("base_score", 0)
        edge_label = edge.get("mechanism_description", f"{score:.2f}")
        if len(edge_label) > 25:
            edge_label = f"{score:.2f}"
        lines.append(f'    {src_id} -.->|\"{_escape_mermaid(edge_label)}\"| {tgt_id}')

    # Draw orphan nodes
    for node in all_nodes:
        if node["id"] not in drawn_nodes:
            safe_id, label, _ = node_labels[node["id"]]
            node_type = node.get("type", "paper")
            if node_type == "score":
                lines.append(f'    {safe_id}("{_escape_mermaid(label[:30])}")')
            else:
                lines.append(f'    {safe_id}["{_escape_mermaid(label)}"]')

    lines.append("```")
    lines.append("")

    # Legend
    lines.append("### Legend")
    lines.append("- **Thick arrow (⇒)**: Causation edge")
    lines.append("- **Dashed arrow (-·→)**: Correlation edge (score > 0.3)")
    lines.append("- **Paper nodes**: `[brackets]` | **Score nodes**: `(rounded)`")
    lines.append("")
    lines.append(f"- Causal edges: {len(causal_edges)}")
    lines.append(f"- Strong correlation edges: {len(strong_correlation)}")
    lines.append(f"- Weak correlation edges (not shown): {len(weak_correlation)}")
    lines.append(f"- Total nodes: {len(all_nodes)}")
    lines.append("")

    (meta_dir / "causal-map.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# B1-based helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _write_papers_index(
    papers_dir: Path,
    scored_papers: list[dict],
) -> None:
    """Generate vault/papers/index.md — ranked list with links."""
    lines = [
        "# Paper Index",
        "",
        "> Ranked list of all scored papers with links to individual notes.",
        "",
        "| Rank | Score | Year | Citations | Title |",
        "|------|-------|------|-----------|-------|",
    ]

    for i, p in enumerate(scored_papers, 1):
        score = p.get("dry_score", 0)
        year = p.get("year", "?")
        citations = p.get("citations", 0) or 0
        title = (p.get("title", "?") or "?")[:60]
        pid = p.get("paper_id", "?")
        passed = p.get("passed", False)
        escaped_title = title.replace("|", "\\|")
        lines.append(
            f"| {'✅' if passed else '⚠️'} {i} | {score:.3f} | {year} "
            f"| {citations} | [[{pid}|{escaped_title}]] |"
        )

    lines.append("")
    lines.append(f"*{len(scored_papers)} papers total*")
    lines.append("")

    (papers_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
