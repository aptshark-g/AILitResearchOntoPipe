"""
Dual-linkage knowledge transfer index.

Generates two complementary views of the knowledge graph:

  **Near Transfer** — papers that share the same L3 method paradigm
    or L2 math foundation or L4 physical constraint.  Essentially
    "people who used similar approaches" — the standard literature
    review axis.

  **Far Transfer** — papers from *different domains* whose structure
    templates (abstract_pattern + mathematical_core) are isomorphic.
    Example: a neural adaptive controller for vibration suppression
    may be structurally identical to an adaptive optimizer for
    portfolio allocation.  The seed ontology (L3 methods → L2 math →
    L1 axioms) provides the semantic backbone for detecting these
    cross-domain analogies.

Output: ``00-meta/near-transfer.md`` and ``00-meta/far-transfer.md``
in the Obsidian vault.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _sanitize(text: str) -> str:
    """Sanitize a paper title into a safe filename stub."""
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(text))[:60]


def _seed_name(seed_id: str, all_nodes: list[dict]) -> str:
    """Look up a human-readable name for a seed node id."""
    for n in all_nodes:
        if n.get("id") == seed_id:
            return n.get("title", seed_id)
    return seed_id


# ═══════════════════════════════════════════════════════════════════════════
# Near Transfer: same paradigm / same math / same physics
# ═══════════════════════════════════════════════════════════════════════════

def _write_near_transfer(
    meta_dir: Path,
    papers: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Generate 00-meta/near-transfer.md.

    Clusters papers that share:
      - Same L3 method seeds (algorithmic paradigm)
      - Same L2 math seeds (mathematical foundation)
      - Same L4 physics seeds (physical constraint)
    """
    # Build paper → seed adjacency from edges
    paper_seeds: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in all_edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        score = edge.get("base_score", 0)
        # Detect seed id by prefix (method-X, math-X, phys-X, axiom-X, meta-X)
        for prefix in ("method-", "math-", "phys-", "axiom-", "meta-"):
            if tgt.startswith(prefix):
                paper_seeds[src][tgt] = max(paper_seeds[src].get(tgt, 0), score)
            if src.startswith(prefix):
                paper_seeds[tgt][src] = max(paper_seeds[tgt].get(src, 0), score)

    # Cluster by seed
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        pid = paper.get("id", paper.get("paper_id", ""))
        for seed_id in paper_seeds.get(pid, {}):
            by_seed[seed_id].append(paper)

    # Sort seeds by cluster size
    ranked = sorted(by_seed.items(), key=lambda kv: -len(kv[1]))

    lines = [
        "# Near Transfer: Shared Paradigm Links",
        "",
        "> Papers clustered by shared L3 method paradigms,",
        "> L2 mathematical foundations, and L4 physical constraints.",
        "> These are the **direct** intellectual neighbors — same toolkit, different problems.",
        "",
    ]

    if not ranked:
        lines.append("*No seed-linked papers found. Run Phase F (lite or full) to establish paper→seed edges.*")
        lines.append("")
        (meta_dir / "near-transfer.md").write_text("\n".join(lines), encoding="utf-8")
        return

    for seed_id, cluster in ranked:
        name = _seed_name(seed_id, all_nodes)
        lines.append(f"## {name}")
        lines.append(f"> `{seed_id}` — {len(cluster)} papers")
        lines.append("")

        for paper in sorted(cluster, key=lambda p: p.get("title", "")):
            pid = paper.get("id", paper.get("paper_id", "?"))
            title = paper.get("title", pid)
            filename = _sanitize(pid)
            year = paper.get("year", "")
            b1 = ""
            # Find B1/B2 scores from edge data
            for edge in all_edges:
                if edge.get("source_id") == pid and "b1-" in edge.get("target_id", ""):
                    b1 = f" — B1:{edge.get('base_score', 0):.3f}"
                    break
                if edge.get("source_id") == pid and "b2-" in edge.get("target_id", ""):
                    b1 = f" — B2:{edge.get('base_score', 0):.1f}"
                    break
            lines.append(f"- 📄 [[../papers/{pid}|{title}]] ({year}){b1}")

        lines.append("")

    # ── Summary stats ──────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    lines.append(f"- **Seed clusters:** {len(ranked)}")
    lines.append(f"- **Total papers linked:** {sum(len(c) for _, c in ranked)}")
    lines.append(f"- **Top cluster:** {_seed_name(ranked[0][0], all_nodes)} ({len(ranked[0][1])} papers)")
    lines.append("")

    (meta_dir / "near-transfer.md").write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Far Transfer: cross-domain structural analogy
# ═══════════════════════════════════════════════════════════════════════════

def _write_far_transfer(
    meta_dir: Path,
    papers: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Generate 00-meta/far-transfer.md.

    Identifies papers from **different domains** that share the same
    underlying structure:

      - Same abstract_pattern (e.g. "perceive → adapt → constrain → actuate")
      - Same mathematical_core (e.g. "gradient_descent_on_manifold")
      - Same control_architecture (e.g. "feedforward_adaptive")

    The key insight: when two papers from different application domains
    (say, vibration control and portfolio optimization) share the same
    abstract_pattern and mathematical_core, they are **structurally
    isomorphic** — a finding in one domain may transfer to the other.

    This is made possible by Phase F's structure_template extraction,
    which strips all domain-specific terminology from the paper.
    """

    # ── Extract structure templates from paper nodes ─────────────
    pattern_groups: dict[str, list[dict]] = defaultdict(list)
    math_core_groups: dict[str, list[dict]] = defaultdict(list)
    arch_groups: dict[str, list[dict]] = defaultdict(list)

    for paper in papers:
        st = paper.get("structure_template", {})
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except (json.JSONDecodeError, TypeError):
                st = {}

        ap = st.get("abstract_pattern", "") if isinstance(st, dict) else ""
        mc = st.get("mathematical_core", "") if isinstance(st, dict) else ""
        arch = st.get("control_architecture", "") if isinstance(st, dict) else ""

        if ap and len(ap) > 5:
            pattern_groups[ap].append(paper)
        if mc and len(mc) > 3:
            math_core_groups[mc].append(paper)
        if arch and len(arch) > 3:
            arch_groups[arch].append(paper)

    lines = [
        "# Far Transfer: Cross-Domain Structural Analogies",
        "",
        "> Papers from **different application domains** that share the same",
        "> underlying mathematical and algorithmic structure.",
        ">",
        "> When a neural adaptive controller for vibration and an adaptive",
        "> optimizer for finance share the same abstract_pattern, a finding",
        "> in one domain may transfer to the other.",
        "",
    ]

    # ── If no structure templates available ──────────────────────
    has_structures = any(
        isinstance(p.get("structure_template", {}), dict)
        and p.get("structure_template", {}).get("abstract_pattern")
        for p in papers
    )
    if not has_structures:
        lines.append("> ⚠️ No structure templates available. Run Phase F in **full mode**")
        lines.append("> (with an LLM API key) to extract abstract_pattern and mathematical_core.")
        lines.append("> Lite-mode keyword matching does not generate far-transfer data.")
        lines.append("")
        (meta_dir / "far-transfer.md").write_text("\n".join(lines), encoding="utf-8")
        return

    # ── By abstract_pattern (most transferable) ───────────────────
    if pattern_groups:
        lines.append("## By Abstract Pattern")
        lines.append("")
        lines.append("*Papers with the same control logic stripped of domain terms.*")
        lines.append("")

        for pattern, group in sorted(pattern_groups.items(), key=lambda kv: -len(kv[1])):
            # Only show clusters with ≥2 papers (lone patterns aren't analogies)
            if len(group) < 2:
                continue
            lines.append(f"### `{pattern}` ({len(group)} papers)")
            lines.append("")
            for paper in sorted(group, key=lambda p: p.get("title", "")):
                pid = paper.get("id", paper.get("paper_id", "?"))
                title = paper.get("title", pid)
                filename = _sanitize(pid)
                year = paper.get("year", "")
                # Show domain info if available
                st = paper.get("structure_template", {})
                if isinstance(st, str):
                    try:
                        st = json.loads(st)
                    except (json.JSONDecodeError, TypeError):
                        st = {}
                da = st.get("domain_abstraction", "") if isinstance(st, dict) else ""
                domain_info = f" — *{da[:60]}*" if da else ""
                lines.append(f"- 📄 [[../papers/{pid}|{title}]] ({year}){domain_info}")
            lines.append("")

    # ── By mathematical_core ──────────────────────────────────────
    if math_core_groups:
        lines.append("## By Mathematical Core")
        lines.append("")
        lines.append("*Papers that reduce to the same mathematical operation.*")
        lines.append("")

        for mc, group in sorted(math_core_groups.items(), key=lambda kv: -len(kv[1])):
            if len(group) < 2:
                continue
            lines.append(f"### `{mc}` ({len(group)} papers)")
            lines.append("")
            for paper in sorted(group, key=lambda p: p.get("title", "")):
                pid = paper.get("id", paper.get("paper_id", "?"))
                title = paper.get("title", pid)
                filename = _sanitize(pid)
                year = paper.get("year", "")
                lines.append(f"- 📄 [[../papers/{pid}|{title}]] ({year})")
            lines.append("")

    # ── By control_architecture ───────────────────────────────────
    if arch_groups:
        lines.append("## By Control Architecture")
        lines.append("")
        lines.append("*Papers using the same control topology.*")
        lines.append("")

        for arch, group in sorted(arch_groups.items(), key=lambda kv: -len(kv[1])):
            if len(group) < 2:
                continue
            lines.append(f"### `{arch}` ({len(group)} papers)")
            lines.append("")
            for paper in sorted(group, key=lambda p: p.get("title", "")):
                pid = paper.get("id", paper.get("paper_id", "?"))
                title = paper.get("title", pid)
                filename = _sanitize(pid)
                year = paper.get("year", "")
                lines.append(f"- 📄 [[../papers/{pid}|{title}]] ({year})")
            lines.append("")

    # ── Tally ─────────────────────────────────────────────────────
    cross_domain_pairs = 0
    for group in pattern_groups.values():
        domains = set()
        for p in group:
            st = p.get("structure_template", {})
            if isinstance(st, str):
                try:
                    st = json.loads(st)
                except (json.JSONDecodeError, TypeError):
                    st = {}
            da = st.get("domain_abstraction", "") if isinstance(st, dict) else ""
            if da:
                domains.add(da)
        if len(domains) >= 2:
            cross_domain_pairs += 1

    lines.append("---")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    lines.append(f"- **Pattern clusters (≥2 papers):** {sum(1 for g in pattern_groups.values() if len(g) >= 2)}")
    lines.append(f"- **Math-core clusters (≥2 papers):** {sum(1 for g in math_core_groups.values() if len(g) >= 2)}")
    lines.append(f"- **Architecture clusters (≥2 papers):** {sum(1 for g in arch_groups.values() if len(g) >= 2)}")
    lines.append(f"- **Cross-domain pattern groups:** {cross_domain_pairs}")
    lines.append("")

    (meta_dir / "far-transfer.md").write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: generate both indices at once
# ═══════════════════════════════════════════════════════════════════════════

def write_transfer_indices(
    meta_dir: Path,
    papers: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> dict[str, int]:
    """Write both near-transfer.md and far-transfer.md.

    Returns counts: {near_clusters, far_pattern_clusters, far_math_clusters,
                     far_arch_clusters, cross_domain_groups}
    """
    _write_near_transfer(meta_dir, papers, all_nodes, all_edges)

    # Count near-transfer clusters for summary
    paper_seeds: dict[str, set[str]] = defaultdict(set)
    for edge in all_edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        for prefix in ("method-", "math-", "phys-", "axiom-", "meta-"):
            if tgt.startswith(prefix):
                paper_seeds[src].add(tgt)
            if src.startswith(prefix):
                paper_seeds[tgt].add(src)

    by_seed: dict[str, list] = defaultdict(list)
    for paper in papers:
        pid = paper.get("id", paper.get("paper_id", ""))
        for seed_id in paper_seeds.get(pid, {}):
            by_seed[seed_id].append(paper)

    _write_far_transfer(meta_dir, papers, all_nodes, all_edges)

    # Count far-transfer clusters
    pattern_groups: dict[str, list] = defaultdict(list)
    math_core_groups: dict[str, list] = defaultdict(list)
    arch_groups: dict[str, list] = defaultdict(list)
    cross_domain = 0

    for paper in papers:
        st = paper.get("structure_template", {})
        if isinstance(st, str):
            try:
                st = json.loads(st)
            except (json.JSONDecodeError, TypeError):
                st = {}
        ap = st.get("abstract_pattern", "") if isinstance(st, dict) else ""
        mc = st.get("mathematical_core", "") if isinstance(st, dict) else ""
        arch = st.get("control_architecture", "") if isinstance(st, dict) else ""
        if ap and len(ap) > 5:
            pattern_groups[ap].append(paper)
        if mc and len(mc) > 3:
            math_core_groups[mc].append(paper)
        if arch and len(arch) > 3:
            arch_groups[arch].append(paper)

    for group in pattern_groups.values():
        domains = set()
        for p in group:
            st = p.get("structure_template", {})
            if isinstance(st, str):
                try:
                    st = json.loads(st)
                except (json.JSONDecodeError, TypeError):
                    st = {}
            da = st.get("domain_abstraction", "") if isinstance(st, dict) else ""
            if da:
                domains.add(da)
        if len(domains) >= 2:
            cross_domain += 1

    return {
        "near_clusters": len(by_seed),
        "far_pattern_clusters": sum(1 for g in pattern_groups.values() if len(g) >= 2),
        "far_math_clusters": sum(1 for g in math_core_groups.values() if len(g) >= 2),
        "far_arch_clusters": sum(1 for g in arch_groups.values() if len(g) >= 2),
        "cross_domain_groups": cross_domain,
    }
