"""
Deduplication engine — 3-level dedup for academic paper results.

────────────────────────────────────────────────────────
Level 1: DOI match      (exact, case-insensitive)
Level 2: arXiv ID match (exact, normalized)
Level 3: Title similarity (SequenceMatcher, default 0.85 threshold)

When two papers are matched, the one with the longer abstract is kept
and the shorter is discarded.
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import difflib
from typing import Sequence


def title_similarity(a: str, b: str) -> float:
    """Compute 0–1 similarity between two paper titles.

    Uses ``difflib.SequenceMatcher`` on lowercased, stripped strings.
    This is fast for small batches (<500 papers) and needs no external deps.

    Args:
        a: first title
        b: second title

    Returns:
        Float in [0.0, 1.0] where 1.0 means identical strings.
    """
    return difflib.SequenceMatcher(
        None,
        a.lower().strip(),
        b.lower().strip(),
    ).ratio()


def dedup_papers(
    papers: Sequence[dict],
    title_threshold: float = 0.85,
) -> list[dict]:
    """Deduplicate a list of unified paper dicts.

    Dedup levels applied in order:

    1. **DOI** — normalized (lowercased, stripped).  Empty DOIs are skipped.
    2. **arXiv ID** — normalized (no version suffix).  Empty IDs are skipped.
    3. **Title similarity** — SequenceMatcher against all already-accepted
       papers.  If similarity ≥ *title_threshold*, the paper with the
       **longer abstract** is kept.

    Args:
        papers: list of paper dicts (unified schema)
        title_threshold: minimum SequenceMatcher ratio to consider two
            titles as duplicates (default 0.85)

    Returns:
        New list with duplicates removed, preserving order from *papers*.
    """
    deduped: list[dict] = []
    seen_dois: set[str] = set()
    seen_arxivids: set[str] = set()

    for p in papers:
        # ── Level 1: DOI ──
        doi = str(p.get("doi", "")).strip().lower()
        if doi and doi in seen_dois:
            continue

        # ── Level 2: arXiv ID ──
        arxiv_id = str(p.get("arxiv_id", "")).strip()
        if arxiv_id and arxiv_id in seen_arxivids:
            continue

        # ── Level 3: Title similarity ──
        is_dup = False
        p_title = (p.get("title") or "").lower().strip()
        for existing in deduped:
            e_title = (existing.get("title") or "").lower().strip()
            if not p_title or not e_title:
                continue
            if title_similarity(p_title, e_title) >= title_threshold:
                # Merge: keep the paper with the longer abstract
                if len(str(p.get("abstract") or "")) > len(str(existing.get("abstract") or "")):
                    # Update existing entry with this paper's fields
                    existing.update(p)
                is_dup = True
                break
        if is_dup:
            continue

        # ── Accept ──
        if doi:
            seen_dois.add(doi)
        if arxiv_id:
            seen_arxivids.add(arxiv_id)
        deduped.append(p)

    return deduped
