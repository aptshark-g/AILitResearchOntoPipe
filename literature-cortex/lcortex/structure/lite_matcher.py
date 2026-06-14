"""
Lite-mode paper→seed matcher — keyword-driven, no LLM required.

Maps papers to ontology seed nodes by scoring title+abstract keywords
against each seed's keywords array.  Designed for dry/lite pipeline modes
where Phase F (LLM-based structure extraction) is skipped.

Produces the same output shape as ``extract_structure`` so run_controller
can use it interchangeably:

.. code-block:: json

    {
        "paper_id": "2507.03854",
        "knowledge_level": ["L3-method", "L4-physics"],
        "knowledge_level_confidence": 0.72,
        "matched_seeds": ["method-7", "method-4", "phys-5"],
        "match_scores": {"method-7": 0.85, "method-4": 0.62, "phys-5": 0.51},
        "structure_template": null
    }
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger("lcortex.structure.lite_matcher")

# ── Stopwords filtered from paper text before matching ───────────────────
_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "its",
    "it", "that", "this", "these", "those", "which", "who", "what",
    "not", "no", "nor", "so", "if", "than", "then", "also", "just",
    "about", "into", "over", "after", "before", "between", "under",
    "within", "without", "through", "during", "such", "both", "each",
    "every", "all", "some", "any", "more", "most", "other", "new",
    "novel", "using", "based", "proposed", "method", "approach",
    "paper", "system", "model", "results", "show", "shown", "use",
    "used", "present", "study", "demonstrate",
}


def _tokenize(text: str, min_len: int = 2) -> set[str]:
    """Lowercase, split, strip stopwords, return unique tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= min_len}


def _load_seeds(seeds_dir: str) -> list[dict]:
    """Load all seed JSON files and return flat list of seed dicts."""
    seeds: list[dict] = []
    seeds_path = Path(seeds_dir)
    for f in sorted(seeds_path.glob("seed_L*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            seeds.extend(data)
    return seeds


def match_paper(
    paper: dict,
    seeds: list[dict],
    top_k: int = 5,
    min_score: float = 0.15,
) -> dict:
    """Match a single paper to ontology seed nodes.

    Algorithm:
      1. Tokenize paper title + abstract (weighted 2:1)
      2. For each seed, compute Jaccard overlap between paper tokens
         and seed keywords
      3. Boost score if seed name/aliases appear as phrases in paper text
      4. Return top-k matches above min_score

    Args:
        paper: Paper dict with 'title', 'abstract', optionally 'keywords'.
        seeds: List of seed dicts (from seed_L*.json).
        top_k: Max number of matched seeds to return.
        min_score: Minimum Jaccard score for a match.

    Returns:
        Result dict compatible with extract_structure output shape.
    """
    paper_id = paper.get("paper_id", paper.get("arxiv_id", paper.get("id", "")))
    title = paper.get("title", "")
    abstract = paper.get("abstract", paper.get("abstract_text", ""))
    paper_kw = paper.get("keywords", [])
    if isinstance(paper_kw, str):
        paper_kw = [k.strip() for k in paper_kw.split(",")]

    # Build paper token set — title tokens weighted 2×
    title_tokens = _tokenize(title)
    abstract_tokens = _tokenize(abstract)
    paper_tokens = title_tokens | abstract_tokens
    paper_text = (title + " " + abstract).lower()

    if not paper_tokens:
        log.debug("lite_matcher: no tokens for paper '%s'", paper_id)
        return {
            "paper_id": paper_id,
            "knowledge_level": [],
            "knowledge_level_confidence": 0.0,
            "matched_seeds": [],
            "match_scores": {},
            "structure_template": None,
        }

    # Score each seed
    scores: dict[str, float] = {}
    for seed in seeds:
        seed_id = seed.get("node_id", "")
        seed_keywords = set(seed.get("keywords", []))
        if not seed_keywords:
            continue

        # Primary score: Jaccard between paper tokens and seed keywords
        jaccard = len(paper_tokens & seed_keywords) / len(paper_tokens | seed_keywords)

        # Boost: phrase-level matches in paper text
        phrase_boost = 0.0
        name = seed.get("name", "").lower()
        aliases = [a.lower() for a in seed.get("aliases", [])]

        # Check if seed name or aliases appear as whole phrases in paper text
        for phrase in [name] + aliases:
            if len(phrase) > 5 and phrase in paper_text:
                phrase_boost += 0.15
            elif len(phrase) > 3 and phrase in paper_text:
                phrase_boost += 0.08

        # Title-only matches get extra weight
        title_boost = 0.0
        title_lower = title.lower()
        for phrase in [name] + aliases:
            if len(phrase) > 4 and phrase in title_lower:
                title_boost += 0.20

        combined = min(1.0, jaccard + phrase_boost + title_boost)

        if combined >= min_score:
            scores[seed_id] = combined

    # Select top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    matched_ids = [sid for sid, _ in ranked]
    match_scores = {sid: round(sc, 3) for sid, sc in ranked}

    # Derive knowledge levels from matched seeds
    seed_index = {s["node_id"]: s for s in seeds}
    knowledge_levels: list[str] = []
    seen_levels: set[str] = set()
    for sid in matched_ids:
        seed = seed_index.get(sid)
        if not seed:
            continue
        level = seed.get("level", "?")
        cat = seed.get("category", "general")
        kl_label = f"L{level}-{cat}"
        if kl_label not in seen_levels:
            knowledge_levels.append(kl_label)
            seen_levels.add(kl_label)

    confidence = round(sum(scores[sid] for sid in matched_ids) / max(len(matched_ids), 1), 3)

    return {
        "paper_id": paper_id,
        "knowledge_level": knowledge_levels,
        "knowledge_level_confidence": confidence,
        "matched_seeds": matched_ids,
        "match_scores": match_scores,
        "structure_template": None,
    }


def match_papers(
    papers: list[dict],
    seeds_dir: str,
    top_k: int = 5,
    min_score: float = 0.15,
) -> list[dict]:
    """Batch match multiple papers to seed ontology nodes.

    Args:
        papers: List of paper dicts.
        seeds_dir: Path to the seeds/ directory containing seed_L*.json.
        top_k: Max matches per paper.
        min_score: Minimum score threshold.

    Returns:
        List of result dicts, one per paper.
    """
    seeds = _load_seeds(seeds_dir)
    if not seeds:
        log.warning("lite_matcher: no seeds found in %s", seeds_dir)
        return []

    results: list[dict] = []
    for paper in papers:
        result = match_paper(paper, seeds, top_k=top_k, min_score=min_score)
        results.append(result)

    return results


# ── Convenience: build paper→seed edges ───────────────────────────────────

def build_seed_edges(match_results: list[dict]) -> list[dict]:
    """Convert match results into graph edge dicts.

    Returns list of edge dicts compatible with GraphStore.add_edge():
        {source_id, target_id, base_type, base_score, mechanism_description}
    """
    edges: list[dict] = []
    for result in match_results:
        paper_id = result.get("paper_id", "")
        for seed_id, score in result.get("match_scores", {}).items():
            edges.append({
                "source_id": paper_id,
                "target_id": seed_id,
                "base_type": "correlation",
                "base_score": score,
                "mechanism_description": f"lite_matcher: keyword overlap score={score:.3f}",
            })
    return edges
