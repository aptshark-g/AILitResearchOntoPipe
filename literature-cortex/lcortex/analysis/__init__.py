"""Analysis module — Phase B1 dry scoring + Phase B2 4C+L scoring.

Pluggable scoring framework:
  - ``scoring.py`` — Scorer + DataSource abstract interfaces, CompositeScorer
  - ``builtin_scorers.py`` — 7 registered scorers (bm25, intent, background,
    recency, impact, venue, author) + 2 data sources
  - ``dry_scorer.py`` — BM25Scorer, IntentWordExtractor, dry_score_batch (legacy) +
    ``build_default_composite()`` bridge to new framework

Usage::

    # Default 5-dim
    from lcortex.analysis import dry_score_batch, PASS_THRESHOLD
    results = dry_score_batch(papers, query)

    # Custom with VenueScorer
    from lcortex.analysis.dry_scorer import build_default_composite
    from lcortex.analysis.builtin_scorers import VenueScorer
    from lcortex.analysis.scoring import OpenAlexSource

    composite = build_default_composite(
        bm25, intent_words, background_words,
        extra_scorers=[VenueScorer(weight=0.05)],
        extra_sources=[OpenAlexSource()],
    )
    result = composite.score(paper, paper_index=0)
"""

from .scorer import score_batch, score_paper
from .dry_scorer import (
    AutoWeightedScorer,
    dry_score_paper,
    dry_score_batch,
    load_dry_scored,
    build_default_composite,
    PASS_THRESHOLD,
)

__all__ = [
    # Phase B2 (LLM)
    "score_paper",
    "score_batch",
    # Phase B1 (dry)
    "AutoWeightedScorer",
    "dry_score_paper",
    "dry_score_batch",
    "load_dry_scored",
    "build_default_composite",
    "PASS_THRESHOLD",
]
