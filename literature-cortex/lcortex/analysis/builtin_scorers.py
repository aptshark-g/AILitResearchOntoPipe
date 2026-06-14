"""
Built-in dry scorers — plug into the scoring framework.

All scorers implement the ``Scorer`` abstract interface from ``lcortex.analysis.scoring``
and are registered via ``@register_scorer`` for name-based discovery.

Default composite (5 dimensions):
  BM25Scorer(0.30) + IntentScorer(0.40) + BackgroundScorer(0.10)
  + RecencyScorer(0.10) + ImpactScorer(0.10)

Optional / example:
  VenueScorer(0.10) — journal/conference tier, needs OpenAlexSource
  AuthorScorer(0.05) — author h-index proxy, needs OpenAlexSource
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from lcortex.analysis.scoring import (
    Scorer,
    ScoreResult,
    register_scorer,
    register_source,
    DataSource,
)

# ═══════════════════════════════════════════════════════════════════════════
# Re-use existing BM25 + Intent implementation from dry_scorer
# ═══════════════════════════════════════════════════════════════════════════


@register_scorer("bm25")
class BM25Scorer(Scorer):
    """BM25 relevance between query and paper text.

    Requires ``bm25`` and ``intent_words`` to be injected via constructor
    (populated from ``dry_scorer.BM25Scorer.fit(papers)``).
    """

    name = "bm25"

    def __init__(
        self,
        bm25: Any = None,
        intent_words: set[str] | None = None,
        weight: float = 0.30,
    ):
        super().__init__(weight=weight)
        self._bm25 = bm25
        self._intent_words = intent_words or set()

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        if self._bm25 is None:
            return ScoreResult(name=self.name, value=0.0, weight=self.weight, weighted=0.0)

        title = paper.get("title", "")
        abstract = paper.get("abstract", paper.get("abstract_text", ""))
        index = paper.get("_index", 0)

        title_scores = [self._bm25.score_sentence(title, index) for _ in range(1)]
        abstract_scores = [self._bm25.score_sentence(abstract, index) for _ in range(1)]
        bm25_title = sum(title_scores) / max(len(title_scores), 1)
        bm25_abstract = sum(abstract_scores) / max(len(abstract_scores), 1)
        bm25_norm = max(bm25_title, bm25_abstract)

        detail = {"bm25_title": round(bm25_title, 4), "bm25_abstract": round(bm25_abstract, 4)}
        val = min(1.0, bm25_norm)
        return ScoreResult(
            name=self.name, value=val, weight=self.weight,
            weighted=val * self.weight, detail=detail,
        )


@register_scorer("intent")
class IntentScorer(Scorer):
    """Intent-word matching: title (3x) + abstract (1.5x).

    Requires ``intent_words`` and ``background_words`` injected via constructor.
    """

    name = "intent"

    def __init__(
        self,
        intent_words: set[str] | None = None,
        background_words: set[str] | None = None,
        weight: float = 0.40,
    ):
        super().__init__(weight=weight)
        self._intent = intent_words or set()
        self._bg = background_words or set()

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        title = paper.get("title", "")
        abstract = paper.get("abstract", paper.get("abstract_text", ""))
        title_lower = title.lower()
        abstract_lower = abstract.lower()

        intent_title = sum(1 for w in self._intent if w in title_lower)
        intent_abstract = sum(1 for w in self._intent if w in abstract_lower)
        total_intent = max(intent_title, intent_abstract)
        val = min(1.0, (intent_title * 3 + intent_abstract * 1.5) / max(len(self._intent) * 3, 1))

        return ScoreResult(
            name=self.name, value=val, weight=self.weight,
            weighted=val * self.weight,
            detail={"intent_title": intent_title, "intent_abstract": intent_abstract,
                     "total_unique": total_intent},
        )


@register_scorer("background")
class BackgroundScorer(Scorer):
    """Background-word matching in title/abstract."""

    name = "background"

    def __init__(self, background_words: set[str] | None = None, weight: float = 0.10):
        super().__init__(weight=weight)
        self._bg = background_words or set()

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        if not self._bg:
            return ScoreResult(name=self.name, value=0.0, weight=self.weight, weighted=0.0)
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", paper.get("abstract_text", "")).lower()
        hits = sum(1 for w in self._bg if w in title or w in abstract)
        val = min(1.0, hits / max(len(self._bg) * 0.3, 1))
        return ScoreResult(name=self.name, value=val, weight=self.weight,
                           weighted=val * self.weight, detail={"hits": hits})


@register_scorer("recency")
class RecencyScorer(Scorer):
    """Recency boost based on publication year.

    Linear decay over 5 years: current_year = 1.0, 5 years ago = 0.0.
    """

    name = "recency"

    def __init__(self, current_year: int | None = None, lookback: int = 5, weight: float = 0.10):
        super().__init__(weight=weight)
        self._cy = current_year or datetime.date.today().year
        self._lookback = lookback

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        year = paper.get("year", 0) or 0
        if not year or year > self._cy:
            val = 0.5
        else:
            age = self._cy - year
            val = max(0.0, 1.0 - age / self._lookback) if age <= self._lookback else 0.0
        return ScoreResult(name=self.name, value=val, weight=self.weight,
                           weighted=val * self.weight, detail={"year": year})


@register_scorer("impact")
class ImpactScorer(Scorer):
    """Citation impact proxy (from arXiv comment or OpenAlex).

    Maps citation count to [0, 1] via log scale: 0→0, 10→0.5, 100→1.0.
    """

    name = "impact"
    _IMPACT_MAX: float = 1.0

    def __init__(self, weight: float = 0.10):
        super().__init__(weight=weight)

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        # First try enriched_meta, then paper-level
        citations = 0
        if enriched_meta:
            citations = max(
                enriched_meta.get("citations", 0),
                enriched_meta.get("cited_by_count", 0),
            )
        if not citations:
            citations = paper.get("citations", 0)

        if citations >= 100:
            val = self._IMPACT_MAX
        elif citations >= 10:
            val = self._IMPACT_MAX * 0.5
        elif citations >= 1:
            val = self._IMPACT_MAX * 0.2
        else:
            val = 0.0

        return ScoreResult(name=self.name, value=val, weight=self.weight,
                           weighted=val * self.weight, detail={"citations": citations})


# ═══════════════════════════════════════════════════════════════════════════
# Optional / example scorers (extend the framework)
# ═══════════════════════════════════════════════════════════════════════════


@register_scorer("venue")
class VenueScorer(Scorer):
    """Journal / conference tier scoring.

    Uses enriched_meta from OpenAlexSource.  Falls back to heuristic
    category matching from arXiv categories.

    Prestige tiers (configurable via constructor):
      - Tier 1 (1.0): Nature, Science, Cell, top IEEE/ACM transactions
      - Tier 2 (0.6): Recognized journals, A* conferences
      - Tier 3 (0.3): Other peer-reviewed venues
      - Tier 0 (0.0): Preprint / unknown
    """

    name = "venue"
    _TIER1: set[str] = {"nature", "science", "cell", "pnas", "physical review letters"}
    _TIER2: set[str] = {"ieee transactions", "acm transactions", "neurips", "icml", "iclr",
                        "cvpr", "iccv", "eccv", "acl", "emnlp", "aaai", "ijcai"}

    def __init__(self, weight: float = 0.10):
        super().__init__(weight=weight)

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        venue_name = ""

        if enriched_meta:
            venue_name = enriched_meta.get("venue_name", "").lower()
        if not venue_name:
            venue_name = paper.get("venue", "").lower()

        if not venue_name:
            val = 0.0
            detail = {"venue": "unknown"}
        elif any(t in venue_name for t in self._TIER1):
            val = 1.0
            detail = {"venue": venue_name, "tier": 1}
        elif any(t in venue_name for t in self._TIER2):
            val = 0.6
            detail = {"venue": venue_name, "tier": 2}
        else:
            val = 0.3
            detail = {"venue": venue_name, "tier": 3}

        return ScoreResult(name=self.name, value=val, weight=self.weight,
                           weighted=val * self.weight, detail=detail)


@register_scorer("author")
class AuthorScorer(Scorer):
    """Author impact proxy — uses author count and cited_by_count.

    Does NOT require per-author h-index lookup (heavy API).  Instead uses
    a lightweight proxy: mean citations per author, log-scaled.
    """

    name = "author"

    def __init__(self, weight: float = 0.05):
        super().__init__(weight=weight)

    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        cited = 0
        author_count = 2  # default

        if enriched_meta:
            cited = enriched_meta.get("cited_by_count", 0)
            author_count = max(1, enriched_meta.get("author_count", 2))

        if not cited:
            cited = paper.get("citations", 0)

        cpa = cited / author_count  # citations per author
        val = min(1.0, cpa / 50.0)  # 50 cites/author → 1.0

        return ScoreResult(name=self.name, value=val, weight=self.weight,
                           weighted=val * self.weight,
                           detail={"cited_by_count": cited, "author_count": author_count})


# ═══════════════════════════════════════════════════════════════════════════
# Data sources — built-in
# ═══════════════════════════════════════════════════════════════════════════

@register_source("arxiv_meta")
class _ArxivMetaSource(DataSource):
    name = "arxiv_meta"

    def enrich(self, paper: dict[str, Any]) -> dict[str, Any]:
        try:
            citations = paper.get("citations", 0)
            if not citations and paper.get("comment"):
                m = re.search(r"(\d+)\s*citations?", str(paper.get("comment", "")), re.I)
                if m:
                    citations = int(m.group(1))
            return {"citations": int(citations) if citations else 0}
        except Exception:
            return {}


@register_source("openalex")
class _OpenAlexSource(DataSource):
    name = "openalex"
    _base_url = "https://api.openalex.org/works/"

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout
        self._cache: dict[str, dict] = {}

    def enrich(self, paper: dict[str, Any]) -> dict[str, Any]:
        doi = paper.get("doi", "")
        if not doi and paper.get("paper_id"):
            doi = f"https://doi.org/10.48550/arXiv.{paper.get('paper_id', '')}"
        if not doi or doi in self._cache:
            return self._cache.get(doi, {})

        import json, urllib.request, urllib.error
        try:
            req = urllib.request.Request(
                f"{self._base_url}doi:{doi}",
                headers={"User-Agent": "literature-cortex/0.1"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            return {}

        venue = data.get("primary_location", {}).get("source", {}) or {}
        result = {
            "cited_by_count": data.get("cited_by_count", 0),
            "venue_name": venue.get("display_name", ""),
            "venue_type": venue.get("type", ""),
            "is_oa": data.get("open_access", {}).get("is_oa", False),
            "author_count": len(data.get("authorships", [])),
        }
        self._cache[doi] = result
        return result
