"""
Dry Scoring Framework — Pluggable scorer + data source architecture.

The scoring pipeline is composed of:
  - **DataSources**: fetch external metadata (citations, venue rank, author
    h-index, etc.) for a paper. Each source is optional and independently
    fail-safe.
  - **Scorers**: compute a single numeric score from a paper dict and
    optional enriched metadata. Scorers are weighted and combined.

Architecture::

    Paper dict
        │
        ├── DataSource.enrich(paper) → enriched_meta dict
        │     ├── ArxivMetaSource (citations, versions)
        │     ├── OpenAlexSource (venue rank, cited_by_count, author h-index)
        │     └── CustomSource (user-defined)
        │
        └── CompositeScorer.score(paper, enriched_meta) → float
              ├── BM25Scorer        (0.30)
              ├── IntentScorer      (0.40)
              ├── BackgroundScorer  (0.10)
              ├── RecencyScorer     (0.10)
              └── VenueScorer       (0.10)
              └── AuthorScorer      (0.05, example custom)

All scorers and sources are **composable at runtime**:
    scorer = CompositeScorer([
        BM25Scorer(bm25=bm25, intent_words=iw, weight=0.30),
        VenueScorer(weight=0.10),
    ])
    scorer.add_source(OpenAlexSource())
    score = scorer.score(paper)

The built-in ``dry_score_paper`` and ``dry_score_batch`` remain available
as convenience wrappers that use the default 5-dimension composite.
"""

from __future__ import annotations

import datetime
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

log = logging.getLogger("lcortex.analysis.scoring")

# ═══════════════════════════════════════════════════════════════════════════
# Data Source Interface
# ═══════════════════════════════════════════════════════════════════════════


class DataSource(ABC):
    """Abstract data source — fetches external metadata for a paper.

    Each source is **optional and fail-safe**: exceptions are caught and
    logged, never propagated.  A source may return partial results.
    """

    # Human-readable name for logging
    name: ClassVar[str] = "base"

    @abstractmethod
    def enrich(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Fetch and return enriched metadata fields.

        Args:
            paper: Paper dict with at least ``paper_id``, ``title``.

        Returns:
            Dict of {field_name: value} to merge into ``enriched_meta``.
            Empty dict on failure or no data available.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class ArxivMetaSource(DataSource):
    """Extract citation count, version count, and category from arXiv paper dict.

    No external API call — works from what's already in the paper dict
    returned by ``arxiv.Search``.
    """

    name: ClassVar[str] = "arxiv_meta"

    def enrich(self, paper: dict[str, Any]) -> dict[str, Any]:
        try:
            citations = paper.get("citations", 0)
            if not citations and paper.get("comment"):
                # Heuristic: look for "X citations" in comment
                import re
                m = re.search(r"(\d+)\s*citations?", str(paper.get("comment", "")), re.I)
                if m:
                    citations = int(m.group(1))
            categories = paper.get("categories", paper.get("primary_category", ""))
            versions = paper.get("versions", 0)
            return {
                "citations": int(citations) if citations else 0,
                "categories": categories,
                "versions": int(versions) if versions else 1,
            }
        except Exception:
            return {}


class OpenAlexSource(DataSource):
    """Fetch venue rank, cited_by_count, and author h-index from OpenAlex API.

    Requires network access.  Falls back gracefully on timeout/error.
    """

    name: ClassVar[str] = "openalex"
    _base_url: str = "https://api.openalex.org/works/"

    def __init__(self, timeout: float = 5.0, cache: dict | None = None):
        self._timeout = timeout
        self._cache: dict[str, dict] = cache or {}

    def enrich(self, paper: dict[str, Any]) -> dict[str, Any]:
        doi = paper.get("doi", "")
        if not doi and "arxiv" in str(paper.get("id", "")):
            doi = f"https://doi.org/10.48550/arXiv.{paper.get('paper_id', '')}"

        if not doi:
            return {}

        if doi in self._cache:
            return self._cache[doi]

        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                f"{self._base_url}doi:{doi}",
                headers={"User-Agent": "literature-cortex/0.1 (mailto:dev@example.com)"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            log.debug("OpenAlex lookup failed for %s: %s", doi[:40], exc)
            return {}

        venue = data.get("primary_location", {}).get("source", {}) or {}
        authorship = data.get("authorships", [])

        # Venue rank proxy (SJR/CiteScore not directly available via free API)
        cited = data.get("cited_by_count", 0)

        result = {
            "cited_by_count": cited,
            "venue_name": venue.get("display_name", ""),
            "venue_type": venue.get("type", ""),
            "is_oa": data.get("open_access", {}).get("is_oa", False),
            "author_count": len(authorship),
        }
        self._cache[doi] = result
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Scorer Interface
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreResult:
    """Single scorer output."""

    name: str           # scorer name (e.g. "bm25", "venue")
    value: float        # raw score
    weight: float       # assigned weight in composite
    weighted: float     # value * weight
    detail: dict = field(default_factory=dict)  # scorer-specific debug info


class Scorer(ABC):
    """Abstract scorer — one dimension of the composite dry score."""

    name: ClassVar[str] = "base"

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    @abstractmethod
    def score(self, paper: dict[str, Any], enriched_meta: dict[str, Any] | None = None) -> ScoreResult:
        """Compute a single score dimension.

        Args:
            paper: Paper dict.
            enriched_meta: Merged metadata from all DataSources.

        Returns:
            ScoreResult with value in [0, 1].
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


# ═══════════════════════════════════════════════════════════════════════════
# Composite Scorer
# ═══════════════════════════════════════════════════════════════════════════


class CompositeScorer:
    """Weighted composite of multiple Scorer instances.

    Usage::

        composite = CompositeScorer([
            BM25Scorer(bm25=bm25, weight=0.30),
            VenueScorer(weight=0.10),
        ])
        composite.add_source(OpenAlexSource())
        result = composite.score(paper, paper_index=0)

    """

    def __init__(self, scorers: list[Scorer] | None = None):
        self._scorers: list[Scorer] = list(scorers) if scorers else []
        self._sources: list[DataSource] = []

    def add_scorer(self, scorer: Scorer) -> CompositeScorer:
        """Add a scorer.  Returns self for chaining."""
        self._scorers.append(scorer)
        return self

    def add_source(self, source: DataSource) -> CompositeScorer:
        """Add a data source.  Returns self for chaining."""
        self._sources.append(source)
        return self

    @property
    def scorers(self) -> list[Scorer]:
        return list(self._scorers)

    @property
    def sources(self) -> list[DataSource]:
        return list(self._sources)

    def score(
        self,
        paper: dict[str, Any],
        paper_index: int = 0,
    ) -> dict[str, Any]:
        """Run all sources, then all scorers, and return a unified result dict.

        Returns a dict compatible with the ``dry_score_paper`` output format.

        Args:
            paper: Paper dict.
            paper_index: Position in batch (for BM25 doc index).

        Returns:
            Dict with keys: paper_id, dry_score, dry_detail, passed, reason.
        """
        # ── Enrich from all data sources ────────────────────────────
        enriched: dict[str, Any] = {}
        for src in self._sources:
            try:
                fields = src.enrich(paper)
                enriched.update(fields)
            except Exception as exc:
                log.debug("DataSource %s failed for %s: %s",
                          src.name, paper.get("paper_id", "?"), exc)

        # ── Partial paper fields that scorers may need ─────────────
        paper["_enriched"] = enriched
        paper["_index"] = paper_index

        # ── Run scorers ────────────────────────────────────────────
        results: list[ScoreResult] = []
        for scorer in self._scorers:
            try:
                result = scorer.score(paper, enriched)
                results.append(result)
            except Exception as exc:
                log.warning("Scorer %s failed for %s: %s",
                            scorer.name, paper.get("paper_id", "?"), exc)
                results.append(ScoreResult(
                    name=scorer.name, value=0.0, weight=scorer.weight,
                    weighted=0.0, detail={"error": str(exc)},
                ))

        total_weighted = sum(r.weighted for r in results)
        total_weight = sum(r.weight for r in results)
        final = total_weighted / total_weight if total_weight > 0 else 0.0

        detail = {r.name: round(r.value, 4) for r in results}
        detail["_weighted_total"] = round(total_weighted, 4)

        passed = final >= _PASS_THRESHOLD

        return {
            "paper_id": paper.get("paper_id", paper.get("id", "")),
            "title": paper.get("title", ""),
            "dry_score": round(final, 4),
            "dry_detail": detail,
            "enriched_meta": {k: v for k, v in enriched.items()
                              if not k.startswith("_")},
            "passed": passed,
            "reason": "; ".join(r.name for r in results) if not passed else "",
        }


# ── Built-in pass threshold ───────────────────────────────────────────────
_PASS_THRESHOLD = 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Registry: discover scorers by name
# ═══════════════════════════════════════════════════════════════════════════

_registry: dict[str, type] = {}


def register_scorer(name: str):
    """Decorator to register a Scorer class by name."""
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator


def register_source(name: str):
    """Decorator to register a DataSource class by name."""
    def decorator(cls):
        _registry[f"source:{name}"] = cls
        return cls
    return decorator


def list_registered() -> dict[str, type]:
    """Return all registered scorers and sources."""
    return dict(_registry)
