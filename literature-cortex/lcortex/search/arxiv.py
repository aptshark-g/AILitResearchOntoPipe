"""
arXiv Search Adapter — streaming + batch modes.

Uses the `arxiv` Python library (Client.results() API).
Rate limit: ~1 QPS handled internally by the arxiv lib.

Unified paper schema keys:
    paper_id, title, authors, year, venue, abstract, abstract_length,
    citations, source, doi, arxiv_id, oa_status, pdf_url, url
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

log = logging.getLogger("lcortex.search.arxiv")

# ---------------------------------------------------------------------------
# arXiv ID normalisation
# ---------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/abs/)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")
_ARXIV_NORM_RE = re.compile(r"^([0-9]{4}\.[0-9]{4,5})v\d+$")


def norm_arxiv_id(raw: str) -> str | None:
    """Normalize an arXiv ID to 'YYYY.xxxxx' without version suffix.

    Accepts raw strings like:
        - ``2401.12345``
        - ``2401.12345v2``
        - ``http://arxiv.org/abs/2401.12345``
        - ``arxiv:2401.12345v3``

    Returns ``None`` if no arXiv ID pattern is found.
    """
    if not raw:
        return None
    m = _ARXIV_ID_RE.search(raw)
    if not m:
        return None
    aid = m.group(1)
    nm = _ARXIV_NORM_RE.match(aid)
    if nm:
        aid = nm.group(1)
    return aid


# ---------------------------------------------------------------------------
# Streaming search (yield)
# ---------------------------------------------------------------------------

def search_arxiv_stream(query: str, max_results: int = 10) -> Iterator[dict]:
    """Stream arXiv results one paper at a time via generator.

    Each yielded paper is a dict following the unified schema.  Results are
    yielded as they arrive — no need to wait for all results to accumulate.

    Args:
        query: free-text search query (e.g. "attention mechanism in transformers")
        max_results: maximum number of papers to yield (default 10)

    Yields:
        dict — one unified paper record per result

    Raises:
        Nothing — all errors are logged and the generator simply stops yielding.
    """
    try:
        import arxiv  # type: ignore[import-untyped]
    except ImportError:
        log.warning("arXiv Python library not installed; skipping arXiv search")
        return

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()
        for result in client.results(search):
            arxiv_id_norm = norm_arxiv_id(result.entry_id) or ""

            yield {
                "paper_id": arxiv_id_norm or result.entry_id,
                "title": result.title or "",
                "authors": [a.name for a in result.authors],
                "year": result.published.year if result.published else 0,
                "venue": "arXiv",
                "abstract": result.summary or "",
                "abstract_length": len(result.summary) if result.summary else 0,
                "citations": 0,
                "source": "arxiv",
                "doi": result.doi or "",
                "arxiv_id": arxiv_id_norm,
                "oa_status": "gold",   # arXiv papers are always OA
                "pdf_url": result.pdf_url or "",
                "url": result.entry_id or "",
            }
    except Exception as exc:
        log.warning("arXiv search failed: %s", exc)


# ---------------------------------------------------------------------------
# Batch search
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv — batch mode (returns a list).

    Convenience wrapper around :func:`search_arxiv_stream`.  Use the stream
    variant directly if you need streaming behavior (JSONL / progressive UI).

    Args:
        query: free-text search query
        max_results: maximum papers to return

    Returns:
        List of unified paper dicts.  Empty list on error.
    """
    return list(search_arxiv_stream(query, max_results))
