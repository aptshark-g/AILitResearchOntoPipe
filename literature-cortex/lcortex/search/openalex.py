"""
OpenAlex REST API Search Adapter.

OpenAlex is a free, open catalog of 250M+ scholarly works.
Rate limit: 100K calls/day (no API key needed for polite use).

Rebuilds abstracts from inverted_index, extracts arxiv_id from locations,
and filters out papers without abstracts.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests

log = logging.getLogger("lcortex.search.openalex")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
OPENALEX_USER_AGENT = "literature-cortex/1.0 (mailto:research@example.com)"

# OpenAlex `select` fields — top-level fields we actually use.
# Dropped: referenced_works (huge), concepts, mesh, topics, grants,
# authorships sub-fields come in full but we strip after extraction.
OA_SELECT = (
    "id,title,publication_year,cited_by_count,doi,"
    "authorships,abstract_inverted_index,"
    "open_access,primary_location,locations"
)

# arXiv ID pattern for extraction from landing_page_url
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/abs/)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")
_ARXIV_NORM_RE = re.compile(r"^([0-9]{4}\.[0-9]{4,5})v\d+$")


def _norm_arxiv_id(raw: str) -> str | None:
    """Normalize an arXiv ID extracted from a URL."""
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
# Abstract rebuild
# ---------------------------------------------------------------------------

def rebuild_abstract(inverted: dict | list | None) -> str:
    """Rebuild a plain-text abstract from OpenAlex inverted_index.

    The inverted_index maps each word to a list of character positions.
    If the field is a plain list (legacy format) we join directly.

    Args:
        inverted: OpenAlex ``abstract_inverted_index`` value (dict, list, or None).

    Returns:
        Plain-text abstract string (empty if no abstract).
    """
    if not inverted:
        return ""
    if isinstance(inverted, list):
        return " ".join(inverted).strip()
    # inverted_index: {word: [positions]}
    try:
        words = sorted(inverted.items(), key=lambda kv: kv[1][0])
        return " ".join(w for w, _ in words)
    except (TypeError, IndexError, KeyError):
        log.debug("Unable to rebuild abstract from inverted_index")
        return ""


# ---------------------------------------------------------------------------
# ArXiv ID extraction
# ---------------------------------------------------------------------------

def _extract_arxiv_id(locations: list | None, primary_location: dict | None) -> str | None:
    """Walk OpenAlex locations to find an arXiv landing page URL.

    Checks ``primary_location`` first, then iterates ``locations``.
    """
    candidates = []
    if primary_location:
        candidates.append(primary_location)
    if locations:
        candidates.extend(locations)

    for loc in candidates:
        if not loc:
            continue
        url = loc.get("landing_page_url") or ""
        if "arxiv" in url.lower():
            aid = _norm_arxiv_id(url)
            if aid:
                return aid
    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_openalex(
    query: str,
    max_results: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[dict]:
    """Search OpenAlex REST API.

    Args:
        query: free-text search query
        max_results: max papers to return (per the API's ``per_page``; capped at 200)
        year_min: minimum publication year (inclusive)
        year_max: maximum publication year (inclusive)

    Returns:
        List of unified paper dicts.  Empty list on error or timeout.
        Papers without an abstract are hard-filtered out.
    """
    papers: list[dict] = []

    params: dict = {
        "search": query.strip(),
        "per_page": min(max_results, 200),
        "sort": "cited_by_count:desc",
        "mailto": "research@example.com",
        "select": OA_SELECT,
    }

    # Build year filter
    if year_min is not None and year_max is not None:
        params["filter"] = f"publication_year:{year_min}-{year_max}"
    elif year_min is not None:
        params["filter"] = f"publication_year:>{year_min - 1}"
    elif year_max is not None:
        params["filter"] = f"publication_year:<{year_max + 1}"

    try:
        resp = requests.get(
            OPENALEX_SEARCH_URL,
            params=params,
            headers={"User-Agent": OPENALEX_USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        log.warning("OpenAlex search timed out after 15s — returning empty")
        return []
    except requests.RequestException as exc:
        log.warning("OpenAlex search failed: %s", exc)
        return []
    except Exception as exc:
        log.warning("OpenAlex parse failed: %s", exc)
        return []

    for w in data.get("results", []):
        # Authors
        authors_raw = w.get("authorships") or []
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in authors_raw
        ]
        authors = [a for a in authors if a]

        # Abstract: rebuild, then discard the heavy inverted index
        iv_idx = w.get("abstract_inverted_index")
        abstract = rebuild_abstract(iv_idx).strip()

        # Hard filter: skip papers without abstracts
        if not abstract:
            continue

        # DOI
        doi_val = w.get("doi") or ""
        doi_norm = doi_val.lower().lstrip("https://doi.org/").lstrip("http://doi.org/") if doi_val else ""

        # arXiv ID
        arxiv_id = _extract_arxiv_id(
            w.get("locations"), w.get("primary_location")
        )

        # Open access
        oa = w.get("open_access") or {}
        oa_status = oa.get("oa_status", "closed")
        pdf_url = oa.get("oa_url") or ""

        # Year
        year = w.get("publication_year") or 0

        # Venue (from primary_location.source)
        primary_loc = w.get("primary_location") or {}
        venue = ""
        if primary_loc and primary_loc.get("source"):
            venue = primary_loc["source"].get("display_name", "")

        papers.append({
            "paper_id": w.get("id", "").replace("https://openalex.org/", ""),
            "title": w.get("title", ""),
            "authors": authors,
            "year": year,
            "venue": venue or "",
            "abstract": abstract,
            "abstract_length": len(abstract),
            "citations": w.get("cited_by_count", 0),
            "source": "openalex",
            "doi": doi_norm,
            "arxiv_id": arxiv_id or "",
            "oa_status": oa_status,
            "pdf_url": pdf_url,
            "url": doi_val or w.get("id", ""),
        })

    log.info("OpenAlex returned %d papers (after abstract filter)", len(papers))
    return papers[:max_results]
