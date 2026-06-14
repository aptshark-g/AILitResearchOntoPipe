"""
Unified Search Facade — multi-source academic search with degradation ladder.

Architecture
────────────
  Layer 1 (API, fast, parallel within ThreadPoolExecutor):
    ├── OpenAlex  — REST API, 100K req/day, JSON
    ├── arXiv     — Python ``arxiv`` lib, 1 QPS, abstract + PDF
    └── S2        — Semantic Scholar REST API, requires ``S2_API_KEY``

  Layer 2 (opencli, slower, sequential, browser-backed):
    ├── dblp           — ``opencli dblp search``  (no browser, 15 s timeout)
    ├── Google Scholar — ``opencli google-scholar search`` (browser, 60 s)
    ├── CNKI           — ``opencli cnki search``  (browser, 60 s)
    └── 百度学术        — ``opencli baidu-scholar search`` (browser, 60 s)

  Degradation ladder
  ──────────────────
  1. Always call OpenAlex + arXiv in parallel.
  2. If ``S2_API_KEY`` is set → call S2 in parallel.
  3. If results < ``MIN_PAPERS`` AND query is CS → call dblp (fast, no browser).
  4. If results < ``MIN_PAPERS`` → call Google Scholar (slow).
  5. If query contains Chinese → also call CNKI.
  6. If still < ``MIN_PAPERS`` + Chinese → try 百度学术 (last resort).

Unified paper schema
────────────────────
Every returned paper dict contains these keys::

    paper_id, title, authors, year, venue,
    abstract, abstract_length, citations, source,
    doi, arxiv_id, oa_status, pdf_url, url
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from lcortex.search.arxiv import search_arxiv  # noqa: F811
from lcortex.search.dedup import dedup_papers
from lcortex.search.openalex import search_openalex

log = logging.getLogger("lcortex.search.adapter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum query length — combined queries can be huge, cap to prevent OOM
MAX_QUERY_LENGTH = 200

# Minimum paper count before triggering Layer-2 fallback
MIN_PAPERS = 15

# CS heuristic keywords for triggering dblp
_CS_KEYWORDS = [
    "neural", "network", "learning", "transformer", "attention", "llm", "gpt",
    "bert", "diffusion", "reinforcement", "gan", "rnn", "cnn", "lstm",
    "computer vision", "nlp", "natural language", "robotics",
    "optimization", "gradient", "backprop", "convolution",
    "machine", "deep", "agent", "token", "embedding",
    "graph", "model", "training", "inference", "fine-tun",
    "generat", "language model", "self-attention", "vision",
]

# Chinese character detection
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")

# ---- Fields we strip after extraction to reduce memory ----
_REDUNDANT_FIELDS = (
    "referenced_works", "locations", "abstract_inverted_index",
    "authorships", "concepts", "mesh", "topics",
    "related_works", "counts_by_year", "grants",
    "ids", "host_venue", "alternate_host_venues",
    "corresponding_author_ids", "corresponding_institution_ids",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env(key: str) -> str | None:
    """Read a single key from the LCORTEX_* environment or .env file.

    Lookup order:
      1. ``LCORTEX_S2_API_KEY`` environment variable
      2. ``.env`` file in the project root (fallback, for skill compat)
    """
    # First check LCORTEX-prefixed env var
    mapped = {
        "S2_API_KEY": ("LCORTEX_S2_API_KEY",),
    }
    if key in mapped:
        for env_name in mapped[key]:
            val = os.environ.get(env_name)
            if val:
                return val
    # Generic env var
    val = os.environ.get(key)
    if val:
        return val

    # Fallback: .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if not os.path.isfile(env_path):
        return None
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return None


def has_chinese(text: str) -> bool:
    """Check whether a string contains Chinese characters."""
    return bool(_CHINESE_RE.search(text))


def is_cs_query(query: str) -> bool:
    """Heuristic: whether a query looks like a CS/ML topic.

    Used to decide whether to trigger dblp as Layer-2 fallback.
    """
    ql = query.lower()
    return any(kw in ql for kw in _CS_KEYWORDS)


def _strip_redundant_fields(paper: dict) -> dict:
    """Remove large/unnecessary fields from a paper dict.

    Fields like ``referenced_works`` (OpenAlex can return 100s of IDs),
    ``abstract_inverted_index`` (already rebuilt), etc. are memory hogs
    and never used downstream.
    """
    for key in _REDUNDANT_FIELDS:
        paper.pop(key, None)
    return paper


# ---------------------------------------------------------------------------
# OpenCLI adapter (Layer 2)
# ---------------------------------------------------------------------------

def _run_opencli(
    args: list[str],
    timeout: int = 60,
) -> list[dict]:
    """Run opencli in a subprocess and parse JSON output.

    Args:
        args:    opencli arguments, e.g. ``["dblp", "search", "attention", "--limit", "10", "-f", "json"]``
        timeout: subprocess timeout in seconds

    Returns:
        List of raw dicts from JSON output.  Empty list on failure.
    """
    cmd = ["opencli"] + args
    log.debug("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            log.warning("opencli exited %d stderr=%s", proc.returncode, proc.stderr[:200])
            return []

        stdout = proc.stdout.strip()
        if not stdout:
            return []

        # Try JSON parse
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            # Maybe JSON lines?
            try:
                parsed = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            except json.JSONDecodeError:
                log.warning("opencli output is not valid JSON: %s", stdout[:200])
                return []

        # Unwrap common wrapper keys
        if isinstance(parsed, dict):
            for key in ("results", "items", "data", "hits"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]

        if not isinstance(parsed, list):
            parsed = [parsed]

        return parsed

    except subprocess.TimeoutExpired:
        log.warning("opencli %s timed out after %ss", args[0], timeout)
    except FileNotFoundError:
        log.warning("opencli not found in PATH — Layer 2 disabled")
    except Exception as exc:
        log.warning("opencli error: %s", exc)

    return []


# arXiv ID pattern reused from openalex.py but needed here for opencli mapping
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/abs/)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")
_ARXIV_NORM_RE = re.compile(r"^([0-9]{4}\.[0-9]{4,5})v\d+$")


def _norm_arxiv_id(raw: str) -> str | None:
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


def _norm_doi(raw: str) -> str | None:
    """Lowercase + strip DOI URL prefix."""
    if not raw:
        return None
    d = raw.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d


def _map_opencli_to_paper(raw: dict, source: str) -> dict:
    """Map opencli raw output to the unified paper schema.

    opencli adapters output varying field names; this function does
    best-effort fuzzy matching across known key names (English + Chinese).
    """
    title = (
        raw.get("title")
        or raw.get("name")
        or raw.get("标题")
        or raw.get("题目")
        or ""
    )

    # Authors: list, semicolon/pipe separated, or comma separated
    authors = raw.get("authors") or raw.get("作者") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in re.split(r"[;,|]\s*", authors) if a.strip()]

    # Year: try several known keys
    year = 0
    year_val = (
        raw.get("year")
        or raw.get("date")
        or raw.get("published")
        or raw.get("日期")
        or raw.get("出版年份")
        or ""
    )
    if year_val:
        m = re.search(r"(19|20)\d{2}", str(year_val))
        if m:
            year = int(m.group(0))

    venue = (
        raw.get("venue")
        or raw.get("journal")
        or raw.get("conference")
        or raw.get("journal_name")
        or raw.get("期刊")
        or raw.get("会议")
        or ""
    )

    abstract = (
        raw.get("abstract")
        or raw.get("snippet")
        or raw.get("摘要")
        or raw.get("description")
        or ""
    )

    doi_val = raw.get("doi") or raw.get("DOI") or ""
    if doi_val:
        doi_val = _norm_doi(doi_val) or doi_val

    arxiv_id = ""
    raw_id = raw.get("arxiv_id") or raw.get("id") or ""
    if "arxiv" in str(raw).lower() or "arxiv" in str(raw_id).lower():
        arxiv_id = _norm_arxiv_id(raw_id) or ""

    citations = raw.get("citations") or raw.get("citation_count") or raw.get("cited_by") or 0
    try:
        citations = int(citations)
    except (ValueError, TypeError):
        citations = 0

    url_val = raw.get("url") or raw.get("link") or raw.get("href") or ""
    pdf_url = raw.get("pdf_url") or raw.get("pdf") or ""

    return {
        "paper_id": str(raw.get("id") or raw.get("paper_id") or hash(title)),
        "title": title,
        "authors": authors,
        "year": year,
        "venue": str(venue) if venue else "",
        "abstract": abstract,
        "abstract_length": len(abstract),
        "citations": citations,
        "source": source,
        "doi": doi_val or "",
        "arxiv_id": arxiv_id,
        "oa_status": "unknown",
        "pdf_url": pdf_url,
        "url": url_val,
    }


def search_via_opencli(source: str, query: str, limit: int = 10) -> list[dict]:
    """Layer 2 search via opencli.

    Args:
        source: one of ``"dblp"``, ``"google-scholar"``, ``"cnki"``, ``"baidu-scholar"``
        query:  search string
        limit:  max results

    Returns:
        List of unified paper dicts. Empty list on failure.
    """
    cmd_map: dict[str, tuple[list[str], int]] = {
        "dblp": (
            ["dblp", "search", query, "--limit", str(limit), "-f", "json"],
            15,  # dblp is fast, no browser
        ),
        "google-scholar": (
            ["google-scholar", "search", query, "--limit", str(limit), "-f", "json"],
            60,  # browser-backed
        ),
        "cnki": (
            ["cnki", "search", query, "--limit", str(limit), "-f", "json"],
            60,  # browser-backed
        ),
        "baidu-scholar": (
            ["baidu-scholar", "search", query, "--limit", str(limit), "-f", "json"],
            60,  # browser-backed, least reliable
        ),
    }

    if source not in cmd_map:
        log.warning("Unknown opencli source: %s", source)
        return []

    args, timeout = cmd_map[source]
    raw_results = _run_opencli(args, timeout=timeout)

    source_label = f"opencli:{source}"
    papers = [_map_opencli_to_paper(r, source_label) for r in raw_results]

    # Filter out completely empty records
    papers = [p for p in papers if p["title"].strip()]
    log.info("opencli:%s returned %d papers", source, len(papers))
    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar adapter (Layer 1)
# ---------------------------------------------------------------------------

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "title,authors,year,venue,abstract,externalIds,url,"
    "citationCount,openAccessPdf,publicationTypes"
)
S2_MAX_PER_REQUEST = 100
S2_RATE_LIMIT_SLEEP = 3.1  # seconds (100 req / 5 min ≈ 1 per 3 s)


def search_s2(
    query: str,
    max_results: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[dict]:
    """Search Semantic Scholar. Requires ``S2_API_KEY`` env var.

    Rate limit: 100 requests per 5 minutes without API key.
    """
    api_key = _load_env("S2_API_KEY")
    if not api_key:
        log.info("S2_API_KEY not set, skipping Semantic Scholar")
        return []

    headers = {"x-api-key": api_key}
    papers: list[dict] = []
    params: dict = {
        "query": query,
        "limit": min(max_results, S2_MAX_PER_REQUEST),
        "fields": S2_FIELDS,
    }
    if year_min is not None and year_max is not None:
        params["year"] = f"{year_min}-{year_max}"
    elif year_min is not None:
        params["year"] = f"{year_min}-"

    try:
        resp = requests.get(S2_SEARCH_URL, headers=headers, params=params, timeout=20)
        if resp.status_code == 429:
            log.warning("S2 rate-limited, retrying after 5s...")
            time.sleep(5)
            resp = requests.get(S2_SEARCH_URL, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        for p in data.get("data", []):
            external_ids = p.get("externalIds") or {}
            doi_val = external_ids.get("DOI", "")
            arxiv_id = _norm_arxiv_id(external_ids.get("ArXiv") or "")

            authors_raw = p.get("authors") or []
            authors = [a.get("name", "") for a in authors_raw]

            year = p.get("year") or 0

            venue_raw = p.get("venue") or ""
            if isinstance(venue_raw, dict):
                venue_raw = venue_raw.get("name", "") or ""

            oa = p.get("openAccessPdf") or {}
            oa_status = "gold" if oa.get("url") else "unknown"
            pdf_url = oa.get("url") or ""

            papers.append({
                "paper_id": p.get("paperId", ""),
                "title": p.get("title", ""),
                "authors": authors,
                "year": year,
                "venue": str(venue_raw) if venue_raw else "",
                "abstract": p.get("abstract") or "",
                "abstract_length": len(p.get("abstract") or ""),
                "citations": p.get("citationCount", 0),
                "source": "semantic_scholar",
                "doi": doi_val.lower() if doi_val else "",
                "arxiv_id": arxiv_id or "",
                "oa_status": oa_status,
                "pdf_url": pdf_url,
                "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
            })

        log.info("Semantic Scholar returned %d results", len(papers))
    except requests.Timeout:
        log.warning("Semantic Scholar timed out — skipping")
    except requests.RequestException as exc:
        log.warning("S2 search failed: %s", exc)
    except Exception as exc:
        log.warning("S2 parse failed: %s", exc)

    # Respect rate limit
    time.sleep(S2_RATE_LIMIT_SLEEP)

    return papers[:max_results]


# ---------------------------------------------------------------------------
# Unified Search
# ---------------------------------------------------------------------------

def unified_search(
    query: str,
    max_per_source: int = 8,
    year_min: int | None = None,
    year_max: int | None = None,
    sources: list[str] | None = None,
    lang: str = "auto",
) -> dict:
    """Unified search across multiple academic sources with degradation ladder.

    This is the primary entry point for Phase A (search).

    Args:
        query:           free-text search query (truncated to 200 chars)
        max_per_source:  max results per source (default 8)
        year_min:        minimum publication year filter
        year_max:        maximum publication year filter
        sources:         Layer-1 sources to use.  Default: all three.
        lang:            ``"auto"`` → auto-detect Chinese; ``"en"`` / ``"zh"`` to force.

    Returns:
        A dict with two keys:
            - ``papers`` — list of deduplicated, year-filtered, sorted paper dicts
            - ``sources_status`` — dict mapping source name → ``{status, count}``
    """
    # ── Guard: truncate long queries ──
    query = query.strip()[:MAX_QUERY_LENGTH]

    if sources is None:
        sources = ["openalex", "arxiv", "semantic_scholar"]

    is_chinese = lang == "zh" or (lang == "auto" and has_chinese(query))
    is_cs = is_cs_query(query)

    all_papers: list[dict] = []
    sources_status: dict[str, dict] = {}

    # ── Layer 1: parallel API calls ──
    s2_enabled = "semantic_scholar" in sources and bool(_load_env("S2_API_KEY"))
    if not s2_enabled and "semantic_scholar" in sources:
        log.info("S2_API_KEY not set, using OpenAlex + arXiv only")

    tasks: dict[str, any] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        if "openalex" in sources:
            tasks["oa"] = ex.submit(
                search_openalex, query, max_per_source, year_min, year_max
            )
        if "arxiv" in sources:
            tasks["ax"] = ex.submit(search_arxiv, query, max_per_source)
        if s2_enabled:
            tasks["s2"] = ex.submit(
                search_s2, query, max_per_source, year_min, year_max
            )

        for label, fut in tasks.items():
            source_name = {"oa": "openalex", "ax": "arxiv", "s2": "semantic_scholar"}[label]
            try:
                results = fut.result(timeout=35)
                results = [_strip_redundant_fields(p) for p in results]
                all_papers.extend(results)
                sources_status[source_name] = {"status": "ok", "count": len(results)}
            except Exception as exc:
                log.warning("Layer-1 %s future failed: %s", label, exc)
                sources_status[source_name] = {"status": "error", "count": 0, "reason": str(exc)[:200]}
            finally:
                gc.collect()

    log.info("Layer 1: %d papers (before dedup)", len(all_papers))

    # Dedup after Layer 1
    all_papers = dedup_papers(all_papers)

    # ── Layer 2: on-demand opencli calls ──
    need_l2 = len(all_papers) < MIN_PAPERS
    l2_limit = min(max_per_source, 10)

    # dblp: CS + need more
    if need_l2 and is_cs:
        try:
            dblp_results = search_via_opencli("dblp", query, limit=l2_limit)
            dblp_results = [_strip_redundant_fields(p) for p in dblp_results]
            all_papers.extend(dblp_results)
            sources_status["opencli:dblp"] = {"status": "ok", "count": len(dblp_results)}
            gc.collect()
        except Exception as exc:
            log.warning("dblp opencli failed: %s", exc)
            sources_status["opencli:dblp"] = {"status": "error", "count": 0, "reason": str(exc)[:200]}

    all_papers = dedup_papers(all_papers)

    # Google Scholar: not CS or still need more
    if len(all_papers) < MIN_PAPERS:
        try:
            gs_results = search_via_opencli("google-scholar", query, limit=l2_limit)
            gs_results = [_strip_redundant_fields(p) for p in gs_results]
            all_papers.extend(gs_results)
            sources_status["opencli:google-scholar"] = {"status": "ok", "count": len(gs_results)}
            gc.collect()
        except Exception as exc:
            log.warning("google-scholar opencli failed: %s", exc)
            sources_status["opencli:google-scholar"] = {"status": "error", "count": 0, "reason": str(exc)[:200]}

    all_papers = dedup_papers(all_papers)

    # CNKI: Chinese query
    if is_chinese:
        try:
            cnki_results = search_via_opencli("cnki", query, limit=l2_limit)
            cnki_results = [_strip_redundant_fields(p) for p in cnki_results]
            all_papers.extend(cnki_results)
            sources_status["opencli:cnki"] = {"status": "ok", "count": len(cnki_results)}
            gc.collect()
        except Exception as exc:
            log.warning("cnki opencli failed: %s", exc)
            sources_status["opencli:cnki"] = {"status": "error", "count": 0, "reason": str(exc)[:200]}

    # 百度学术: last resort for Chinese
    if is_chinese and len(all_papers) < MIN_PAPERS:
        try:
            baidu_results = search_via_opencli("baidu-scholar", query, limit=l2_limit)
            baidu_results = [_strip_redundant_fields(p) for p in baidu_results]
            all_papers.extend(baidu_results)
            sources_status["opencli:baidu-scholar"] = {"status": "ok", "count": len(baidu_results)}
            gc.collect()
        except Exception as exc:
            log.warning("baidu-scholar opencli failed: %s", exc)
            sources_status["opencli:baidu-scholar"] = {"status": "error", "count": 0, "reason": str(exc)[:200]}

    # Final dedup
    all_papers = dedup_papers(all_papers)

    # Enforce year filter on combined results
    all_papers = [
        p for p in all_papers
        if (year_min is None or int(p.get("year") or 0) >= year_min)
        and (year_max is None or int(p.get("year") or 0) <= year_max)
    ]

    # Sort: citations desc, then year desc
    all_papers.sort(
        key=lambda p: (int(p.get("citations") or 0), int(p.get("year") or 0)),
        reverse=True,
    )

    sources_status["__total__"] = len(all_papers)
    log.info("Final: %d papers after dedup + filtering", len(all_papers))

    return {
        "papers": all_papers,
        "sources_status": sources_status,
    }
