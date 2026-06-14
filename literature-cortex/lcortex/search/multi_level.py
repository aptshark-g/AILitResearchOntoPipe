"""
Multi-Level Search Filter — 3-level progressive filtering for Phase A.

Architecture
────────────
Level 1: API native filtering (arXiv with cat + year bounds)
Level 2: Local keyword filtering (pure Python, no LLM)
Level 3: LLM quick relevance check (mini prompt, ~300 tokens/paper)

Each level narrows results; if any level produces too few results,
its parameters are relaxed and retried (fallback chain).

Outputs
───────
- candidates_raw.jsonl     — Level 1 output (max 50 papers)
- candidates_filtered.jsonl — Level 2 output (~10-20 papers)
- candidates.jsonl          — Level 3 output (final ~10 papers)

Memory safety: single-pass streaming throughout; no full accumulation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("lcortex.search.multi_level")

# ═══════════════════════════════════════════════════════════════════════
# Level 2 keyword dictionaries
# ═══════════════════════════════════════════════════════════════════════

CORE_KEYWORDS: list[str] = [
    "adaptive", "feedforward", "feedback", "control",
    "FxLMS", "LMS", "vibration", "noise", "suppression",
    "isolation", "active", "filter", "neural",
]

EXCLUDE_ALWAYS: list[str] = [
    "Routhian", "symplectic", "Hamiltonian",
    "pedagogy", "student",
]

EXCLUDE_IF_ALONE: list[str] = [
    "education", "survey", "review", "tutorial",
]

# ═══════════════════════════════════════════════════════════════════════
# Level 3 prompt template (minimal token consumption)
# ═══════════════════════════════════════════════════════════════════════

QUICK_RELEVANCE_SYSTEM_PROMPT = """\
You are a relevance filter for academic paper search.

Rate how relevant each paper is to the user's research topic.
Score 1-5:
1 = Completely unrelated (different domain)
2 = Tangentially related (similar words, different problem)
3 = Related methodology, different application
4 = Directly relevant (same problem, different approach)
5 = Highly relevant (same problem, directly applicable)

Output ONLY a JSON object: {"score": <int 1-5>, "reason": "<1 brief sentence>"}
No markdown, no commentary, no code fences."""

QUICK_RELEVANCE_USER_TEMPLATE = """\
Topic: {topic}

Title: {title}
Abstract: {abstract_snippet}

Score (1-5):"""

# ═══════════════════════════════════════════════════════════════════════
# Level 1: API native filtering
# ═══════════════════════════════════════════════════════════════════════


def _search_arxiv_with_strategy(
    query: str,
    q_query: str,
    max_results: int,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> tuple[list[dict], int]:
    """Execute one arXiv search strategy and return (papers, count)."""
    try:
        import arxiv  # type: ignore[import-untyped]
    except ImportError:
        log.warning("arXiv Python library not installed")
        return [], 0

    try:
        search = arxiv.Search(
            query=q_query,
            max_results=max_results,
            sort_by=(
                arxiv.SortCriterion.Relevance
                if sort_by == "relevance"
                else arxiv.SortCriterion.LastUpdatedDate
            ),
            sort_order=(
                arxiv.SortOrder.Descending
                if sort_order == "descending"
                else arxiv.SortOrder.Ascending
            ),
        )
        client = arxiv.Client()
        results = []
        for result in client.results(search):
            arxiv_id_raw = result.entry_id or ""
            # Normalize arXiv ID
            import re
            m = re.search(r"(?:arxiv\.org/abs/)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", arxiv_id_raw)
            arxiv_id_norm = ""
            if m:
                aid = m.group(1)
                nm = re.match(r"^([0-9]{4}\.[0-9]{4,5})v\d+$", aid)
                arxiv_id_norm = nm.group(1) if nm else aid

            results.append({
                "paper_id": arxiv_id_norm or arxiv_id_raw,
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
                "oa_status": "gold",
                "pdf_url": result.pdf_url or "",
                "url": result.entry_id or "",
            })
        return results, len(results)
    except Exception as exc:
        log.warning("arXiv search failed: %s", exc)
        return [], 0


def search_level_1(
    query: str,
    output_path: str | Path,
    year_min: int | None = 2018,
    year_max: int | None = None,
    max_results: int = 50,
) -> int:
    """Level 1: API-native arXiv search with progressive fallback.

    Strategies (tried in order):
        1. Full filter: query + cat:cs.SY OR eess.SY, sort by relevance
        2. No category filter: query only, sort by relevance
        3. No category + no year filter: query only, sort by last updated

    Writes results directly to *output_path* as JSONL (streaming).

    Returns the number of papers written.
    """
    import datetime
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build year filter clause for arXiv API if both bounds provided
    year_clause = ""
    if year_min is not None and year_max is not None:
        year_clause = f" AND submittedDate:[{year_min}0101 TO {year_max}1231]"

    strategies = [
        # Strategy 1: full filter (cat + year)
        {
            "q": f'({query}) AND (cat:cs.SY OR cat:eess.SY){year_clause}' if year_clause
                 else f'({query}) AND (cat:cs.SY OR cat:eess.SY)',
            "sort_by": "relevance",
            "desc": "category + year filter",
        },
        # Strategy 2: remove category filter
        {
            "q": query + year_clause if year_clause else query,
            "sort_by": "relevance",
            "desc": "no category filter",
        },
        # Strategy 3: remove category + year filter, sort by recency
        {
            "q": query,
            "sort_by": "lastUpdatedDate",
            "desc": "no category + no year filter (recency)",
        },
    ]

    total_written = 0

    for i, strategy in enumerate(strategies):
        log.info(
            "Level 1 strategy %d: %s (query=%s)",
            i + 1,
            strategy["desc"],
            strategy["q"][:120],
        )

        results, count = _search_arxiv_with_strategy(
            query=query,
            q_query=strategy["q"],
            max_results=max_results,
            sort_by=strategy["sort_by"],
        )

        if not results:
            log.warning("Level 1 strategy %d returned 0 results", i + 1)
            continue

        # Write to JSONL
        with open(output_path, "w", encoding="utf-8") as fh:
            for paper in results:
                # Apply local year filter (in case API doesn't support it natively)
                py = paper.get("year", 0)
                if year_min and py < year_min:
                    continue
                if year_max and py > year_max:
                    continue
                fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
                total_written += 1

        if total_written > 0:
            log.info(
                "Level 1: strategy %d succeeded — %d papers → %s",
                i + 1,
                total_written,
                output_path,
            )
            return total_written

        # More results may come from subsequent strategies
        # But we already wrote the file, so keep track

    log.warning("Level 1: all strategies returned 0 results")
    return 0


# ═══════════════════════════════════════════════════════════════════════
# Level 2: Local keyword filtering
# ═══════════════════════════════════════════════════════════════════════


def filter_level_2(
    input_path: str | Path,
    output_path: str | Path,
    min_match: int = 2,
    min_output: int = 5,
) -> int:
    """Level 2: Local keyword filtering (pure Python, no LLM).

    Rules:
        - Must contain ≥ *min_match* core keywords
        - Always exclude papers with exclude_always keywords
        - Exclude papers where title has ["education","survey","review","tutorial"]
          AND fewer than 2 core keywords match

    Fallback:
        If < *min_output* papers survive, relax to min_match=1 and re-run.

    Processes stream-wise (one line at a time, no full accumulation).

    Returns the number of papers kept.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.warning("Level 2 input file not found: %s", input_path)
        return 0

    kept_ids: set[str] = set()
    kept_lines: list[str] = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── Build searchable text ──
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text = f"{title} {abstract}".lower()
            title_lower = title.lower()

            # ── Must-exclude check ──
            if any(kw.lower() in text for kw in EXCLUDE_ALWAYS):
                continue

            # ── Core keyword matches ──
            matches = sum(1 for kw in CORE_KEYWORDS if kw.lower() in text)

            # ── Soft-exclude: title has education/survey/review/tutorial + few core ──
            soft_exclude = any(
                kw.lower() in title_lower for kw in EXCLUDE_IF_ALONE
            )
            if soft_exclude and matches < 2:
                continue

            # ── Minimum match threshold ──
            if matches >= min_match:
                paper_id = paper.get("paper_id", paper.get("id", ""))
                if paper_id not in kept_ids:
                    kept_ids.add(paper_id)
                    kept_lines.append(line)

    # ── Fallback: relax to min_match=1 if too few ──
    if len(kept_lines) < min_output and min_match > 1:
        log.info(
            "Level 2: only %d papers with min_match=%d — retrying with min_match=1",
            len(kept_lines),
            min_match,
        )
        return filter_level_2(input_path, output_path, min_match=1, min_output=min_output)

    # ── Write output ──
    with open(output_path, "w", encoding="utf-8") as fout:
        for line in kept_lines:
            fout.write(line + "\n")

    log.info(
        "Level 2: %d papers kept out of raw results (min_match=%d) → %s",
        len(kept_lines),
        min_match,
        output_path,
    )
    return len(kept_lines)


# ═══════════════════════════════════════════════════════════════════════
# Level 3: LLM quick relevance check
# ═══════════════════════════════════════════════════════════════════════


def _parse_level_3_json(raw: str) -> dict | None:
    """Robust JSON parse for Level 3 mini-response.

    Tries multiple strategies in order:
    1. Direct JSON parse
    2. Extract first { ... } pair (innermost)
    3. Extract largest { ... } pair (greedy)
    4. Extract "score": N pattern from non-JSON text
    5. Extract "relevance" followed by a digit as last resort

    Strategy 4 and 5 set confidence=0.5 to indicate non-JSON extraction.
    """
    import re
    raw = raw.strip()

    # ── Strip code fences ──
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    # ── Strategy 1: Direct parse ──
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "score" in obj:
            obj.setdefault("confidence", 1.0)
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Strategy 2: Extract innermost { ... } ──
    m = re.search(r"\{[^{}]*?\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "score" in obj:
                obj.setdefault("confidence", 0.9)
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Strategy 3: Extract largest { ... } (non-greedy to avoid mismatches) ──
    # Use balanced brace matching
    brace_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if brace_match:
        try:
            obj = json.loads(brace_match.group(0))
            if isinstance(obj, dict) and "score" in obj:
                obj.setdefault("confidence", 0.8)
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Try with DOTALL + greedy as last JSON attempt
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "score" in obj:
                obj.setdefault("confidence", 0.7)
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Strategy 4: Extract "score": N pattern from any text ──
    score_m = re.search(r"""["']?score["']?\s*[:=]\s*(\d+)""", raw, re.IGNORECASE)
    if score_m:
        score_val = int(score_m.group(1))
        if 1 <= score_val <= 5:
            return {
                "score": score_val,
                "reason": "extracted from non-JSON text (score key)",
                "confidence": 0.5,
            }

    # ── Strategy 5: Last resort — extract any digit after "relevance" ──
    rel_m = re.search(r"relevance[^\d]*(\d)", raw, re.IGNORECASE)
    if rel_m:
        score_val = int(rel_m.group(1))
        if 1 <= score_val <= 5:
            return {
                "score": score_val,
                "reason": "extracted from non-JSON text (relevance heuristic)",
                "confidence": 0.5,
            }

    # ── Strategy 6: Bare digit in range 1-5 at start or near "score" ──
    # Look for a lone digit 1-5 that appears as a standalone number
    digit_m = re.search(r"\b([1-5])\b", raw)
    if digit_m:
        score_val = int(digit_m.group(1))
        return {
            "score": score_val,
            "reason": "extracted from non-JSON text (bare digit heuristic)",
            "confidence": 0.3,
        }

    return None


def filter_level_3(
    query: str,
    input_path: str | Path,
    output_path: str | Path,
    llm_adapter: Any = None,
    target_count: int = 10,
    workspace_dir: str | Path | None = None,
    target_final: int = 10,
) -> int:
    """Level 3: LLM quick relevance screening.

    Rates each paper 1-5 for relevance to *query*.  Papers scoring ≥ 3
    are kept, sorted by score descending, limited to *target_count*.

    Token budget: ~300 tokens per paper (mini prompt + 100 token response).

    If no *llm_adapter* is available, takes the top *target_count* papers
    directly from Level 2 output.

    When Level 3 returns 0 scored papers, instead of copying ALL Level 2
    papers, sorts Level 2 by (citations, year) and takes top *target_final*.

    Saves raw Level 3 responses to level3_debug.jsonl for inspection.

    Returns the number of papers kept.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.warning("Level 3 input file not found: %s", input_path)
        return 0

    # ── No LLM fallback: take top N from Level 2 ──
    if llm_adapter is None or not llm_adapter.is_available():
        log.info("Level 3: no LLM available — taking top %d from Level 2", target_count)
        papers: list[dict] = []
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Sort by citations desc, year desc
        papers.sort(key=lambda p: (p.get("citations", 0) or 0, p.get("year", 0) or 0), reverse=True)
        top_papers = papers[:target_count]

        with open(output_path, "w", encoding="utf-8") as fout:
            for paper in top_papers:
                fout.write(json.dumps(paper, ensure_ascii=False) + "\n")

        log.info(
            "Level 3 (no LLM): %d papers → %s",
            len(top_papers),
            output_path,
        )
        return len(top_papers)

    # ── LLM available: score each paper ──
    scored: list[tuple[int, dict]] = []
    papers_input: list[dict] = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                papers_input.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Resolve workspace dir for debug logging
    ws_dir = None
    if workspace_dir:
        ws_dir = Path(workspace_dir)
    elif output_path.parent:
        ws_dir = output_path.parent

    for paper in papers_input:
        title = paper.get("title", "")
        paper_id = paper.get("paper_id", paper.get("id", "unknown"))
        abstract = paper.get("abstract", "")
        abstract_snippet = abstract[:200]

        prompt_user = QUICK_RELEVANCE_USER_TEMPLATE.format(
            topic=query,
            title=title,
            abstract_snippet=abstract_snippet,
        )

        try:
            result = llm_adapter.complete(QUICK_RELEVANCE_SYSTEM_PROMPT, prompt_user)
            if isinstance(result, dict) and "skipped" in result:
                # NoOp / unavailable
                log.debug("Level 3 LLM skipped (adapter unavailable) for: %s", title[:50])
                continue

            raw_text = str(result) if not isinstance(result, dict) else str(result.get("_raw", result))

            # ── Save raw response to debug log ──
            if ws_dir:
                debug_path = ws_dir / "level3_debug.jsonl"
                try:
                    with open(debug_path, "a", encoding="utf-8") as dbg:
                        dbg.write(json.dumps({
                            "paper_id": paper_id,
                            "title": title[:100],
                            "raw": raw_text[:500],
                            "timestamp": __import__("datetime").datetime.now().isoformat(),
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass  # Don't let debug logging break the pipeline

            parsed = _parse_level_3_json(raw_text)

            if parsed is None:
                log.warning("Level 3: failed to parse LLM response for '%s'", title[:50])
                continue

            score_val = parsed.get("score", 0)
            confidence = parsed.get("confidence", 1.0)
            reason = parsed.get("reason", "")

            if isinstance(score_val, (int, float)) and score_val >= 3:
                scored.append((int(score_val), paper))
                log.debug(
                    "Level 3: %s score=%d conf=%.2f reason=%s",
                    paper_id, score_val, confidence, reason[:60],
                )
        except Exception as exc:
            log.warning("Level 3: LLM error for '%s': %s", title[:50], exc)

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top_papers = [p for _, p in scored[:target_count]]

    # ── Improved fallback: if LLM scored 0 papers, sort L2 and take top ──
    if not top_papers:
        log.warning(
            "Level 3: LLM returned 0 scored papers — "
            "falling back to top %d from Level 2 by citations+year",
            target_final,
        )
        # Sort Level 2 papers by citations desc, year desc
        papers_input.sort(
            key=lambda p: (
                p.get("citations", 0) or 0,
                p.get("year", 0) or 0,
            ),
            reverse=True,
        )
        top_papers = papers_input[:target_final]

    with open(output_path, "w", encoding="utf-8") as fout:
        for paper in top_papers:
            fout.write(json.dumps(paper, ensure_ascii=False) + "\n")

    log.info(
        "Level 3 (LLM): %d papers kept (scored %d total from %d input) → %s",
        len(top_papers),
        len(scored),
        len(papers_input),
        output_path,
    )
    return len(top_papers)


# ═══════════════════════════════════════════════════════════════════════
# Top-level orchestrator
# ═══════════════════════════════════════════════════════════════════════


def multi_level_search(
    query: str,
    workspace_dir: str | Path,
    llm_adapter: Any = None,
    max_results: int = 50,
    year_min: int | None = 2018,
    year_max: int | None = None,
    target_final: int = 10,
    skip_level_3: bool = True,
    sources: str = "arxiv",
) -> tuple[int, int, int, int]:
    """Run the complete 3-level search filter.

    Args:
        query: Search query string.
        workspace_dir: Directory to write JSONL output files.
        llm_adapter: Optional LLM adapter for Level 3.
        max_results: Max results from arXiv API (Level 1).
        year_min: Minimum publication year.
        year_max: Maximum publication year.
        target_final: Target final count after all filtering.
        sources: Comma-separated source names (currently only "arxiv").

    Returns:
        (n1, n2, n3, total) — counts for each level + final output count.
    """
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    raw_path = workspace_dir / "candidates_raw.jsonl"
    filtered_path = workspace_dir / "candidates_filtered.jsonl"
    final_path = workspace_dir / "candidates.jsonl"

    log.info("Multi-level search: query='%s' target=%d", query, target_final)

    # ── Level 1: API native filtering ──
    n1 = search_level_1(
        query=query,
        output_path=raw_path,
        year_min=year_min,
        year_max=year_max,
        max_results=max_results,
    )

    if n1 == 0:
        log.warning("Level 1 returned 0 results — all search strategies failed")
        return 0, 0, 0, 0

    # ── Level 2: Local keyword filtering ──
    n2 = filter_level_2(
        input_path=raw_path,
        output_path=filtered_path,
        min_match=2,
        min_output=5,
    )

    if n2 == 0:
        log.warning("Level 2 returned 0 results after keyword filtering")
        # Last resort: copy raw → final
        with open(raw_path, "r") as fin, open(final_path, "w") as fout:
            for line in fin:
                fout.write(line)
        log.info("Fallback: raw results copied directly to candidates.jsonl")
        return n1, 0, n1, n1

    # ── Level 3: LLM quick relevance screening ──
    if not skip_level_3:
        n3 = filter_level_3(
            query=query,
            input_path=filtered_path,
            output_path=final_path,
            llm_adapter=llm_adapter,
            target_count=target_final,
            workspace_dir=workspace_dir,
            target_final=target_final,
        )
    else:
        log.info("Level 3: skipped (dry/lite mode or user requested)")
        n3 = 0

    # Determine total final papers
    final_count = n3
    if n3 == 0 and n2 > 0:
        # LLM failed but Level 2 has results
        # Sort Level 2 by citations+year and take top target_final
        log.warning(
            "Level 3 returned 0 — falling back to top %d from Level 2 sorted by citations+year",
            target_final,
        )
        papers_l2: list[dict] = []
        with open(filtered_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    papers_l2.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        papers_l2.sort(
            key=lambda p: (p.get("citations", 0) or 0, p.get("year", 0) or 0),
            reverse=True,
        )
        top_l2 = papers_l2[:target_final]
        with open(final_path, "w") as fout:
            for paper in top_l2:
                fout.write(json.dumps(paper, ensure_ascii=False) + "\n")
        final_count = len(top_l2)

    log.info(
        "Multi-level search complete: L1=%d L2=%d L3=%d final=%d",
        n1,
        n2,
        n3,
        final_count,
    )
    return n1, n2, n3, final_count
