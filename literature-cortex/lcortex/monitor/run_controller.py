"""Run Controller — orchestrates a full A→G run with monitoring.

The RunController wraps each pipeline phase with:
  - Phase start/end bookkeeping via PipelineMonitor
  - Item-level progress tracking
  - Token counting (for LLM phases)
  - Error recovery (skip failing papers, continue pipeline)
  - Graceful degradation when LLM is unavailable (NoOpAdapter)
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from lcortex.core.config import Config, get_config
from lcortex.core.state import (
    PHASES,
    PHASE_LABELS,
    PipelineState,
    PhaseStatus,
    checkpoint_marker,
    save_state,
)
from lcortex.intelligence.base import LLMAdapter
from lcortex.intelligence.adapters.noop import NoOpAdapter
from lcortex.monitor.monitor import PipelineMonitor

log = logging.getLogger("lcortex.run_controller")

# ────────────────────────────────────────────────────────────────────────
# Shared stopwords for CD fallback query extraction
# ────────────────────────────────────────────────────────────────────────
_STOPS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to",
    "for", "with", "by", "at", "from", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "based", "using", "via", "new",
    "novel", "approach", "method", "system", "design",
    "control", "analysis", "model", "study", "towards",
    "toward", "application", "experimental", "proposed",
    "adaptive", "active", "robust", "improved", "efficient",
    "modified", "enhanced", "nonlinear", "linear", "hybrid",
}

# ────────────────────────────────────────────────────────────────────────
# Phase icons for CLI output
# ────────────────────────────────────────────────────────────────────────
_ICON = {
    "A": "🔍",
    "B1": "🧮",
    "B2": "🤖",
    "C": "📚",
    "D": "📚",
    "E": "📝",
    "F": "🏗️",
    "G": "📊",
    "F2": "🔄",
}

_PHASE_NAME = {
    "A": "Search",
    "B1": "Dry Scoring",
    "B2": "Scoring (LLM)",
    "C": "Limitations",
    "D": "Extensions",
    "E": "Synthesis",
    "F": "Structure",
    "G": "Export",
    "F2": "Conflict",
}


def _format_duration(seconds: float) -> str:
    """Pretty-print a duration in seconds."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{s:.0f}s"


def _build_fallback_review(
    topic: str,
    core_papers: list[dict],
    lim_papers: list[dict],
    ext_papers: list[dict],
    mode: str = "lite",
) -> str:
    """Build a structured fallback review with actual paper data.

    Used when LLM synthesis fails or is unavailable.
    Always produces >500 bytes of real content.
    """
    import time as _time

    # Rank core papers: prefer B2 mean_score, fallback to B1 dry_score
    def _rank_key(p: dict) -> float:
        b2 = p.get("mean_score", None)
        if b2 is not None and b2 > 0:
            return float(b2)
        return float(p.get("dry_score", 0))

    ranked = sorted(core_papers, key=_rank_key, reverse=True)
    top = ranked[:10]

    lines = [f"# Literature Review: {topic}\n\n"]
    lines.append(
        "*Auto-generated structured review (LLM not available). "
        "Sections are derived from pipeline scores and metadata.*\n\n"
    )
    lines.append(
        f"**Pipeline mode:** {mode}  |  "
        f"**Core papers:** {len(core_papers)}  |  "
        f"**Limitations:** {len(lim_papers)}  |  "
        f"**Extensions:** {len(ext_papers)}  |  "
        f"**Total:** {len(core_papers) + len(lim_papers) + len(ext_papers)}\n\n"
    )
    lines.append("---\n\n")

    # ── Overview section ─────────────────────────────────────────────
    lines.append("## Overview\n\n")
    if ranked:
        year_counts: dict[str, int] = {}
        for p in ranked:
            y = str(p.get("year", p.get("published_year", "?")))
            year_counts[y] = year_counts.get(y, 0) + 1
        year_summary = ", ".join(
            f"{y}: {c}" for y, c in sorted(year_counts.items(), reverse=True)
        )
        lines.append(f"- **Year distribution:** {year_summary}\n")

        avg_b1 = sum(float(p.get("dry_score", 0)) for p in ranked) / max(len(ranked), 1)
        lines.append(f"- **Average B1 dry score:** {avg_b1:.2f}\n")

        b2_scored = [p for p in ranked if p.get("mean_score") is not None]
        if b2_scored:
            avg_b2 = sum(float(p.get("mean_score", 0)) for p in b2_scored) / len(b2_scored)
            lines.append(f"- **Average B2 mean score (scored only):** {avg_b2:.2f} ({len(b2_scored)} papers)\n")

        kw_counter: dict[str, int] = {}
        for p in ranked:
            for kw in p.get("keywords", []):
                kw_counter[kw] = kw_counter.get(kw, 0) + 1
        top_kws = sorted(kw_counter.items(), key=lambda x: x[1], reverse=True)[:8]
        if top_kws:
            kw_line = ", ".join(f"{kw} ({n})" for kw, n in top_kws)
            lines.append(f"- **Top keywords:** {kw_line}\n")
    lines.append("\n")

    # ── Top Papers (ranked) ──────────────────────────────────────────
    lines.append("## Top Papers\n\n")
    lines.append(
        "Papers ranked by B2 mean score (when available), then by B1 dry score.\n\n"
    )

    for i, p in enumerate(top, 1):
        pid = p.get("paper_id", p.get("id", f"paper-{i}"))
        title = p.get("title", pid)
        b1_score = p.get("dry_score", 0)
        b2_score = p.get("mean_score", None)
        year = p.get("year", p.get("published_year", "?"))
        abstract = p.get("abstract", "")
        keywords = p.get("keywords", [])

        lines.append(f"### {i}. {title}\n\n")
        lines.append(f"| Field | Value |\n")
        lines.append(f"|-------|-------|\n")
        lines.append(f"| **Paper ID** | `{pid}` |\n")
        lines.append(f"| **Year** | {year} |\n")
        lines.append(f"| **B1 Dry Score** | {b1_score:.3f} |\n")
        if b2_score is not None:
            lines.append(f"| **B2 Mean Score** | {b2_score:.3f} |\n")
        if keywords:
            lines.append(f"| **Keywords** | {', '.join(keywords)} |\n")
        lines.append("\n")

        if abstract:
            ab = abstract[:500].rstrip()
            lines.append(f"**Abstract:** {ab}{'...' if len(abstract) > 500 else ''}\n\n")

        lines.append("\n")

    # ── Limitation Papers ────────────────────────────────────────────
    if lim_papers:
        lines.append("---\n\n")
        lines.append("## Limitation Papers\n\n")
        lines.append(
            "Papers identified as critiques, gaps, or limitations "
            "related to the core topic.\n\n"
        )
        for p in lim_papers[:8]:
            title = p.get("title", p.get("paper_id", "?"))
            pid = p.get("paper_id", p.get("id", "?"))
            year = p.get("year", p.get("published_year", "?"))
            abstract = p.get("abstract", "")
            lines.append(f"### {title}\n\n")
            lines.append(f"- **ID:** `{pid}` | **Year:** {year}\n")
            if abstract:
                ab = abstract[:200].rstrip()
                lines.append(f"- {ab}{'...' if len(abstract) > 200 else ''}\n")
            lines.append("\n")

    # ── Extension / Recent Papers ────────────────────────────────────
    if ext_papers:
        lines.append("---\n\n")
        lines.append("## Recent Advances / Extensions\n\n")
        lines.append(
            "Papers representing recent advances, improvements, "
            "or extensions of methods in the core topic.\n\n"
        )
        for p in ext_papers[:8]:
            title = p.get("title", p.get("paper_id", "?"))
            pid = p.get("paper_id", p.get("id", "?"))
            year = p.get("year", p.get("published_year", "?"))
            abstract = p.get("abstract", "")
            lines.append(f"### {title}\n\n")
            lines.append(f"- **ID:** `{pid}` | **Year:** {year}\n")
            if abstract:
                ab = abstract[:200].rstrip()
                lines.append(f"- {ab}{'...' if len(abstract) > 200 else ''}\n")
            lines.append("\n")

    # ── References section ───────────────────────────────────────────
    all_papers = list(core_papers) + list(lim_papers) + list(ext_papers)
    if all_papers:
        lines.append("---\n\n")
        lines.append("## References\n\n")
        seen_ids: set[str] = set()
        ref_num = 0
        for p in all_papers:
            pid = p.get("paper_id", p.get("id", ""))
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            ref_num += 1
            title = p.get("title", pid)
            year = p.get("year", p.get("published_year", "?"))
            authors = p.get("authors", [])
            if isinstance(authors, list) and authors:
                author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
            else:
                author_str = "Unknown"
            lines.append(f"{ref_num}. {author_str} ({year}). *{title}*. `{pid}`\n")
        lines.append("\n")

    lines.append("---\n\n")
    lines.append(
        f"*Generated by Literature Cortex pipeline "
        f"(mode={mode}) on {_time.strftime('%Y-%m-%d %H:%M')}.*\n"
    )

    return "".join(lines)


class RunController:
    """Orchestrates a full A→G run with monitoring.

    Handles LLM/no-LLM modes gracefully — when using a ``NoOpAdapter``,
    LLM-requiring phases (B, E, F, F-2) are skipped with a log message.

    Parameters
    ----------
    config:
        Literature Cortex configuration.
    adapter:
        LLM adapter instance (may be NoOpAdapter for dry/no-LLM modes).
    monitor:
        PipelineMonitor instance for progress tracking.
    """

    def __init__(
        self,
        config: Config,
        adapter: LLMAdapter,
        monitor: PipelineMonitor,
        resource_profile: "ResourceProfile | None" = None,
    ):
        self._config = config
        self._adapter = adapter
        self._monitor = monitor
        self._mode = monitor._mode if monitor else "lite"
        self._no_llm = isinstance(adapter, NoOpAdapter) or not adapter.is_available()

        # Resource profile (auto-detect or user-specified)
        if resource_profile is None:
            from lcortex.core.resources import ResourceProfile
            resource_profile = ResourceProfile.from_env()
        self._rp = resource_profile

    # ────────────────────────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────────────────────────
    def run_full_pipeline(
        self,
        query: str,
        mode: str = "lite",
        max_results: int = 10,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sources: str = "arxiv",
        export_format: str = "obsidian",
    ) -> dict:
        """Run phases A through G.

        Parameters
        ----------
        query:
            Search query string.
        mode:
            Pipeline mode: ``"dry"``, ``"lite"``, or ``"full"``.
        max_results:
            Maximum number of papers to retrieve in Phase A.
        year_min, year_max:
            Publication year range.
        sources:
            Comma-separated search sources.
        export_format:
            Export format (``"obsidian"`` or ``"json"``).

        Returns
        -------
        dict
            Final summary from ``monitor.final_summary()``.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:80]
        ws = Path(self._config.workspace.path)
        run_dir = ws / slug

        # ── State ──────────────────────────────────────────────────
        state = PipelineState(
            query=query,
            slug=slug,
            mode=mode,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        self._monitor.start_run(slug, query)

        # ── Banner ──────────────────────────────────────────────────
        mode_label = mode.upper()
        llm_status = "no LLM" if self._no_llm else "LLM available"
        print(f"\n{'='*60}")
        print(f"⚙️  Resource: {self._rp.profile_summary}")
        print(f"Literature Cortex Pipeline — {mode_label} mode")
        print(f"Query: {query}")
        print(f"Workspace: {run_dir}")
        print(f"LLM: {llm_status}")
        print(f"⚙️  Resource: {self._rp.profile_summary}")
        print(f"{'='*60}")

        try:
            # ── Phase A: Multi-Level Search  ────────────────────────
            state, papers = self.run_phase_a(
                query, state, max_results, year_min, year_max, sources,
                workspace=run_dir,
            )
        except Exception as exc:
            log.exception("Phase A failed: %s", exc)
            state.mark_error("A", str(exc))
            self._monitor.record_error("A", str(exc), recoverable=False)
            state, papers = state, []

        # Determine which phases to run based on mode
        run_b1 = True  # B1 always runs (dry NLP scoring)
        run_b2 = mode in ("lite", "full")  # B2 LLM 4C+L
        run_cd = mode in ("lite", "full")  # C/D needs arXiv search → OOM risk in dry
        run_e = mode in ("lite", "full")  # E runs in lite+ (needs LLM or fallback)
        run_f = mode in ("dry", "lite", "full")  # F: lite-matcher in dry/lite, LLM in full
        run_f2 = (mode == "full")  # F-2: divergence detection (needs F results + LLM) + seed edges

        # Phase F templates (carried forward to Phase F-2 and G)
        templates: list[dict] = []

        # ── Phase B1: Dry NLP Scoring (always)  ────────────────────
        dry_results: list[dict] = []
        if run_b1 and papers:
            try:
                state, dry_results = self.run_phase_b1(
                    papers, state, query, workspace=run_dir,
                )
            except Exception as exc:
                log.exception("Phase B1 failed: %s", exc)
                state.mark_error("B1", str(exc))
                self._monitor.record_error("B1", str(exc), recoverable=False)
                dry_results = []

        # ── Phase B2: LLM 4C+L Scoring (top-N from B1)  ───────────
        scored: list[dict] = []
        if run_b2:
            try:
                # Select top papers from B1 for LLM scoring
                from lcortex.analysis.dry_scorer import PASS_THRESHOLD
                b2_top_n = self._rp.b2_top_n
                top_for_b2 = [p for p in papers
                              if any(r.get("paper_id") == p.get("paper_id", p.get("id", ""))
                                     and r.get("dry_score", 0) >= PASS_THRESHOLD
                                     for r in dry_results)][:b2_top_n]
                if not top_for_b2:
                    top_for_b2 = papers[:b2_top_n]

                state, scored = self.run_phase_b2(
                    top_for_b2, state, workspace=run_dir,
                )
            except Exception as exc:
                log.exception("Phase B2 failed: %s", exc)
                state.mark_error("B2", str(exc))
                self._monitor.record_error("B2", str(exc), recoverable=False)
                scored = papers

        # ── Phase C/D: Limitations & Extensions ──────────────────
        lim_papers: list[dict] = []
        ext_papers: list[dict] = []
        if run_cd:
            try:
                state, lim_ext = self.run_phase_c_d(
                    state, dry_results=dry_results, workspace=run_dir,
                )
                lim_papers = lim_ext.get("lim_papers", [])
                ext_papers = lim_ext.get("ext_papers", [])
            except Exception as exc:
                log.exception("Phase C/D failed: %s", exc)
                state.mark_error("C", str(exc))
                self._monitor.record_error("C", str(exc), recoverable=False)

        # ── Determine core passed papers for E and F ─────────────
        # Use B2 scored results if available, otherwise B1 dry results
        from lcortex.analysis.dry_scorer import PASS_THRESHOLD
        if scored:
            core_papers = [
                s for s in scored
                if s.get("mean_score", 0) >= PASS_THRESHOLD or s.get("passed", False)
            ]
            if not core_papers:
                core_papers = scored  # Use all if none pass threshold
        else:
            core_papers = [
                r for r in dry_results
                if r.get("dry_score", 0) >= PASS_THRESHOLD or r.get("passed", False)
            ]
            if not core_papers and dry_results:
                core_papers = sorted(
                    dry_results,
                    key=lambda r: r.get("dry_score", 0),
                    reverse=True,
                )[:10]

        # ── Phase E: Synthesis  ───────────────────────────────────
        if run_e:
            try:
                state, review = self.run_phase_e(
                    state,
                    core_papers=core_papers,
                    lim_papers=lim_papers,
                    ext_papers=ext_papers,
                    b1_results=dry_results,
                    workspace=run_dir,
                )
            except Exception as exc:
                log.exception("Phase E failed: %s", exc)
                state.mark_error("E", str(exc))
                self._monitor.record_error("E", str(exc), recoverable=False)

        # ── Phase F: Structure  ───────────────────────────────────
        if run_f:
            try:
                state, templates = self.run_phase_f(
                    state, core_papers, workspace=run_dir,
                )
            except Exception as exc:
                log.exception("Phase F failed: %s", exc)
                state.mark_error("F", str(exc))
                self._monitor.record_error("F", str(exc), recoverable=False)
                templates = []

        # ── Phase F-2: Conflict Detection  ───────────────────────
        conflict_reports: list[dict] = []
        if run_f2 and not self._no_llm and templates:
            # ── Ensure graph.db exists before F-2 (seeds + papers needed for K) ──
            ws_run = run_dir / state.slug
            ws_run.mkdir(parents=True, exist_ok=True)
            db_path = run_dir / "graph.db"
            if not db_path.exists():
                # Quick init: seeds only (papers come from Phase A)
                from lcortex.graph.store import GraphStore
                tmp_store = GraphStore(str(db_path))
                from lcortex.seeds import auto_initialize as auto_init_seeds
                seeds_dir = Path(__file__).resolve().parents[1] / "seeds"
                auto_init_seeds(tmp_store, str(seeds_dir))
                # Also insert Phase A papers as nodes
                for p in (papers or []):
                    nid = p.get("paper_id", p.get("arxiv_id", p.get("id", "")))
                    if nid:
                        tmp_store.add_node({
                            "id": nid, "type": "paper",
                            "title": p.get("title", ""),
                            "year": p.get("year", 0),
                            "knowledge_level": [],
                            "structure_template": {},
                        })

            try:
                state, conflict_reports = self.run_phase_f2(
                    state, core_papers, templates, workspace=run_dir,
                )
            except Exception as exc:
                log.exception("Phase F-2 failed: %s", exc)
                state.mark_error("F2", str(exc))
                self._monitor.record_error("F2", str(exc), recoverable=False)

        # ── Phase G: Export + GraphStore persistence ──────────────
        state, export_result = self.run_phase_g(
            state, export_format, workspace=run_dir,
            papers=papers,
            dry_results=dry_results,
            core_papers=core_papers,
            lim_papers=lim_papers,
            ext_papers=ext_papers,
        )

        # ── Finalize ────────────────────────────────────────────────
        save_state(str(run_dir), state)
        checkpoint_marker(str(run_dir), "G")
        self._monitor.end_run()

        # Print summary
        summary = self._print_summary(state, papers, run_dir)
        return summary

    # ────────────────────────────────────────────────────────────────
    # Phase A: Multi-Level Search (replaces single-source search)
    # ────────────────────────────────────────────────────────────────
    def run_phase_a(
        self,
        query: str,
        state: PipelineState,
        max_results: int = 10,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sources: str = "arxiv",
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, list[dict]]:
        """Phase A: Multi-level search filter (3 levels with fallback).

        Level 1: arXiv API with cat + year filter → candidates_raw.jsonl
        Level 2: Local keyword filtering → candidates_filtered.jsonl
        Level 3: LLM quick relevance check → candidates.jsonl
        """
        phase = "A"
        label = _PHASE_NAME[phase]
        icon = _ICON[phase]

        state.mark(phase, PhaseStatus.IN_PROGRESS)
        ws = workspace or Path(self._config.workspace.path)
        ws_run = ws / state.slug
        ws_run.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        self._monitor.phase_start(phase, label)

        papers: list[dict] = []
        degradations: list[str] = []
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

        # Use multi-level search if arxiv is the source
        if "arxiv" in source_list:
            from lcortex.search.multi_level import multi_level_search

            # Prepare LLM adapter for Level 3 (only if available)
            llm_for_l3 = self._adapter if not self._no_llm else None

            try:
                n1, n2, n3, total = multi_level_search(
                    query=query,
                    workspace_dir=ws_run,
                    llm_adapter=llm_for_l3,
                    max_results=max(50, max_results * 5),  # Level 1 gets more
                    year_min=year_min or 2018,
                    year_max=year_max,
                    target_final=max_results,
                    skip_level_3=(self._no_llm or self._mode == "lite"),  # L3 only in full mode
                    sources=sources,
                )

                self._monitor.log_event("source_ok", {
                    "multi_level": True,
                    "level_1": n1,
                    "level_2": n2,
                    "level_3": n3,
                    "final": total,
                }, phase=phase)

                # Load final candidates
                candidates_file = ws_run / "candidates.jsonl"
                if candidates_file.exists():
                    with open(candidates_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                papers.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                log.info("Phase A multi-level: L1=%d L2=%d L3=%d final=%d", n1, n2, n3, len(papers))

            except Exception as exc:
                log.warning("Multi-level search failed: %s — falling back to single-source", exc)
                self._monitor.record_degradation(phase, "multi_level", str(exc))
                degradations.append(f"multi_level: {exc}")

                # Fallback: single-source arXiv search
                try:
                    from lcortex.search.arxiv import search_arxiv
                    results = search_arxiv(query, max_results=max_results)
                    papers.extend(results)
                    # Save as candidates.jsonl
                    with open(ws_run / "candidates.jsonl", "w") as f:
                        for p in papers:
                            f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    self._monitor.log_event("source_ok", {
                        "source": "arxiv",
                        "count": len(results),
                        "fallback": True,
                    }, phase=phase)
                except Exception as exc2:
                    log.exception("Fallback arXiv search also failed: %s", exc2)

        # Handle non-arxiv sources (if any remain)
        for src in source_list:
            if src == "arxiv":
                continue  # already handled
            try:
                if src == "openalex":
                    log.info("Phase A: OpenAlex search for '%s'", query)
                    from lcortex.search.openalex import search_openalex
                    results = search_openalex(query, max_results=max_results, year_min=year_min, year_max=year_max)
                    papers.extend(results)
                    self._monitor.log_event("source_ok", {"source": "openalex", "count": len(results)}, phase=phase)
            except Exception as exc:
                log.warning("Search source '%s' failed: %s", src, exc)
                self._monitor.record_degradation(phase, src, str(exc))
                degradations.append(f"{src}: {exc}")

        # Dedup if multiple sources
        if len(source_list) > 1:
            from lcortex.search.dedup import dedup_papers
            papers = dedup_papers(papers)

        # Year filter (redundant with multi_level but safe)
        if year_min or year_max:
            papers = [p for p in papers
                      if (year_min is None or p.get("year", 0) >= year_min)
                      and (year_max is None or p.get("year", 0) <= year_max)]

        # Sort by citations desc, then year desc
        papers.sort(key=lambda p: (p.get("citations", 0), p.get("year", 0)), reverse=True)

        elapsed = time.monotonic() - t0
        state.paper_count = len(papers)

        # Build phase-end summary
        summary = {
            "papers_found": len(papers),
            "sources_used": source_list,
            "degradations": len(degradations),
            "multi_level": "arxiv" in source_list,
        }
        self._monitor.phase_end(phase, summary)

        state.mark_completed(phase)
        save_state(str(ws), state)
        checkpoint_marker(str(ws), phase)

        # Print progress line
        if degradations:
            deg_parts = []
            for s in degradations:
                parts = s.split(":", 1)
                src_name = parts[0]
                reason = parts[1] if len(parts) > 1 else "timeout"
                deg_parts.append(f"{src_name}: {reason}")
            extra = " | " + " ".join(deg_parts)
        else:
            extra = ""
        deg_note = " (降级)" if degradations else ""
        print(f"📊 [{phase}] {label:12s} ✅ {len(papers)} papers ({_format_duration(elapsed)}){deg_note}{extra}")

        return state, papers

    # ────────────────────────────────────────────────────────────────
    # Phase B1: Dry NLP Scoring (always runs, zero LLM tokens)
    # ────────────────────────────────────────────────────────────────
    def run_phase_b1(
        self,
        papers: list[dict],
        state: PipelineState,
        query: str = "",
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, list[dict]]:
        """Phase B1: NLP multi-dimensional dry scoring (no LLM).

        Scores every paper on 6 dimensions:
          - TF-IDF relevance (0.30)
          - Keyword overlap (0.20)
          - Recency (0.15)
          - Citation impact (0.15)
          - Structural completeness (0.15)
          - Venue signal (0.05)

        Runs in ALL modes (dry/lite/full).  Zero token cost.
        """
        phase = "B1"
        label = _PHASE_NAME[phase]
        icon = _ICON[phase]

        state.mark(phase, PhaseStatus.IN_PROGRESS)
        ws = workspace or Path(self._config.workspace.path)
        ws_run = ws / state.slug
        ws_run.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        if not papers:
            state.mark_skipped(phase, "no papers to score")
            self._monitor.phase_start(phase, label, total_items=0)
            self._monitor.phase_end(phase, {"skipped": True, "reason": "no papers"})
            print(f"📊 [{phase}] {label:12s} ⏭ skipped (no papers)")
            return state, []

        self._monitor.phase_start(phase, label, total_items=len(papers))

        from lcortex.analysis.dry_scorer import dry_score_batch, PASS_THRESHOLD

        ws_run = (workspace or Path(self._config.workspace.path)) / state.slug
        ws_run.mkdir(parents=True, exist_ok=True)
        output_path = ws_run / "analysis_b1.jsonl"

        # Auto-detect domain constraints from query
        domain_require, domain_exclude = _infer_domain(query)

        # Use batch scorer for BM25 indexing + intent extraction
        results = dry_score_batch(
            papers, query, output_path=output_path,
            domain_require=domain_require, domain_exclude=domain_exclude,
        )
        passed_count = sum(1 for r in results if r.get("passed", False))

        # Per-paper monitor tracking
        for r in results:
            pid = r.get("paper_id", "?")
            title = r.get("title", pid)[:60]
            self._monitor.item_done(phase, pid, r, tokens_in=0, tokens_out=0)
            ds = r.get("dry_score", 0)
            pass_str = "✅" if r.get("passed") else "⚠️"
            print(f"📊 [{phase}] {label:12s} {pass_str} {pid} dry={ds:.3f} | {title[:50]}")

        elapsed = time.monotonic() - t0
        state.passed_count = passed_count

        avg_score = sum(r.get("dry_score", 0) for r in results) / max(len(results), 1)
        summary = {
            "total": len(papers),
            "scored": len(results),
            "passed": passed_count,
            "avg_dry_score": round(avg_score, 3),
            "threshold": PASS_THRESHOLD,
            "elapsed_s": round(elapsed, 2),
        }
        self._monitor.phase_end(phase, summary)

        state.mark_completed(phase)
        save_state(str(ws), state)
        checkpoint_marker(str(ws), phase)

        print(f"📊 [{phase}] {label:12s} ✅ {passed_count}/{len(papers)} passed (avg={avg_score:.3f}, ≥{PASS_THRESHOLD}) ({_format_duration(elapsed)})")

        return state, results

    # ────────────────────────────────────────────────────────────────
    # Phase B2: LLM 4C+L Scoring (only top-N from B1)
    # ────────────────────────────────────────────────────────────────
    def run_phase_b2(
        self,
        papers: list[dict],
        state: PipelineState,
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, list[dict]]:
        """Phase B2: 4C+L scoring via LLM (only on top papers from B1)."""
        phase = "B2"
        label = _PHASE_NAME[phase]
        icon = _ICON[phase]

        state.mark(phase, PhaseStatus.IN_PROGRESS)
        ws = workspace or Path(self._config.workspace.path)
        t0 = time.monotonic()

        if self._no_llm or not papers:
            state.mark_skipped(phase, "no LLM adapter configured")
            self._monitor.phase_start(phase, label, total_items=0)
            self._monitor.phase_end(phase, {"skipped": True, "reason": "no LLM"})
            print(f"📊 [{phase}] {label:12s} ⏭ skipped (no LLM)")
            return state, papers

        self._monitor.phase_start(phase, label, total_items=len(papers))

        from lcortex.analysis.scorer import score_paper

        scored: list[dict] = []
        passed_count = 0
        flagged_count = 0

        for idx, paper in enumerate(papers):
            paper_id = paper.get("id", paper.get("paper_id", f"paper-{idx}"))
            title = paper.get("title", paper_id)[:60]

            self._monitor.item_start(phase, paper_id, title)

            # Print intermediate progress
            if idx > 0 and idx % 3 == 0:
                print(f"📊 [{phase}] {label:12s} 🔄 {idx}/{len(papers)} papers")

            try:
                result = score_paper(paper, self._adapter)

                if "error" in result:
                    self._monitor.item_error(phase, paper_id, result["error"])
                    result["paper_id"] = paper_id
                    scored.append(result)
                    continue

                result.setdefault("paper_id", paper_id)
                result.setdefault("title", title)
                scored.append(result)

                # Track tokens from result if available
                tki = result.get("tokens_in", 0)
                tko = result.get("tokens_out", 0)

                self._monitor.item_done(
                    phase, paper_id, result,
                    tokens_in=tki, tokens_out=tko,
                )

                # Check pass/fail
                mean_s = result.get("mean_score", 0)
                # Fallback: compute mean_score from scores if missing or zero
                if mean_s == 0 and "scores" in result and result["scores"]:
                    score_vals = [v for v in result["scores"].values()
                                  if isinstance(v, (int, float)) and v > 0]
                    if score_vals:
                        mean_s = round(sum(score_vals) / len(score_vals), 2)
                        result["mean_score"] = mean_s

                if mean_s >= 3.0:
                    passed_count += 1
                else:
                    flagged_count += 1

                # Show per-paper summary line
                scores_str = ""
                if "scores" in result:
                    s = result["scores"]
                    scores_str = f"s={s} mean={mean_s:.1f}"
                print(f"📊 [{phase}] {label:12s} ✅ {paper_id} {scores_str} | {title[:50]}")

            except Exception as exc:
                log.exception("Failed to score paper %s: %s", paper_id, exc)
                self._monitor.item_error(phase, paper_id, str(exc))
                self._monitor.record_error(phase, f"{paper_id}: {exc}", recoverable=True)
                scored.append({"paper_id": paper_id, "error": str(exc)})

        elapsed = time.monotonic() - t0
        state.passed_count = passed_count

        summary = {
            "total": len(papers),
            "scored": len(scored),
            "passed": passed_count,
            "flagged": flagged_count,
            "elapsed_s": round(elapsed, 2),
        }
        self._monitor.phase_end(phase, summary)

        state.mark_completed(phase)
        save_state(str(ws), state)
        checkpoint_marker(str(ws), phase)

        print(f"📊 [{phase}] {label:12s} ✅ {passed_count}/{len(papers)} passed, {flagged_count} flagged ({_format_duration(elapsed)})")

        return state, scored

    # ────────────────────────────────────────────────────────────────
    # Phase C/D: Limitations & Extensions
    # ────────────────────────────────────────────────────────────────
    def run_phase_c_d(
        self,
        state: PipelineState,
        dry_results: list[dict] = None,
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, dict]:
        """Phase C + D: Generate limitation/extension queries from B1 top papers,
        run arXiv search, and save results.

        Parameters
        ----------
        state: PipelineState
        dry_results: B1 scored paper list (from run_phase_b1). Must include
                     ``dry_score``, ``title``, ``keywords``, etc.
        workspace: Run directory override.

        Returns
        -------
        (state, {"limitations": count, "extensions": count})
        """
        dry_results = dry_results or []
        ws_root = workspace or Path(self._config.workspace.path)
        ws = ws_root / state.slug
        ws.mkdir(parents=True, exist_ok=True)
        t0_total = time.monotonic()

        # ── Pick top-5 by dry_score for query generation ────────────
        sorted_papers = sorted(
            dry_results,
            key=lambda r: r.get("dry_score", 0),
            reverse=True,
        )[:5]

        if not sorted_papers:
            for ph in ("C", "D"):
                state.mark(ph, PhaseStatus.IN_PROGRESS)
                self._monitor.phase_start(ph, _PHASE_NAME[ph], total_items=0)
                self._monitor.phase_end(ph, {"skipped": True, "reason": "no scored papers"})
                state.mark_completed(ph)
            print(f"📊 [C/D] Lim/Ext      ⏭ skipped (no scored papers)")
            return state, {"limitations": 0, "extensions": 0}

        # ── Generate C/D queries from each top paper ────────────────
        all_c_queries: list[dict] = []
        all_d_queries: list[dict] = []

        for paper in sorted_papers:
            pid = paper.get("paper_id", paper.get("id", "?"))
            title = paper.get("title", pid)[:60]

            if self._no_llm:
                # ── Fallback: extract meaningful terms from title / intent ──
                keywords = paper.get("keywords", [])
                method_terms: list[str] = []

                # 1) Try B1 dry_detail intent words
                dry_detail = paper.get("dry_detail", {})
                intent_words = dry_detail.get("intent_words", [])
                if intent_words:
                    method_terms = [w for w in intent_words if len(w) > 2]

                # 2) Fallback: extract key nouns from title (skip stopwords)
                if not method_terms and title:
                    title_words = [
                        w for w in title.lower().split()
                        if len(w) > 2 and w not in _STOPS
                    ]
                    method_terms = title_words[:4]

                if not method_terms:
                    method = title.split()[0] if title else "control"
                    method_terms = [method]

                # Build queries from the best 1-2 terms
                method = " ".join(method_terms[:2])
                c_queries = [{"query": f"{method} limitations convergence stability"}]
                d_queries = [{"query": f"{method} recent advances improvement 2024 2025"}]

                # Add keyword combos if available
                top_kws = keywords[:3] if keywords else []
                for kw in top_kws:
                    c_queries.append({"query": f"{kw} limitation convergence"})
                    d_queries.append({"query": f"{kw} improvement novel"})
                # Dedup
                seen = set()
                c_queries = [q for q in c_queries
                             if q["query"] not in seen and not seen.add(q["query"])]
                d_queries = [q for q in d_queries
                             if q["query"] not in seen and not seen.add(q["query"])]

                log.info("Phase C/D fallback for '%s': %d C + %d D keyword queries (terms=%s)",
                         pid, len(c_queries), len(d_queries), method_terms[:3])
            else:
                # ── LLM-powered query generation ────────────────────
                from lcortex.review.search_query import generate_queries
                try:
                    result = generate_queries(paper, self._adapter)
                    if "error" in result:
                        log.warning("Query generation failed for '%s': %s — using keyword fallback",
                                    pid, result.get("error", ""))
                        self._monitor.record_degradation(
                            "C", "query_gen", f"{pid}: {result.get('error', '')}"
                        )
                        # Fallback: extract meaningful terms from title
                        title = paper.get("title", "")
                        title_words = [
                            w for w in title.lower().split()
                            if len(w) > 2 and w not in _STOPS
                        ]
                        method = " ".join(title_words[:2]) if title_words else "control"
                        c_queries = [{"query": f"{method} limitations convergence stability"}]
                        d_queries = [{"query": f"{method} recent advances improvement 2024 2025"}]
                    else:
                        c_queries = result.get("phase_c_queries", [])
                        d_queries = result.get("phase_d_queries", [])
                except Exception as exc:
                    log.exception("Query generation exception for '%s': %s", pid, exc)
                    self._monitor.record_degradation("C", "query_gen", f"{pid}: {exc}")
                    # Fallback: extract meaningful terms from title
                    title = paper.get("title", "")
                    title_words = [
                        w for w in title.lower().split()
                        if len(w) > 2 and w not in _STOPS
                    ]
                    method = " ".join(title_words[:2]) if title_words else "control"
                    c_queries = [{"query": f"{method} limitations convergence stability"}]
                    d_queries = [{"query": f"{method} recent advances improvement 2024 2025"}]

            all_c_queries.extend(c_queries)
            all_d_queries.extend(d_queries)

        # Normalize: ensure all queries are dicts
        all_c_queries = [{"query": q} if isinstance(q, str) else q for q in all_c_queries]
        all_d_queries = [{"query": q} if isinstance(q, str) else q for q in all_d_queries]

        # Dedup across all papers
        seen_c = set()
        unique_c = []
        for q in all_c_queries:
            qtext = q.get("query", "")
            if qtext and qtext not in seen_c:
                seen_c.add(qtext)
                unique_c.append(q)
        all_c_queries = unique_c

        seen_d = set()
        unique_d = []
        for q in all_d_queries:
            qtext = q.get("query", "")
            if qtext and qtext not in seen_d:
                seen_d.add(qtext)
                unique_d.append(q)
        all_d_queries = unique_d

        log.info("Phase C/D: %d unique C queries, %d unique D queries",
                 len(all_c_queries), len(all_d_queries))

        # ── Run arXiv search for each query ─────────────────────────
        from lcortex.search.arxiv import search_arxiv
        
        # Use resource profile to control concurrency
        MAX_CD_SEARCHES = self._rp.max_cd_searches

        # Phase C: Limitation search
        state.mark("C", PhaseStatus.IN_PROGRESS)
        self._monitor.phase_start("C", _PHASE_NAME["C"], total_items=len(all_c_queries))
        t0 = time.monotonic()

        lim_papers: list[dict] = []
        seen_lim_ids = set()

        for idx, qd in enumerate(all_c_queries[:MAX_CD_SEARCHES]):
            qtext = qd.get("query", "")
            qid = f"C-q{idx}"
            self._monitor.item_start("C", qid, qtext[:60])

            try:
                results = search_arxiv(qtext, max_results=3)
                new_papers = []
                for r in results:
                    pid = r.get("paper_id", r.get("id", ""))
                    if pid and pid not in seen_lim_ids:
                        seen_lim_ids.add(pid)
                        r["_phase"] = "C"
                        r["_source_query"] = qtext
                        new_papers.append(r)
                lim_papers.extend(new_papers)

                self._monitor.item_done("C", qid, {
                    "query": qtext,
                    "results": len(new_papers),
                    "total": len(results),
                })
                self._monitor.log_event("c_search", {
                    "query": qtext,
                    "found": len(results),
                    "new": len(new_papers),
                }, phase="C")

                log.info("Phase C query '%s': found %d papers (%d new)",
                         qtext[:60], len(results), len(new_papers))

            except Exception as exc:
                log.warning("Phase C search failed for '%s': %s", qtext[:60], exc)
                self._monitor.item_error("C", qid, str(exc))
                self._monitor.record_degradation("C", f"search_{idx}", str(exc))

        elapsed_c = time.monotonic() - t0

        # Save C results
        c_path = ws / "limitation_papers.json"
        with open(c_path, "w", encoding="utf-8") as f:
            json.dump(lim_papers, f, ensure_ascii=False, indent=2)

        self._monitor.phase_end("C", {
            "queries": len(all_c_queries),
            "limitations_found": len(lim_papers),
            "elapsed_s": round(elapsed_c, 2),
        })
        state.mark_completed("C")
        save_state(str(ws_root), state)
        checkpoint_marker(str(ws_root), "C")

        print(f"📊 [C] Limitations    ✅ {len(lim_papers)} papers from {len(all_c_queries)} queries ({_format_duration(elapsed_c)})")

        # Phase D: Extension search
        state.mark("D", PhaseStatus.IN_PROGRESS)
        self._monitor.phase_start("D", _PHASE_NAME["D"], total_items=len(all_d_queries))
        t0 = time.monotonic()

        ext_papers: list[dict] = []
        seen_ext_ids = set()

        for idx, qd in enumerate(all_d_queries[:MAX_CD_SEARCHES]):
            qtext = qd.get("query", "")
            qid = f"D-q{idx}"
            self._monitor.item_start("D", qid, qtext[:60])

            try:
                results = search_arxiv(qtext, max_results=3)
                new_papers = []
                for r in results:
                    pid = r.get("paper_id", r.get("id", ""))
                    if pid and pid not in seen_ext_ids:
                        seen_ext_ids.add(pid)
                        r["_phase"] = "D"
                        r["_source_query"] = qtext
                        new_papers.append(r)
                ext_papers.extend(new_papers)

                self._monitor.item_done("D", qid, {
                    "query": qtext,
                    "results": len(new_papers),
                    "total": len(results),
                })
                self._monitor.log_event("d_search", {
                    "query": qtext,
                    "found": len(results),
                    "new": len(new_papers),
                }, phase="D")

                log.info("Phase D query '%s': found %d papers (%d new)",
                         qtext[:60], len(results), len(new_papers))

            except Exception as exc:
                log.warning("Phase D search failed for '%s': %s", qtext[:60], exc)
                self._monitor.item_error("D", qid, str(exc))
                self._monitor.record_degradation("D", f"search_{idx}", str(exc))

        elapsed_d = time.monotonic() - t0

        # Save D results
        d_path = ws / "extension_papers.json"
        with open(d_path, "w", encoding="utf-8") as f:
            json.dump(ext_papers, f, ensure_ascii=False, indent=2)

        self._monitor.phase_end("D", {
            "queries": len(all_d_queries),
            "extensions_found": len(ext_papers),
            "elapsed_s": round(elapsed_d, 2),
        })
        state.mark_completed("D")
        save_state(str(ws_root), state)
        checkpoint_marker(str(ws_root), "D")

        print(f"📊 [D] Extensions     ✅ {len(ext_papers)} papers from {len(all_d_queries)} queries ({_format_duration(elapsed_d)})")

        elapsed_total = time.monotonic() - t0_total
        print(f"📊 [C/D] Lim/Ext      ✅ {len(lim_papers)} lim + {len(ext_papers)} ext ({_format_duration(elapsed_total)})")

        return state, {
            "limitations": len(lim_papers),
            "extensions": len(ext_papers),
            "lim_papers": lim_papers,
            "ext_papers": ext_papers,
        }

    # ────────────────────────────────────────────────────────────────
    # Phase E: Synthesis
    # ────────────────────────────────────────────────────────────────
    def run_phase_e(
        self,
        state: PipelineState,
        core_papers: list[dict] = None,
        lim_papers: list[dict] = None,
        ext_papers: list[dict] = None,
        b1_results: list[dict] = None,
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, str]:
        """Phase E: Synthesize a structured literature review.

        Loads all collected data (B1 scored, B2 scored if available,
        limitation/extension papers) and calls the synthesis engine.

        Uses LLM when available; falls back to a template review from
        the top-5 B1 papers otherwise.

        Parameters
        ----------
        state: PipelineState
        core_papers: Core papers that passed B1/B2 gate.
        lim_papers: Limitation/critique papers from Phase C.
        ext_papers: Extension/advancement papers from Phase D.
        b1_results: B1 scored results (for fallback template).
        workspace: Run directory override.

        Returns
        -------
        (state, review_markdown_string)
        """
        phase = "E"
        label = _PHASE_NAME[phase]
        core_papers = core_papers or []
        lim_papers = lim_papers or []
        ext_papers = ext_papers or []
        b1_results = b1_results or []
        ws_root = workspace or Path(self._config.workspace.path)
        ws = ws_root / state.slug
        ws.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        state.mark(phase, PhaseStatus.IN_PROGRESS)

        # ── Resolve core papers: prefer B2 scored, then B1 passed ──
        # Try loading analysis.jsonl (B2 output) first
        b2_path = ws / "analysis.jsonl"
        if b2_path.exists() and not core_papers:
            b2_papers = []
            with open(b2_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        b2_papers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            # Filter: B2 papers with mean_score >= 3.0 are "passed"
            from lcortex.analysis.dry_scorer import PASS_THRESHOLD
            for p in b2_papers:
                if p.get("mean_score", 0) >= PASS_THRESHOLD or p.get("passed", False):
                    core_papers.append(p)

        # Fallback: use B1 passed papers if no core papers resolved
        if not core_papers and b1_results:
            from lcortex.analysis.dry_scorer import PASS_THRESHOLD
            core_papers = [r for r in b1_results
                           if r.get("dry_score", 0) >= PASS_THRESHOLD
                           or r.get("passed", False)]

        if not core_papers and b1_results:
            # Ultimate fallback: take top-5 by dry_score
            core_papers = sorted(
                b1_results,
                key=lambda r: r.get("dry_score", 0),
                reverse=True,
            )[:5]

        topic = getattr(state, "query", "") or "Research Topic"

        if self._no_llm:
            # ── Fallback: structured template review ─────────────────
            self._monitor.phase_start(phase, label, total_items=1)
            self._monitor.log_event("phase_e_fallback", {
                "mode": "template",
                "core_count": len(core_papers),
                "lim_count": len(lim_papers),
                "ext_count": len(ext_papers),
            }, phase=phase)
            review_text = _build_fallback_review(topic, core_papers, lim_papers, ext_papers, self._mode)
        else:
            # ── LLM-powered synthesis ───────────────────────────────
            self._monitor.phase_start(phase, label, total_items=1)

            from lcortex.review.synthesizer import synthesize_review

            try:
                review_text = synthesize_review(
                    topic=topic,
                    core_papers=core_papers,
                    lim_papers=lim_papers,
                    ext_papers=ext_papers,
                    adapter=self._adapter,
                )
                if isinstance(review_text, dict) and "_raw" in review_text:
                    review_text = review_text["_raw"]
                if isinstance(review_text, dict) and "error" in review_text:
                    log.warning("Synthesis returned error: %s — generating fallback", review_text["error"])
                    self._monitor.record_degradation(phase, "synthesize", review_text["error"][:100])
                    review_text = _build_fallback_review(topic, core_papers, lim_papers, ext_papers, self._mode)
                if not review_text or review_text.startswith("# Review Synthesis Failed"):
                    log.warning("Synthesis returned empty/error — generating fallback template")
                    self._monitor.record_degradation(phase, "synthesize", "empty_result")
                    review_text = _build_fallback_review(topic, core_papers, lim_papers, ext_papers, self._mode)
            except Exception as exc:
                log.exception("Synthesis failed: %s", exc)
                self._monitor.record_degradation(phase, "synthesize", str(exc))
                review_text = _build_fallback_review(topic, core_papers, lim_papers, ext_papers, self._mode)

        # Clean and write review.md (handle raw LLM wrapper + markdown fences)
        review_text = _unwrap_markdown_fence(review_text)
        review_path = ws / "review.md"
        review_path.write_text(review_text, encoding="utf-8")

        line_count = review_text.count("\n")
        self._monitor.phase_end(phase, {
            "review_lines": line_count,
            "core_papers": len(core_papers),
            "lim_papers": len(lim_papers),
            "ext_papers": len(ext_papers),
            "llm_used": not self._no_llm,
        })
        state.mark_completed(phase)
        save_state(str(ws_root), state)
        checkpoint_marker(str(ws_root), phase)

        elapsed = time.monotonic() - t0
        print(f"📊 [{phase}] {label:12s} ✅ {line_count} lines "
              f"({len(core_papers)} core + {len(lim_papers)} lim + {len(ext_papers)} ext) "
              f"({_format_duration(elapsed)}) | review.md written")

        return state, review_text

    # ────────────────────────────────────────────────────────────────
    # Phase F: Structure extraction
    # ────────────────────────────────────────────────────────────────
    def run_phase_f(
        self,
        state: PipelineState,
        papers: list[dict] = None,
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, list[dict]]:
        """Phase F: Structure template extraction + knowledge level inference.

        In dry/lite mode (no LLM): uses ``lite_matcher`` for keyword-based
        paper→seed mapping with 49 ontology nodes.

        In full mode (LLM available): uses ``extract_structure`` for deep
        structure template extraction via LLM.
        """
        phase = "F"
        label = _PHASE_NAME[phase]
        papers = papers or []
        ws_root = workspace or Path(self._config.workspace.path)
        ws = ws_root / state.slug
        ws.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        state.mark(phase, PhaseStatus.IN_PROGRESS)

        # ── Lite mode: keyword-driven paper→seed matching ──────────────
        if self._no_llm:
            self._monitor.phase_start(phase, label, total_items=len(papers))

            from lcortex.structure.lite_matcher import match_papers, build_seed_edges

            # Find seeds directory relative to project
            seeds_dir = Path(__file__).resolve().parents[1] / "seeds"

            templates = match_papers(papers, str(seeds_dir), top_k=5, min_score=0.15)

            # Build seed edges for graph
            seed_edges = build_seed_edges(templates)

            # Enrich templates with metadata
            for t in templates:
                pid = t.get("paper_id", "")
                matched = t.get("matched_seeds", [])
                titles = [p.get("title", pid)[:50] for p in papers
                          if p.get("paper_id", p.get("id", "")) == pid]
                t["title"] = titles[0] if titles else pid
                t["lite_mode"] = True

                matched_str = ", ".join(matched[:3])
                if len(matched) > 3:
                    matched_str += f" +{len(matched)-3}"
                kl = t.get("knowledge_level", [])
                kl_str = ", ".join(kl) if kl else "none"
                print(f"📊 [{phase}] {label:12s} ✅ {pid} seeds=[{matched_str}] "
                      f"KL=[{kl_str}] | {t.get('title', pid)[:40]}")

            # Save
            analysis_path = ws / "analysis.json"
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(templates, f, ensure_ascii=False, indent=2)

            struct_dir = ws / "structures"
            struct_dir.mkdir(parents=True, exist_ok=True)
            for t in templates:
                pid = t.get("paper_id", "unknown")
                safe_pid = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(pid))[:80]
                spath = struct_dir / f"{safe_pid}.json"
                with open(spath, "w", encoding="utf-8") as f:
                    json.dump(t, f, ensure_ascii=False, indent=2, default=str)

            elapsed = time.monotonic() - t0
            matched_count = sum(1 for t in templates if t.get("matched_seeds"))
            total_edges = len(seed_edges)

            self._monitor.phase_end(phase, {
                "templates": len(templates),
                "matched": matched_count,
                "seed_edges": total_edges,
                "saved_to": str(analysis_path),
            })
            state.mark_completed(phase)
            save_state(str(ws_root), state)
            checkpoint_marker(str(ws_root), phase)

            print(f"📊 [{phase}] {label:12s} ✅ {len(templates)} papers ({_format_duration(elapsed)}) "
                  f"| {matched_count} matched | {total_edges} seed edges | analysis.json + structures/")

            return state, templates

        # ── Full mode: LLM-based structure extraction ──────────────────
        self._monitor.phase_start(phase, label, total_items=len(papers))

        from lcortex.structure.extractor import extract_structure

        templates: list[dict] = []
        for idx, paper in enumerate(papers):
            paper_id = paper.get("id", paper.get("paper_id", f"paper-{idx}"))
            title = paper.get("title", paper_id)[:60]

            self._monitor.item_start(phase, paper_id, title)

            if idx > 0 and idx % 3 == 0:
                print(f"📊 [{phase}] {label:12s} 🔄 {idx}/{len(papers)} papers")

            try:
                result = extract_structure(paper, self._adapter)

                if "error" in result:
                    log.warning("Structure extraction failed for '%s': %s",
                                paper_id, result.get("error", ""))
                    self._monitor.item_error(phase, paper_id, result.get("error", ""))
                    self._monitor.record_error(phase, f"{paper_id}: {result.get('error', '')}", recoverable=True)
                    result["paper_id"] = paper_id
                    result["title"] = title
                    templates.append(result)
                    continue

                result["paper_id"] = paper_id
                result["title"] = title
                if "scores" in paper:
                    result["scores"] = paper["scores"]
                if "dry_score" in paper:
                    result["dry_score"] = paper["dry_score"]

                self._monitor.item_done(phase, paper_id, {
                    "knowledge_level": result.get("knowledge_level", []),
                    "has_template": bool(result.get("structure_template")),
                })
                templates.append(result)

                kl = result.get("knowledge_level", [])
                kl_str = ", ".join(kl) if kl else "none"
                st = result.get("structure_template", {})
                has_st = bool(st and any(st.values()))
                print(f"📊 [{phase}] {label:12s} ✅ {paper_id} KL=[{kl_str}] "
                      f"template={'yes' if has_st else 'no'} | {title[:50]}")

            except Exception as exc:
                log.exception("Phase F extraction exception for '%s': %s", paper_id, exc)
                self._monitor.item_error(phase, paper_id, str(exc))
                self._monitor.record_error(phase, f"{paper_id}: {exc}", recoverable=True)
                templates.append({
                    "paper_id": paper_id,
                    "title": title,
                    "error": str(exc),
                })

        elapsed = time.monotonic() - t0

        # ── Save enriched papers ────────────────────────────────────
        analysis_path = ws / "analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)

        # ── Augment with seed matching (full mode also gets paper→seed edges) ──
        try:
            from lcortex.structure.lite_matcher import match_papers, build_seed_edges
            seeds_dir = Path(__file__).resolve().parents[1] / "seeds"
            seen = {t.get("paper_id", ""): t for t in templates}
            paper_lookup = {p.get("id", p.get("paper_id", "")): p for p in papers}
            matched = match_papers(papers, str(seeds_dir), top_k=5, min_score=0.15)
            for m in matched:
                pid = m.get("paper_id", "")
                if pid in seen:
                    seen[pid]["matched_seeds"] = m.get("matched_seeds", [])
                    seen[pid]["match_scores"] = m.get("match_scores", {})
                    seen[pid]["knowledge_level"] = seen[pid].get("knowledge_level", []) or m.get("knowledge_level", [])
            # Re-save with match data
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(templates, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        struct_dir = ws / "structures"
        struct_dir.mkdir(parents=True, exist_ok=True)
        for t in templates:
            pid = t.get("paper_id", "unknown")
            safe_pid = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(pid))[:80]
            spath = struct_dir / f"{safe_pid}.json"
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(t, f, ensure_ascii=False, indent=2, default=str)

        novel = sum(1 for t in templates
                    if t.get("structure_template")
                    and not t.get("error"))

        self._monitor.phase_end(phase, {
            "templates": len(templates),
            "novel_patterns": novel,
            "saved_to": str(analysis_path),
        })
        state.mark_completed(phase)
        save_state(str(ws_root), state)
        checkpoint_marker(str(ws_root), phase)

        print(f"📊 [{phase}] {label:12s} ✅ {len(templates)} papers ({_format_duration(elapsed)}) "
              f"| {novel} novel patterns | analysis.json + structures/")

        return state, templates

    # ────────────────────────────────────────────────────────────────
    # Phase F-2: Conflict detection (double-loop)
    # ────────────────────────────────────────────────────────────────
    def run_phase_f2(
        self,
        state: PipelineState,
        papers: list[dict] = None,
        templates: list[dict] = None,
        workspace: Optional[Path] = None,
    ) -> tuple[PipelineState, list[dict]]:
        """Phase F-2: Deconstruct-reconstruct conflict detection.

        For each paper that passed Phase F, runs the full 5-step divergence
        pipeline: Deconstruction → Retrieval → Reconstruction → Conflict
        Assessment → Decision (single_loop/double_loop/seed_anchored/degraded).

        Skips papers whose structure_template confidence is high (>0.7) and
        whose knowledge_level confidence is high (>0.7) — those are already
        well-understood by the ontology.
        """
        phase = "F2"
        label = _PHASE_NAME[phase]
        papers = papers or []
        templates = templates or []
        ws_root = workspace or Path(self._config.workspace.path)
        ws = ws_root / state.slug
        ws.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        state.mark(phase, PhaseStatus.IN_PROGRESS)

        if self._no_llm:
            state.mark_skipped(phase, "no LLM adapter configured")
            self._monitor.phase_start(phase, label, total_items=0)
            self._monitor.phase_end(phase, {"skipped": True, "reason": "no LLM"})
            print(f"📊 [{phase}] {label:12s} ⏭ skipped (no LLM)")
            return state, []

        from lcortex.structure.deconstructor import detect_conflict
        from lcortex.graph.store import GraphStore

        # Open graph store for knowledge graph queries
        db_path = ws_root / "graph.db"
        store = GraphStore(str(db_path)) if db_path.exists() else None

        # Build template index by paper_id
        template_index: dict[str, dict] = {}
        for t in templates:
            pid = t.get("paper_id", "")
            if pid:
                template_index[pid] = t

        # Determine which papers need F-2 evaluation
        candidates = []
        for paper in papers:
            pid = paper.get("paper_id", paper.get("id", ""))
            tmpl = template_index.get(pid, {})

            # Skip if confidence is high (well-understood by ontology)
            kl_conf = tmpl.get("knowledge_level_confidence", 0)
            st = tmpl.get("structure_template")
            has_template = st and isinstance(st, dict) and any(st.values())

            # Always evaluate if: low confidence OR no template extracted
            if kl_conf < 0.6 or not has_template:
                candidates.append((paper, tmpl))
            else:
                log.info("Phase F-2: skipping '%s' — high confidence (%.2f) + template",
                         pid[:40], kl_conf)

        if not candidates:
            self._monitor.phase_start(phase, label, total_items=0)
            self._monitor.phase_end(phase, {"conflicts": 0, "skipped_all_high_confidence": True})
            state.mark_completed(phase)
            print(f"📊 [{phase}] {label:12s} ✅ all high-confidence — skipped")
            return state, []

        self._monitor.phase_start(phase, label, total_items=len(candidates))

        reports: list[dict] = []
        double_loop_count = 0
        degraded_count = 0

        for idx, (paper, tmpl) in enumerate(candidates):
            paper_id = paper.get("paper_id", paper.get("id", f"paper-{idx}"))
            title = paper.get("title", paper_id)[:60]

            self._monitor.item_start(phase, paper_id, title)

            # Build structure dict with everything F-2 needs
            structure = {
                "knowledge_level": tmpl.get("knowledge_level", []),
                "knowledge_level_confidence": tmpl.get("knowledge_level_confidence", 0),
                "structure_template": tmpl.get("structure_template"),
            }

            try:
                result = detect_conflict(
                    paper=paper,
                    structure=structure,
                    graph_store=store,
                    adapter=self._adapter,
                )

                if "error" in result:
                    log.warning("F-2 conflict detection failed for '%s': %s",
                                paper_id, result.get("error", ""))
                    self._monitor.item_error(phase, paper_id, result.get("error", ""))
                    result["paper_id"] = paper_id
                    reports.append(result)
                    continue

                result["paper_id"] = paper_id
                result["title"] = title

                # Check decision
                ca = result.get("conflict_assessment", {})
                action = ca.get("recommended_action", "single_loop")
                triggers = ca.get("triggers_double_loop", False)
                degraded = ca.get("downgraded_by_meta", False)
                unexplain = ca.get("unexplainability_score", 0)
                conditions = ca.get("conditions_true_count", 0)

                self._monitor.item_done(phase, paper_id, {
                    "action": action,
                    "unexplainability": unexplain,
                    "conditions_true": conditions,
                })
                reports.append(result)

                if action == "double_loop":
                    double_loop_count += 1
                elif degraded:
                    degraded_count += 1

                icon = {"double_loop": "🔴", "degraded_by_meta": "🟡",
                        "single_loop": "🟢", "seed_anchored": "⚪"}.get(action, "❓")
                print(f"📊 [{phase}] {label:12s} {icon} {paper_id} action={action} "
                      f"unexplain={unexplain:.2f} conditions={conditions}/4 | {title[:40]}")

            except Exception as exc:
                log.exception("F-2 conflict detection exception for '%s': %s", paper_id, exc)
                self._monitor.item_error(phase, paper_id, str(exc))
                self._monitor.record_error(phase, f"{paper_id}: {exc}", recoverable=True)
                reports.append({
                    "paper_id": paper_id,
                    "title": title,
                    "error": str(exc),
                })

        elapsed = time.monotonic() - t0

        # ── Save conflict reports ────────────────────────────────────
        f2_path = ws / "conflict_report.json"
        with open(f2_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2, default=str)

        self._monitor.phase_end(phase, {
            "reports": len(reports),
            "double_loop": double_loop_count,
            "degraded": degraded_count,
            "single_loop": len(reports) - double_loop_count - degraded_count,
            "saved_to": str(f2_path),
        })
        state.mark_completed(phase)
        save_state(str(ws_root), state)
        checkpoint_marker(str(ws_root), phase)

        print(f"📊 [{phase}] {label:12s} ✅ {len(reports)} papers ({_format_duration(elapsed)}) "
              f"| {double_loop_count} double-loop | {degraded_count} degraded | "
              f"{len(reports) - double_loop_count - degraded_count} single-loop | conflict_report.json")

        return state, reports

    # ────────────────────────────────────────────────────────────────
    # Phase G: Export
    # ────────────────────────────────────────────────────────────────
    def run_phase_g(
        self,
        state: PipelineState,
        export_format: str = "obsidian",
        workspace: Optional[Path] = None,
        papers: list[dict] = None,
        dry_results: list[dict] = None,
        core_papers: list[dict] = None,
        lim_papers: list[dict] = None,
        ext_papers: list[dict] = None,
    ) -> tuple[PipelineState, dict]:
        """Phase G: GraphStore persistence + export to Obsidian vault or JSON."""
        phase = "G"
        label = _PHASE_NAME[phase]
        ws = workspace or Path(self._config.workspace.path)

        state.mark(phase, PhaseStatus.IN_PROGRESS)
        self._monitor.phase_start(phase, label)

        result: dict[str, Any] = {"format": export_format, "files": []}

        # ═══════════════════════════════════════════════════════════
        # GraphStore persistence (always)
        # ═══════════════════════════════════════════════════════════
        graph_nodes = 0
        graph_edges = 0
        try:
            from lcortex.graph.store import GraphStore
            from lcortex.graph.edge import create_correlation_edge
            
            db_path = ws / "graph.db"
            store = GraphStore(str(db_path))

            # ── Seed initialization (first-run only) ─────────────────
            try:
                from lcortex.seeds import auto_initialize as auto_init_seeds
                seeds_dir = Path(__file__).resolve().parents[1] / "seeds"
                seed_stats = auto_init_seeds(store, str(seeds_dir))
                if seed_stats.get("inserted", 0) > 0:
                    log.info("Phase G: Seeds — %d nodes injected into graph", seed_stats["inserted"])
                    print(f"📊 [G] Seeds        ✅ {seed_stats['inserted']} ontology nodes loaded")
                elif seed_stats.get("skipped", 0) > 0:
                    log.info("Phase G: Seeds — graph already populated (%d existing)", seed_stats["skipped"])
            except Exception as seed_exc:
                log.warning("Phase G: Seed initialization failed: %s", seed_exc)

            # 写入论文节点
            for p in (papers or []):
                nid = p.get('paper_id', p.get('arxiv_id', p.get('id', '')))
                if nid:
                    store.add_node({'id': nid, 'type': 'paper', 'title': p.get('title',''), 'year': p.get('year',0), 'knowledge_level': [], 'structure_template': {}})
                    graph_nodes += 1
            
            # 写入 B1 评分边
            for r in (dry_results or []):
                pid = r.get('paper_id', '')
                ds = r.get('dry_score', 0)
                if pid and ds > 0:
                    det = r.get('dry_detail', {})
                    snid = f"b1-{pid}"
                    store.add_node({'id': snid, 'type': 'score', 'title': f'B1:{ds:.3f}', 'year': 0, 'knowledge_level': [], 'structure_template': {}})
                    edge_data = create_correlation_edge(pid, snid, score=ds, mechanism_desc=f"BM25={det.get('bm25',0):.2f} Intent={det.get('intent',0):.2f}")
                    store.add_edge(edge_data['source_id'], edge_data['target_id'], edge_data)
                    graph_edges += 1
            
            # 写入 B2 评分边
            for r in (core_papers or []):
                pid = r.get('paper_id', '')
                ms = r.get('mean_score', 0)
                if pid and ms > 0:
                    s = r.get('scores', {})
                    snid = f"b2-{pid}"
                    store.add_node({'id': snid, 'type': 'score', 'title': f'B2:{ms:.1f}', 'year': 0, 'knowledge_level': [], 'structure_template': {}})
                    edge_data = create_correlation_edge(pid, snid, score=ms/5.0, mechanism_desc=f"4C+L: {s}")
                    store.add_edge(edge_data['source_id'], edge_data['target_id'], edge_data)
                    graph_edges += 1
            
            # 写入论文间关联边
            paper_ids = [r.get('paper_id','') for r in (dry_results or [])]
            for i in range(len(paper_ids)):
                for j in range(i+1, len(paper_ids)):
                    a, b = paper_ids[i], paper_ids[j]
                    sa = dry_results[i].get('dry_score', 0) if dry_results else 0
                    sb = dry_results[j].get('dry_score', 0) if dry_results else 0
                    if sa > 0.2 and sb > 0.2:
                        edge_data = create_correlation_edge(a, b, score=min(sa, sb), mechanism_desc=f"B1: {sa:.2f}↔{sb:.2f}")
                        store.add_edge(edge_data['source_id'], edge_data['target_id'], edge_data)
                        graph_edges += 1

            # ── Paper → Seed edges (from Phase F lite matcher) ──────────
            analysis_file = ws / "analysis.json"
            if not analysis_file.exists():
                # Check slug subdirectory
                analysis_file = ws / state.slug / "analysis.json"
            if analysis_file.exists():
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        phase_f_results = json.load(f)
                    seed_edge_count = 0
                    for result in phase_f_results:
                        paper_id = result.get("paper_id", "")
                        for seed_id, score in result.get("match_scores", {}).items():
                            edge_data = create_correlation_edge(
                                paper_id, seed_id, score=score,
                                mechanism_desc=f"lite_matcher: {score:.3f}",
                            )
                            store.add_edge(edge_data['source_id'], edge_data['target_id'], edge_data)
                            seed_edge_count += 1
                            graph_edges += 1
                    if seed_edge_count > 0:
                        log.info("Phase G: Seed edges — %d paper→seed edges from lite matcher", seed_edge_count)
                        result["seed_edges"] = seed_edge_count
                except Exception as exc:
                    log.warning("Phase G: Seed edge creation failed: %s", exc)
            
            result["graph_nodes"] = graph_nodes
            result["graph_edges"] = graph_edges
            result["graph_path"] = str(db_path)
            log.info("Phase G: GraphStore — %d nodes, %d edges → %s", graph_nodes, graph_edges, db_path)
        except Exception as exc:
            log.warning("GraphStore persistence failed: %s", exc)
            result["graph_error"] = str(exc)

        # ═══════════════════════════════════════════════════════════
        # Obsidian vault export
        # ═══════════════════════════════════════════════════════════
        if export_format == "obsidian":
            vault_dir = ws / "vault"

            slug_dir = ws / state.slug
            b1_path = ws / "analysis_b1.jsonl"
            if not b1_path.exists() and slug_dir.exists():
                b1_path = slug_dir / "analysis_b1.jsonl"
            if b1_path.exists():
                try:
                    from lcortex.export.obsidian import generate_vault_from_b1
                    export_result = generate_vault_from_b1(
                        b1_jsonl_path=b1_path,
                        output_dir=vault_dir,
                        query=state.query,
                    )
                    result["papers_exported"] = export_result.get("papers_exported", 0)
                    result["meta_files"] = export_result.get("meta_files", 0)
                    result["total_files"] = export_result.get("total_files", 0)
                    result["errors"] = export_result.get("errors", [])
                    log.info(
                        "Phase G: B1 vault export — %d papers, %d meta, %d total files",
                        result["papers_exported"],
                        result["meta_files"],
                        result["total_files"],
                    )
                except Exception as exc:
                    log.warning("B1 vault export failed: %s — falling back to placeholder", exc)
                    self._monitor.record_degradation(phase, "b1_export", str(exc))
                    _fallback_placeholder_export(vault_dir, ws, result)
            else:
                log.info("Phase G: no analysis_b1.jsonl found — using placeholder export")
                _fallback_placeholder_export(vault_dir, ws, result)

            # ── GraphStore → vault enrichment (seeds + causal map + graph.json) ──
            # Re-use the store we already opened for seed init + paper insertion
            try:
                from lcortex.export.obsidian import generate_vault_from_graphstore
                gs_export = generate_vault_from_graphstore(
                    store=store,
                    output_dir=vault_dir,
                    query=state.query,
                )
                result.setdefault("graphstore_files", 0)
                result["graphstore_files"] = gs_export.get("total_files", 0)
                result["graph_seed_nodes"] = gs_export.get("seed_nodes", 0)
                log.info(
                    "Phase G: GraphStore vault enrichment — %d files, %d seed nodes",
                    gs_export.get("total_files", 0),
                    gs_export.get("seed_nodes", 0),
                )
            except Exception as exc:
                log.warning("GraphStore vault enrichment failed: %s", exc)

            # ── Final file collection ──────────────────────────────────
            try:
                for md_file in vault_dir.rglob("*.md"):
                    try:
                        result["files"].append(str(md_file.relative_to(ws)))
                    except ValueError:
                        result["files"].append(str(md_file))
                result["file_count"] = len(result["files"])
            except Exception:
                pass
        else:
            # JSON export
            export_path = ws / "export.json"
            export_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["files"].append(str(export_path.relative_to(ws)))
            result["file_count"] = 1

        self._monitor.phase_end(phase, {"files": len(result.get("files", []))})
        state.mark_completed(phase)
        save_state(str(ws), state)
        checkpoint_marker(str(ws), phase)

        print(f"📊 [{phase}] {label:12s} ✅ {result.get('file_count', len(result.get('files', [])))} files | {export_format} export")

        return state, result

    # ────────────────────────────────────────────────────────────────
    # Summary printing
    # ────────────────────────────────────────────────────────────────
    def _print_summary(
        self,
        state: PipelineState,
        papers: list[dict],
        run_dir: Path,
    ) -> dict:
        """Print the final summary table and return the dict."""
        summary = self._monitor.final_summary()
        elapsed = summary.get("elapsed_s", 0.0)
        errors = summary.get("total_errors", 0)
        degradations = summary.get("total_degradations", 0)

        # Determine which phases to show based on mode
        show_phases = ["A", "B1"]
        if self._mode in ("lite", "full"):
            show_phases.append("B2")
        show_phases += ["C", "D", "E", "F", "F2", "G"]

        # Build phase status line
        phase_parts: list[str] = []
        for p in show_phases:
            status = state.phases.get(p, PhaseStatus.PENDING)
            if status == PhaseStatus.COMPLETED:
                phase_parts.append(f"{p} ✅")
            elif status == PhaseStatus.SKIPPED:
                phase_parts.append(f"{p} ⏭")
            elif status == PhaseStatus.ERROR:
                phase_parts.append(f"{p} ❌")
            elif status == PhaseStatus.PENDING:
                phase_parts.append(f"{p} ⏳")
            else:
                phase_parts.append(f"{p} 🔄")

        phases_str = "  ".join(phase_parts)

        # Paper counts
        paper_count = len(papers)
        scored_count = state.passed_count if state.passed_count else 0
        if self._mode == "dry" or self._no_llm:
            paper_line = f"{paper_count} found, {scored_count} passed dry (no LLM)"
        else:
            paper_line = f"{paper_count} found, {scored_count} passed"

        # Degradation notes
        deg_notes = summary.get("degradations", [])
        deg_str = ""
        if deg_notes:
            deg_str = f" | Degradations: {len(deg_notes)} ({', '.join(deg_notes[:3])})"

        # Output count
        vault_dir = run_dir / "vault"
        output_count = 0
        if vault_dir.exists():
            output_count = len(list(vault_dir.rglob("*.md")))

        mode_label = {
            "dry": "dry (no LLM)",
            "lite": "lite (A→G single-loop)",
            "full": "full (A→G with structure extraction)",
        }.get(self._mode, self._mode)

        def _pad(s: str, width: int) -> str:
            """Right-pad string to width, accounting for CJK wide chars."""
            vis = 0
            for ch in s:
                if ord(ch) > 0x7FF or ord(ch) in (0x2705, 0x23F3, 0x2B55, 0x26D4, 0x274C, 0x21BA, 0x2192):
                    vis += 2
                else:
                    vis += 1
            pad_len = max(0, width - vis)
            return s + " " * pad_len

        w = 56  # inner box width
        lines = [
            "  📊 Literature Cortex — Run Summary",
            f"  Mode: {mode_label}",
            f"  Phases: {phases_str}",
            f"  Papers: {paper_line}",
        ]
        if errors or deg_str:
            lines.append(f"  Errors: {errors}  {deg_str}")
        else:
            lines.append(f"  Errors: {errors}")
        lines.append(f"  Output: vault/ → {output_count} .md files")
        lines.append(f"  Duration: {_format_duration(elapsed)}")

        # Build box
        top_border = "╔" + "═" * w + "╗"
        mid_border = "╠" + "═" * w + "╣"
        bot_border = "╚" + "═" * w + "╝"

        print(f"\
{'='*60}")
        print(top_border)
        for i, line in enumerate(lines):
            padded = _pad(line, w)
            print(f"║{padded}║")
            if i == 0:
                print(mid_border)
        print(bot_border)
        print(f"{'='*60}")


    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────
    def _placeholder_papers(self, query: str, count: int) -> list[dict]:
        """Generate placeholder papers for testing when no search adapter exists."""
        papers = []
        titles = [
            f"Recent advances in {query} — a survey",
            f"A novel {query} framework using deep learning",
            f"Comparative analysis of {query} methods",
            f"Robust {query} for real-time applications",
            f"Stability analysis of {query} systems",
        ]
        for i in range(min(count, len(titles))):
            papers.append({
                "id": f"placeholder-{i + 1:03d}",
                "paper_id": f"placeholder-{i + 1:03d}",
                "title": titles[i],
                "abstract": f"Abstract for {titles[i]}",
                "keywords": query.split(),
                "year": 2024,
                "source": "arxiv",
                "citation_count": 0,
            })
        return papers


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def mode_has_search(config: Config) -> bool:
    """Check if the user wants placeholder papers when search is broken."""
    return True  # Always include placeholder set for now


def _fallback_placeholder_export(
    vault_dir: Path,
    ws: Path,
    result: dict,
) -> None:
    """Fallback: write placeholder .md files when B1 data is unavailable."""
    papers_dir = vault_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        fpath = papers_dir / f"paper-{i + 1:03d}.md"
        fpath.write_text(
            f"# Paper {i + 1:03d}\n\n"
            f"*Placeholder — B1 scoring data not available. Run 'lcortex run <query>' first.*\n",
            encoding="utf-8",
        )
        result["files"].append(str(fpath.relative_to(ws)))

    # Write 00-meta files
    meta_dir = vault_dir / "00-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "dry-summary.md").write_text(
        "# Dry Score Summary\n\n*No B1 scoring data available.*\n",
        encoding="utf-8",
    )
    result["files"].append(str((meta_dir / "dry-summary.md").relative_to(ws)))

    # Write README
    (vault_dir / "README.md").write_text(
        _VAULT_README if "_VAULT_README" in dir() else
        "# Literature Cortex Vault\n\n*Placeholder vault.*\n",
        encoding="utf-8",
    )
    result["files"].append(str((vault_dir / "README.md").relative_to(ws)))

    # Write causal-map.md
    (vault_dir / "causal-map.md").write_text(
        "# Causal Map\n\n"
        "```mermaid\ngraph TD\n"
        "  A[Search] --> B[Scoring]\n"
        "  B --> G[Export]\n"
        "```\n",
        encoding="utf-8",
    )
    result["files"].append(str((vault_dir / "causal-map.md").relative_to(ws)))

    result["file_count"] = 5 + 4  # 5 papers + README + summary + causal-map + index


def _infer_domain(query: str) -> tuple[set[str] | None, set[str] | None]:
    """Auto-detect domain constraints from user query.

    If the query is about vibration control, adds domain-require words
    (vibration/structural/mechanical) and domain-exclude words
    (wildfire/satellite/robot/network) to filter out cross-domain noise.

    Returns (domain_require, domain_exclude) — both can be None if no
    domain detected.
    """
    q = query.lower()

    # ─── Domain definitions ───
    domains = {
        "vibration": {
            "require": {"vibration", "vibrating", "structural", "mechanical",
                       "cantilever", "beam", "plate", "rotor", "isolator",
                       "damping", "modal", "resonance", "seismic"},
            "exclude": {"wildfire", "satellite", "robot", "network",
                       "image", "video", "speech", "audio", "language",
                       "cancer", "drug", "clinical", "patient",
                       "financial", "stock", "market", "trading",
                       "social", "user", "recommend", "sentiment"},
        },
        "power": {
            "require": {"power", "voltage", "current", "converter", "inverter",
                       "grid", "battery", "motor", "generator", "transformer"},
            "exclude": {"wildfire", "satellite", "network", "image", "speech",
                       "cancer", "drug", "financial", "social"},
        },
        "aerospace": {
            "require": {"aircraft", "satellite", "launch", "spacecraft", "rocket",
                       "aero", "flight", "wing", "turbine", "propulsion"},
            "exclude": {"wildfire", "cancer", "drug", "financial"},
        },
    }

    # ─── Detect domain from query keywords ───
    for domain_name, domain_def in domains.items():
        require_set = domain_def["require"]
        exclude_set = domain_def["exclude"]
        # Check if query contains at least 2 domain-require words
        matches = sum(1 for w in require_set if w in q)
        if matches >= 1:
            return require_set, exclude_set

    # ─── Default: no domain detection ───
    return None, None

def _unwrap_markdown_fence(text: str) -> str:
    """Strip markdown code fences from LLM synthesis output."""
    if not isinstance(text, str):
        return str(text) if text else ""
    text = text.strip()
    fences = ["```markdown", "```md", "```"]
    for f in fences:
        if text.startswith(f):
            text = text[len(f):].lstrip("\n")
            if text.endswith("```"):
                text = text[:-3].rstrip()
            break
    return text.strip()

