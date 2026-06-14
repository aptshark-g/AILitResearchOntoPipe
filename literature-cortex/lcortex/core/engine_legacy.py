"""Main pipeline engine — orchestrates phases A→G.

The engine is the heart of Literature Cortex. It coordinates:
  Phase A  → Multi-source paper search
  Phase B  → 4C+L scoring (LLM)
  Phase C  → Limitation search
  Phase D  → Extension search
  Phase E  → Synthesis (LLM)
  Phase F  → Structure template extraction (LLM)
  Phase F-2 → Deconstruct-reconstruct-conflict detection (LLM)
  Phase G  → Dual-linkage knowledge graph + Obsidian export

Each phase can run independently or as part of a full pipeline.
Phases that require LLM gracefully skip when no adapter is configured.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lcortex.core.config import Config, get_config
from lcortex.core.state import (
    PHASES,
    PHASE_LABELS,
    PipelineState,
    PhaseStatus,
    checkpoint_marker,
    load_state,
    resume_from_checkpoint,
    save_state,
)

logger = logging.getLogger("lcortex.engine")


# ---------------------------------------------------------------------------
# LLM adapter detection (lightweight — real adapters in intelligence/)
# ---------------------------------------------------------------------------
def _has_llm(config: Config) -> bool:
    """Check if an LLM adapter is available."""
    return config.llm.is_configured


def _llm_skip_message(phase: str) -> str:
    """Printable skip message when LLM is unavailable."""
    label = PHASE_LABELS.get(phase, phase)
    return f"⚠️  LLM not configured — skipping Phase {phase} ({label})"


# ---------------------------------------------------------------------------
# Individual phase runners (placeholders)
# ---------------------------------------------------------------------------
def _run_phase_a(query: str, config: Config, state: PipelineState,
                 max_results: int = 10, year_min: Optional[int] = None,
                 year_max: Optional[int] = None, sources: str = "arxiv",
                 stream: bool = False,
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase A: Multi-source paper search. No LLM required."""
    state.mark("A", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n🔍 Phase A: Searching '{query}'")
    print(f"   Sources: {sources} | Max: {max_results} "
          f"| Year: {year_min or '—'}–{year_max or '—'}")

    # TODO: Replace with actual search adapter calls
    candidates_file = ws / "01-candidates.jsonl"
    if not candidates_file.exists():
        candidates_file.write_text("", encoding="utf-8")

    print(f"   📄 Found 0 papers (search adapter not yet implemented)")
    print(f"   ✅ Phase A complete")
    state.mark_completed("A")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "A")
    return state


def _run_phase_b(config: Config, state: PipelineState,
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase B: 4C+L scoring via LLM."""
    if not _has_llm(config):
        msg = _llm_skip_message("B")
        print(f"\n{msg}")
        state.mark_skipped("B", "no LLM adapter configured")
        return state

    state.mark("B", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n🤖 Phase B: 4C+L Scoring")
    # TODO: Load candidates, call LLM adapter for each
    print(f"   ⚠️  Scoring adapter not yet implemented — skipping")
    state.mark_skipped("B", "scoring adapter not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "B")
    return state


def _run_phase_c(config: Config, state: PipelineState,
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase C: Limitation search."""
    state.mark("C", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n📚 Phase C: Limitation Search")
    print(f"   ⚠️  Limitation search not yet implemented — skipping")
    state.mark_skipped("C", "not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "C")
    return state


def _run_phase_d(config: Config, state: PipelineState,
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase D: Extension search."""
    state.mark("D", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n📚 Phase D: Extension Search")
    print(f"   ⚠️  Extension search not yet implemented — skipping")
    state.mark_skipped("D", "not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "D")
    return state


def _run_phase_e(config: Config, state: PipelineState,
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase E: Synthesis — LLM generates structured review."""
    if not _has_llm(config):
        msg = _llm_skip_message("E")
        print(f"\n{msg}")
        state.mark_skipped("E", "no LLM adapter configured")
        return state

    state.mark("E", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n📝 Phase E: Synthesis")
    print(f"   ⚠️  Synthesis not yet implemented — skipping")
    state.mark_skipped("E", "not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "E")
    return state


def _run_phase_f(config: Config, state: PipelineState,
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase F: Structure template extraction + knowledge level inference."""
    if not _has_llm(config):
        msg = _llm_skip_message("F")
        print(f"\n{msg}")
        state.mark_skipped("F", "no LLM adapter configured")
        return state

    state.mark("F", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n🏗️  Phase F: Structure Template Extraction")
    print(f"   ⚠️  Structure extraction not yet implemented — skipping")
    state.mark_skipped("F", "not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "F")
    return state


def _run_phase_f2(config: Config, state: PipelineState,
                  workspace: Optional[Path] = None) -> PipelineState:
    """Phase F-2: Deconstruct-Reconstruct-Conflict Detection (Double-Loop)."""
    if not _has_llm(config):
        msg = _llm_skip_message("F2")
        print(f"\n{msg}")
        state.mark_skipped("F2", "no LLM adapter configured")
        return state

    state.mark("F2", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n🔄 Phase F-2: Conflict Detection (Double-Loop)")
    print(f"   ⚠️  Conflict detection not yet implemented — skipping")
    state.mark_skipped("F2", "not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "F2")
    return state


def _run_phase_g(config: Config, state: PipelineState,
                 export_format: str = "obsidian",
                 workspace: Optional[Path] = None) -> PipelineState:
    """Phase G: Dual-linkage knowledge graph + export."""
    state.mark("G", PhaseStatus.IN_PROGRESS)
    ws = workspace or Path(config.workspace.path)

    print(f"\n📊 Phase G: Graph Generation + Export")
    print(f"   Format: {export_format}")
    print(f"   ⚠️  Graph generation + export not yet implemented — skipping")
    state.mark_skipped("G", "not yet implemented")
    save_state(str(ws), state)
    checkpoint_marker(str(ws), "G")
    return state


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    query: str,
    mode: str = "lite",
    config: Optional[Config] = None,
    max_results: int = 10,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    sources: str = "arxiv",
    stream: bool = False,
    export_format: str = "obsidian",
    resume: bool = False,
) -> PipelineState:
    """Run the full A→G literature research pipeline.

    Args:
        query: Search query string.
        mode: Pipeline mode (lite | auto | full | double-loop).
        config: Configuration. Loaded from defaults if None.
        max_results: Max papers to retrieve in Phase A.
        year_min: Minimum publication year.
        year_max: Maximum publication year.
        sources: Comma-separated search sources.
        stream: Stream results as JSONL.
        export_format: Export format (obsidian | json).
        resume: Resume from last checkpoint if available.

    Returns:
        PipelineState with final status of all phases.
    """
    if config is None:
        config = get_config()

    # Determine slug and run workspace
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:80]
    run_ws = Path(config.workspace.path) / slug
    run_ws.mkdir(parents=True, exist_ok=True)

    # Resume or create new state
    if resume:
        state = _resume_pipeline(str(run_ws), query, slug, mode)
        if state is None:
            state = PipelineState(
                query=query, slug=slug, mode=mode,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
    else:
        state = PipelineState(
            query=query, slug=slug, mode=mode,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    print(f"\n{'='*60}")
    print(f"Literature Cortex Pipeline — {mode.upper()} mode")
    print(f"Query: {query}")
    print(f"Workspace: {run_ws}")
    print(f"LLM: {'configured' if _has_llm(config) else 'not configured (no-AI mode)'}")
    print(f"{'='*60}")

    # ── Phase A: Search (always runs, no LLM) ──
    state = _run_phase_a(
        query, config, state,
        max_results=max_results, year_min=year_min,
        year_max=year_max, sources=sources, stream=stream,
        workspace=run_ws,
    )

    # ── Phase B: Scoring (LLM) ──
    if mode in ("lite", "auto", "full", "double-loop"):
        state = _run_phase_b(config, state, workspace=run_ws)

    # ── Phase C/D: Limitation/Extension search ──
    if mode in ("auto", "full", "double-loop"):
        state = _run_phase_c(config, state, workspace=run_ws)
        state = _run_phase_d(config, state, workspace=run_ws)

    # ── Phase E: Synthesis (LLM) ──
    if mode in ("full", "double-loop"):
        state = _run_phase_e(config, state, workspace=run_ws)

    # ── Phase F: Structure extraction (LLM) ──
    if mode in ("full", "double-loop"):
        state = _run_phase_f(config, state, workspace=run_ws)

    # ── Phase F-2: Conflict detection (LLM, double-loop) ──
    if mode == "double-loop":
        state = _run_phase_f2(config, state, workspace=run_ws)

    # ── Phase G: Graph + Export ──
    state = _run_phase_g(config, state, export_format=export_format, workspace=run_ws)

    # Summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete: {query}")
    _print_phase_summary(state)
    print(f"State saved to: {run_ws / 'state.json'}")
    print(f"{'='*60}")

    return state


def _resume_pipeline(workspace: str, query: str, slug: str,
                     mode: str) -> Optional[PipelineState]:
    """Try to resume from a previous run. Returns None if no state found."""
    state = load_state(workspace)
    if state is not None:
        print(f"📋 Resuming pipeline from state.json (slug={state.slug})")
        return state
    resume_phase = resume_from_checkpoint(workspace)
    if resume_phase is not None:
        print(f"📋 Found checkpoints — would resume from Phase {resume_phase}")
    return None


def _print_phase_summary(state: PipelineState):
    """Print a compact phase status table."""
    print("\nPhase Summary:")
    for p in PHASES:
        status = state.phases[p]
        label = PHASE_LABELS.get(p, p)
        icon = {
            PhaseStatus.COMPLETED: "✅",
            PhaseStatus.PENDING: "⏳",
            PhaseStatus.IN_PROGRESS: "🔄",
            PhaseStatus.SKIPPED: "⏭️",
            PhaseStatus.ERROR: "❌",
        }.get(status, "❓")
        print(f"  {icon} Phase {p} ({label}): {status.value}")
