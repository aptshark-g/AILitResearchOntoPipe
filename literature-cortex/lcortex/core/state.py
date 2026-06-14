"""Pipeline state machine with checkpoint/resume support.

Manages the state of a literature research pipeline across phases A→G.
Supports atomic writes and checkpoint markers for safe resume.

Usage:
    state = PipelineState(query="FxLMS vibration")
    state.mark_completed("A")
    save_state("/path/to/vault/my-slug", state)

    state = load_state("/path/to/vault/my-slug")
    next_phase = state.next_phase()
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------
class PhaseStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"      # e.g. no LLM adapter available
    ERROR = "error"


PHASES = ["A", "B1", "B2", "C", "D", "E", "F", "F2", "G"]

PHASE_LABELS = {
    "A": "search",
    "B1": "dry_scoring",
    "B2": "scoring",
    "C": "limitations",
    "D": "extensions",
    "E": "synthesis",
    "F": "structure",
    "F2": "conflict",
    "G": "graph",
}


@dataclass
class PipelineState:
    """Current state of a literature research pipeline."""

    query: str = ""
    slug: str = ""
    mode: str = "lite"          # lite | auto | full | double-loop

    # Per-phase status
    phases: dict[str, PhaseStatus] = field(default_factory=lambda: {
        p: PhaseStatus.PENDING for p in PHASES
    })

    # Metadata
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    paper_count: int = 0
    passed_count: int = 0
    notes: list[str] = field(default_factory=list)

    def mark(self, phase: str, status: PhaseStatus):
        """Mark a phase with a status."""
        if phase not in PHASES:
            raise ValueError(f"Unknown phase: {phase}. Known: {PHASES}")
        self.phases[phase] = status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, phase: str):
        """Shorthand to mark a phase as completed."""
        self.mark(phase, PhaseStatus.COMPLETED)

    def mark_skipped(self, phase: str, reason: str = ""):
        """Mark a phase as skipped (e.g. missing LLM adapter)."""
        self.mark(phase, PhaseStatus.SKIPPED)
        if reason:
            self.notes.append(f"[{phase}] SKIPPED: {reason}")

    def mark_error(self, phase: str, reason: str = ""):
        """Mark a phase as errored."""
        self.mark(phase, PhaseStatus.ERROR)
        if reason:
            self.notes.append(f"[{phase}] ERROR: {reason}")

    def next_phase(self) -> Optional[str]:
        """Return the first pending phase, or None if all done."""
        for p in PHASES:
            if self.phases[p] in (PhaseStatus.PENDING, PhaseStatus.IN_PROGRESS):
                return p
        return None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON."""
        return {
            "query": self.query,
            "slug": self.slug,
            "mode": self.mode,
            "phases": {p: v.value for p, v in self.phases.items()},
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "paper_count": self.paper_count,
            "passed_count": self.passed_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        """Deserialize from a plain dict."""
        phases_raw = data.get("phases", {})
        phases = {}
        for p in PHASES:
            raw = phases_raw.get(p, "pending")
            try:
                phases[p] = PhaseStatus(raw)
            except ValueError:
                phases[p] = PhaseStatus.PENDING

        return cls(
            query=data.get("query", ""),
            slug=data.get("slug", ""),
            mode=data.get("mode", "lite"),
            phases=phases,
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at"),
            paper_count=data.get("paper_count", 0),
            passed_count=data.get("passed_count", 0),
            notes=data.get("notes", []),
        )


# ---------------------------------------------------------------------------
# Persistence (atomic write)
# ---------------------------------------------------------------------------
def state_file(workspace: str) -> Path:
    """Full path to state.json inside workspace."""
    return Path(workspace) / "state.json"


def save_state(workspace: str, state: PipelineState):
    """Atomically save pipeline state to workspace/state.json."""
    state.updated_at = datetime.now(timezone.utc).isoformat()
    sf = state_file(workspace)
    sf.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
    # Atomic: write to .tmp then rename
    fd, tmpname = tempfile.mkstemp(dir=str(sf.parent), suffix=".tmp", prefix=".state.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmpname, sf)
    except Exception:
        os.unlink(tmpname)
        raise


def load_state(workspace: str) -> Optional[PipelineState]:
    """Load pipeline state from workspace/state.json, or None."""
    sf = state_file(workspace)
    if not sf.exists():
        return None
    try:
        with open(sf, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return PipelineState.from_dict(data)
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# ---------------------------------------------------------------------------
# Checkpoint / Resume
# ---------------------------------------------------------------------------
def checkpoint_marker(workspace: str, phase: str) -> Path:
    """Create a checkpoint marker file for a given phase.

    The marker is an empty file named .checkpoint_{phase}.
    Returns the Path to the marker.
    """
    marker = Path(workspace) / f".checkpoint_{phase}"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return marker


def list_checkpoints(workspace: str) -> list[str]:
    """List phases that have checkpoint markers, sorted A→G."""
    ws = Path(workspace)
    found = []
    if not ws.exists():
        return found
    for p in PHASES:
        if (ws / f".checkpoint_{p}").exists():
            found.append(p)
    return found


def resume_from_checkpoint(workspace: str) -> Optional[str]:
    """Determine the phase to resume from.

    Priority:
      1. Load state.json → return next_phase()
      2. Checkpoint markers → return first phase where marker doesn't exist
      3. None (start fresh)

    Returns the phase key (e.g. "A", "B", …) or None.
    """
    # Priority 1: state.json
    state = load_state(workspace)
    if state is not None:
        nxt = state.next_phase()
        if nxt is not None:
            return nxt
        # All phases completed in state
        return None

    # Priority 2: checkpoint markers (if no state.json)
    cps = list_checkpoints(workspace)
    if cps:
        # Find first phase without a marker
        for p in PHASES:
            if p not in cps:
                return p
        # All phases have markers
        return None

    # Nothing to resume from
    return None
