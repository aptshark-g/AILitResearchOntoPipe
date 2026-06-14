"""Pipeline Monitor — tracks a running pipeline.

Writes to:
  - workspace/{slug}/state.json         — updated after each phase boundary
  - workspace/{slug}/events.jsonl       — appended after each item/error/degradation
  - workspace/{slug}/monitor_report.md  — final detailed report

All timestamps are ISO 8601 with timezone (Asia/Shanghai by default, UTC fallback).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("lcortex.monitor")


def _now_iso() -> str:
    """Return current time as ISO 8601 string with timezone.

    Uses local Asia/Shanghai timezone (UTC+8) as default,
    falling back to UTC with marker if TZ env is not set.
    """
    try:
        # Prefer local timezone if available (Asia/Shanghai from env)
        local_tz = os.environ.get("TZ", "")
        if "+" in local_tz or "-" in local_tz:
            return datetime.now(timezone.utc).isoformat()
        return datetime.now().astimezone().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


class PipelineMonitor:
    """Tracks a running pipeline: progress, token usage, errors, events.

    Parameters
    ----------
    workspace:
        The root workspace directory.  Per-run state lives in
        ``workspace/{slug}/``.
    mode:
        Pipeline mode — ``"dry"``, ``"lite"``, or ``"full"``.
    """

    def __init__(self, workspace: Path, mode: str = "full"):
        self._workspace = Path(workspace)
        self._mode = mode
        self._slug: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._events_fh = None           # Open file handle for events.jsonl

        # ── internal counters ──────────────────────────────────────
        self._phase_order: list[str] = []
        self._phase_labels: dict[str, str] = {}
        self._phase_start_times: dict[str, float] = {}
        self._phase_item_counts: dict[str, int] = {}
        self._phase_completed: dict[str, int] = {}
        self._phase_summaries: dict[str, dict] = {}

        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._errors: list[dict] = []
        self._degradations: list[dict] = []
        self._warnings: list[str] = []

        self._start_time: float = 0.0
        self._end_time: float = 0.0

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────
    def start_run(self, slug: str, query: str) -> None:
        """Begin tracking a pipeline run.

        Creates the run directory and opens the events.jsonl file for
        append-only writing.
        """
        self._slug = slug
        self._run_dir = self._workspace / slug
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.monotonic()

        # Open events.jsonl in append mode — atomically safe across restarts
        events_path = self._run_dir / "events.jsonl"
        self._events_fh = open(str(events_path), "a", encoding="utf-8")

        self.log_event("run_start", {
            "query": query,
            "mode": self._mode,
        })

    def end_run(self) -> None:
        """Close the run and finalize reporting."""
        self._end_time = time.monotonic()

        self.log_event("run_end", {
            "total_errors": len(self._errors),
            "total_degradations": len(self._degradations),
        })

        if self._events_fh:
            self._events_fh.close()
            self._events_fh = None

        # Write final monitor_report.md
        self._write_monitor_report()

    def close(self) -> None:
        """Close without finalizing (for error exits)."""
        if self._events_fh:
            self._events_fh.close()
            self._events_fh = None

    # ────────────────────────────────────────────────────────────────────
    # Phase tracking
    # ────────────────────────────────────────────────────────────────────
    def phase_start(self, phase: str, label: str = "", total_items: int = 0) -> None:
        """Mark the beginning of a pipeline phase."""
        self._phase_order.append(phase)
        self._phase_labels[phase] = label
        self._phase_start_times[phase] = time.monotonic()
        self._phase_item_counts[phase] = total_items
        self._phase_completed[phase] = 0

        self.log_event("phase_start", {
            "phase": phase,
            "label": label,
            "total_items": total_items,
        })

    def phase_progress(self, phase: str, item_idx: int, item_label: str = "") -> None:
        """Update phase progress (intermediate step)."""
        if phase in self._phase_completed:
            self._phase_completed[phase] = max(
                self._phase_completed[phase], item_idx
            )

    def phase_end(self, phase: str, summary: Optional[dict] = None) -> None:
        """Mark the end of a pipeline phase.

        Parameters
        ----------
        phase:
            Phase key (A, B, C, …).
        summary:
            Phase-specific summary dict (e.g. ``{"passed": 5, "flagged": 2}``).
        """
        elapsed = 0.0
        if phase in self._phase_start_times:
            elapsed = time.monotonic() - self._phase_start_times[phase]

        self._phase_summaries[phase] = summary or {}

        self.log_event("phase_end", {
            "phase": phase,
            "elapsed_s": round(elapsed, 3),
            "summary": summary or {},
        })

    # ────────────────────────────────────────────────────────────────────
    # Item-level tracking (per paper)
    # ────────────────────────────────────────────────────────────────────
    def item_start(self, phase: str, item_id: str, item_title: str) -> None:
        """Log the start of processing an individual item (paper)."""
        # (Lightweight — just track count)
        if phase in self._phase_completed:
            self._phase_completed[phase] += 1

    def item_done(self, phase: str, item_id: str, result: Optional[dict] = None,
                  tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Log successful completion of an item."""
        self.record_tokens(phase, tokens_in, tokens_out)

        event_data: dict[str, Any] = {
            "paper_id": item_id,
        }
        if result:
            if "scores" in result:
                event_data["scores"] = result["scores"]
            if "mean_score" in result:
                event_data["mean_score"] = result["mean_score"]
            if "title" in result:
                event_data["title"] = result["title"]
            if "knowledge_level" in result:
                event_data["knowledge_level"] = result["knowledge_level"]

        self.log_event("item_scored", event_data, phase=phase)

    def item_error(self, phase: str, item_id: str, error: str) -> None:
        """Log that an item failed processing."""
        self.log_event("item_error", {
            "paper_id": item_id,
            "error": error[:500],
        }, phase=phase)

    # ────────────────────────────────────────────────────────────────────
    # Token tracking
    # ────────────────────────────────────────────────────────────────────
    def record_tokens(self, phase: str, tokens_in: int, tokens_out: int) -> None:
        """Accumulate token usage counters."""
        self._total_tokens_in += tokens_in
        self._total_tokens_out += tokens_out

    # ────────────────────────────────────────────────────────────────────
    # Event logging (append-only JSONL)
    # ────────────────────────────────────────────────────────────────────
    def log_event(self, event_type: str, data: dict,
                  phase: Optional[str] = None) -> None:
        """Append a structured event to ``events.jsonl``.

        The event object contains at minimum ``t`` (timestamp), ``e``
        (event_type), and all keys from *data*.  If *phase* is provided
        it is added as the ``p`` key.
        """
        record: dict[str, Any] = {
            "t": _now_iso(),
            "e": event_type,
        }
        if phase:
            record["p"] = phase
        record.update(data)

        if self._events_fh:
            # Append immediately — atomic per-line
            self._events_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._events_fh.flush()

    # ────────────────────────────────────────────────────────────────────
    # Degradation
    # ────────────────────────────────────────────────────────────────────
    def record_degradation(self, phase: str, source: str, reason: str) -> None:
        """Record a degradation event (partial failure, fallback used)."""
        self._degradations.append({
            "phase": phase,
            "source": source,
            "reason": reason,
            "t": _now_iso(),
        })
        self.log_event("degradation", {
            "source": source,
            "reason": reason,
        }, phase=phase)

    # ────────────────────────────────────────────────────────────────────
    # Error handling
    # ────────────────────────────────────────────────────────────────────
    def record_error(self, phase: str, error: str,
                     recoverable: bool = True) -> None:
        """Record an error event."""
        self._errors.append({
            "phase": phase,
            "error": error[:500],
            "recoverable": recoverable,
            "t": _now_iso(),
        })
        self.log_event("error", {
            "error": error[:500],
            "recoverable": recoverable,
        }, phase=phase)

    # ────────────────────────────────────────────────────────────────────
    # Reporting
    # ────────────────────────────────────────────────────────────────────
    def progress_report(self) -> str:
        """Human-readable one-liner describing current progress.

        Example::

            [B] Scoring  3/12  ✅2 ⚠1  |  tokens: 12 345 in / 4 567 out
        """
        parts: list[str] = []
        for phase in self._phase_order:
            label = self._phase_labels.get(phase, phase)
            total = self._phase_item_counts.get(phase, 0)
            done = self._phase_completed.get(phase, 0)
            errors_here = sum(
                1 for e in self._errors if e["phase"] == phase
            )
            parts.append(
                f"[{phase}] {label}: {done}/{total}"
                + (f" ⚠{errors_here}" if errors_here else "")
            )
        suffix = ""
        if self._total_tokens_in or self._total_tokens_out:
            suffix = f"  |  tki:{self._total_tokens_in} tko:{self._total_tokens_out}"
        return "  ".join(parts) + suffix

    def detailed_report(self) -> str:
        """Full Markdown progress report."""
        lines: list[str] = [
            "# 📊 Literature Cortex — Monitor Report",
            "",
            f"**Mode:** `{self._mode}`",
            f"**Slug:** `{self._slug or '—'}`",
            "",
            "## Phase Summary",
            "",
        ]

        # Table header
        lines.append("| Phase | Label | Items | Status | Time |")
        lines.append("|-------|-------|-------|--------|------|")

        for phase in self._phase_order:
            label = self._phase_labels.get(phase, phase)
            total = self._phase_item_counts.get(phase, 0)
            done = self._phase_completed.get(phase, 0)
            elapsed_str = ""
            if phase in self._phase_start_times:
                elapsed = time.monotonic() - self._phase_start_times[phase]
                elapsed_str = f"{elapsed:.1f}s"

            status = "✅" if done >= total and total > 0 else (
                "🔄" if done > 0 else ("⏭" if total == 0 else "⏳")
            )
            lines.append(
                f"| {phase} | {label} | {done}/{total} | {status} | {elapsed_str} |"
            )

        lines.extend([
            "",
            "## Token Usage",
            "",
            f"- **Input tokens:** {self._total_tokens_in:,}",
            f"- **Output tokens:** {self._total_tokens_out:,}",
            f"- **Total:** {self._total_tokens_in + self._total_tokens_out:,}",
        ])

        if self._errors:
            lines.extend([
                "",
                "## Errors",
                "",
            ])
            for err in self._errors:
                rec = "🔄 recoverable" if err.get("recoverable") else "❌ fatal"
                lines.append(
                    f"- **[{err['phase']}]** ({rec}) {err['error'][:120]}"
                )

        if self._degradations:
            lines.extend([
                "",
                "## Degradations",
                "",
            ])
            for d in self._degradations:
                lines.append(
                    f"- **[{d['phase']}]** {d['source']}: {d['reason'][:120]}"
                )

        return "\n".join(lines)

    def final_summary(self) -> dict:
        """Return a JSON-serializable summary dict suitable for merging
        into ``state.json``.
        """
        elapsed = self._end_time - self._start_time if self._end_time else 0.0

        phases_status: dict[str, str] = {}
        for phase in self._phase_order:
            total = self._phase_item_counts.get(phase, 0)
            done = self._phase_completed.get(phase, 0)
            if total == 0 and done == 0:
                phases_status[phase] = "skipped"
            elif done >= total:
                phases_status[phase] = "completed"
            elif done > 0:
                phases_status[phase] = "partial"
            else:
                phases_status[phase] = "pending"

        return {
            "mode": self._mode,
            "slug": self._slug,
            "phases": phases_status,
            "total_errors": len(self._errors),
            "total_degradations": len(self._degradations),
            "tokens_in": self._total_tokens_in,
            "tokens_out": self._total_tokens_out,
            "elapsed_s": round(elapsed, 2),
            "errors": [e["error"][:200] for e in self._errors],
            "degradations": [
                f"{d['source']}: {d['reason']}" for d in self._degradations
            ],
        }

    # ────────────────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────────────────
    def _write_monitor_report(self) -> None:
        """Write the detailed report to ``monitor_report.md``."""
        if not self._run_dir:
            return
        report_path = self._run_dir / "monitor_report.md"
        try:
            report_path.write_text(self.detailed_report(), encoding="utf-8")
            log.info("Monitor report written to %s", report_path)
        except OSError as exc:
            log.error("Failed to write monitor report: %s", exc)
