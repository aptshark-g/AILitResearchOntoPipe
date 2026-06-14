"""
JSONL Streaming Output — write/read search results as newline-delimited JSON.

Format
──────
Each search session produces one JSONL file with a header, body lines,
and a status footer::

    {"type": "search_metadata", "query": "...", ...}
    {"paper_id": "...", "title": "...", ...}
    {"paper_id": "...", "title": "...", ...}
    {"type": "sources_status", "statuses": {...}, "total_papers": N}

This format is append-friendly (each line is self-contained) and easy to
read with ``tail -f`` during streaming.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Iterator, TextIO, Optional

log = logging.getLogger("lcortex.search.stream")

# Character budget for error reason strings in sources_status
MAX_REASON_CHARS = 200


def stream_to_jsonl(
    generator: Iterator[dict],
    output_path: str | Path,
    *,
    query: str = "",
    year_min: int | None = None,
    year_max: int | None = None,
    sources: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Write a paper generator to a JSONL file.

    Writes three sections:
        1. **Metadata header** — search parameters and start timestamp
        2. **Paper records** — one unified-schema dict per line
        3. **Sources status footer** — per-source counts and errors

    Args:
        generator:   iterator yielding paper dicts (unified schema)
        output_path: where to write the JSONL file (will be overwritten)
        query:       original search query (for metadata header)
        year_min:    year filter minimum
        year_max:    year filter maximum
        sources:     list of source names used
        metadata:    extra key-value pairs to include in the header (optional)

    Returns:
        ``sources_status`` dict with per-source counts and ``total_papers``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_counts: dict[str, int] = {}
    paper_count = 0

    with open(output_path, "w", encoding="utf-8") as fh:
        # ── Header ──
        header: dict = {
            "type": "search_metadata",
            "query": query[:200],
            "year_min": year_min,
            "year_max": year_max,
            "sources": sources or [],
            "started_at": datetime.datetime.now().isoformat(),
            "streaming": True,
        }
        if metadata:
            header.update(metadata)
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        fh.flush()

        # ── Papers ──
        for paper in generator:
            src = paper.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

            fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
            fh.flush()
            paper_count += 1

        # ── Footer ──
        sources_status = _build_sources_status(source_counts, sources or [], paper_count)
        status_line: dict = {
            "type": "sources_status",
            "statuses": sources_status,
            "total_papers": paper_count,
        }
        fh.write(json.dumps(status_line, ensure_ascii=False) + "\n")
        fh.flush()

    log.info("JSONL stream written: %d papers → %s", paper_count, output_path)
    return source_counts


def _build_sources_status(
    source_counts: dict[str, int],
    expected_sources: list[str],
    total_papers: int,
) -> dict[str, dict]:
    """Build the sources_status block for the JSONL footer."""
    statuses: dict[str, dict] = {}
    for src in expected_sources:
        count = source_counts.get(src, 0)
        statuses[src] = {"status": "ok", "count": count}
    # Include any unexpected sources that actually returned results
    for src in source_counts:
        if src not in statuses:
            statuses[src] = {"status": "ok", "count": source_counts[src]}
    return statuses


def read_jsonl_stream(path: str | Path) -> Iterator[dict]:
    """Read a JSONL file and yield paper dicts (skips metadata/status lines).

    Lines with ``"type"`` set to ``"search_metadata"`` or ``"sources_status"``
    are silently skipped — only paper records are yielded.

    Args:
        path: path to a JSONL file produced by :func:`stream_to_jsonl`

    Yields:
        dict — one unified paper record per line

    Example::

        for paper in read_jsonl_stream("results.jsonl"):
            print(paper["title"])
    """
    path = Path(path)
    if not path.exists():
        log.warning("JSONL file not found: %s", path)
        return

    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("JSONL line %d is not valid JSON — skipping", line_num)
                continue

            rtype = record.get("type", "")
            if rtype in ("search_metadata", "sources_status"):
                continue

            yield record


def read_jsonl_metadata(path: str | Path) -> dict | None:
    """Extract the metadata header from a JSONL file.

    Args:
        path: path to a JSONL file

    Returns:
        Metadata dict (the first line's content) or ``None`` if the file
        cannot be read or the first line is not metadata.
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
        if not first_line:
            return None
        try:
            record = json.loads(first_line)
        except json.JSONDecodeError:
            return None

        if record.get("type") == "search_metadata":
            return record
    return None


def read_jsonl_sources_status(path: str | Path) -> dict | None:
    """Extract the sources_status footer from a JSONL file.

    Reads the *last* line of the file (the footer).
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path, "rb") as fh:
        # Seek to last few KB to find the footer line
        try:
            fh.seek(0, 2)  # end
            file_size = fh.tell()
            chunk_size = min(file_size, 4096)
            fh.seek(max(0, file_size - chunk_size))
            tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return None

        # Last non-empty line
        lines = tail.strip().split("\n")
        if not lines:
            return None
        try:
            record = json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

        if record.get("type") == "sources_status":
            return record
    return None
