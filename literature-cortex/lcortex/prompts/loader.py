"""Prompt loader — reads .md prompt templates and renders them with variable substitution.

All prompt files follow the same structure:

    # Title (comment)
    # ...
    ---
    ## SYSTEM PROMPT
    <system prompt content>
    ## USER PROMPT TEMPLATE
    <user template content with {{variables}}>
    ## OUTPUT SCHEMA (optional)
    ...
    ## VARIABLES (optional)
    ...

Template variables use ``{{variable_name}}`` syntax.
"""

from __future__ import annotations

import os
import re
from typing import Any


_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Known document-level section headers (case-insensitive, normalized to underscore form).
# Only these headers split the document; any ``## `` inside the body is kept as content.
_KNOWN_SECTIONS: set[str] = {
    "system_prompt",
    "user_prompt_template",
    "output_schema",
    "output_format",
    "variables",
    "notes",
}

# Variable placeholder pattern: {{variable_name}}
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


def _normalize_heading(heading: str) -> str:
    """Normalize a ``## HEADING`` line to a lookup key."""
    return heading.strip().lower().replace(" ", "_")


def _split_document_sections(text: str) -> dict[str, str]:
    """Split a prompt .md file into document-level sections.

    Only lines matching ``## <KNOWN_SECTION>`` at the start of a line trigger
    a section boundary.  ``## `` lines inside code fences or that don't match
    a known section name are treated as body content, not section headers.

    Returns:
        dict mapping normalized section name → body text (whitespace-trimmed).
        Any preamble before the first section header is stored under key ``""``.
    """
    lines = text.split("\n")
    sections: dict[str, str] = {}
    current_section = ""
    current_body: list[str] = []

    for line in lines:
        # Check if this is a document-level section header
        stripped = line.strip()
        if stripped.startswith("## "):
            potential_heading = stripped[3:]  # remove "## "
            normalized = _normalize_heading(potential_heading)
            if normalized in _KNOWN_SECTIONS:
                # Save current section
                if current_body:
                    sections[current_section] = "\n".join(current_body).strip()
                    current_body = []
                current_section = normalized
                continue

        # Not a known section boundary → accumulate in current section body
        current_body.append(line)

    # Save the last section
    if current_body:
        sections[current_section] = "\n".join(current_body).strip()

    return sections


def load_prompt(name: str) -> dict[str, str]:
    """Load a prompt .md file and parse its SYSTEM PROMPT and USER PROMPT TEMPLATE sections.

    Args:
        name: Prompt file stem, e.g. ``"stage_b_scoring"``.
              The ``.md`` extension is added automatically.

    Returns:
        dict with keys ``"system_prompt"`` and ``"user_prompt_template"``.
        Also includes ``"title"`` (first-line comment, stripped of leading ``#``),
        ``"output_schema"``, ``"output_format"``, and ``"variables"`` if present.

    Raises:
        FileNotFoundError: if the prompt file does not exist.
        ValueError: if required sections are missing.
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Extract title from the first ``# ...`` line
    title = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            title = stripped.lstrip("#").strip()
            break

    sections = _split_document_sections(raw)

    system_prompt = sections.get("system_prompt", "")
    user_prompt_template = sections.get("user_prompt_template", "")

    if not system_prompt:
        raise ValueError(
            f"Prompt file {path} is missing a ## SYSTEM PROMPT section"
        )
    if not user_prompt_template:
        raise ValueError(
            f"Prompt file {path} is missing a ## USER PROMPT TEMPLATE section"
        )

    result: dict[str, str] = {
        "title": title,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }

    # Include other known sections
    for key in ("output_schema", "output_format", "variables"):
        if key in sections:
            result[key] = sections[key]

    return result


def render_prompt(name: str, variables: dict[str, Any] | None = None) -> tuple[str, str]:
    """Load a prompt and render its template variables.

    Args:
        name: Prompt file stem (e.g. ``"stage_b_scoring"``).
        variables: Dict mapping variable names to their string values.
                   Any ``{{varname}}`` in the user template is replaced by
                   ``str(variables["varname"])``.  Lists are joined with
                   newlines; dicts are serialized as JSON.

    Returns:
        ``(system_prompt, rendered_user_message)`` tuple.
        The system prompt is returned as-is (no variable substitution).
    """
    prompt = load_prompt(name)
    system = prompt["system_prompt"]
    user_template = prompt["user_prompt_template"]
    vars_dict = variables or {}

    def _replacer(m: re.Match) -> str:
        varname = m.group(1)
        val = vars_dict.get(varname)
        if val is None:
            # Leave unresolved variables as-is (visible to LLM as {{var}})
            return f"{{{{{varname}}}}}"
        if isinstance(val, list):
            return "\n".join(str(item) for item in val)
        if isinstance(val, dict):
            import json
            return json.dumps(val, ensure_ascii=False, indent=2)
        return str(val)

    user_message = _VAR_RE.sub(_replacer, user_template)
    return system, user_message


# ---------------------------------------------------------------------------
# Convenience — list available prompts
# ---------------------------------------------------------------------------

def list_prompts() -> list[str]:
    """Return a list of available prompt names (without ``.md`` extension)."""
    result: list[str] = []
    for fname in os.listdir(_PROMPTS_DIR):
        if fname.endswith(".md") and not fname.startswith("__"):
            result.append(fname[:-3])  # strip .md
    result.sort()
    return result
