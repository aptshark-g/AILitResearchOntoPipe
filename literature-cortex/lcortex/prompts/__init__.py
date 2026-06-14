"""Prompt loader — reads .md prompt templates and renders with variable substitution.

All prompt files follow the same structure:
    ## SYSTEM PROMPT → system prompt content
    ## USER PROMPT TEMPLATE → template with {{variables}}
    ## OUTPUT SCHEMA → JSON schema (optional)
"""

from lcortex.prompts.loader import render_prompt, load_prompt, list_prompts

__all__ = [
    "render_prompt",
    "load_prompt",
    "list_prompts",
]
