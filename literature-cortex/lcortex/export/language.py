"""Language converter for Literature Cortex outputs.

Converts review.md / Obsidian paper notes / metadata between languages.
Supports 3 modes:
  1. LLM-powered translation (DeepSeek/OpenAI/Claude — ~2-4K tokens per output)
  2. Hybrid: translate only top-level sections (综述/日次/说明), rest stays English
  3. LLM-powered glossary: translate key technical terms inline (~500 tokens)

Design: only triggered at export time (Phase G), not during pipeline.
         Source documents stay in original language (English).
         Translated copies are written to vault/lang/{lang}/.

Usage:
    lcortex export --lang zh       # Translate vault to Chinese
    lcortex run --lang zh          # Export with Chinese translation
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("lcortex.lang")


# ═══════════════════════════════════════════════════════════════════════
# Translation prompts (compact, optimized for token efficiency)
# ═══════════════════════════════════════════════════════════════════════

TRANSLATE_SYSTEM = {
    "zh": """You are a technical translator specializing in control systems and engineering.
Translate the following academic text from English to Simplified Chinese.
- Preserve ALL technical terms: algorithm names (FxLMS, MPC, LQR, Koopman), mathematical notation, paper IDs, citations.
- Keep markdown formatting, tables, YAML frontmatter, wikilinks intact.
- Translate section headers and body text naturally.
- Output the translated text ONLY. No explanation.""",

    "ja": """You are a technical translator specializing in control systems and engineering.
Translate the following academic text from English to Japanese.
- Preserve ALL technical terms: algorithm names (FxLMS, MPC, LQR, Koopman), mathematical notation, paper IDs, citations.
- Keep markdown formatting, tables, YAML frontmatter, wikilinks intact.
- Translate section headers and body text naturally.
- Output the translated text ONLY. No explanation.""",

    "ko": """You are a technical translator specializing in control systems and engineering.
Translate the following academic text from English to Korean.
- Preserve ALL technical terms: algorithm names (FxLMS, MPC, LQR), mathematical notation, paper IDs, citations.
- Keep markdown formatting, tables, YAML frontmatter, wikilinks intact.
- Translate section headers and body text naturally.
- Output the translated text ONLY. No explanation.""",
}

GLOSSARY_PROMPT = """Translate ONLY the following technical terms from English to {lang_name}. 
Output as JSON: {{"term": "translated"}}.
Terms: {terms}"""

# ═══════════════════════════════════════════════════════════════════════
# Token estimation
# ═══════════════════════════════════════════════════════════════════════

def estimate_translation_tokens(text: str) -> dict[str, int]:
    """Estimate token cost for translating a text block.
    
    Rough estimates (validated against typical DeepSeek usage):
    - System prompt: ~150 tokens
    - Input text: ~1 token per 4 chars (English)
    - Output: ~1.2x input (Chinese is denser)
    """
    chars = len(text)
    input_tokens = max(100, chars // 4) + 150  # text + system prompt
    output_tokens = max(50, input_tokens * 6 // 5)  # ~1.2x
    return {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}


# ═══════════════════════════════════════════════════════════════════════
# Main converter
# ═══════════════════════════════════════════════════════════════════════

def convert_vault_language(
    vault_dir: str | Path,
    target_lang: str = "zh",
    adapter: Any = None,
    mode: str = "generate",  # generate | glossary | none
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convert an Obsidian vault from English to target language.

    Args:
        vault_dir: Path to vault/ directory.
        target_lang: Language code (zh, ja, ko, fr, de, es...).
        adapter: LLMAdapter instance. If None, auto-detect or use glossary mode.
        mode:
            - "generate": LLM translates all files (requires adapter)
            - "glossary": LLM generates a term glossary only (~500 tokens)
            - "none": Skip translation, just copy files
        dry_run: Estimate tokens without making LLM calls.

    Returns:
        Dict with stats: files_translated, tokens_used, lang_dir.
    """
    vault_dir = Path(vault_dir)
    lang_dir = vault_dir / "lang" / target_lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "target_lang": target_lang,
        "mode": mode,
        "files_translated": 0,
        "tokens_used": 0,
        "lang_dir": str(lang_dir),
        "errors": [],
    }

    # ── Collect source files ──
    source_files: list[Path] = []
    for pattern in ["papers/*.md", "00-meta/*.md", "review.md"]:
        for f in vault_dir.glob(pattern) if vault_dir.exists() else []:
            if f.is_file():
                source_files.append(f)

    if not source_files:
        log.warning("No source files found in %s", vault_dir)
        return stats

    # ── LLM adapter check ──
    if mode == "generate" and (adapter is None or not adapter.is_available()):
        log.info("LLM not available — falling back to glossary mode")
        mode = "glossary"

    # ── Mode: generate (LLM translation) ──
    if mode == "generate" and adapter is not None:
        system_prompt = TRANSLATE_SYSTEM.get(
            target_lang,
            TRANSLATE_SYSTEM["zh"],
        )

        for src_file in source_files:
            rel_path = src_file.relative_to(vault_dir)
            dst_file = lang_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                text = src_file.read_text(encoding="utf-8")
                if len(text) < 50:
                    # Skip tiny files (just copy)
                    dst_file.write_text(text, encoding="utf-8")
                    continue

                if dry_run:
                    est = estimate_translation_tokens(text)
                    stats["tokens_used"] += est["total"]
                    stats["files_translated"] += 1
                    continue

                # Split long texts into chunks to avoid context overflow
                chunks = _split_text(text, max_chars=3000)
                translated_chunks = []

                for i, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        user_msg = f"Part {i+1}/{len(chunks)}. Translate this academic text:\n\n{chunk}"
                    else:
                        user_msg = f"Translate this academic text:\n\n{chunk}"

                    result = adapter.complete(system_prompt, user_msg)

                    if isinstance(result, dict):
                        if "_raw" in result:
                            translated_chunks.append(result["_raw"])
                        elif "error" in result:
                            log.warning("Translation failed for %s: %s", rel_path, result["error"])
                            stats["errors"].append(str(rel_path))
                            translated_chunks.append(chunk)  # fallback to original
                        elif "skipped" in result:
                            translated_chunks.append(chunk)
                    else:
                        translated_chunks.append(str(result))

                    est = estimate_translation_tokens(chunk)
                    stats["tokens_used"] += est["total"]

                dst_file.write_text("".join(translated_chunks), encoding="utf-8")
                stats["files_translated"] += 1

            except Exception as exc:
                log.warning("Translate %s failed: %s", rel_path, exc)
                stats["errors"].append(str(rel_path))
                # Copy original as fallback
                dst_file.write_text(text, encoding="utf-8")

    # ── Mode: glossary (LLM term glossary, cheap) ──
    elif mode == "glossary":
        # Collect all unique technical terms from papers
        terms = _collect_technical_terms(vault_dir)
        lang_names = {"zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
        lang_name = lang_names.get(target_lang, target_lang)

        if terms and adapter is not None and not dry_run:
            prompt = GLOSSARY_PROMPT.format(
                lang_name=lang_name,
                terms=", ".join(sorted(terms)[:50]),
            )
            result = adapter.complete("", prompt)
            glossary_text = str(result) if not isinstance(result, dict) else json.dumps(result, ensure_ascii=False)

            glossary_file = lang_dir / "00-meta" / "glossary.md"
            glossary_file.parent.mkdir(parents=True, exist_ok=True)
            glossary_file.write_text(
                f"# Technical Term Glossary: EN → {lang_name}\n\n"
                f"Auto-generated. Terms extracted from {len(terms)} unique paper keywords.\n\n"
                f"```json\n{glossary_text}\n```\n",
                encoding="utf-8",
            )
            stats["tokens_used"] = 500
            stats["glossary_terms"] = len(terms)

        # Copy source files untranslated
        for src_file in source_files:
            rel_path = src_file.relative_to(vault_dir)
            dst_file = lang_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(src_file.read_text(encoding="utf-8"))
        stats["files_translated"] = len(source_files)

    # ── Mode: none (just copy) ──
    else:
        for src_file in source_files:
            rel_path = src_file.relative_to(vault_dir)
            dst_file = lang_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(src_file.read_text(encoding="utf-8"))
        stats["files_translated"] = len(source_files)

    log.info(
        "Language conversion: %d files → %s (mode=%s, tokens=%d)",
        stats["files_translated"], lang_dir, mode, stats["tokens_used"],
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _split_text(text: str, max_chars: int = 3000) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current += ("\n\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


def _collect_technical_terms(vault_dir: Path) -> set[str]:
    """Extract unique technical terms from paper .md YAML frontmatter."""
    terms: set[str] = set()
    papers_dir = vault_dir / "papers"
    if not papers_dir.exists():
        return terms

    import re
    for md_file in papers_dir.glob("*.md"):
        if md_file.name == "index.md":
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
            # Extract keywords from YAML frontmatter
            m = re.search(r"^keywords:\s*\n((?:\s+-.+\n?)*)", text, re.MULTILINE)
            if m:
                for line in m.group(1).split("\n"):
                    kw = line.strip().lstrip("- ")
                    if kw and len(kw) > 2:
                        terms.add(kw)
            # Extract tags
            m = re.search(r"^tags:\s*\n((?:\s+-.+\n?)*)", text, re.MULTILINE)
            if m:
                for line in m.group(1).split("\n"):
                    tag = line.strip().lstrip("- ")
                    if tag and len(tag) > 2:
                        terms.add(tag)
        except Exception:
            pass

    return terms
