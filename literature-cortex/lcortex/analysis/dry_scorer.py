"""
Phase B1: Auto-Weighted NLP Dry Scoring (BM25 + Intent Words + Zero LLM).

Replaces TF-IDF with BM25 (Robertson et al. 1994) to fix rare-word
over-weighting and common-domain-word under-weighting.

Key innovations:
- BM25: term-frequency saturation + document-length normalization
- IntentWordExtractor: heuristic rules detect methods, acronyms, model
  names from query (no AI, no external deps)
- Title boost: 3x for intent-word match in title, 1.5x for abstract
- Zero external dependencies: pure Python

Dimensions:
  1. BM25 relevance (0.30)
  2. Intent-word matching with title/abstract weighting (0.40)
  3. Background-word matching (0.10)
  4. Recency boost (0.10)
  5. Citation impact proxy (0.10)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger("lcortex.analysis.dry_scorer")

# ═══════════════════════════════════════════════════════════════════════
# Intent Word Extractor
# ═══════════════════════════════════════════════════════════════════════

class IntentWordExtractor:
    """Extracts intent words from user query via heuristic rules.

    Intent words = methods, models, acronyms, specific techniques.
    Background words = generic domain descriptors.

    Rules (no AI, no external dep):
    1. Acronyms (ALL_CAPS or CamelCase with >=2 uppercase letters) → intent
    2. Hyphenated or compound technical terms → intent
    3. Numbers with units (e.g. H2, Hinf) → intent
    4. Common English fluff → background
    """

    BACKGROUND_WORDS: Set[str] = {
        "a","an","and","are","as","at","be","by","for","from","in",
        "is","it","of","on","or","that","the","to","with",
        "using","based","via","over","under","between","among",
        "new","novel","recent","modern","advanced","improved",
        "approach","method","methodology","framework","scheme",
        "study","paper","article","work","research","analysis",
        "proposed","presented","investigated","demonstrated","discussed",
    }

    ACRONYM_RE = re.compile(r"\b[A-Z]{2,}[a-zA-Z0-9]*\b")
    MIXED_RE = re.compile(r"\b[A-Z][a-z]*[A-Z][a-zA-Z0-9]*\b")
    NUMBER_RE = re.compile(r"\b[A-Z]?\d+[a-zA-Z]*\b")

    def extract(self, query: str) -> Tuple[Set[str], Set[str]]:
        """Returns (intent_words, background_words), all lowercased."""
        tokens = self._tokenize(query)
        intent: Set[str] = set()
        background: Set[str] = set()

        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower in self.BACKGROUND_WORDS:
                background.add(tok_lower)
                continue

            if self.ACRONYM_RE.match(tok):
                intent.add(tok_lower)
                continue

            if self.MIXED_RE.match(tok):
                intent.add(tok_lower)
                continue

            if self.NUMBER_RE.match(tok):
                intent.add(tok_lower)
                continue

            if "-" in tok:
                parts = tok.lower().split("-")
                for p in parts:
                    if p not in self.BACKGROUND_WORDS and len(p) > 2:
                        intent.add(p)
                continue

            # Length heuristic: >6 chars = more specific
            if len(tok) >= 6 and tok_lower not in self.BACKGROUND_WORDS:
                intent.add(tok_lower)
                continue

            background.add(tok_lower)

        return intent, background

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text)


# ═══════════════════════════════════════════════════════════════════════
# BM25 Scorer
# ═══════════════════════════════════════════════════════════════════════

class BM25Scorer:
    """Lightweight BM25 implementation (Robertson et al. 1994).
    Tuned for short academic abstracts (150-300 words).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.5):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.documents: List[List[str]] = []
        self.idf_cache: Dict[str, float] = {}

    def fit(self, papers: List[dict]) -> None:
        """Index all papers for BM25 scoring (expects dict with title/abstract)."""
        self.N = len(papers)
        total_len = 0

        for p in papers:
            text = f'{p.get("title", "")} {p.get("abstract", "")}'
            tokens = self._tokenize(text)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avgdl = total_len / self.N if self.N > 0 else 1.0

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """BM25 score for a single document."""
        if doc_idx >= len(self.documents):
            return 0.0
        doc = self.documents[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        freq = Counter(doc)
        score = 0.0
        for q in query_tokens:
            if q not in freq:
                continue
            idf = self._idf(q)
            f = freq[q]
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (num / den)
        return score

    def _idf(self, term: str) -> float:
        if term in self.idf_cache:
            return self.idf_cache[term]
        df = self.df.get(term, 0)
        idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        self.idf_cache[term] = idf
        return idf

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ═══════════════════════════════════════════════════════════════════════
# Phase B1 Auto-Weighted Scorer
# ═══════════════════════════════════════════════════════════════════════

PASS_THRESHOLD: float = 0.25

# Weights
_INTENT_TITLE_BOOST = 3.0
_INTENT_ABSTRACT_BOOST = 1.5
_BACKGROUND_MATCH_WEIGHT = 0.2
_RECENCY_BOOST_MAX = 0.15
_IMPACT_MAX = 0.10
_BM25_NORM_DIVISOR = 10.0


def dry_score_paper(
    paper: dict[str, Any],
    query: str,
    scorer: Any = None,
    bm25: BM25Scorer | None = None,
    intent_words: Set[str] | None = None,
    background_words: Set[str] | None = None,
    domain_exclude_words: Set[str] | None = None,
    domain_require_words: Set[str] | None = None,
    doc_idx: int = 0,
    current_year: int | None = None,
) -> dict[str, Any]:
    """Score a single paper with BM25 + intent matching + domain detection.
    
    Args:
        domain_exclude_words: 论文中出现这些词 → 跨领域，大幅降权
        domain_require_words: 论文必须至少包含1个 → 否则跨领域
    """
    import datetime

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    title_tokens = BM25Scorer._tokenize(title)
    abstract_tokens = BM25Scorer._tokenize(abstract)
    all_tokens = title_tokens + abstract_tokens
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    cy = current_year or datetime.date.today().year

    # Extract intent/background if not provided
    if intent_words is None or background_words is None:
        extractor = IntentWordExtractor()
        intent_words, background_words = extractor.extract(query)

    # ═══════════════════════════════════════════════════════════
    # Domain Detection (NEW — 关联性判断 vs 存在性判断)
    # ═══════════════════════════════════════════════════════════
    domain_penalty = 1.0  # multiplier, 1.0 = no penalty
    domain_reason = ""

    # Layer 1: Domain require — 必须至少匹配1个领域基础词，或至少是近邻领域
    if domain_require_words:
        require_match = sum(1 for w in domain_require_words if w in abstract_lower or w in title_lower)
        if require_match == 0:
            # Check if it's a neighbor domain (shares control/adaptive/filter words)
            neighbor_words = {"control", "adaptive", "filter", "noise", "anc", "avc",
                            "suppression", "damping", "isolat", "actuator", "sensor",
                            "feedback", "feedforward", "fx-lms", "fxlms", "lms", "rls"}
            neighbor_match = sum(1 for w in neighbor_words if w in abstract_lower or w in title_lower)
            if neighbor_match >= 2:
                domain_penalty *= 0.7  # 近邻领域: 轻微降权 (如 ANC → AVC)
                domain_reason = f"neighbor_domain(missing_require, has_control_terms:{neighbor_match})"
            else:
                domain_penalty *= 0.3  # 完全无关领域: 大幅降权
                domain_reason = f"no_domain_word(missing:{len(domain_require_words)})"

    # Layer 2: Domain exclude — 出现其他领域词 → 降权
    if domain_exclude_words:
        exclude_hits = [w for w in domain_exclude_words if w in abstract_lower or w in title_lower]
        if exclude_hits:
            # 每命中1个排除词，额外降低15%
            domain_penalty *= max(0.1, 1.0 - len(exclude_hits) * 0.15)
            if not domain_reason:
                domain_reason = f"exclude_words:{','.join(exclude_hits[:3])}"

    # 1. BM25
    bm25_raw = 0.0
    if bm25 is not None:
        query_tokens = BM25Scorer._tokenize(query)
        bm25_raw = bm25.score(query_tokens, doc_idx)
    bm25_norm = min(bm25_raw / _BM25_NORM_DIVISOR, 1.0)

    # 2. Intent-word matching
    intent_title = sum(1 for w in intent_words if w in title_tokens)
    intent_abstract = sum(1 for w in intent_words if w in abstract_tokens)
    intent_total = len(intent_words)
    intent_score = 0.0
    if intent_total > 0:
        intent_score = (
            intent_title * _INTENT_TITLE_BOOST + intent_abstract * _INTENT_ABSTRACT_BOOST
        ) / (intent_total * _INTENT_TITLE_BOOST)
        intent_score = min(intent_score, 1.0)

    # 3. Background matching
    bg_matches = sum(1 for w in background_words if w in all_tokens)
    bg_score = min(bg_matches / max(len(background_words), 1), 1.0) * _BACKGROUND_MATCH_WEIGHT

    # 4. Recency
    year = paper.get("year", cy)
    age = cy - year if year else 0
    if age <= 0:
        recency = _RECENCY_BOOST_MAX
    elif age == 1:
        recency = _RECENCY_BOOST_MAX * 0.7
    elif age == 2:
        recency = _RECENCY_BOOST_MAX * 0.4
    else:
        recency = 0.0

    # 5. Citation impact
    citations = paper.get("citations", 0) or 0
    if citations >= 100:
        impact = _IMPACT_MAX
    elif citations >= 50:
        impact = _IMPACT_MAX * 0.7
    elif citations >= 10:
        impact = _IMPACT_MAX * 0.4
    elif citations > 0:
        impact = _IMPACT_MAX * 0.2
    else:
        impact = 0.0

    # Final weighted combination (with domain penalty)
    final = (
        bm25_norm * 0.30 +
        intent_score * 0.40 +
        bg_score * 0.10 +
        recency * 0.10 +
        impact * 0.10
    ) * domain_penalty

    passed = final >= PASS_THRESHOLD

    reason_parts = []
    if intent_title > 0:
        reason_parts.append(f"intent_in_title:{intent_title}")
    if intent_abstract > 0:
        reason_parts.append(f"intent_in_abstract:{intent_abstract}")
    if not passed:
        reason_parts.append(f"below_threshold({final:.3f}<{PASS_THRESHOLD})")
    if domain_reason:
        reason_parts.append(domain_reason)

    return {
        "paper_id": paper.get("paper_id", paper.get("id", "")),
        "title": title,
        "dry_score": round(final, 4),
        "dry_detail": {
            "bm25": round(bm25_norm, 4),
            "intent": round(intent_score, 4),
            "bg_match": round(bg_score, 4),
            "recency": round(recency, 4),
            "impact": round(impact, 4),
        },
        "passed": passed,
        "reason": "; ".join(reason_parts) if reason_parts else "no_match",
    }


def dry_score_batch(
    papers: list[dict[str, Any]],
    query: str,
    output_path: str | Path | None = None,
    current_year: int | None = None,
    domain_require: Set[str] | None = None,
    domain_exclude: Set[str] | None = None,
) -> list[dict[str, Any]]:
    """Score a batch with BM25 + intent matching + domain detection.
    
    Args:
        domain_require: 论文必须包含 ≥1 个词，否则跨领域降权
        domain_exclude: 论文中包含这些词 → 跨领域降权
    """
    import datetime
    cy = current_year or datetime.date.today().year

    # Extract intent/background
    extractor = IntentWordExtractor()
    intent_words, background_words = extractor.extract(query)

    # Build BM25 index
    bm25 = BM25Scorer()
    bm25.fit(papers)

    # Debug output
    print()
    print(f"🔍 Intent words: {sorted(intent_words)}")
    if domain_require:
        print(f"📍 Domain require: {sorted(domain_require)}")
    if domain_exclude:
        print(f"🚫 Domain exclude: {sorted(domain_exclude)}")
    print()

    results: list[dict[str, Any]] = []
    fh = None
    try:
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fh = open(str(out), "w", encoding="utf-8")
    except OSError as exc:
        log.warning("Cannot open dry_scorer output %s: %s", output_path, exc)

    try:
        for idx, paper in enumerate(papers):
            result = dry_score_paper(
                paper, query,
                bm25=bm25, intent_words=intent_words,
                background_words=background_words,
                domain_require_words=domain_require,
                domain_exclude_words=domain_exclude,
                doc_idx=idx, current_year=cy,
            )
            result["year"] = paper.get("year", 0)
            result["citations"] = paper.get("citations", 0)
            result["venue"] = paper.get("venue", "")
            result["source"] = paper.get("source", "arxiv")
            results.append(result)
            if fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()

    results.sort(key=lambda r: r.get("dry_score", 0.0), reverse=True)
    passed = sum(1 for r in results if r.get("passed", False))
    top3_avg = round(sum(r["dry_score"] for r in results[:3]) / 3, 3) if len(results) >= 3 else 0
    log.info(
        "Dry scorer (BM25+Intent): %d papers → %d passed (≥%.2f) | top3_avg=%.3f",
        len(results), passed, PASS_THRESHOLD, top3_avg,
    )
    return results


def load_dry_scored(input_path: str | Path, min_score: float = 0.0) -> list[dict[str, Any]]:
    """Load dry-scored results from JSONL file."""
    input_path = Path(input_path)
    results: list[dict[str, Any]] = []
    if not input_path.exists():
        return results
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("dry_score", 0.0) >= min_score:
                    results.append(record)
            except json.JSONDecodeError:
                continue
    results.sort(key=lambda r: r.get("dry_score", 0.0), reverse=True)
    return results


# Backward compat
class AutoWeightedScorer:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Backward-compat bridge to pluggable scoring framework
# ═══════════════════════════════════════════════════════════════════════════


def build_default_composite(
    bm25: "BM25Scorer",
    intent_words: set,
    background_words: set,
    current_year: int | None = None,
    scorer_overrides: dict | None = None,
    extra_scorers: list | None = None,
    extra_sources: list | None = None,
):
    """Build a CompositeScorer with the default 5-dim profile + optional extras.

    Returns a ``CompositeScorer`` ready for ``.score(paper)``.

    Example adding a VenueScorer::

        composite = build_default_composite(
            bm25, iw, bw,
            extra_scorers=[VenueScorer(weight=0.05)],
            extra_sources=[OpenAlexSource()],
        )
    """
    from lcortex.analysis.scoring import CompositeScorer
    from lcortex.analysis.builtin_scorers import (
        BM25Scorer as PBM25Scorer,
        IntentScorer,
        BackgroundScorer,
        RecencyScorer,
        ImpactScorer,
    )
    import datetime

    cy = current_year or datetime.date.today().year
    over = scorer_overrides or {}

    scorers = [
        PBM25Scorer(bm25=bm25, intent_words=intent_words, weight=over.get("bm25", 0.30)),
        IntentScorer(intent_words=intent_words, background_words=background_words, weight=over.get("intent", 0.40)),
        BackgroundScorer(background_words=background_words, weight=over.get("background", 0.10)),
        RecencyScorer(current_year=cy, weight=over.get("recency", 0.10)),
        ImpactScorer(weight=over.get("impact", 0.10)),
    ]
    if extra_scorers:
        scorers.extend(extra_scorers)

    composite = CompositeScorer(scorers)
    if extra_sources:
        for src in extra_sources:
            composite.add_source(src)
    return composite
