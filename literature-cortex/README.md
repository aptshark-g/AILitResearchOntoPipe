<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/papers-arXiv-red" alt="arXiv">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Claude-purple" alt="LLM">
</p>

# 🧠 Literature Cortex

**AI-powered literature research pipeline with knowledge graph, structural analogy, and double-loop learning.**

Not just another paper search tool. Literature Cortex builds a *living ontology* from academic papers — classifying them by their underlying mathematical paradigms, anchoring them in physical constraints, and detecting when a new paper fundamentally challenges the framework.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture Overview](#architecture-overview)
  - [The 7-Phase Pipeline](#the-7-phase-pipeline)
  - [Ontology Seed Library (49 Nodes)](#ontology-seed-library-49-nodes)
  - [Double-Loop Learning](#double-loop-learning)
- [Quick Start](#quick-start)
- [Pipeline Modes](#pipeline-modes)
- [Core Concepts](#core-concepts)
  - [Dry Scoring (Zero LLM)](#dry-scoring-zero-llm)
  - [Phase F: Paper → Ontology Mapping](#phase-f-paper--ontology-mapping)
  - [Phase F-2: Conflict Detection](#phase-f-2-conflict-detection)
  - [Knowledge Graph & Vault Export](#knowledge-graph--vault-export)
- [Pluggable Scoring Framework](#pluggable-scoring-framework)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [License](#license)

---

## Why This Exists

Traditional literature review tools stop at search and summarization. That's insufficient when you're trying to answer deeper questions:

- **Does this paper use the same *abstract structure* as another paper in a completely different field?** (e.g., is active vibration control structurally identical to portfolio optimization?)
- **Does this paper challenge foundational assumptions, or merely tune parameters?** (Double-loop vs single-loop learning)
- **Where does this paper sit in the hierarchy from axioms → math → algorithms → physics → engineering?**
- **What are the *shared limitations* across papers, and which ones are genuinely unsolved?**

Literature Cortex addresses these by maintaining a structured ontology with 49 pre-built seed nodes (L0–L4) and a pipeline that maps every paper onto them.

---

## Architecture Overview

### The 7-Phase Pipeline

```
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────┐
│ Phase A │ → │ Phase B1 │ → │ Phase B2 │ → │ Phases C/D    │
│ Search  │   │ Dry Score│   │ LLM Score│   │ Limits/Extend │
└─────────┘   └──────────┘   └──────────┘   └───────────────┘
     │                                 │               │
     ▼                                 ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────────────────────────┐
│ Phase G │ ← │ Phase F2 │ ← │ Phase F                      │
│ Export  │   │ Conflict │   │ Structure Extraction         │
└─────────┘   └──────────┘   └──────────────────────────────┘
                     │
                     ▼
              ┌──────────┐
              │ Phase E  │
              │ Synthesis│
              └──────────┘
```

| Phase | What it does | LLM needed? |
|-------|-------------|:---:|
| **A** | Multi-level arXiv search (3 layers: API → keyword → LLM filter) | Optional |
| **B1** | BM25 + Intent-word dry scoring (5 dimensions) | No |
| **B2** | 4C+L LLM scoring (Contribution, Correctness, Clarity, Connectedness, Limitations) | Yes |
| **C/D** | Find limitation literature + extension/advance papers via arXiv | Optional |
| **E** | Synthesize a structured review with cross-paper comparison matrix | Yes |
| **F** | Extract structure template + knowledge level (LLM or keyword matcher) | Mode-dependent |
| **F-2** | 5-step divergence detection: Deconstruct → Retrieve → Reconstruct → Assess → Decide | Yes |
| **G** | Persist to SQLite graph store + export Obsidian vault with causal maps | No |

### Ontology Seed Library (49 Nodes)

The seed nodes form a **hierarchical, cross-domain knowledge backbone**. They are not isolated tags — they form directed dependency chains. See [`lcortex/seeds/ONTOLOGY_GUIDE.md`](lcortex/seeds/ONTOLOGY_GUIDE.md) for the full cross-layer causal map.

```
L0 ─ Meta (6)
    How the system thinks — double-loop learning, causation theory,
    epistemic boundaries, hierarchy & emergence.

    ↓ depends on

L1 ─ Axioms (12)
    Mathematical foundations — ZFC set theory, Peano induction,
    Gödel incompleteness, Noether's theorem, HoTT, category theory.

    ↓ generates

L2 ─ Mathematics (10)
    Mathematical toolkits — function approximation, ODE/PDE,
    optimization, probability, spectral analysis, graph theory,
    information theory, topology.

    ↓ supports

L3 ─ Methods (13)
    Algorithmic paradigms — search, dynamic programming, adaptive update,
    feedback, feedforward, model-based simulation, data-driven,
    spectral decomposition, dimensionality reduction.

    ↓ constrained by

L4 ─ Physics (8)
    Reality layer — mechanical oscillation, thermal transport,
    electromagnetic coupling, material response, noise floor,
    causal delay, phase transitions, structural stability.
```

**Key design principle**: Upper layers depend on lower ones, never the reverse. L1 doesn't know about L3's algorithms. L3's methods must trace back through L2 math to L1 axioms.

### Double-Loop Learning

Most systems do **single-loop learning** — tune parameters within a fixed framework. Literature Cortex detects when a paper requires **double-loop learning** — revising the framework itself.

```python
# Phase F-2: 5-step divergence detection
result = detect_conflict(
    paper=paper,
    structure=structure_template,
    graph_store=knowledge_graph,
    adapter=llm_adapter,
)
# → {recommended_action: "double_loop", unexplainability_score: 0.85, ...}
```

The pipeline evaluates 4 **Posner conditions** for conceptual change:
- **C1 (Dissatisfaction)**: Does the paper reveal a fundamental flaw in existing approaches?
- **C2 (Intelligibility)**: Is the new concept self-consistent and understandable?
- **C3 (Plausibility)**: Is it more reasonable than alternatives?
- **C4 (Fruitfulness)**: Would adopting it open significant new research directions?

**Meta-policy safeguards** prevent runaway ontology changes: consecutive trigger limits, impact ratio caps, and cold-start detection automatically downgrade or block unsafe double-loop proposals.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/literature-cortex.git
cd literature-cortex
pip install -e .

# ── Dry mode: no LLM required ──────────────────────────
lcortex run "active vibration control" --mode dry --max 8

# ── Lite mode: LLM for scoring + synthesis ─────────────
export DEEPSEEK_API_KEY="sk-..."
lcortex run "FxLMS vibration suppression" --mode lite --max 10

# ── Full mode: complete pipeline with divergence detection
lcortex run "Koopman operator control" --mode full --max 5

# ── Seed library ───────────────────────────────────────
lcortex seed list              # List all 49 seeds by level
lcortex seed show method-4     # Show seed details
lcortex seed add ./my_seeds/   # Add custom seeds

# ── Export ─────────────────────────────────────────────
lcortex export --format obsidian  # Obsidian vault
lcortex export --format json      # Graph JSON
```

---

## Pipeline Modes

| Mode | Search | B1 Dry | B2 LLM | C/D | E Synthesis | F Structure | F-2 Conflict | LLM Required |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dry` | ✅ | ✅ | ⏭ | ✅ | ⏭ | Lite matcher | ⏭ | **None** |
| `lite` | ✅ | ✅ | ✅ | ✅ | ✅ | Lite matcher | ⏭ | DeepSeek/OpenAI |
| `full` | ✅ | ✅ | ✅ | ✅ | ✅ | LLM extract | ✅ | DeepSeek/OpenAI |

**What's "Lite matcher"?** In dry/lite mode, Phase F uses a keyword-driven matcher (`lcortex/structure/lite_matcher.py`) that maps papers to ontology seeds using Jaccard overlap + phrase matching — no LLM needed, ~10ms per paper.

---

## Core Concepts

### Dry Scoring (Zero LLM)

Phase B1 evaluates papers without any AI call. It uses five dimensions:

```python
from lcortex.analysis import dry_score_batch

results = dry_score_batch(papers, "active vibration control")
# → [{paper_id: "2507.03854", dry_score: 0.337, passed: True, ...}, ...]
```

| Dimension | Weight | What it measures |
|-----------|:------:|------------------|
| **BM25** | 0.30 | Term-frequency relevance with saturation (replaces TF-IDF) |
| **Intent** | 0.40 | Method/acronym matching with title (3×) and abstract (1.5×) boost |
| **Background** | 0.10 | Domain keyword co-occurrence |
| **Recency** | 0.10 | Linear decay over 5 years |
| **Impact** | 0.10 | Citation count proxy (log-scaled) |

The IntentWordExtractor uses heuristic rules (ALL_CAPS detection, CamelCase, hyphenated compounds, numbers with units like `H2`/`H∞`) — no AI, no external deps.

### Phase F: Paper → Ontology Mapping

Every paper that passes the scoring gate gets mapped to ontology seed nodes.

**Lite mode** (no LLM):
```
2512.03990 "Vibration Suppression by Neural Control"
  → method-9 (Data-Driven Function Approximation)  0.455
  → math-7   (Graph Theory & Network Topology)      0.403
  → phys-1   (Mechanical Oscillation & Wave Dynamics) 0.350
```

**Full mode** (LLM-powered):
```json
{
  "knowledge_level": ["L3-Algorithm", "L5-Engineering"],
  "knowledge_level_confidence": 0.70,
  "structure_template": {
    "control_architecture": "feedforward_adaptive",
    "abstract_pattern": "encode → adapt → decode → actuate",
    "mathematical_core": "gradient_descent_on_learned_manifold",
    "domain_abstraction": "neural network learns inverse dynamics..."
  }
}
```

### Phase F-2: Conflict Detection

The deconstructor breaks a paper into indivisible atoms, matches each against the knowledge graph, attempts reconstruction using only existing concepts, and assesses whether a fundamental gap exists:

```python
from lcortex.structure.deconstructor import detect_conflict

result = detect_conflict(paper, structure, graph_store, adapter)
# Step 1: Deconstruction  → [assumptions, axioms, methodology atoms, ...]
# Step 2: Retrieval       → atom ↔ knowledge graph matches
# Step 3: Reconstruction  → 15% reconstructable, 85% unexplainable
# Step 4: Conflict        → Posner C1-C4 evaluation
# Step 5: Decision        → single_loop / double_loop / seed_anchored / degraded_by_meta
```

### Knowledge Graph & Vault Export

Phase G persists everything to a **SQLite graph store** and exports an **Obsidian vault**:

```
vault/
├── README.md
├── graph.json                          # Full serialized graph (65 nodes, 35 edges)
├── papers/
│   ├── index.md                        # Ranked table of all papers
│   ├── 2507.03854.md                   # Paper note with YAML frontmatter + wikilinks
│   └── ...
└── 00-meta/
    ├── knowledge-tree.md               # Papers + seeds organized by ontology level
    ├── causal-map.md                   # Mermaid flowchart of all edges
    └── dry-summary.md                  # B1 scoring summary
```

The knowledge-tree integrates papers and seeds into one view:

```markdown
## Ontology Levels

### L1 — axiom (12 nodes)
- 🌱 `axiom-10` Symmetry & Conservation (Noether's Theorem)
- 🌱 `axiom-11` Linearization & Local Approximation
...

### L3 — method (13 nodes)
- 🌱 `method-4` Adaptive Update & Online Learning
- 🌱 `method-7` Feedforward & Predictive Compensation
- 📄 [[../papers/2507.03854|Latent FxLMS...]] (2025) — B1:0.337
```

---

## Pluggable Scoring Framework

The scoring system is **fully composable**. You can add custom data sources (OpenAlex, Semantic Scholar, your own API) and custom scoring dimensions (venue prestige, author h-index, user alignment) without modifying the pipeline.

```python
from lcortex.analysis.dry_scorer import build_default_composite
from lcortex.analysis.builtin_scorers import VenueScorer, AuthorScorer
from lcortex.analysis.scoring import CompositeScorer, DataSource, Scorer, register_scorer

# ── Option 1: Extend defaults ──────────────────────────────
composite = build_default_composite(
    bm25, intent_words, background_words,
    extra_scorers=[VenueScorer(weight=0.10), AuthorScorer(weight=0.05)],
    extra_sources=[OpenAlexSource()],  # Fetches venue rank + author count via API
)
result = composite.score(paper, paper_index=0)

# ── Option 2: Full custom ──────────────────────────────────
@register_scorer("my_metric")
class MyScorer(Scorer):
    name = "my_metric"
    def score(self, paper, enriched_meta=None) -> ScoreResult:
        # Your scoring logic
        return ScoreResult(name=self.name, value=0.8, weight=self.weight,
                           weighted=0.8 * self.weight)

composite = CompositeScorer([MyScorer(weight=0.5), ...])
```

**Built-in scorers** (all registered for name-based discovery):

| Scorer | Weight | Description |
|--------|:------:|-------------|
| `bm25` | 0.30 | BM25 relevance with saturation |
| `intent` | 0.40 | Method/acronym matching |
| `background` | 0.10 | Domain keyword matching |
| `recency` | 0.10 | Publication year decay |
| `impact` | 0.10 | Citation count proxy |
| `venue` | 0.10 | Journal/conference tier (IEEE Trans → 0.6, Nature → 1.0) |
| `author` | 0.05 | Mean citations per author |

**Built-in data sources**:
| Source | Data | API |
|--------|------|-----|
| `arxiv_meta` | Citation count, versions, categories | Local |
| `openalex` | Venue name/type, cited_by_count, author_count, OA status | Network |

---

## Project Structure

```
literature-cortex/
├── lcortex/
│   ├── cli.py                       # Click CLI entry point (12 commands)
│   ├── analysis/
│   │   ├── scoring.py               # Scorer + DataSource interfaces + CompositeScorer
│   │   ├── builtin_scorers.py       # 7 registered scorers + 2 data sources
│   │   ├── dry_scorer.py            # BM25 scorer, IntentWordExtractor, batch scoring
│   │   └── scorer.py                # Phase B2 LLM scorer (4C+L)
│   ├── core/
│   │   ├── config.py                # Config from env vars + TOML
│   │   ├── engine_legacy.py         # Legacy standalone engine
│   │   ├── resources.py             # Adaptive resource profiling (CPU/mem-aware)
│   │   └── state.py                 # PipelineState + checkpoint management
│   ├── export/
│   │   └── obsidian.py              # Obsidian vault + graph.json + causal maps
│   ├── graph/
│   │   ├── store.py                 # SQLite graph store (WAL, thread-safe)
│   │   ├── edge.py                  # Edge factory (correlation/causation)
│   │   └── schema.sql              # DB schema
│   ├── intelligence/
│   │   ├── factory.py               # LLM auto-detection + fallback chain
│   │   └── adapters/                # DeepSeek, OpenAI, Ollama, Claude, NoOp
│   ├── monitor/
│   │   ├── run_controller.py        # 7-phase run orchestrator (96 KB)
│   │   └── monitor.py               # PipelineMonitor + report generation
│   ├── ontology/
│   │   ├── engine.py                # Double-loop change executor
│   │   ├── evolution.py             # insert/reparent/merge/split operations
│   │   └── distillation.py          # Ontology distillation from paper clusters
│   ├── prompts/                     # LLM prompt templates (Markdown)
│   ├── search/
│   │   ├── multi_level.py           # 3-layer search (API → keyword → LLM)
│   │   ├── arxiv.py                 # arXiv API client
│   │   └── dedup.py                 # DOI → arXiv → title deduplication
│   ├── seeds/
│   │   ├── seed_L0_meta.json        # 6 meta-policy nodes
│   │   ├── seed_L1_axioms.json      # 12 mathematical foundation nodes
│   │   ├── seed_L2_math.json        # 10 mathematical tool nodes
│   │   ├── seed_L3_methods.json     # 13 algorithm paradigm nodes
│   │   ├── seed_L4_physics.json     # 8 physical constraint nodes
│   │   ├── loader.py                # SeedLoader + auto_initialize
│   │   ├── ONTOLOGY_GUIDE.md        # Full cross-layer causal chain guide
│   │   └── meta/control_policy.json # Meta-policy thresholds
│   └── structure/
│       ├── extractor.py             # Phase F LLM structure extractor
│       ├── lite_matcher.py          # Phase F keyword matcher (no LLM)
│       └── deconstructor.py         # Phase F-2 divergence detection pipeline
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation & Requirements

```bash
pip install -e .
```

**Dependencies** (all pure Python, no compiled extensions):
- `click ≥ 8.1` — CLI framework
- `arxiv ≥ 2.1` — arXiv API client
- `requests ≥ 2.31` — HTTP client
- `pyyaml ≥ 6.0` — YAML config support

**Optional** (for Phase B2/E/F/F-2):
- DeepSeek API key (`DEEPSEEK_API_KEY` env var)
- Or: OpenAI API key, Ollama local server, Claude API key

**Python**: ≥ 3.10

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built with ❤️ for researchers who think in structures, not summaries.</sub>
</p>
