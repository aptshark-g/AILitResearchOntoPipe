---
name: literature-cortex
description: |
  End-to-end AI-powered literature research pipeline with ontological reasoning,
  knowledge graph, and double-loop learning. 7-phase pipeline: multi-level
  arXiv search → dry NLP scoring → LLM 4C+L scoring → limitation/extension
  search → structured synthesis → structure template extraction → divergence
  detection with Posner conditions. 49 pre-built ontology seed nodes (L0-L4).
  Dual-linkage export with near-transfer and far-transfer cross-domain analogies.
  Pluggable scoring framework with custom data sources and scorers.
  Triggered when the user wants to run a comprehensive literature review with
  ontological classification, needs structural analogy between papers, or
  wants to detect when research fundamentally challenges existing frameworks.
version: 0.1.0
---

# Literature Cortex — AI-Powered Literature Research Pipeline

## Overview

Literature Cortex transforms academic paper search into a structured knowledge
graph with ontological reasoning. It maps every paper onto 49 pre-built
ontology seed nodes (L0 meta-policy → L4 physics), detects cross-domain
structural analogies, and identifies when a paper challenges foundational
assumptions (double-loop learning).

**7-phase pipeline:**

```
A (Search) → B1 (Dry Score) → B2 (LLM 4C+L) → C/D (Limits/Extensions)
    → E (Synthesis) → F (Structure) → F2 (Conflict) → G (Export)
```

## Quick Start

```bash
# Capture dependencies
pip install -e .

# Dry mode — no LLM, keyword-driven paper→seed matching
lcortex run "active vibration control" --mode dry --max 8

# Lite mode — LLM for scoring + synthesis, keyword matching for structure
export DEEPSEEK_API_KEY="***"
lcortex run "FxLMS vibration suppression" --mode lite --max 10

# Full mode — complete pipeline with LLM structure extraction + divergence detection
lcortex run "Koopman operator control" --mode full --max 5

# List ontology seeds
lcortex seed list

# Export to Obsidian vault + graph.json
lcortex export --format obsidian
```

## Pipeline Modes

| Mode | Search | Dry Score | LLM Score | C/D | Synthesis | Structure | Conflict | LLM? |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dry` | ✅ | ✅ | ⏭ | ✅ | ⏭ | Lite matcher | ⏭ | **No** |
| `lite` | ✅ | ✅ | ✅ | ✅ | ✅ | Lite matcher | ⏭ | Yes |
| `full` | ✅ | ✅ | ✅ | ✅ | ✅ | LLM extract | ✅ | Yes |

## Core Modules

Install location (Python package): `pip install -e .` from the skill directory.

```bash
lcortex run "<query>" --mode dry|lite|full --max 5-20
lcortex seed list            # List 49 ontology seeds
lcortex seed show <id>       # Show seed details
lcortex export --format obsidian  # Obsidian vault output
lcortex status               # Show workspace state
```

## Output Structure

After a run, the workspace contains:

```
vault/
├── graph.json                 # Full serialized knowledge graph
├── papers/                    # Per-paper Obsidian notes
│   ├── index.md               # Ranked paper table
│   └── {paper_id}.md          # YAML frontmatter + wikilinks
└── 00-meta/
    ├── knowledge-tree.md      # Papers + seeds by ontology level
    ├── near-transfer.md       # Shared paradigm clusters
    ├── far-transfer.md        # Cross-domain structural analogies
    └── causal-map.md          # Mermaid flowchart of all edges
```

## Configuration

No config file needed. Set environment variables for LLM:

```bash
export DEEPSEEK_API_KEY="***"        # DeepSeek (default)
export OPENAI_API_KEY="***"          # OpenAI fallback
export LCORTEX_LLM_PROVIDER=ollama   # Use Ollama local server
```

## Constraints

- **LLM required for**: Phase B2 (4C+L scoring), Phase E (synthesis),
  Phase F full-mode (structure template extraction), Phase F-2 (conflict detection)
- **Dry mode** works fully offline — BM25 + IntentWordExtractor + keyword matching
- **arXiv API** may rate-limit; the search module handles automatic retry with exponential backoff
- **Workspace**: defaults to `~/lcortex-vault`, override with `-w <path>`
