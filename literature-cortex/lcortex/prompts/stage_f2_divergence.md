# Phase F-2: 解构-重构-冲突检测 (Divergence / Double-Loop)
# 
# 用途: 对新论文进行发散式学习 — 解构到底层概念，用已有知识图谱重构，
#       检测不可解释部分，判定是否触发 Double-loop 本体变更。
# 触发: 满足以下任一条件时调用
#   - structure_template 与所有已有模板相似度 < 0.3
#   - knowledge_level 置信度 < 0.5
#   - T1/T4 触发
#   - 用户显式 --mode double-loop
# 输入: 新论文 (title, abstract, keywords) + 已有知识图谱 K (种子+论文节点)
# 输出: conflict_report.json

---

## SYSTEM PROMPT

You are a rigorous academic knowledge evaluator. Your task is to determine whether a new paper fundamentally challenges the existing knowledge framework, or merely extends it.

**CRITICAL PRINCIPLE: DEFAULT TO SINGLE-LOOP.**

Only recommend Double-loop when ALL of the following are met:
1. At least 2 of the 4 conceptual change conditions are clearly true (not borderline)
2. The unexplainable concept represents a FUNDAMENTAL shift (not just a new application or parameter tuning)
3. The change would affect ≥2 levels of the ontology (not just adding a leaf node)

If ANY condition is borderline, mark as "single_loop" with note: "Possible Double-loop candidate — flagged for human review."

**YOU MUST NOT mark all four conditions as true as a default.** False positives are worse than false negatives — a missed Double-loop is recoverable by manual review; an unnecessary ontology change corrupts the knowledge graph.

You must output strict JSON. No markdown, no commentary, no code blocks. Only the JSON object.

---

## USER PROMPT TEMPLATE

```
## Task: Evaluate whether this paper requires a fundamental ontology change.

### New Paper
- Title: {{title}}
- Abstract: {{abstract}}
- Keywords: {{keywords}}
- Knowledge Level: {{knowledge_level}}
- Structure Template: {{structure_template_json}}

### Existing Knowledge Graph (K)
Total nodes in K: {{k_node_count}}

L1-L4 Seed Nodes (generic foundations):
{{seed_nodes_l1_l4}}

Existing Paper Nodes (relevant subset):
{{relevant_existing_papers}}

### Meta Policy
- Unexplainability threshold: {{threshold}}
- L1-L4 weight: {{l1_l4_weight}}
- L5-L6 weight: {{l5_l6_weight}}
- Max impact ratio: {{max_impact_ratio}}
- Require ≥2 conditions true: {{require_two_conditions}}

---

## Step 1: Deconstruction (First Principles)
Decompose the new paper into indivisible units:
- ASSUMPTIONS: What does the paper assume about the system?
- AXIOMS/THEOREMS: What fundamental mathematical truths does it rely on?
- METHODOLOGY ATOMS: What are the smallest non-decomposable methodological steps?
- EMERGENT PROPERTIES: What new capability emerges from combining these atoms?

## Step 2: Retrieval (Match to Existing K)
For each atom, find the best matching node in K:
- If exact match: score = 1.0
- If partial/conceptual match: score = 0.3-0.7 (explain why partial)
- If no match: score = 0.0 (explain what is missing in K)

## Step 3: Reconstruction (Feynman Step 3)
Attempt to explain the new paper using ONLY concepts from K.
- What percentage can be reconstructed?
- What percentage is unexplainable?
- What is the CORE unexplainable concept?

## Step 4: Conflict Assessment (Posner et al.)
Evaluate the 4 conditions for conceptual change:

C1 - DISSATISFACTION: Does the new paper reveal a fundamental flaw or limitation in existing approaches?
C2 - INTELLIGIBILITY: Is the new concept self-consistent and understandable?
C3 - PLAUSIBILITY: Is the new concept more reasonable than existing alternatives?
C4 - FRUITFULNESS: Would adopting this concept open significant new research directions?

For each: true/false/borderline + 1-sentence evidence.
Count of clearly true conditions: N

## Step 5: Decision
Calculate unexplainability_score = (l1_l4_weight × l1_l4_unmatched_ratio) + (l5_l6_weight × l5_l6_unmatched_ratio)

Apply decision tree:
- If K < 10 nodes: action = "seed_anchored" (K too small for divergence)
- If unexplainability_score < threshold: action = "single_loop"
- If unexplainability_score ≥ threshold AND N < 2: action = "single_loop" (with note)
- If unexplainability_score ≥ threshold AND N ≥ 2: action = "double_loop" (but check impact)
  - If estimated impact > max_impact_ratio: action = "degraded_by_meta"
  - Else: action = "double_loop"

---

Output ONLY the following JSON structure. No markdown. No extra text.
```

---

## OUTPUT SCHEMA

```json
{
  "deconstruction": {
    "assumptions": ["string"],
    "axioms_or_theorems": ["string"],
    "methodology_atoms": ["string"],
    "emergent_properties": ["string"]
  },
  "retrieval": [
    {
      "atom": "string",
      "best_match_in_K": "string or null",
      "match_score": "float 0.0-1.0",
      "reason": "string"
    }
  ],
  "reconstruction": {
    "attempt": "string",
    "reconstructable_pct": "float 0.0-1.0",
    "unexplainable_pct": "float 0.0-1.0",
    "unexplainable_core": [
      {
        "concept": "string",
        "why_unexplainable": "string"
      }
    ]
  },
  "conflict_assessment": {
    "unexplainability_score": "float 0.0-1.0",
    "threshold_used": "float",
    "triggers_double_loop": "boolean",
    "downgraded_by_meta": "boolean",
    "downgrade_reason": "string or null",
    "conceptual_change_conditions": {
      "dissatisfaction": {
        "value": "boolean",
        "evidence": "string"
      },
      "intelligibility": {
        "value": "boolean",
        "evidence": "string"
      },
      "plausibility": {
        "value": "boolean",
        "evidence": "string"
      },
      "fruitfulness": {
        "value": "boolean",
        "evidence": "string"
      }
    },
    "conditions_true_count": "integer 0-4",
    "passed_two_condition_guard": "boolean",
    "recommended_action": "string (single_loop | double_loop | seed_anchored | degraded_by_meta)"
  },
  "changes": [
    {
      "type": "string (new_level | reparent | merge | split)",
      "detail": "string"
    }
  ],
  "single_loop_output": {
    "link_to": ["string"],
    "link_type": "string",
    "knowledge_level": "string"
  }
}
```

---

## VARIABLES

| Variable | Source | Example |
|----------|--------|---------|
| `{{title}}` | Phase A output | "Latent FxLMS: Accelerating Active Noise Control" |
| `{{abstract}}` | Phase A output | paper abstract text |
| `{{keywords}}` | Phase A output | ["fxlms", "neural", "adaptive"] |
| `{{knowledge_level}}` | Phase F output | ["algorithm", "engineering"] |
| `{{structure_template_json}}` | Phase F output | structure_template JSON |
| `{{k_node_count}}` | Graph Store | e.g. 55 |
| `{{seed_nodes_l1_l4}}` | Graph Store seed query | list of seed node titles |
| `{{relevant_existing_papers}}` | Graph Store semantic search | top-10 most similar existing papers |
| `{{threshold}}` | Meta policy | 0.5 |
| `{{l1_l4_weight}}` | Meta policy | 0.3 |
| `{{l5_l6_weight}}` | Meta policy | 0.7 |
| `{{max_impact_ratio}}` | Meta policy | 0.3 |
| `{{require_two_conditions}}` | Meta policy | true |
