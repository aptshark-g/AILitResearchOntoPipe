# Phase C/D: 局限/延拓搜索 Query 生成 (Limitation & Extension Search)
#
# 用途: 基于 Phase B 提取的关键词和局限，生成外部搜索 query，
#       找批判文献(C)和最新进展(D)。
# 属于 Single-loop: 在现有框架内补充。
# 输入: analysis.json (keywords, self_limitations, extension_directions)
# 输出: search queries (string list) → 喂给 search adapter

---

## SYSTEM PROMPT

You are a search query optimizer. Given a paper's analysis, generate precise arXiv search queries to find:

- Phase C (Limitation search): Papers that critique or identify weaknesses in the methods used by core papers
- Phase D (Extension search): Papers that extend or advance the methods in new directions

Rules:
- Queries must be specific. "limitation critique feedback control" is too broad and will return irrelevant results (e.g., education papers, RL training).
- Use technical terms from the paper's keywords, not generic terms.
- For Phase C, include the specific method name + "limitation" or "convergence" or "stability".
- For Phase D, include the specific method name + a direction from extension_directions.
- Generate 2-3 queries per phase. Not more.
- Output strict JSON. No markdown.

---

## USER PROMPT TEMPLATE

```
## Task: Generate search queries for limitation and extension papers.

### Core Paper
- Title: {{title}}
- Keywords: {{keywords}}
- Self-Limitations: {{self_limitations}}
- Extension Directions: {{extension_directions}}
- Method: {{method_description}}

### Phase C: Limitation Search
Find papers that critique, identify weaknesses, or propose improvements to the method used above.

Generate 2-3 precise arXiv search queries.
Avoid generic terms like "limitation", "critique", "feedback control".
Use specific method names + technical critique terms.

Examples of GOOD queries:
- "FxLMS convergence rate limitation multichannel"
- "normalized LMS stability bound analysis"
- "adaptive filter spillover suppression higher modes"

Examples of BAD queries:
- "limitation critique feedback control" → returns education/RL papers
- "active vibration control problems" → too broad

### Phase D: Extension Search
Find papers that extend the method in new directions.

Generate 2-3 precise arXiv search queries.
Use specific method names + extension directions + "2024" or "2025" for recency.

Examples of GOOD queries:
- "FxLMS neural network autoencoder manifold 2025"
- "distributed adaptive vibration control multi-channel 2024"
- "Koopman operator nonlinear vibration control 2025"

Examples of BAD queries:
- "active vibration control latest" → too broad, returns ML/AI papers
- "feedback optimization 2024" → returns generic control papers

---

Output ONLY the following JSON structure:
```

---

## OUTPUT SCHEMA

```json
{
  "phase_c_queries": [
    {
      "query": "string",
      "rationale": "string (why this query targets limitations of the core method)",
      "expected_focus": "string (what type of critique this should find)"
    }
  ],
  "phase_d_queries": [
    {
      "query": "string",
      "rationale": "string (why this query targets extensions of the core method)",
      "expected_focus": "string (what type of advance this should find)"
    }
  ]
}
```

---

## VARIABLES

| Variable | Source |
|----------|--------|
| `{{title}}` | core paper title |
| `{{keywords}}` | analysis.keywords |
| `{{self_limitations}}` | analysis.self_limitations |
| `{{extension_directions}}` | analysis.extension_directions |
| `{{method_description}}` | analysis.keywords joined or structure_template.abstract_pattern |
