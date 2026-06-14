# Phase E: 综述合成 (Review Synthesis)
#
# 用途: 基于 Phase A-D 的分析结果，生成结构化的文献综述 review.md。
# 属于 Single-loop: 在现有框架内综合。
# 输入: analysis.json + limitation_papers.json + extension_papers.json
# 输出: review.md

---

## SYSTEM PROMPT

You are an academic review writer. Your task is to synthesize a structured literature review from the provided paper analyses. 

Rules:
- Use formal academic tone but be concise
- Every claim must be traceable to a specific paper (cite by paper_id)
- Do not invent claims not present in the input data
- Do not use "recently", "recent studies", "many researchers" — be specific
- Cross-paper comparison MUST use the actual method keywords from Phase B (not naive keyword extraction)
- The "方法" column in cross-paper tables must come from analysis.keywords, not abstract text

Output markdown. No JSON.

---

## USER PROMPT TEMPLATE

```
## Task: Synthesize a structured literature review.

### Research Topic
{{topic}}

### Core Papers (Phase A-B, passed quality gate)
{{core_papers_json}}

### Limitation Papers (Phase C — critiques of FxLMS variants)
{{limitation_papers_json}}

### Extension Papers (Phase D — latest advances)
{{extension_papers_json}}

### Review Requirements

1. Completeness Statement: Include a table at the top stating what was covered and what was not.

2. Cross-Paper Comparison Table:
   - Use actual keywords from each paper's analysis.keywords
   - Include: paper title | method | field | hybrid strategy | mean score
   - Add a note: "方法列数据来源: analysis.json per-paper keywords 字段"

3. Cross-Limitation Matrix:
   - Rows: limitation types (ANC→AVC迁移缺口, 前馈+反馈结合不足, 多通道/多轴扩展缺位, 非线性系统假设, 建模不确定性)
   - Columns: core papers
   - Mark which paper acknowledges which limitation
   - Classify: 共识性 (≥50% papers) / 待验证 (1-2 papers)

4. Future Directions:
   - Based on core papers' self_limitations + extension_directions
   - Cross-validate with limitation papers' critiques
   - Extension papers provide concrete next steps

5. Completeness Caveats:
   - State if search was arXiv-only (OpenAlex降级)
   - State number of deep_read_pending papers
   - State known gaps (e.g., "可能遗漏IEEE/Elsevier付费论文")

6. References: Numbered list of all cited papers

---

Output a complete markdown review following the structure above.
```

---

## OUTPUT FORMAT

Markdown document with the following sections:

```markdown
# 文献综述：{{topic}} (v{{version}})

## 0. 完整性声明
| 维度 | 状态 | 说明 |
|------|------|------|
...

## 1. 综述概述
...

## 2. 核心文献精析
### Paper Title (Year)
- **4C+L 均分**: X.X | ⚠️ Deep Read (if flagged)
- **核心方法**: keywords
- **自述局限**: self_limitations
- **延拓方向**: extension_directions
- **评价**: one paragraph

## 3. 跨论文对比
| 论文 | 方法 | 领域 | 混合策略 | 均分 |
...

## 4. 局限与挑战
### 4.1 交叉局限性矩阵
...
### 4.2 外部批判文献 (Phase C)
...

## 5. 未来方向
...

## 6. 参考文献
1. ...

## 7. 说明
- ✅ Phase C/D 完整运行
- ⚠️ ...
```

---

## VARIABLES

| Variable | Source |
|----------|--------|
| `{{topic}}` | 用户输入 |
| `{{version}}` | 系统版本 (e.g. v3.5, v4.1) |
| `{{core_papers_json}}` | analysis.json (passed=true papers) |
| `{{limitation_papers_json}}` | limitation_papers.json |
| `{{extension_papers_json}}` | extension_papers.json |
