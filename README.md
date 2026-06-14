Literature Cortex
AI-powered literature research pipeline with ontological reasoning. Not just search — builds a living knowledge graph that maps papers to their mathematical paradigms, anchors them in physical constraints, and detects when research fundamentally challenges the framework.

Key capabilities:

🔍 3-layer arXiv search (API → keyword → LLM filter)

📊 5-dimension dry scoring (BM25 + Intent + Background + Recency + Impact) — zero LLM

🧠 LLM-powered 4C+L paper scoring + structure extraction + cross-paper synthesis

🌱 49 pre-built ontology seed nodes (L0 Meta → L1 Axioms → L2 Math → L3 Methods → L4 Physics)

🔄 Double-loop learning: 5-step divergence detection (Deconstruct → Retrieve → Reconstruct → Assess → Decide)

🗺️ Dual-linkage export: near-transfer (shared paradigms) + far-transfer (cross-domain structural analogies)

📦 Obsidian vault export with causal maps, wikilinks, and graph.json

🔌 Pluggable scoring framework: custom data sources + scorers via @register_scorer

Pipeline modes: Dry (no LLM) / Lite (LLM scoring) / Full (LLM + divergence detection)

pip install -e . → lcortex run "active vibration control" --mode dry --max 8

Pure Python. No compiled extensions. 

基于人工智能的文献研究流程，具备本体推理功能。不仅能进行搜索，还能构建一个动态知识图谱，将论文映射到其数学范式中，依据物理约束条件进行锚定，并在研究从根本上挑战现有框架时进行检测。
关键能力：
三层 Arxiv 搜索（通过 API→关键词→大语言模型筛选）
五维干性评分（BM25 + 意图 + 背景 + 推荐 + 影响）→ 零大语言模型
基于大语言模型的 4C+L 论文评分、结构提取及跨论文合成
49 个预构建的本体种子节点（L0 元数据→L1 公理→L2 数学→L3 方法→L4 物理）
双循环学习：5 步差异检测（解构→检索→重建→评估→决策）
双重链接导出：近迁移（共享范式）、远迁移（跨领域结构类比）
Obsidian 库导出，包含因果图、维基链接和 graph.json
可插拔评分框架，通过 @register_scorer
流程模式：干性（无大语言模型）/ 轻度（少量大语言模型评分）/ 完整（大语言模型 + 差异检测）
pip install -e . ；lortex run "主动振动控制" --mode dry --max 8
纯 Python 实现，无编译扩展。
