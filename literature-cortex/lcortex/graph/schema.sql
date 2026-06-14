-- Literature Cortex — Graph Storage Schema
-- SQLite single-file, zero-configuration knowledge graph backend

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,                -- paper_id or seed_id
    type TEXT NOT NULL DEFAULT 'paper', -- paper | seed | meta_control
    title TEXT,
    year INTEGER,
    knowledge_level TEXT DEFAULT '[]',  -- JSON array: ["algorithm","engineering"]
    structure_template TEXT DEFAULT '{}', -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    base_type TEXT NOT NULL DEFAULT 'correlation', -- correlation | causation
    base_score REAL DEFAULT 0.0,
    causal_promotion TEXT DEFAULT '{}', -- JSON: {is_causal, conditions_met, confidence, last_evaluated, promotion_path}
    mechanism_description TEXT,
    history TEXT DEFAULT '[]',          -- JSON array of state change records
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ontology_evolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trigger_paper TEXT,
    change_type TEXT NOT NULL,          -- reparent | merge | split | rename
    description TEXT,
    details TEXT DEFAULT '{}'           -- JSON: detailed change metadata
);

CREATE TABLE IF NOT EXISTS meta_policy_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_description TEXT,
    old_policy TEXT DEFAULT '{}',       -- JSON snapshot of previous policy
    new_policy TEXT DEFAULT '{}'        -- JSON snapshot of new policy
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_year ON nodes(year);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_base_type ON edges(base_type);
CREATE INDEX IF NOT EXISTS idx_ontology_evolution_timestamp ON ontology_evolution(timestamp);
CREATE INDEX IF NOT EXISTS idx_meta_policy_history_timestamp ON meta_policy_history(timestamp);
