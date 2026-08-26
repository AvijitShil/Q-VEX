-- Q-VEX SQLite Schema
-- Uses FTS5 for BM25 text search and foreign-key cascades for graph integrity.

-- Enable WAL mode and foreign keys (must be run via PRAGMA, not in schema file)
-- PRAGMA journal_mode=WAL;
-- PRAGMA foreign_keys=ON;

-- Core Nodes
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    vector_idx INTEGER UNIQUE,
    metadata JSON,
    is_deleted BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edges (relationships between nodes)
CREATE TABLE IF NOT EXISTS edges (
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,
    edge_type TEXT DEFAULT 'related',
    confidence FLOAT DEFAULT 1.0,
    PRIMARY KEY (source, target, edge_type),
    FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
);

-- Indexes for fast graph traversal
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);

-- FTS5 Virtual Table for BM25 Search (Zero-RAM impact)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(
    text,
    content='nodes',
    content_rowid='id'
);

-- Triggers to auto-update FTS index on node changes
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO fts_nodes(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO fts_nodes(fts_nodes, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO fts_nodes(fts_nodes, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO fts_nodes(rowid, text) VALUES (new.id, new.text);
END;
