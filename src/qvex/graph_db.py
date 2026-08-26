"""SQLite-backed graph database with FTS5 full-text search and recursive CTE traversal."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

from qvex.models import EdgeData, NodeData

logger = logging.getLogger("qvex.graph_db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class GraphDB:
    """Thin wrapper around SQLite providing graph storage, FTS5 BM25 search,
    and recursive k-hop traversal.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.  Use ``":memory:"`` for an
        ephemeral in-memory database (useful for tests).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode and foreign keys
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()
        logger.info("GraphDB initialized at %s", self._db_path)

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Read and execute the SQL schema file."""
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        text: str,
        vector_idx: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a node and return its ID.

        The FTS5 index is updated automatically via the ``nodes_ai`` trigger.
        """
        meta_json = json.dumps(metadata) if metadata else None
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO nodes (text, vector_idx, metadata) VALUES (?, ?, ?)",
                (text, vector_idx, meta_json),
            )
            self._conn.commit()
            node_id = cursor.lastrowid
        logger.debug("Added node %d (vector_idx=%s)", node_id, vector_idx)
        return node_id  # type: ignore[return-value]

    def get_node(self, node_id: int) -> NodeData | None:
        """Fetch a single node by ID, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT id, text, vector_idx, metadata, is_deleted, created_at "
            "FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_nodes_by_ids(self, ids: Sequence[int]) -> list[NodeData]:
        """Batch-fetch nodes by a collection of IDs."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, text, vector_idx, metadata, is_deleted, created_at "
            f"FROM nodes WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def update_node(
        self,
        node_id: int,
        text: str | None = None,
        vector_idx: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a node's text, vector_idx, and/or metadata.  Returns True if updated."""
        parts: list[str] = []
        params: list[Any] = []
        if text is not None:
            parts.append("text = ?")
            params.append(text)
        if vector_idx is not None:
            parts.append("vector_idx = ?")
            params.append(vector_idx)
        if metadata is not None:
            parts.append("metadata = ?")
            params.append(json.dumps(metadata))
        if not parts:
            return False

        params.append(node_id)
        sql = f"UPDATE nodes SET {', '.join(parts)} WHERE id = ?"
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.debug("Updated node %d", node_id)
        return updated

    def delete_node(self, node_id: int) -> bool:
        """Hard-delete a node.  Cascade deletes edges; triggers clean FTS5."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM nodes WHERE id = ?", (node_id,)
            )
            self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted node %d", node_id)
        return deleted

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: str = "related",
        confidence: float = 1.0,
    ) -> None:
        """Insert an edge (or ignore if it already exists)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (source, target, edge_type, confidence) "
                "VALUES (?, ?, ?, ?)",
                (source, target, edge_type, confidence),
            )
            self._conn.commit()
        logger.debug("Added edge %d -> %d (%s)", source, target, edge_type)

    def get_all_nodes(self) -> list[NodeData]:
        """Return every active (non-deleted) node in the graph."""
        rows = self._conn.execute(
            "SELECT id, text, vector_idx, metadata, is_deleted, created_at "
            "FROM nodes WHERE is_deleted = 0"
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_all_edges(self) -> list[EdgeData]:
        """Return every edge in the graph."""
        rows = self._conn.execute(
            "SELECT source, target, edge_type, confidence FROM edges"
        ).fetchall()
        return [
            EdgeData(
                source=r["source"],
                target=r["target"],
                edge_type=r["edge_type"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    def get_edges(self, node_id: int) -> list[EdgeData]:
        """Return all edges originating from *node_id*."""
        rows = self._conn.execute(
            "SELECT source, target, edge_type, confidence "
            "FROM edges WHERE source = ?",
            (node_id,),
        ).fetchall()
        return [
            EdgeData(
                source=r["source"],
                target=r["target"],
                edge_type=r["edge_type"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # BM25 full-text search via FTS5
    # ------------------------------------------------------------------

    def bm25_search(self, query: str, limit: int = 50) -> list[int]:
        """Return node IDs ranked by BM25 relevance to *query*.

        Uses the SQLite FTS5 ``MATCH`` operator and ``bm25()`` ranking
        function.  Returns at most *limit* results.
        """
        import re
        # Tokenise and clean input to prevent FTS5 syntax errors and increase recall
        tokens = re.findall(r'\b\w+\b', query)
        if not tokens:
            return []

        # Suffix wildcard '*' for prefix/partial match, joined by OR to maximize recall.
        # Vector reranking down the pipeline will compute precise semantic similarities.
        fts_query = " OR ".join(f"{token}*" for token in tokens)

        try:
            rows = self._conn.execute(
                "SELECT rowid FROM fts_nodes WHERE fts_nodes MATCH ? "
                "ORDER BY bm25(fts_nodes) LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(
                "FTS5 MATCH with query '%s' failed: %s. Retrying with literal query.",
                fts_query,
                e,
            )
            # Fallback to double-quoted search, escaping internal quotes
            escaped_query = f'"{query.replace(chr(34), chr(34)+chr(34))}"'
            try:
                rows = self._conn.execute(
                    "SELECT rowid FROM fts_nodes WHERE fts_nodes MATCH ? "
                    "ORDER BY bm25(fts_nodes) LIMIT ?",
                    (escaped_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

        ids = [r["rowid"] for r in rows]
        logger.debug(
            "BM25 search '%s' (FTS query: '%s') returned %d hits",
            query,
            fts_query,
            len(ids),
        )
        return ids

    # ------------------------------------------------------------------
    # Recursive k-hop graph expansion
    # ------------------------------------------------------------------

    def k_hop_expand(self, seed_ids: Sequence[int], hops: int = 2) -> set[int]:
        """Expand a set of seed node IDs by *hops* via a recursive CTE.

        This runs entirely inside the SQLite engine — no Python loops.
        """
        if not seed_ids or hops < 1:
            return set(seed_ids)

        # Build seed SELECT statements compatible with all SQLite versions
        seed_selects = " UNION ALL ".join(
            f"SELECT {sid}, 0" for sid in seed_ids
        )
        sql = f"""
        WITH RECURSIVE k_hop(node_id, hop) AS (
            -- Seed set at hop 0
            {seed_selects}
            UNION
            -- Forward expansion
            SELECT e.target, k.hop + 1
            FROM edges e
            JOIN k_hop k ON e.source = k.node_id
            WHERE k.hop < ?
            UNION
            -- Backward expansion (treat edges as undirected)
            SELECT e.source, k.hop + 1
            FROM edges e
            JOIN k_hop k ON e.target = k.node_id
            WHERE k.hop < ?
        )
        SELECT DISTINCT node_id FROM k_hop;
        """
        rows = self._conn.execute(sql, (hops, hops)).fetchall()
        expanded = {r["node_id"] for r in rows}
        logger.debug(
            "K-hop expand (%d seeds, %d hops) -> %d nodes",
            len(seed_ids),
            hops,
            len(expanded),
        )
        return expanded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> NodeData:
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return NodeData(
            id=row["id"],
            text=row["text"],
            vector_idx=row["vector_idx"],
            metadata=meta,
            is_deleted=bool(row["is_deleted"]),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        logger.info("GraphDB connection closed.")

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
