"""QVEX — the main orchestrator tying GraphDB and VectorStore together."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from qvex.graph_db import GraphDB
from qvex.models import IngestResult, SearchResult
from qvex.vector_store import VectorStore
from qvex.extractor.structural import StructuralGraphBuilder
from qvex.extractor.lexical import LexicalCooccurrenceBuilder

logger = logging.getLogger("qvex")


class QVEX:
    """A hybrid graph-vector database for local GraphRAG pipelines.

    Combines:
    - **SQLite FTS5** for BM25 full-text search
    - **Recursive CTEs** for lightning-fast k-hop graph traversal
    - **TurboVec** quantized vector index for similarity search

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding vectors (required, no default).
    storage_dir : str | Path
        Directory to persist the SQLite database and vector index files.
    bit_width : int
        Quantization bit-width for the vector index (2 or 4). Default: 4.

    Example
    -------
    >>> tg = QVEX(dim=384, storage_dir="./my_graph")
    >>> doc_id = tg.add("Transformers use self-attention.", vector=embedding)
    >>> results = tg.search("attention mechanism", vector=query_vec, k=5)
    """

    def __init__(
        self,
        dim: int,
        storage_dir: str | Path = "./qvex_data",
        bit_width: int = 4,
    ) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        db_path = self._storage_dir / "qvex.db"
        index_path = self._storage_dir / "vectors.tq"

        self._graph = GraphDB(db_path)
        self._vectors = VectorStore(
            dim=dim, bit_width=bit_width, index_path=index_path
        )
        self._dim = dim
        self._lock = threading.RLock()

        logger.info(
            "QVEX initialized (dim=%d, bit_width=%d, storage=%s)",
            dim,
            bit_width,
            self._storage_dir,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def graph_db(self) -> GraphDB:
        """Public access to the underlying GraphDB (for advanced/eval use)."""
        return self._graph

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a document node with its embedding vector.

        Parameters
        ----------
        text : str
            The text content of the document chunk.
        vector : np.ndarray
            The embedding vector for this chunk.
        metadata : dict | None
            Optional JSON-serialisable metadata.

        Returns
        -------
        int
            The node ID assigned to this document.

        Raises
        ------
        ValueError
            If the vector dimension does not match the configured ``dim``.
        """
        vec = np.asarray(vector, dtype=np.float32).squeeze()
        if vector.shape[-1] != self._dim:
            raise ValueError(f"Dimension mismatch: expected {self._dim}, got {vector.shape[-1]}")

        with self._lock:
            # Store node in the graph first without vector index
            node_id = self._graph.add_node(
                text, metadata=metadata
            )
            try:
                # Store vector
                vec_idx = self._vectors.add(vec)
                # Link vector index to graph node
                self._graph.update_node(node_id, vector_idx=vec_idx)
            except Exception as e:
                # Rollback: hard-delete the graph node
                self._graph.delete_node(node_id)
                raise e

        logger.debug("add: node_id=%d, vec_idx=%d", node_id, vec_idx)
        return node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str = "related",
        confidence: float = 1.0,
    ) -> None:
        """Create a directed edge between two existing nodes."""
        self._graph.add_edge(source_id, target_id, edge_type, confidence)

    def delete(self, doc_id: int) -> bool:
        """Delete a document by its node ID.

        Hard-deletes from SQLite (cascading edges and FTS5 via triggers).
        Soft-deletes from the vector index (ID is masked from future searches).

        Returns True if the node existed and was deleted.
        """
        with self._lock:
            node = self._graph.get_node(doc_id)
            if node is None:
                return False

            # Soft-delete from vector index
            if node.vector_idx is not None:
                self._vectors.soft_delete(node.vector_idx)

            # Hard-delete from SQLite (triggers clean up FTS5 + cascade edges)
            self._graph.delete_node(doc_id)

        logger.info("Deleted doc_id=%d (vec_idx=%s)", doc_id, node.vector_idx)
        return True

    def clear(self) -> None:
        """Clear all nodes, edges, and vectors in the database."""
        with self._lock:
            self._graph.clear()
            self._vectors.clear()
        logger.info("Cleared all data from database.")

    def update(
        self,
        doc_id: int,
        new_text: str,
        new_vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Update a document by deleting the old one and adding a new one.

        Returns the new node ID.
        """
        with self._lock:
            # delete and add are called without their own lock acquisition
            # because we already hold the lock here
            node = self._graph.get_node(doc_id)
            if node is not None:
                if node.vector_idx is not None:
                    self._vectors.soft_delete(node.vector_idx)
                self._graph.delete_node(doc_id)

            vec = np.asarray(new_vector, dtype=np.float32).squeeze()
            if new_vector.shape[-1] != self._dim:
                raise ValueError(f"Dimension mismatch: expected {self._dim}, got {new_vector.shape[-1]}")
            
            node_id = self._graph.add_node(
                new_text, metadata=metadata
            )
            try:
                vec_idx = self._vectors.add(vec)
                self._graph.update_node(node_id, vector_idx=vec_idx)
            except Exception as e:
                self._graph.delete_node(node_id)
                raise e

        return node_id

    # ------------------------------------------------------------------
    # Hybrid Search Pipeline
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        vector: np.ndarray,
        k: int = 10,
        hops: int = 2,
        bm25_k: int = 50,
    ) -> list[SearchResult]:
        """Execute the hybrid BM25 → Graph Expansion → Vector Reranking pipeline.

        Pipeline steps:

        1. **BM25 text match** — Query SQLite FTS5 for the top *bm25_k* node IDs.
        2. **Graph expansion** — Feed those IDs into a recursive CTE to grab
           their *hops*-hop neighbours.
        3. **Vector reranking** — Pass the expanded ID set to TurboVec's
           ``search(allowlist=…)`` to rank by embedding similarity, filtering
           out any soft-deleted vectors.

        Parameters
        ----------
        query : str
            Text query for BM25 matching.
        vector : np.ndarray
            Query embedding vector for similarity ranking.
        k : int
            Number of results to return.
        hops : int
            Number of graph hops to expand from the BM25 seed set.
        bm25_k : int
            Maximum number of BM25 seed results.

        Returns
        -------
        list[SearchResult]
            Ranked search results with scores and metadata.
        """
        # Step 1: BM25 text search (with vector seed fallback)
        bm25_ids: list[int] = []
        if query and query.strip():
            bm25_ids = self._graph.bm25_search(query, limit=bm25_k)

        if not bm25_ids:
            # Semantic fallback: seed from vector search
            vec_seed_results = self._vectors.search(vector, k=min(bm25_k, max(k, 10)))
            if not vec_seed_results:
                logger.debug("search: no seed hits found for query='%s'", query)
                return []
            vec_indices = [idx for idx, _ in vec_seed_results]
            seed_nodes = self._graph.get_nodes_by_vector_indices(vec_indices)
            bm25_ids = [n.id for n in seed_nodes]
            if not bm25_ids:
                return []

        # Track which nodes were direct BM25 hits (hop 0)
        bm25_set = set(bm25_ids)

        # Step 2: Graph expansion via recursive CTE
        expanded_ids = self._graph.k_hop_expand(bm25_ids, hops=hops)

        # Map node IDs to their vector indices for the allowlist
        nodes = self._graph.get_nodes_by_ids(list(expanded_ids))
        id_to_node = {n.id: n for n in nodes}
        vec_idx_to_node_id: dict[int, int] = {}
        allowlist_vec_ids: set[int] = set()

        for n in nodes:
            if n.vector_idx is not None:
                vec_idx_to_node_id[n.vector_idx] = n.id
                allowlist_vec_ids.add(n.vector_idx)

        if not allowlist_vec_ids:
            logger.debug("search: no valid vector indices in expanded set")
            return []

        # Step 3: Vector reranking with allowlist
        vec_results = self._vectors.search(
            query_vector=vector,
            k=k,
            allowlist=allowlist_vec_ids,
        )

        # Build SearchResult objects
        results: list[SearchResult] = []
        for vec_idx, score in vec_results:
            node_id = vec_idx_to_node_id.get(vec_idx)
            if node_id is None:
                continue
            node = id_to_node.get(node_id)
            if node is None:
                continue

            hop_distance = 0 if node_id in bm25_set else 1  # simplified
            results.append(
                SearchResult(
                    id=node_id,
                    text=node.text,
                    score=score,
                    metadata=node.metadata,
                    hop_distance=hop_distance,
                )
            )

        logger.info(
            "search '%s': %d BM25 seeds -> %d expanded -> %d results",
            query,
            len(bm25_ids),
            len(expanded_ids),
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # High-level ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        embed_fn: Callable[[str], np.ndarray],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> IngestResult:
        """Ingest raw text: chunk, embed, store, and automatically build graph edges.

        Parameters
        ----------
        text : str
            The full document text to ingest.
        embed_fn : callable
            A function that takes a text string and returns its embedding
            as a numpy array of shape ``(dim,)``.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Number of overlapping characters between consecutive chunks.

        Returns
        -------
        IngestResult
            Summary with ``node_ids``, ``edge_count``, and ``chunk_count``.
        """
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        node_ids: list[int] = []
        edge_count = 0

        # Initialize builders
        structural_builder = StructuralGraphBuilder()
        lexical_builder = LexicalCooccurrenceBuilder(top_k=15, idf_threshold=2.0)

        with self._lock:
            # Pass 1: Add each chunk as a node and feed to lexical builder
            for chunk in chunks:
                vec = embed_fn(chunk)
                nid = self.add(chunk, vec)
                node_ids.append(nid)
                lexical_builder.add_chunk_pass1(nid, chunk)

            # Pass 2: Compute TF-IDF
            lexical_builder.compute_idf_and_pass2()

            # Build structural edges
            structural_edges = structural_builder.build_edges(node_ids)
            for src, tgt, edge_type, confidence in structural_edges:
                self.add_edge(src, tgt, edge_type=edge_type, confidence=confidence)
                edge_count += 1

            # Build lexical edges
            lexical_edges = lexical_builder.build_edges()
            for src, tgt, edge_type, confidence in lexical_edges:
                self.add_edge(src, tgt, edge_type=edge_type, confidence=confidence)
                edge_count += 1

        result = IngestResult(
            node_ids=node_ids,
            edge_count=edge_count,
            chunk_count=len(chunks),
        )
        logger.info(
            "Ingested %d chunks, %d edges", result.chunk_count, result.edge_count
        )
        return result

    # ------------------------------------------------------------------
    # Persistence & lifecycle
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the vector index to disk.

        The SQLite database is auto-persisted via WAL mode.
        """
        self._vectors.save()
        logger.info("QVEX saved to %s", self._storage_dir)

    def close(self) -> None:
        """Save and close all resources."""
        try:
            self.save()
        except Exception:
            logger.warning("Could not save vector index on close.", exc_info=True)
        self._graph.close()
        logger.info("QVEX closed.")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_graph(self, output_html: str = "qvex_graph.html") -> str:
        """Generate a standalone interactive HTML graph visualization.

        Uses ``vis-network`` (loaded from CDN) to render an interactive,
        draggable network map of all nodes and edges in the database.

        Parameters
        ----------
        output_html : str
            Path for the output HTML file.  Defaults to ``qvex_graph.html``
            in the current working directory.

        Returns
        -------
        str
            Absolute path to the generated HTML file.
        """
        import html as html_mod

        nodes = self._graph.get_all_nodes()
        edges = self._graph.get_all_edges()

        # Build vis-network node data
        node_entries: list[str] = []
        for n in nodes:
            label = n.text[:50].replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            title = html_mod.escape(n.text[:200])
            node_entries.append(
                f"  {{id: {n.id}, label: '{label}', title: '{title}'}}"
            )

        # Edge colors by type
        edge_type_colors = {
            "next_chunk": "#6366f1",      # indigo
            "shared_entity": "#f59e0b",   # amber
            "related": "#10b981",         # emerald
            "related_to": "#ef4444",      # red
        }

        edge_entries: list[str] = []
        for e in edges:
            color = edge_type_colors.get(e.edge_type, "#94a3b8")
            label = e.edge_type.replace("_", " ")
            edge_entries.append(
                f"  {{from: {e.source}, to: {e.target}, "
                f"label: '{label}', color: {{color: '{color}'}}, "
                f"arrows: 'to'}}"
            )

        nodes_js = ",\n".join(node_entries)
        edges_js = ",\n".join(edge_entries)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Q-VEX Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }}
    #header {{
      padding: 16px 24px;
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border-bottom: 1px solid #334155;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    #header h1 {{
      font-size: 1.25rem;
      font-weight: 600;
      background: linear-gradient(135deg, #818cf8, #6366f1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    #header .stats {{
      font-size: 0.85rem;
      color: #94a3b8;
    }}
    #graph-container {{
      width: 100%;
      height: calc(100vh - 56px);
    }}
  </style>
</head>
<body>
  <div id="header">
    <h1>Q-VEX Knowledge Graph</h1>
    <span class="stats">{len(nodes)} nodes &middot; {len(edges)} edges</span>
  </div>
  <div id="graph-container"></div>
  <script>
    var nodes = new vis.DataSet([
{nodes_js}
    ]);
    var edges = new vis.DataSet([
{edges_js}
    ]);
    var container = document.getElementById('graph-container');
    var data = {{ nodes: nodes, edges: edges }};
    var options = {{
      nodes: {{
        shape: 'dot',
        size: 16,
        font: {{ size: 12, color: '#e2e8f0', face: 'Inter, system-ui, sans-serif' }},
        color: {{
          background: '#6366f1',
          border: '#818cf8',
          highlight: {{ background: '#818cf8', border: '#a5b4fc' }},
          hover: {{ background: '#818cf8', border: '#a5b4fc' }}
        }},
        borderWidth: 2
      }},
      edges: {{
        width: 1.5,
        font: {{ size: 10, color: '#64748b', strokeWidth: 0 }},
        smooth: {{ type: 'continuous' }}
      }},
      physics: {{
        forceAtlas2Based: {{
          gravitationalConstant: -30,
          centralGravity: 0.005,
          springLength: 150,
          springConstant: 0.02
        }},
        solver: 'forceAtlas2Based',
        stabilization: {{ iterations: 150 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true
      }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>"""

        output_path = Path(output_html).resolve()
        output_path.write_text(html_content, encoding="utf-8")
        logger.info("Graph visualization written to %s", output_path)
        return str(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split *text* into overlapping chunks."""
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        return chunks

    def __repr__(self) -> str:
        return (
            f"QVEX(dim={self._dim}, storage='{self._storage_dir}')"
        )

    def __enter__(self) -> QVEX:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
