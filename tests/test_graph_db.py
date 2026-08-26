"""Tests for qvex.graph_db.GraphDB."""

from __future__ import annotations

import pytest

from qvex.graph_db import GraphDB


class TestNodeCRUD:
    """Node create, read, update, delete operations."""

    def test_add_and_get_node(self, graph_db: GraphDB) -> None:
        nid = graph_db.add_node("Hello world", vector_idx=0)
        assert nid is not None
        node = graph_db.get_node(nid)
        assert node is not None
        assert node.text == "Hello world"
        assert node.vector_idx == 0

    def test_add_node_with_metadata(self, graph_db: GraphDB) -> None:
        meta = {"source": "test", "page": 42}
        nid = graph_db.add_node("Test doc", vector_idx=1, metadata=meta)
        node = graph_db.get_node(nid)
        assert node is not None
        assert node.metadata == meta

    def test_get_nonexistent_node(self, graph_db: GraphDB) -> None:
        assert graph_db.get_node(9999) is None

    def test_update_node_text(self, graph_db: GraphDB) -> None:
        nid = graph_db.add_node("Original text", vector_idx=0)
        updated = graph_db.update_node(nid, text="Updated text")
        assert updated is True
        node = graph_db.get_node(nid)
        assert node.text == "Updated text"

    def test_update_node_metadata(self, graph_db: GraphDB) -> None:
        nid = graph_db.add_node("Test", vector_idx=0, metadata={"v": 1})
        graph_db.update_node(nid, metadata={"v": 2, "new_key": "value"})
        node = graph_db.get_node(nid)
        assert node.metadata == {"v": 2, "new_key": "value"}

    def test_update_nonexistent_returns_false(self, graph_db: GraphDB) -> None:
        assert graph_db.update_node(9999, text="x") is False

    def test_delete_node(self, graph_db: GraphDB) -> None:
        nid = graph_db.add_node("To be deleted", vector_idx=0)
        deleted = graph_db.delete_node(nid)
        assert deleted is True
        assert graph_db.get_node(nid) is None

    def test_delete_nonexistent_returns_false(self, graph_db: GraphDB) -> None:
        assert graph_db.delete_node(9999) is False

    def test_get_nodes_by_ids(self, graph_db: GraphDB) -> None:
        id1 = graph_db.add_node("A", vector_idx=0)
        id2 = graph_db.add_node("B", vector_idx=1)
        graph_db.add_node("C", vector_idx=2)
        nodes = graph_db.get_nodes_by_ids([id1, id2])
        assert len(nodes) == 2
        texts = {n.text for n in nodes}
        assert texts == {"A", "B"}

    def test_get_nodes_by_ids_empty(self, graph_db: GraphDB) -> None:
        assert graph_db.get_nodes_by_ids([]) == []


class TestEdges:
    """Edge operations and cascade behaviour."""

    def test_add_and_get_edges(self, graph_db: GraphDB) -> None:
        n1 = graph_db.add_node("Node A", vector_idx=0)
        n2 = graph_db.add_node("Node B", vector_idx=1)
        graph_db.add_edge(n1, n2, edge_type="knows", confidence=0.9)
        edges = graph_db.get_edges(n1)
        assert len(edges) == 1
        assert edges[0].target == n2
        assert edges[0].edge_type == "knows"
        assert edges[0].confidence == 0.9

    def test_duplicate_edge_ignored(self, graph_db: GraphDB) -> None:
        n1 = graph_db.add_node("A", vector_idx=0)
        n2 = graph_db.add_node("B", vector_idx=1)
        graph_db.add_edge(n1, n2)
        graph_db.add_edge(n1, n2)  # Should not raise
        edges = graph_db.get_edges(n1)
        assert len(edges) == 1

    def test_cascade_delete_removes_edges(self, graph_db: GraphDB) -> None:
        n1 = graph_db.add_node("A", vector_idx=0)
        n2 = graph_db.add_node("B", vector_idx=1)
        n3 = graph_db.add_node("C", vector_idx=2)
        graph_db.add_edge(n1, n2)
        graph_db.add_edge(n2, n3)
        # Delete n2 — should cascade-remove both edges
        graph_db.delete_node(n2)
        assert graph_db.get_edges(n1) == []


class TestFTS5BM25:
    """Full-text search via FTS5."""

    def test_bm25_search_basic(self, graph_db: GraphDB) -> None:
        graph_db.add_node("The quick brown fox jumps over the lazy dog", vector_idx=0)
        graph_db.add_node("A fast cat ran across the field", vector_idx=1)
        graph_db.add_node("Dogs and foxes are friends", vector_idx=2)

        results = graph_db.bm25_search("fox")
        assert len(results) >= 1

    def test_bm25_search_no_match(self, graph_db: GraphDB) -> None:
        graph_db.add_node("Nothing relevant here", vector_idx=0)
        results = graph_db.bm25_search("quantum")
        assert results == []

    def test_bm25_search_respects_limit(self, graph_db: GraphDB) -> None:
        for i in range(10):
            graph_db.add_node(f"Document about python topic {i}", vector_idx=i)
        results = graph_db.bm25_search("python", limit=3)
        assert len(results) <= 3

    def test_fts5_updates_on_node_update(self, graph_db: GraphDB) -> None:
        nid = graph_db.add_node("Artificial intelligence rocks", vector_idx=0)
        # Should find it
        assert len(graph_db.bm25_search("artificial")) >= 1
        # Update text
        graph_db.update_node(nid, text="Machine learning is cool")
        # Old term gone
        assert len(graph_db.bm25_search("artificial")) == 0
        # New term found
        assert len(graph_db.bm25_search("machine")) >= 1

    def test_fts5_cleans_up_on_delete(self, graph_db: GraphDB) -> None:
        nid = graph_db.add_node("Unique xylophone content", vector_idx=0)
        assert len(graph_db.bm25_search("xylophone")) >= 1
        graph_db.delete_node(nid)
        assert len(graph_db.bm25_search("xylophone")) == 0


class TestKHopExpansion:
    """Recursive CTE graph traversal."""

    def _build_chain(self, graph_db: GraphDB) -> list[int]:
        """Build A -> B -> C -> D chain."""
        ids = []
        for i, label in enumerate(["A", "B", "C", "D"]):
            ids.append(graph_db.add_node(label, vector_idx=i))
        for i in range(len(ids) - 1):
            graph_db.add_edge(ids[i], ids[i + 1])
        return ids

    def test_1_hop(self, graph_db: GraphDB) -> None:
        ids = self._build_chain(graph_db)
        expanded = graph_db.k_hop_expand([ids[0]], hops=1)
        # Should include A and B (1 hop forward)
        assert ids[0] in expanded
        assert ids[1] in expanded

    def test_2_hop(self, graph_db: GraphDB) -> None:
        ids = self._build_chain(graph_db)
        expanded = graph_db.k_hop_expand([ids[0]], hops=2)
        assert ids[0] in expanded
        assert ids[1] in expanded
        assert ids[2] in expanded

    def test_0_hops_returns_seeds(self, graph_db: GraphDB) -> None:
        ids = self._build_chain(graph_db)
        expanded = graph_db.k_hop_expand([ids[0]], hops=0)
        assert expanded == {ids[0]}

    def test_empty_seeds(self, graph_db: GraphDB) -> None:
        assert graph_db.k_hop_expand([], hops=2) == set()

    def test_no_cycles(self, graph_db: GraphDB) -> None:
        """A -> B -> A loop shouldn't cause infinite recursion."""
        a = graph_db.add_node("A", vector_idx=0)
        b = graph_db.add_node("B", vector_idx=1)
        graph_db.add_edge(a, b)
        graph_db.add_edge(b, a)
        expanded = graph_db.k_hop_expand([a], hops=10)
        # Should just be A and B, no crash
        assert expanded == {a, b}

    def test_multiple_seeds(self, graph_db: GraphDB) -> None:
        ids = self._build_chain(graph_db)
        # Seeds A and D, 1 hop => A, B, C, D
        expanded = graph_db.k_hop_expand([ids[0], ids[3]], hops=1)
        assert set(ids) == expanded
