"""Integration tests for qvex.core.QVEX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvex.core import QVEX

TEST_DIM = 64


@pytest.fixture
def qvex(tmp_dir: Path) -> QVEX:
    """Provide a fresh QVEX instance."""
    tg = QVEX(dim=TEST_DIM, storage_dir=tmp_dir / "tg_data", bit_width=4)
    yield tg
    tg.close()


def _embed(text: str) -> np.ndarray:
    """Deterministic mock embedding function based on text hash."""
    rng = np.random.default_rng(hash(text) % (2**31))
    vec = rng.random(TEST_DIM, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class TestAddAndSearch:
    """End-to-end add → search pipeline."""

    def test_add_returns_id(self, qvex: QVEX) -> None:
        vec = _embed("test document")
        doc_id = qvex.add("test document", vec)
        assert isinstance(doc_id, int)
        assert doc_id > 0

    def test_search_returns_results(self, qvex: QVEX) -> None:
        texts = [
            "Neural networks use backpropagation for training",
            "Transformers rely on self-attention mechanisms",
            "Convolutional networks are great for image recognition",
            "Recurrent networks process sequential data",
            "Graph neural networks operate on graph structures",
        ]
        for t in texts:
            qvex.add(t, _embed(t))

        results = qvex.search(
            "attention transformers",
            vector=_embed("attention transformers"),
            k=3,
        )
        assert len(results) > 0
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_search_empty_db(self, qvex: QVEX) -> None:
        results = qvex.search(
            "anything", vector=_embed("anything"), k=5
        )
        assert results == []


class TestDeleteAndUpdate:
    """Mutation operations."""

    def test_delete_removes_from_search(self, qvex: QVEX) -> None:
        doc_id = qvex.add("unique zephyr content", _embed("unique zephyr content"))
        # Should find it
        results = qvex.search(
            "zephyr", vector=_embed("zephyr"), k=5
        )
        assert any(r.id == doc_id for r in results)

        # Delete it
        assert qvex.delete(doc_id) is True

        # Should no longer appear
        results = qvex.search(
            "zephyr", vector=_embed("zephyr"), k=5
        )
        assert not any(r.id == doc_id for r in results)

    def test_delete_nonexistent_returns_false(self, qvex: QVEX) -> None:
        assert qvex.delete(9999) is False

    def test_update_replaces_content(self, qvex: QVEX) -> None:
        old_id = qvex.add(
            "Old content about pandas", _embed("Old content about pandas")
        )
        new_id = qvex.update(
            old_id,
            "New content about polars dataframes",
            _embed("New content about polars dataframes"),
        )
        assert new_id != old_id

        # Search for new content should find the new doc
        results = qvex.search(
            "polars", vector=_embed("polars"), k=5
        )
        found_ids = {r.id for r in results}
        assert new_id in found_ids
        assert old_id not in found_ids


class TestGraphExpansion:
    """Graph edges affect search results."""

    def test_edges_expand_search(self, qvex: QVEX) -> None:
        # Create two docs that are textually dissimilar
        doc_a = qvex.add(
            "Quantum computing uses qubits",
            _embed("Quantum computing uses qubits"),
        )
        doc_b = qvex.add(
            "Machine learning models train on data",
            _embed("Machine learning models train on data"),
        )
        # Link them via a graph edge
        qvex.add_edge(doc_a, doc_b, edge_type="related")

        # Search for "quantum" — BM25 finds doc_a, graph expansion should
        # bring in doc_b
        results = qvex.search(
            "quantum",
            vector=_embed("quantum"),
            k=5,
            hops=1,
        )
        found_ids = {r.id for r in results}
        assert doc_a in found_ids
        # doc_b should appear via graph expansion
        assert doc_b in found_ids


class TestIngest:
    """High-level ingestion pipeline."""

    def test_ingest_chunks_text(self, qvex: QVEX) -> None:
        long_text = "word " * 200  # ~1000 chars
        result = qvex.ingest(
            long_text,
            embed_fn=_embed,
            chunk_size=100,
            chunk_overlap=20,
        )
        assert result.chunk_count > 1
        assert len(result.node_ids) == result.chunk_count
        # Consecutive chunks should be linked
        assert result.edge_count >= result.chunk_count - 1

    def test_ingest_empty_text(self, qvex: QVEX) -> None:
        result = qvex.ingest("", embed_fn=_embed)
        assert result.chunk_count == 0
        assert result.node_ids == []

    def test_ingest_single_chunk(self, qvex: QVEX) -> None:
        result = qvex.ingest(
            "Short text", embed_fn=_embed, chunk_size=1000
        )
        assert result.chunk_count == 1
        assert result.edge_count == 0  # No next_chunk edges for single chunk


class TestContextManager:
    """QVEX as a context manager."""

    def test_context_manager(self, tmp_dir: Path) -> None:
        with QVEX(dim=TEST_DIM, storage_dir=tmp_dir / "ctx_data") as tg:
            tg.add("Test document", _embed("Test document"))
        # Should not raise after close


class TestRepr:
    """String representation."""

    def test_repr(self, qvex: QVEX) -> None:
        r = repr(qvex)
        assert "QVEX" in r
        assert str(TEST_DIM) in r
