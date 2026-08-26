"""Tests for qvex.vector_store.VectorStore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qvex.vector_store import VectorStore

TEST_DIM = 64


@pytest.fixture
def vector_store(tmp_dir: Path) -> VectorStore:
    """Provide a fresh VectorStore instance."""
    return VectorStore(dim=TEST_DIM, bit_width=4, index_path=tmp_dir / "test.tq")


class TestAddAndSearch:
    """Basic add and search functionality."""

    def test_add_returns_sequential_index(self, vector_store: VectorStore) -> None:
        rng = np.random.default_rng(1)
        idx0 = vector_store.add(rng.random(TEST_DIM, dtype=np.float32))
        idx1 = vector_store.add(rng.random(TEST_DIM, dtype=np.float32))
        assert idx0 == 0
        assert idx1 == 1

    def test_search_returns_results(
        self, vector_store: VectorStore, random_vectors
    ) -> None:
        vecs = random_vectors(20)
        for v in vecs:
            vector_store.add(v)

        query = vecs[0]
        results = vector_store.search(query, k=5)
        assert len(results) > 0
        assert len(results) <= 5
        # First result should be the closest (ideally itself)
        indices = [idx for idx, _ in results]
        assert 0 in indices  # The query vector itself

    def test_search_with_2d_query(
        self, vector_store: VectorStore, random_vectors
    ) -> None:
        vecs = random_vectors(5)
        for v in vecs:
            vector_store.add(v)
        # 2D query
        query = vecs[0].reshape(1, -1)
        results = vector_store.search(query, k=3)
        assert len(results) > 0


class TestAllowlistAndBlocking:
    """Filtered search with allowlist and blocked IDs."""

    def test_search_with_allowlist(
        self, vector_store: VectorStore, random_vectors
    ) -> None:
        vecs = random_vectors(10)
        for v in vecs:
            vector_store.add(v)

        # Only allow indices 0, 1, 2
        results = vector_store.search(vecs[0], k=5, allowlist={0, 1, 2})
        indices = {idx for idx, _ in results}
        assert indices <= {0, 1, 2}

    def test_search_with_empty_allowlist(
        self, vector_store: VectorStore, random_vectors
    ) -> None:
        vecs = random_vectors(5)
        for v in vecs:
            vector_store.add(v)
        results = vector_store.search(vecs[0], k=5, allowlist=set())
        assert results == []

    def test_search_with_blocked_ids(
        self, vector_store: VectorStore, random_vectors
    ) -> None:
        vecs = random_vectors(5)
        for v in vecs:
            vector_store.add(v)
        results = vector_store.search(vecs[0], k=5, blocked_ids={0})
        indices = {idx for idx, _ in results}
        assert 0 not in indices


class TestSoftDelete:
    """Soft-delete masking."""

    def test_soft_delete_excludes_from_search(
        self, vector_store: VectorStore, random_vectors
    ) -> None:
        vecs = random_vectors(5)
        for v in vecs:
            vector_store.add(v)

        vector_store.soft_delete(0)
        results = vector_store.search(vecs[0], k=5)
        indices = {idx for idx, _ in results}
        assert 0 not in indices


class TestPersistence:
    """Save and load index from disk."""

    def test_save_and_load(self, tmp_dir: Path, random_vectors) -> None:
        path = tmp_dir / "persist_test.tq"
        store1 = VectorStore(dim=TEST_DIM, bit_width=4, index_path=path)
        vecs = random_vectors(10)
        for v in vecs:
            store1.add(v)
        store1.save()

        # Load into a new store
        store2 = VectorStore(dim=TEST_DIM, bit_width=4, index_path=path)
        results = store2.search(vecs[0], k=3)
        assert len(results) > 0

    def test_save_without_path_raises(self) -> None:
        store = VectorStore(dim=TEST_DIM, bit_width=4)
        with pytest.raises(ValueError, match="No index path"):
            store.save()
