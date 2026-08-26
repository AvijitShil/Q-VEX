"""Shared pytest fixtures for Q-VEX tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qvex.graph_db import GraphDB


TEST_DIM = 64


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def graph_db(tmp_dir: Path) -> GraphDB:
    """Provide a fresh GraphDB instance backed by a temp SQLite file."""
    db = GraphDB(tmp_dir / "test.db")
    yield db
    db.close()


@pytest.fixture
def random_vector() -> np.ndarray:
    """Generate a single random unit vector of TEST_DIM dimensions."""
    rng = np.random.default_rng(42)
    vec = rng.random(TEST_DIM, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def random_vectors():
    """Factory fixture: call with (n,) to get n random unit vectors."""

    def _make(n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vecs = rng.random((n, TEST_DIM), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    return _make
