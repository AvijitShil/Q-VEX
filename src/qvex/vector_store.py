"""TurboVec-backed vector store with soft-delete support."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("qvex.vector_store")


class VectorStore:
    """Wrapper around TurboVec's ``TurboQuantIndex`` providing add, search,
    soft-delete, and persistence.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors.  Must be a positive multiple of 8.
    bit_width : int
        Quantization bit-width (2 or 4).
    index_path : str | Path | None
        If provided and the file exists, the index is loaded from disk.
        ``save()`` writes to this path.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 4,
        index_path: str | Path | None = None,
    ) -> None:
        from turbovec import TurboQuantIndex

        self.dim = dim
        self.bit_width = bit_width
        self._index_path = Path(index_path) if index_path else None
        self._lock = threading.Lock()
        self._next_idx: int = 0
        self.deleted_ids: set[int] = set()

        # Load existing index or create a fresh one
        if self._index_path and self._index_path.exists():
            self._index = TurboQuantIndex.load(str(self._index_path))
            # Infer next index from the loaded index size
            self._next_idx = len(self._index)
            logger.info(
                "Loaded vector index from %s (%d vectors)",
                self._index_path,
                self._next_idx,
            )
        else:
            self._index = TurboQuantIndex(dim=dim, bit_width=bit_width)
            logger.info(
                "Created new vector index (dim=%d, bit_width=%d)", dim, bit_width
            )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray) -> int:
        """Add a single vector and return its sequential index.

        Parameters
        ----------
        vector : np.ndarray
            A 1-D array of shape ``(dim,)`` or a 2-D array of shape ``(1, dim)``.

        Returns
        -------
        int
            The index assigned to this vector.
        """
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        with self._lock:
            idx = self._next_idx
            self._index.add(vec)
            self._next_idx += 1

        logger.debug("Added vector at index %d", idx)
        return idx

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        allowlist: set[int] | list[int] | np.ndarray | None = None,
        blocked_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Search for the *k* nearest vectors.

        Parameters
        ----------
        query_vector : np.ndarray
            Query vector of shape ``(dim,)`` or ``(1, dim)``.
        k : int
            Number of results to return.
        allowlist : set | list | ndarray | None
            If provided, only search within these indices.
        blocked_ids : set[int] | None
            Indices to exclude from results (in addition to ``deleted_ids``).

        Returns
        -------
        list[tuple[int, float]]
            Pairs of ``(index, score)`` sorted by descending similarity.
        """
        qvec = np.asarray(query_vector, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)

        total_vectors = len(self._index)
        if total_vectors == 0:
            return []

        # Combine blocked_ids with deleted_ids
        all_blocked = set(self.deleted_ids)
        if blocked_ids:
            all_blocked |= blocked_ids

        # Build a boolean mask for TurboVec's search API
        # mask[i] == True means slot i is included in the search
        mask: np.ndarray | None = None

        if allowlist is not None:
            effective = set(allowlist) - all_blocked
            if not effective:
                return []
            mask = np.zeros(total_vectors, dtype=bool)
            for idx in effective:
                if 0 <= idx < total_vectors:
                    mask[idx] = True
        elif all_blocked:
            mask = np.ones(total_vectors, dtype=bool)
            for idx in all_blocked:
                if 0 <= idx < total_vectors:
                    mask[idx] = False

        # Call TurboVec search
        search_kwargs: dict = {}
        if mask is not None:
            search_kwargs["mask"] = mask

        effective_k = min(k, total_vectors)
        if mask is not None:
            effective_k = min(effective_k, int(mask.sum()))
        if effective_k <= 0:
            return []

        scores, indices = self._index.search(qvec, effective_k, **search_kwargs)

        # Flatten results (search returns 2D arrays: [num_queries, k])
        flat_scores = scores[0] if scores.ndim > 1 else scores
        flat_indices = indices[0] if indices.ndim > 1 else indices

        results: list[tuple[int, float]] = []
        for idx, score in zip(flat_indices, flat_scores):
            idx_int = int(idx)
            if idx_int < 0:
                continue  # padding sentinel
            if idx_int in all_blocked:
                continue
            results.append((idx_int, float(score)))

        logger.debug("Vector search returned %d results (k=%d)", len(results), k)
        return results

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def soft_delete(self, vector_idx: int) -> None:
        """Mark a vector index as deleted (excluded from future searches)."""
        with self._lock:
            self.deleted_ids.add(vector_idx)
        logger.debug("Soft-deleted vector index %d", vector_idx)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Write the index to disk."""
        target = Path(path) if path else self._index_path
        if target is None:
            raise ValueError("No index path specified for saving.")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(target))
        logger.info("Saved vector index to %s", target)

    def load(self, path: str | Path | None = None) -> None:
        """Load the index from disk (replacing the current in-memory index)."""
        from turbovec import TurboQuantIndex

        target = Path(path) if path else self._index_path
        if target is None or not target.exists():
            raise FileNotFoundError(f"Index file not found: {target}")
        self._index = TurboQuantIndex.load(str(target))
        self._next_idx = len(self._index)
        logger.info("Loaded vector index from %s", target)
