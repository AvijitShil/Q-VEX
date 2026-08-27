"""LlamaIndex vector store adapter for Q-VEX.

Allows using Q-VEX as a drop-in vector store within LlamaIndex pipelines::

    from qvex import QVEX
    from qvex.integrations import QVEXLlamaIndexStore

    qvex = QVEX(dim=384, storage_dir="./my_graph")
    vector_store = QVEXLlamaIndexStore(qvex=qvex)

Requires ``llama-index-core`` to be installed::

    pip install llama-index-core
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger("qvex.integrations.llamaindex")

# ---------------------------------------------------------------------------
# Attempt to import LlamaIndex base class; fall back to a plain object
# so the adapter can still be *defined* without llama-index installed.
# ---------------------------------------------------------------------------
try:
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    try:
        from llama_index.core.vector_stores.types import BasePydanticVectorStore as _LlamaBaseStore
    except ImportError:
        try:
            from llama_index.core.vector_stores.types import BaseVectorStore as _LlamaBaseStore
        except ImportError:
            from llama_index.core.vector_stores.types import VectorStore as _LlamaBaseStore

    from llama_index.core.schema import TextNode

    _HAS_LLAMA_INDEX = True
    _BASE_CLASS = _LlamaBaseStore
except ImportError:
    _HAS_LLAMA_INDEX = False
    _BASE_CLASS = object  # type: ignore[assignment,misc]


def _require_llama_index() -> None:
    if not _HAS_LLAMA_INDEX:
        raise ImportError(
            "LlamaIndex is not installed. Install it with:\n"
            "  pip install llama-index-core\n"
            "or:\n"
            "  pip install llama-index"
        )


class QVEXLlamaIndexStore(_BASE_CLASS):  # type: ignore[misc]
    """LlamaIndex-compatible vector store backed by a Q-VEX instance.

    Parameters
    ----------
    qvex : QVEX
        An initialised Q-VEX database instance.
    """

    # LlamaIndex protocol attributes
    stores_text: bool = True
    is_embedding_required: bool = True

    def __init__(self, qvex: Any, **kwargs: Any) -> None:
        _require_llama_index()
        if _HAS_LLAMA_INDEX and hasattr(_BASE_CLASS, "__init__"):
            super().__init__(**kwargs)
        object.__setattr__(self, "_qvex", qvex)

    @classmethod
    def class_name(cls) -> str:
        return "QVEXLlamaIndexStore"

    @property
    def client(self) -> Any:
        return getattr(self, "_qvex", None)

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        nodes: List[Any],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add LlamaIndex ``TextNode`` objects to the Q-VEX store.

        Parameters
        ----------
        nodes : list[TextNode]
            Nodes with ``.embedding`` and ``.get_content()`` populated.

        Returns
        -------
        list[str]
            The Q-VEX node IDs (as strings) for each added node.
        """
        _require_llama_index()
        ids: List[str] = []
        for node in nodes:
            text = node.get_content()
            embedding = node.embedding
            if embedding is None:
                raise ValueError(
                    f"Node {node.node_id} has no embedding. "
                    "Set embeddings before adding to QVEXLlamaIndexStore."
                )
            vec = np.asarray(embedding, dtype=np.float32)
            metadata = node.metadata if hasattr(node, "metadata") else None
            doc_id = self._qvex.add(text, vec, metadata=metadata)
            ids.append(str(doc_id))
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document by its Q-VEX node ID."""
        self._qvex.delete(int(ref_doc_id))

    def query(
        self,
        query: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a vector store query and return LlamaIndex-typed results.

        Parameters
        ----------
        query : VectorStoreQuery
            A LlamaIndex query object with ``.query_embedding`` and
            ``.similarity_top_k``.

        Returns
        -------
        VectorStoreQueryResult
            Contains ``nodes``, ``similarities``, and ``ids``.
        """
        _require_llama_index()
        from llama_index.core.vector_stores.types import VectorStoreQueryResult
        from llama_index.core.schema import TextNode

        query_vec = np.asarray(query.query_embedding, dtype=np.float32)
        k = query.similarity_top_k or 10

        # Use empty string for BM25 query text — the vector reranking is
        # the primary ranking signal in this adapter path.
        query_str = query.query_str if hasattr(query, "query_str") and query.query_str else ""
        results = self._qvex.search(
            query=query_str,
            vector=query_vec,
            k=k,
        )

        nodes = []
        similarities = []
        ids = []
        for r in results:
            node = TextNode(
                text=r.text,
                id_=str(r.id),
                metadata=r.metadata or {},
            )
            nodes.append(node)
            similarities.append(r.score)
            ids.append(str(r.id))

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
