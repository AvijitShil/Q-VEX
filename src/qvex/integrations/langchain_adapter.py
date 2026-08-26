"""LangChain vector store adapter for Q-VEX.

Allows using Q-VEX as a drop-in vector store within LangChain pipelines::

    from qvex import QVEX
    from qvex.integrations import QVEXLangChainVectorStore

    qvex = QVEX(dim=384, storage_dir="./my_graph")
    vector_store = QVEXLangChainVectorStore(qvex=qvex, embedding=my_embeddings)

Requires ``langchain-core`` to be installed::

    pip install langchain-core
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional, Type

import numpy as np

logger = logging.getLogger("qvex.integrations.langchain")

# ---------------------------------------------------------------------------
# Attempt to import LangChain base class
# ---------------------------------------------------------------------------
try:
    from langchain_core.vectorstores import VectorStore as LangChainVectorStore
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    _HAS_LANGCHAIN = True
    _BASE_CLASS = LangChainVectorStore
except ImportError:
    _HAS_LANGCHAIN = False
    _BASE_CLASS = object  # type: ignore[assignment,misc]


def _require_langchain() -> None:
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "LangChain is not installed. Install it with:\n"
            "  pip install langchain-core"
        )


class QVEXLangChainVectorStore(_BASE_CLASS):  # type: ignore[misc]
    """LangChain-compatible vector store backed by a Q-VEX instance.

    Parameters
    ----------
    qvex : QVEX
        An initialised Q-VEX database instance.
    embedding : Embeddings
        A LangChain ``Embeddings`` instance used to embed texts and queries.
    """

    def __init__(self, qvex: Any, embedding: Any, **kwargs: Any) -> None:
        _require_langchain()
        self._qvex = qvex
        self._embedding = embedding

    @property
    def embeddings(self) -> Any:
        """Return the embeddings model used by this store."""
        return self._embedding

    # ------------------------------------------------------------------
    # LangChain VectorStore interface
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the Q-VEX store.

        Parameters
        ----------
        texts : Iterable[str]
            Texts to embed and store.
        metadatas : list[dict] | None
            Optional per-text metadata.

        Returns
        -------
        list[str]
            The Q-VEX node IDs (as strings) for each added document.
        """
        _require_langchain()
        text_list = list(texts)
        embeddings = self._embedding.embed_documents(text_list)
        ids: List[str] = []

        for i, (text, emb) in enumerate(zip(text_list, embeddings)):
            vec = np.asarray(emb, dtype=np.float32)
            meta = metadatas[i] if metadatas and i < len(metadatas) else None
            doc_id = self._qvex.add(text, vec, metadata=meta)
            ids.append(str(doc_id))

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Any]:
        """Return the k most similar documents to the query.

        Parameters
        ----------
        query : str
            Natural-language query.
        k : int
            Number of results to return.

        Returns
        -------
        list[Document]
            LangChain ``Document`` objects with ``page_content`` and ``metadata``.
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Any, float]]:
        """Return the k most similar documents with their similarity scores.

        Parameters
        ----------
        query : str
            Natural-language query.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[Document, float]]
            Pairs of ``(Document, score)``.
        """
        _require_langchain()
        from langchain_core.documents import Document

        query_embedding = self._embedding.embed_query(query)
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        results = self._qvex.search(
            query=query,
            vector=query_vec,
            k=k,
        )

        docs_and_scores: List[tuple[Any, float]] = []
        for r in results:
            doc = Document(
                page_content=r.text,
                metadata=r.metadata or {},
            )
            docs_and_scores.append((doc, r.score))

        return docs_and_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "QVEXLangChainVectorStore":
        """Create a QVEXLangChainVectorStore from a list of texts.

        This is a convenience class method required by the LangChain interface.

        Parameters
        ----------
        texts : list[str]
            Documents to add.
        embedding : Embeddings
            Embeddings model.
        metadatas : list[dict] | None
            Optional metadata per document.
        **kwargs
            Passed through to ``QVEX()`` constructor.  Must include ``dim``.

        Returns
        -------
        QVEXLangChainVectorStore
        """
        _require_langchain()
        from qvex.core import QVEX

        dim = kwargs.pop("dim", None)
        if dim is None:
            # Infer dimension from a test embedding
            test_emb = embedding.embed_query("test")
            dim = len(test_emb)

        storage_dir = kwargs.pop("storage_dir", "./qvex_langchain_data")
        qvex = QVEX(dim=dim, storage_dir=storage_dir, **kwargs)
        store = cls(qvex=qvex, embedding=embedding)
        store.add_texts(texts, metadatas=metadatas)
        return store
