"""Q-VEX — a hyper-compressed graph-vector database for local GraphRAG.

Quick start::

    from qvex import QVEX

    tg = QVEX(dim=384, storage_dir="./my_graph")
    doc_id = tg.add("Transformers use self-attention.", vector=embedding)
    results = tg.search("attention", vector=query_vec, k=5)
"""

from qvex.core import QVEX
from qvex.extractor.base import BaseExtractor
from qvex.graph_db import GraphDB
from qvex.models import EdgeData, IngestResult, NodeData, SearchResult
from qvex.vector_store import VectorStore
from qvex.banner import show_banner

# Framework adapters are available via lazy import:
#   from qvex.integrations import QVEXLlamaIndexStore
#   from qvex.integrations import QVEXLangChainVectorStore

__all__ = [
    "QVEX",
    "GraphDB",
    "VectorStore",
    "BaseExtractor",
    "NodeData",
    "EdgeData",
    "SearchResult",
    "IngestResult",
    "show_banner",
]

__version__ = "0.3.0"
