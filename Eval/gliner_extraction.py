"""Evaluation 3: GLiNER Entity Extraction (RAM-Safe Small Model).

Demonstrates Q-VEX's knowledge-graph construction capability by ingesting a
short AI-domain document, extracting entities + relationships with the tiny
`urchade/gliner_small-v2.1` model, and printing the resulting graph edges.

Usage (from project root):
    python Eval/gliner_extraction.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root + src/ are on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from qvex import QVEX

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "eval_gliner_db")
GLINER_MODEL = "urchade/gliner_small-v2.1"  # RAM-safe English model (~150 MB)

DEMO_DOCUMENT = """
In AI engineering, Large Language Models (LLMs) rely on Retrieval-Augmented
Generation (RAG) to reduce hallucinations. Systems like FAISS and Q-VEX provide
the vector indexing required for this architecture. Techniques like quantization
and graph-based retrieval further improve the efficiency of these pipelines.
Researchers at institutions such as Stanford and Google have shown that combining
BM25 with dense retrieval improves recall significantly.
"""

# Cache the model at module level so embed_fn doesn't reload it on every call
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("[EMBED] Loading all-MiniLM-L6-v2 embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def embed_fn(text: str) -> np.ndarray:
    """Single-text embedding using cached model."""
    model = _get_embed_model()
    return model.encode([text], convert_to_numpy=True)[0].astype(np.float32)


def test_gliner_extraction() -> None:
    print("=" * 60)
    print("GLiNER ENTITY EXTRACTION EVALUATION (SMALL MODEL)")
    print("=" * 60)

    # Pre-load embedding model before GLiNER to avoid repeated loads
    _get_embed_model()

    # Load GLiNER extractor -- uses the actual class name: GraphExtractor
    from qvex.extractor.gliner_extractor import GraphExtractor

    print(f"\n[MODEL] Loading {GLINER_MODEL} ...")
    extractor = GraphExtractor(model_name=GLINER_MODEL)
    print("  -> Model loaded.\n")

    print("[SETUP] Creating QVEX index...")
    import shutil
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
    os.makedirs(STORAGE_DIR, exist_ok=True)
    qvex = QVEX(dim=384, storage_dir=STORAGE_DIR, bit_width=4)

    print("[INGEST] Ingesting demo document with GLiNER extractor...")
    result = qvex.ingest(
        DEMO_DOCUMENT,
        embed_fn=embed_fn,
        chunk_size=200,
        chunk_overlap=30,
        extractor=extractor,
    )

    print(f"\n  -> Chunks / nodes created: {result.chunk_count}  ({len(result.node_ids)} IDs)")
    print(f"  -> Edges created:          {result.edge_count}")

    print("\n[GRAPH] All edges in the knowledge graph:")
    edges = qvex.graph_db.get_all_edges()
    if not edges:
        print("  (no edges extracted -- try a longer document or adjust GLiNER labels)")
    else:
        for edge in edges:
            print(f"  Node {edge.source:>3} --[{edge.edge_type}]--> Node {edge.target}")

    print(f"\n[OK] GLiNER extraction evaluation complete.")
    qvex.close()


if __name__ == "__main__":
    test_gliner_extraction()
