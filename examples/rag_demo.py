"""Q-VEX RAG Demo -- full ingestion pipeline with graph expansion.

Demonstrates:
- High-level `.ingest()` method with automatic chunking
- Graph edge creation between consecutive chunks
- Hybrid search with BM25 -> graph expansion -> vector reranking
- How graph edges pull in contextually related but textually dissimilar chunks

Run:
    pip install -e .
    python examples/rag_demo.py
"""

import numpy as np

from qvex import QVEX

DIM = 64


def mock_embed(text: str) -> np.ndarray:
    """Deterministic mock embedding function."""
    rng = np.random.default_rng(hash(text) % (2**31))
    vec = rng.random(DIM, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# A sample document about neural network architectures
SAMPLE_DOCUMENT = """
Neural networks are the foundation of modern deep learning. They consist of
layers of interconnected neurons that learn to transform input data into useful
representations. The most basic form is the feedforward network, where data
flows in one direction from input to output.

Convolutional Neural Networks (CNNs) revolutionized computer vision by using
filters that slide across images to detect local patterns like edges, textures,
and shapes. Key innovations include residual connections (ResNet), which allow
training of very deep networks by providing skip connections that bypass layers.

Recurrent Neural Networks (RNNs) were designed for sequential data like text
and time series. They maintain a hidden state that captures information from
previous time steps. However, they suffer from vanishing gradients when
processing long sequences, which led to the development of LSTM and GRU cells.

The Transformer architecture, introduced in "Attention Is All You Need" (2017),
replaced recurrence with self-attention mechanisms. This allows the model to
attend to all positions in the input simultaneously, enabling massive
parallelism and better long-range dependency modeling. Transformers are the
backbone of modern large language models like GPT and BERT.

Graph Neural Networks (GNNs) extend deep learning to graph-structured data.
They use message passing between nodes to aggregate neighborhood information.
GNNs are widely used in molecular property prediction, social network analysis,
and recommendation systems. Recent advances combine attention mechanisms from
Transformers with graph structure for more expressive models.
"""


def main() -> None:
    print("=" * 60)
    print("  Q-VEX RAG Demo")
    print("=" * 60)

    # 1. Initialize
    tg = QVEX(dim=DIM, storage_dir="./rag_demo_data", bit_width=4)

    # 2. Ingest the document with automatic chunking
    print("\n[*] Ingesting document with automatic chunking...")
    result = tg.ingest(
        SAMPLE_DOCUMENT,
        embed_fn=mock_embed,
        chunk_size=300,
        chunk_overlap=50,
    )
    print(f"   Chunks created: {result.chunk_count}")
    print(f"   Edges created:  {result.edge_count}")
    print(f"   Node IDs:       {result.node_ids}")

    # 3. Search without graph expansion
    query = "attention mechanism transformer"
    print(f"\n[?] Search WITHOUT graph expansion (hops=0): '{query}'")
    results_no_expand = tg.search(
        query=query,
        vector=mock_embed(query),
        k=3,
        hops=0,  # No expansion
    )
    print(f"   Found {len(results_no_expand)} results:")
    for r in results_no_expand:
        snippet = r.text[:80].replace("\n", " ")
        print(f"   - [score={r.score:.4f}] {snippet}...")

    # 4. Search WITH graph expansion
    print(f"\n[?] Search WITH graph expansion (hops=2): '{query}'")
    results_expanded = tg.search(
        query=query,
        vector=mock_embed(query),
        k=5,
        hops=2,  # Expand 2 hops
    )
    print(f"   Found {len(results_expanded)} results:")
    for r in results_expanded:
        snippet = r.text[:80].replace("\n", " ")
        hop_label = "DIRECT" if r.hop_distance == 0 else f"HOP-{r.hop_distance}"
        print(f"   - [{hop_label}] [score={r.score:.4f}] {snippet}...")

    # 5. Demonstrate delete
    if result.node_ids:
        del_id = result.node_ids[0]
        print(f"\n[x] Deleting first chunk (node {del_id})...")
        tg.delete(del_id)
        print("   Deleted. Re-searching...")
        results_after_del = tg.search(
            query=query,
            vector=mock_embed(query),
            k=5,
            hops=2,
        )
        remaining_ids = {r.id for r in results_after_del}
        assert del_id not in remaining_ids, "Deleted node should not appear!"
        print(f"   [OK] Deleted node {del_id} no longer appears in results.")

    # 6. Clean up
    tg.close()
    print("\n[OK] RAG demo complete! Data saved to ./rag_demo_data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
