"""Q-VEX Quickstart -- minimal usage example.

Run this script after installing turbograph:

    pip install -e .
    python examples/quickstart.py
"""

import numpy as np

from qvex import QVEX

DIM = 64  # Use a small dimension for this demo


def mock_embed(text: str) -> np.ndarray:
    """Simple deterministic mock embedding (replace with a real model)."""
    rng = np.random.default_rng(hash(text) % (2**31))
    vec = rng.random(DIM, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def main() -> None:
    # 1. Create a TurboGraph instance
    print("[*] Creating QVEX...")
    tg = QVEX(dim=DIM, storage_dir="./quickstart_data", bit_width=4)

    # 2. Add some documents
    docs = [
        "Transformers use self-attention for sequence modeling.",
        "Convolutional networks excel at image recognition tasks.",
        "Graph neural networks operate on non-Euclidean data structures.",
        "Recurrent networks process sequential data with hidden states.",
        "Attention mechanisms allow models to focus on relevant parts.",
    ]

    doc_ids = []
    for doc in docs:
        doc_id = tg.add(doc, mock_embed(doc))
        doc_ids.append(doc_id)
        print(f"  Added doc {doc_id}: {doc[:50]}...")

    # 3. Add some graph edges
    tg.add_edge(doc_ids[0], doc_ids[4], edge_type="related")  # transformers <-> attention
    tg.add_edge(doc_ids[2], doc_ids[0], edge_type="related")  # GNN <-> transformers
    print(f"\n[+] Added 2 edges between related documents.")

    # 4. Search!
    query = "attention mechanism"
    print(f"\n[?] Searching for: '{query}'")
    results = tg.search(
        query=query,
        vector=mock_embed(query),
        k=3,
        hops=1,
    )

    print(f"\n[=] Top {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [score={r.score:.4f}] (hop={r.hop_distance}) {r.text[:60]}...")

    # 5. Save and close
    tg.save()
    tg.close()
    print("\n[OK] Done! Data saved to ./quickstart_data/")


if __name__ == "__main__":
    main()
