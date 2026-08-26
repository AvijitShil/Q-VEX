"""Evaluation 2: Recall & Speed (Semantic Traversal on Real Data).

Loads the O'Reilly AI Engineering PDF, builds a Q-VEX graph, then runs
a grounded evaluation measuring Hit Rate @ K and Mean Reciprocal Rank (MRR).

Usage (from project root):
    python Eval/recall_and_speed.py
"""

from __future__ import annotations

import os
import random
import sys
import time

# Ensure project root + src/ are on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd

from qvex import QVEX
from Eval.utils import load_and_chunk_pdf, embed_chunks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_PATH    = os.path.join(os.path.dirname(__file__), "AI Engineering.pdf")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "eval_recall_db")
OUTPUT_CSV  = os.path.join(os.path.dirname(__file__), "speed_latency_results.csv")
MAX_CHUNKS  = 10_000

N_EVAL_QUERIES = 50  # Number of synthetic ground-truth queries to generate
K_VALUES = [1, 3, 5, 10]


def test_recall_and_latency() -> None:
    print("=" * 70)
    print("RECALL & LATENCY EVALUATION (GROUNDED HR@K & MRR)")
    print("=" * 70)

    chunks, vectors, embed_model = _build_index()
    n = len(chunks)
    dim = vectors.shape[1]

    print(f"\n[SETUP] Building Q-VEX index ({n:,} chunks, dim={dim})...")
    import shutil
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
    os.makedirs(STORAGE_DIR, exist_ok=True)
    qvex = QVEX(dim=dim, storage_dir=STORAGE_DIR, bit_width=4)

    # Store mapping of chunk index to assigned node_id
    idx_to_node_id = {}
    for i, (doc, vec) in enumerate(zip(chunks, vectors)):
        node_id = qvex.add(doc, vec)
        idx_to_node_id[i] = node_id

    print(f"  -> Index ready.\n")

    # Generate synthetic queries (first 15 words of random chunks)
    random.seed(42)
    eval_indices = random.sample(range(n), min(N_EVAL_QUERIES, n))
    
    test_queries = []
    ground_truths = []
    
    for idx in eval_indices:
        chunk_text = chunks[idx]
        words = chunk_text.split()
        # Use a short snippet as a synthetic query
        query = " ".join(words[:15]) + ("..." if len(words) > 15 else "")
        test_queries.append(query)
        ground_truths.append(idx_to_node_id[idx])

    # Embed all queries at once
    print(f"[SETUP] Embedding {len(test_queries)} synthetic queries...")
    query_vectors = embed_model.encode(
        test_queries, convert_to_numpy=True
    ).astype(np.float32)

    results = []
    hit_counts = {k: 0 for k in K_VALUES}
    mrr_sum = 0.0

    print("[RUN] Executing search queries...\n")
    for i, (query, q_vec, gt_node_id) in enumerate(zip(test_queries, query_vectors, ground_truths)):
        t0 = time.perf_counter()
        search_results = qvex.search(query, q_vec, k=max(K_VALUES), hops=2)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids = [res.id for res in search_results]
        
        # Calculate rank
        rank = None
        try:
            rank = retrieved_ids.index(gt_node_id) + 1
        except ValueError:
            pass

        # Update Hit Rates
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hit_counts[k] += 1
                
        # Update MRR
        rr = (1.0 / rank) if rank is not None else 0.0
        mrr_sum += rr

        results.append({
            "query_id": i + 1,
            "query": query,
            "latency_ms": latency_ms,
            "gt_node_id": gt_node_id,
            "rank": rank if rank is not None else -1,
            "reciprocal_rank": rr
        })

    num_queries = len(test_queries)
    
    # Calculate final metrics
    hr_metrics = {f"HR@{k}": (hit_counts[k] / num_queries) for k in K_VALUES}
    mrr = mrr_sum / num_queries
    avg_latency = sum(r["latency_ms"] for r in results) / num_queries

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Queries : {num_queries}")
    print(f"Avg Latency   : {avg_latency:.2f} ms")
    print(f"MRR           : {mrr:.4f}")
    for k in K_VALUES:
        print(f"Hit Rate @ {k:2d} : {hr_metrics[f'HR@{k}']:.4f} ({hit_counts[k]}/{num_queries})")
    
    print(f"\n[OK] Detailed results saved to: {OUTPUT_CSV}")
    qvex.close()


def _build_index():
    """Shared helper: extract PDF, embed chunks, return (chunks, vectors, model)."""
    chunks = load_and_chunk_pdf(PDF_PATH)
    n      = min(MAX_CHUNKS, len(chunks))
    chunks = chunks[:n]
    vectors, embed_model = embed_chunks(chunks)
    return chunks, vectors, embed_model


if __name__ == "__main__":
    test_recall_and_latency()
