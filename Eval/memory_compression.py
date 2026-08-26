"""Evaluation 1: Memory Compression (Disk-Based).

Measures actual on-disk size of SQLite + TurboVec files vs raw float32 baseline.
This proves Q-VEX's quantized storage compression with bulletproof disk metrics.

Usage (from project root):
    python Eval/memory_compression.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path so `from qvex import QVEX` works
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Also add src/ so `qvex` package is found
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd

from qvex import QVEX
from Eval.utils import load_and_chunk_pdf, embed_chunks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(os.path.dirname(__file__), "AI Engineering.pdf")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "memory_compression_results.csv")
MAX_CHUNKS = 10_000  # cap to keep benchmark fast


def get_disk_size_mb(directory: str) -> float:
    """Return total size of all files under *directory* in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            fp = os.path.join(dirpath, fname)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def get_file_size_mb(path: str) -> float:
    """Return size of a single file in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def test_memory_compression() -> None:
    print("=" * 70)
    print("MEMORY COMPRESSION EVALUATION (DISK-BASED)")
    print("=" * 70)

    # Measure the actual source PDF size
    book_size_mb = get_file_size_mb(PDF_PATH)
    print(f"\n[SOURCE] PDF on disk: {book_size_mb:.2f} MB  ({PDF_PATH})")

    chunks = load_and_chunk_pdf(PDF_PATH)
    vectors, _ = embed_chunks(chunks)

    # Cap corpus size
    n = min(MAX_CHUNKS, len(chunks))
    chunks = chunks[:n]
    vectors = vectors[:n]

    dim = vectors.shape[1]  # 384 for all-MiniLM-L6-v2
    float32_mb = (n * dim * 4) / (1024 * 1024)

    print(f"\n[CORPUS]  {n:,} chunks | embedding dim={dim}")
    print(f"  Raw float32 vectors baseline : {float32_mb:.2f} MB  ({n} x {dim} x 4 bytes)")
    print(f"  Source PDF on disk           : {book_size_mb:.2f} MB")
    print()

    results = []

    for bw in [4, 2]:
        storage_dir = os.path.join(
            os.path.dirname(__file__), f"eval_mem_db_{bw}bit"
        )
        import shutil
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)

        print(f"[{bw}-bit] Ingesting {n:,} documents into QVEX...")
        qvex = QVEX(dim=dim, storage_dir=storage_dir, bit_width=bw)

        for doc, vec in zip(chunks, vectors):
            qvex.add(doc, vec)

        # Flush to disk
        qvex.save()
        qvex.close()

        disk_mb           = round(get_disk_size_mb(storage_dir), 3)
        vs_float32        = round(float32_mb / disk_mb, 2) if disk_mb > 0 else 0
        vs_book           = round(book_size_mb / disk_mb, 2) if disk_mb > 0 else 0
        savings_mb        = round(float32_mb - disk_mb, 3)
        savings_pct       = round(100 * savings_mb / float32_mb, 1) if float32_mb > 0 else 0
        index_per_chunk_kb = round(disk_mb * 1024 / n, 3) if n > 0 else 0

        print(f"  -> Q-VEX disk size           : {disk_mb:.3f} MB")
        print(f"  -> vs raw float32            : {vs_float32:.2f}x smaller")
        print(f"  -> vs source PDF             : {vs_book:.2f}x smaller")
        print(f"  -> Savings over float32      : {savings_mb:.3f} MB ({savings_pct}%)")
        print(f"  -> Storage per chunk         : {index_per_chunk_kb:.3f} KB")
        print()

        results.append({
            "bit_width"           : bw,
            "corpus_chunks"       : n,
            "embedding_dim"       : dim,
            "source_pdf_mb"       : round(book_size_mb, 3),
            "raw_float32_mb"      : round(float32_mb, 3),
            "qvex_disk_mb"        : disk_mb,
            "compression_vs_float32" : vs_float32,
            "compression_vs_pdf"  : vs_book,
            "savings_mb"          : savings_mb,
            "savings_pct"         : savings_pct,
            "storage_per_chunk_kb": index_per_chunk_kb,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Results saved to: {OUTPUT_CSV}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    test_memory_compression()
