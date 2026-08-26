"""Shared utilities for Q-VEX evaluation scripts.

Provides:
- PDF text extraction via PyMuPDF
- LangChain RecursiveCharacterTextSplitter for chunking
- Sentence-transformers embedding (all-MiniLM-L6-v2, dim=384)
"""

from __future__ import annotations

import os

import numpy as np


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[str]:
    """Extract text from a PDF and split it into overlapping chunks.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    chunk_size : int
        Maximum characters per chunk (default: 512).
    chunk_overlap : int
        Overlap characters between consecutive chunks (default: 50).

    Returns
    -------
    list[str]
        List of text chunks.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install pymupdf"
        ) from exc

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "LangChain text splitters not installed. Run: pip install langchain-text-splitters"
        ) from exc

    print(f"[EXTRACT] Reading PDF: {pdf_path}")
    text = ""
    n_pages = 0
    with fitz.open(pdf_path) as pdf:
        n_pages = len(pdf)
        for page in pdf:
            text += page.get_text()

    print(f"  -> Extracted {len(text):,} characters from {n_pages} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f"  -> Generated {len(chunks):,} text chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def embed_chunks(chunks: list[str]):
    """Embed text chunks using all-MiniLM-L6-v2 (dim=384, RAM-friendly).

    Parameters
    ----------
    chunks : list[str]
        List of text strings to embed.

    Returns
    -------
    tuple[np.ndarray, SentenceTransformer]
        (vectors of shape (N, 384), the loaded model for reuse)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        ) from exc

    print("[EMBED] Loading sentence-transformers model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  -> Embedding {len(chunks):,} chunks (this may take a moment)...")
    vectors = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)
    print(f"  -> Done. Vector shape: {vectors.shape}")
    return vectors, model
