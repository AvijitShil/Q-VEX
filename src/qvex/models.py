"""Pydantic models for Q-VEX data types and search results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NodeData(BaseModel):
    """Represents a node stored in the graph database."""

    id: int
    text: str
    vector_idx: int | None = None
    metadata: dict[str, Any] | None = None
    is_deleted: bool = False
    created_at: str | None = None


class EdgeData(BaseModel):
    """Represents an edge (relationship) between two nodes."""

    source: int
    target: int
    edge_type: str = "related"
    confidence: float = 1.0


class SearchResult(BaseModel):
    """A single result from a hybrid search query."""

    id: int
    text: str
    score: float = Field(description="Combined relevance score from vector similarity.")
    metadata: dict[str, Any] | None = None
    hop_distance: int = Field(
        default=0,
        description="How many hops away this result is from the BM25 seed set.",
    )


class IngestResult(BaseModel):
    """Summary of a batch ingestion operation."""

    node_ids: list[int] = Field(default_factory=list)
    edge_count: int = 0
    chunk_count: int = 0
