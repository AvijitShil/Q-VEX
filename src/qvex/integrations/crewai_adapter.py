# File: src/qvex/integrations/crewai_adapter.py
from typing import Any, Dict, List, Optional
import numpy as np

class QVEXCrewAIMemoryStorage:
    """CrewAI Working Memory Storage Provider for Q-VEX."""

    def __init__(self, qvex_instance: Any, embed_fn: Optional[Any] = None):
        self.qvex = qvex_instance
        self.embed_fn = embed_fn

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        text = str(value)
        meta = metadata or {}
        if self.embed_fn:
            vector = np.array(self.embed_fn(text), dtype=np.float32)
        else:
            vector = np.zeros(self.qvex._dim, dtype=np.float32)
        self.qvex.add(text=text, vector=vector, metadata=meta)

    def search(
        self, query: str, limit: int = 3, score_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        if self.embed_fn:
            query_vec = np.array(self.embed_fn(query), dtype=np.float32)
        else:
            query_vec = np.zeros(self.qvex._dim, dtype=np.float32)

        results = self.qvex.search(query=query, vector=query_vec, k=limit, hops=2)
        
        output = []
        for r in results:
            if r.score >= score_threshold:
                output.append({
                    "id": r.id,
                    "context": r.text,
                    "metadata": r.metadata,
                    "score": float(r.score)
                })
        return output

    def reset(self) -> None:
        if hasattr(self.qvex, "clear"):
            self.qvex.clear()
