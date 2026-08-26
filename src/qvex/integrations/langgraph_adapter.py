# File: src/qvex/integrations/langgraph_adapter.py
from typing import Any, Callable, Dict, List, Optional
import numpy as np

try:
    from langchain_core.tools import tool
except ImportError:
    # Dummy decorator if langchain is not installed
    def tool(name=None, description=None):
        def decorator(func):
            func.name = name
            func.description = description
            return func
        return decorator

class QVEXSemanticMemorySaver:
    """LangGraph Semantic Memory Saver for long-term episodic memory extraction."""
    
    def __init__(self, qvex_instance: Any, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        self.qvex = qvex_instance
        self.embed_fn = embed_fn

    def save_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Saves a string of episodic memory."""
        meta = metadata or {}
        if self.embed_fn:
            vector = np.asarray(self.embed_fn(content), dtype=np.float32)
        else:
            vector = np.zeros(self.qvex._dim, dtype=np.float32)
        return self.qvex.add(text=content, vector=vector, metadata=meta)

    def retrieve_memory(self, query: str, k: int = 3, hops: int = 2) -> List[Dict[str, Any]]:
        """Retrieves episodic memories related to a query."""
        if self.embed_fn:
            query_vec = np.asarray(self.embed_fn(query), dtype=np.float32)
        else:
            query_vec = np.zeros(self.qvex._dim, dtype=np.float32)
            
        results = self.qvex.search(query=query, vector=query_vec, k=k, hops=hops)
        return [{"id": r.id, "content": r.text, "metadata": r.metadata, "score": r.score} for r in results]

    def as_retriever_tool(
        self, 
        name: str = "memory_retrieval", 
        description: str = "Search long-term semantic episodic memory for related facts or past context."
    ) -> Callable:
        """Returns a LangGraph/LangChain compatible Tool for querying this memory."""
        
        # We define a function closure and wrap it with the tool decorator.
        # This will be recognized as a valid Tool by langgraph agents.
        
        @tool(name=name, description=description)
        def memory_tool(query: str) -> str:
            results = self.retrieve_memory(query)
            if not results:
                return "No relevant memories found."
            
            # Combine retrieved contexts into a string for the agent
            return "\n\n".join([f"Memory (Score {r['score']:.2f}): {r['content']}" for r in results])
            
        return memory_tool
