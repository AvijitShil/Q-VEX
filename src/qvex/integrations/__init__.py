"""Q-VEX framework integration adapters."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy imports to avoid pulling in heavy framework dependencies."""
    if name == "QVEXLlamaIndexStore":
        from qvex.integrations.llamaindex_adapter import QVEXLlamaIndexStore

        return QVEXLlamaIndexStore

    if name == "QVEXLangChainVectorStore":
        from qvex.integrations.langchain_adapter import QVEXLangChainVectorStore

        return QVEXLangChainVectorStore

    if name == "QVEXCrewAIMemoryStorage":
        from qvex.integrations.crewai_adapter import QVEXCrewAIMemoryStorage
        return QVEXCrewAIMemoryStorage

    if name == "QVEXSemanticMemorySaver":
        from qvex.integrations.langgraph_adapter import QVEXSemanticMemorySaver
        return QVEXSemanticMemorySaver

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QVEXLlamaIndexStore", 
    "QVEXLangChainVectorStore",
    "QVEXCrewAIMemoryStorage",
    "QVEXSemanticMemorySaver"
]
