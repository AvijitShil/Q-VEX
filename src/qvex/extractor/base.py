"""Abstract base class for entity/relation extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseExtractor(ABC):
    """Protocol for extracting entities and relationships from text.

    Subclasses must implement ``extract_edges`` which takes raw text and
    returns a list of ``(source_entity, relation, target_entity)`` triples.
    """

    @abstractmethod
    def extract_edges(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entity-relation triples from *text*.

        Parameters
        ----------
        text : str
            Raw text to extract entities and relationships from.

        Returns
        -------
        list[tuple[str, str, str]]
            A list of ``(source_entity, relation_type, target_entity)`` triples.
        """
        ...
