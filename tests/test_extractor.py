"""Tests for the extractor module."""

from __future__ import annotations

import pytest

from qvex.extractor.base import BaseExtractor


class DummyExtractor(BaseExtractor):
    """A trivial extractor for testing the protocol."""

    def extract_edges(self, text: str):
        words = text.split()
        edges = []
        for i in range(len(words) - 1):
            edges.append((words[i], "next_word", words[i + 1]))
        return edges


class TestBaseExtractor:
    """BaseExtractor protocol tests."""

    def test_dummy_implements_protocol(self) -> None:
        ext = DummyExtractor()
        edges = ext.extract_edges("hello world foo")
        assert len(edges) == 2
        assert edges[0] == ("hello", "next_word", "world")
        assert edges[1] == ("world", "next_word", "foo")

    def test_empty_text(self) -> None:
        ext = DummyExtractor()
        assert ext.extract_edges("") == []

    def test_single_word(self) -> None:
        ext = DummyExtractor()
        assert ext.extract_edges("alone") == []
