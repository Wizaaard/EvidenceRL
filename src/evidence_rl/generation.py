"""Simple response generation over retrieved evidence.

This module intentionally keeps the generator lightweight.  The goal is to show
how a response can be composed from retrieved evidence while tracking which
pieces were used.  For a production system you would plug in a proper language
model here.
"""

from __future__ import annotations

from typing import Iterable, List

from .documents import RetrievedDocument


class EvidenceConcatenationGenerator:
    """Generate an answer by concatenating snippets from retrieved evidence."""

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        self.max_sentences = max_sentences

    def _take_sentences(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return '. '.join(sentences[: self.max_sentences]) + ('.' if sentences else '')

    def generate(self, query: str, retrieved: Iterable[RetrievedDocument]) -> str:
        """Return a short answer built from the retrieved evidence."""

        snippets: List[str] = []
        for doc in retrieved:
            snippet = self._take_sentences(doc.document.text)
            if snippet:
                snippets.append(snippet)
        if not snippets:
            return f"I could not find evidence for: {query}"
        return f"Answer: {' '.join(snippets)}"


__all__ = ["EvidenceConcatenationGenerator"]
