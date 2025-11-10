"""Core data structures used across the EvidenceRL package.

This module provides lightweight containers for representing text documents and
retrieval results.  Using dataclasses keeps the code explicit while providing a
clear contract for other modules that operate on documents and retrieval hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Document:
    """Container for a text document in the corpus.

    Attributes
    ----------
    doc_id:
        Unique identifier for the document.  Identifiers are opaque strings but
        typically correspond to a filename or database key.
    text:
        The textual content of the document.
    metadata:
        Optional arbitrary metadata that may be useful for inspection or
        downstream processing.
    """

    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RetrievedDocument:
    """Represents a document returned by the retriever.

    In addition to the :class:`Document`, we store the similarity score that was
    used to rank the document.  Keeping the score makes downstream analysis and
    debugging easier.
    """

    document: Document
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be a numeric value")


__all__ = ["Document", "RetrievedDocument"]
