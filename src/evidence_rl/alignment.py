"""Identify which evidence appears most relevant to a generated answer."""

from __future__ import annotations

from typing import Iterable, List

from .documents import RetrievedDocument
from .retrieval import DocumentStore, cosine_similarity


class EvidenceAligner:
    """Select evidence passages that best match the generated output."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def align(
        self,
        generated_text: str,
        candidate_documents: Iterable[RetrievedDocument],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """Return the top-k documents that align with the generated text."""

        docs = list(candidate_documents)
        if not docs:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        generated_vector = self.store.vectorize(generated_text)
        scores = [cosine_similarity(generated_vector, self.store.get_document_vector(doc.document.doc_id)) for doc in docs]
        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [RetrievedDocument(doc.document, float(score)) for doc, score in ranked]


__all__ = ["EvidenceAligner"]
