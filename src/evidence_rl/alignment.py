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
        candidate_documents: Iterable[RetrievedDocument] | None,
        top_k: int,
    ) -> List[RetrievedDocument]:
        """Return the top-k documents that align with the generated text."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        generated_vector = self.store.vectorize(generated_text)
        if candidate_documents is not None:
            docs = list(candidate_documents)
            if not docs:
                return []
            scored = [
                (
                    retrieved.document,
                    cosine_similarity(
                        generated_vector, self.store.get_document_vector(retrieved.document.doc_id)
                    ),
                )
                for retrieved in docs
            ]
        else:
            scored = [
                (document, cosine_similarity(generated_vector, self.store.get_document_vector(document.doc_id)))
                for document in self.store.documents
            ]

        if not scored:
            return []

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return [RetrievedDocument(document, float(score)) for document, score in ranked]


__all__ = ["EvidenceAligner"]
