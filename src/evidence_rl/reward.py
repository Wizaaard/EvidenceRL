"""Reward computation for aligning pre- and post-generation evidence."""

from __future__ import annotations

from typing import Iterable, Sequence

from .documents import RetrievedDocument
from .retrieval import DocumentStore, cosine_similarity, mean_vector


def _to_signed_range(score: float) -> float:
    """Scale a [0, 1] score into the [-1, 1] range."""

    return max(-1.0, min(1.0, (score * 2.0) - 1.0))


def reward_overlap(pre_docs: Iterable[RetrievedDocument], post_docs: Iterable[RetrievedDocument]) -> float:
    """Compute the fraction of overlapping document identifiers."""

    pre_ids = {doc.document.doc_id for doc in pre_docs}
    post_ids = {doc.document.doc_id for doc in post_docs}
    if not pre_ids or not post_ids:
        return 0.0
    return len(pre_ids & post_ids) / len(pre_ids)


def reward_embedding_similarity(
    store: DocumentStore,
    pre_docs: Sequence[RetrievedDocument],
    post_docs: Sequence[RetrievedDocument],
) -> float:
    """Compute cosine similarity between the mean embeddings of two evidence sets."""

    if not pre_docs or not post_docs:
        return 0.0
    pre_vectors = [store.get_document_vector(doc.document.doc_id) for doc in pre_docs]
    post_vectors = [store.get_document_vector(doc.document.doc_id) for doc in post_docs]
    pre_mean = mean_vector(pre_vectors)
    post_mean = mean_vector(post_vectors)
    return cosine_similarity(pre_mean, post_mean)


def combined_reward(
    store: DocumentStore,
    pre_docs: Sequence[RetrievedDocument],
    post_docs: Sequence[RetrievedDocument],
    weight_overlap: float = 0.5,
) -> float:
    """Combine identifier overlap with embedding similarity into a single reward."""

    if not 0.0 <= weight_overlap <= 1.0:
        raise ValueError("weight_overlap must be within [0, 1]")

    overlap = _to_signed_range(reward_overlap(pre_docs, post_docs))
    embedding = _to_signed_range(reward_embedding_similarity(store, pre_docs, post_docs))
    combined = weight_overlap * overlap + (1.0 - weight_overlap) * embedding
    return max(-1.0, min(1.0, combined))


__all__ = [
    "reward_overlap",
    "reward_embedding_similarity",
    "combined_reward",
    "_to_signed_range",
]
