"""Utilities for indexing documents and performing retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .documents import Document, RetrievedDocument

TokenVector = Dict[str, float]
_WORD_RE = re.compile(r"\b\w+\b")


def _normalize_concepts(concepts: Iterable[str] | None) -> Set[str]:
    if not concepts:
        return set()
    return {concept.strip().lower() for concept in concepts if concept and concept.strip()}


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _normalize(vector: TokenVector) -> TokenVector:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return vector
    return {term: value / norm for term, value in vector.items()}


def _tfidf(counter: Counter[str], idf: Dict[str, float], default_idf: float) -> TokenVector:
    total = sum(counter.values())
    if total == 0:
        return {}
    vector = {
        term: (count / total) * idf.get(term, default_idf)
        for term, count in counter.items()
    }
    return _normalize(vector)


def cosine_similarity(vec_a: TokenVector, vec_b: TokenVector) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) < len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())


def mean_vector(vectors: Iterable[TokenVector]) -> TokenVector:
    vectors = list(vectors)
    if not vectors:
        return {}
    accumulator: Dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, value in vector.items():
            accumulator[term] += value
    mean = {term: value / len(vectors) for term, value in accumulator.items()}
    return _normalize(mean)


@dataclass
class DocumentStore:
    """In-memory document store backed by a simple TF-IDF index."""

    documents: Sequence[Document]

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("DocumentStore requires at least one document")

        tokenized = [_tokenize(doc.text) for doc in self.documents]
        document_frequencies: Dict[str, int] = defaultdict(int)
        self._term_frequencies: List[Counter[str]] = []
        for tokens in tokenized:
            counter = Counter(tokens)
            self._term_frequencies.append(counter)
            for term in counter:
                document_frequencies[term] += 1

        self._idf = {
            term: math.log((1 + len(self.documents)) / (1 + df)) + 1.0
            for term, df in document_frequencies.items()
        }
        self._default_idf = math.log(1 + len(self.documents)) + 1.0
        self._doc_vectors = [
            _tfidf(counter, self._idf, self._default_idf) for counter in self._term_frequencies
        ]
        self._id_to_index = {doc.doc_id: idx for idx, doc in enumerate(self.documents)}
        self._concepts_by_index: List[Set[str]] = [
            _normalize_concepts(doc.metadata.get("concepts") if doc.metadata else None)
            for doc in self.documents
        ]
        self._concept_vocabulary: Set[str] = set().union(*self._concepts_by_index)
        self._has_concepts = any(self._concepts_by_index)

    def vectorize(self, text: str) -> TokenVector:
        tokens = _tokenize(text)
        counter = Counter(tokens)
        return _tfidf(counter, self._idf, self._default_idf)

    def extract_concepts(self, text: str) -> Set[str]:
        """Return the subset of known concepts mentioned in ``text``.

        The detection is intentionally lightweight: if a concept string from the corpus
        vocabulary appears as a substring in the lowercased text, it is included. When
        no concepts are known, this returns an empty set without constraining retrieval.
        """

        if not self._concept_vocabulary:
            return set()

        lowered = text.lower()
        return {concept for concept in self._concept_vocabulary if concept in lowered}

    def document_concepts(self, doc_id: str) -> Set[str]:
        if doc_id not in self._id_to_index:
            raise KeyError(f"Unknown document id: {doc_id}")
        return self._concepts_by_index[self._id_to_index[doc_id]]

    def concepts_for_index(self, index: int) -> Set[str]:
        if index < 0 or index >= len(self._concepts_by_index):
            raise IndexError("Document index out of range")
        return self._concepts_by_index[index]

    def get_document_vector(self, doc_id: str) -> TokenVector:
        if doc_id not in self._id_to_index:
            raise KeyError(f"Unknown document id: {doc_id}")
        return self._doc_vectors[self._id_to_index[doc_id]]

    def document_vectors(self, doc_ids: Iterable[str]) -> List[TokenVector]:
        vectors = [self.get_document_vector(doc_id) for doc_id in doc_ids]
        if not vectors:
            raise ValueError("No document identifiers provided")
        return vectors

    def as_retrieved(self, indices: List[int], scores: List[float]) -> List[RetrievedDocument]:
        return [RetrievedDocument(self.documents[idx], score) for idx, score in zip(indices, scores)]


class TfidfRetriever:
    """Retrieve the top-k documents most similar to a query."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.store.vectorize(query)
        query_concepts = self.store.extract_concepts(query)
        scores: List[float] = []
        for idx, doc_vec in enumerate(self.store._doc_vectors):
            doc_concepts = self.store.concepts_for_index(idx)
            if self.store._has_concepts and query_concepts:
                if not doc_concepts or doc_concepts.isdisjoint(query_concepts):
                    scores.append(float("-inf"))
                    continue
            scores.append(cosine_similarity(query_vector, doc_vec))

        ranked = [item for item in enumerate(scores) if math.isfinite(item[1])]
        if not ranked:
            return []

        ranked = sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]
        indices = [idx for idx, _ in ranked]
        top_scores = [float(score) for _, score in ranked]
        return self.store.as_retrieved(indices, top_scores)


__all__ = ["DocumentStore", "TfidfRetriever", "cosine_similarity", "mean_vector", "TokenVector"]
