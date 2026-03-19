"""Utilities for indexing documents and performing retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Sequence, Set, Union

from .documents import Document, RetrievedDocument

TokenVector = Union[Dict[str, float], List[float]]
_WORD_RE = re.compile(r"\b\w+\b")


class TextEmbedder(Protocol):
    """Protocol for text embedders used by the semantic retriever."""

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        ...


def _normalize_concepts(concepts: Iterable[str] | None) -> Set[str]:
    if not concepts:
        return set()
    return {concept.strip().lower() for concept in concepts if concept and concept.strip()}


def _chunk_text(text: str, chunk_size: int = 320, overlap: int = 64) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _normalize(vector: TokenVector) -> TokenVector:
    if isinstance(vector, dict):
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0.0:
            return vector
        return {term: value / norm for term, value in vector.items()}
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


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
    if isinstance(vec_a, dict) and isinstance(vec_b, dict):
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) < len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())

    arr_a = vec_a if not isinstance(vec_a, dict) else []
    arr_b = vec_b if not isinstance(vec_b, dict) else []
    if not arr_a or not arr_b:
        return 0.0
    dot = sum(a * b for a, b in zip(arr_a, arr_b))
    denom = math.sqrt(sum(a * a for a in arr_a)) * math.sqrt(sum(b * b for b in arr_b))
    if denom == 0.0:
        return 0.0
    return float(dot / denom)


def mean_vector(vectors: Iterable[TokenVector]) -> TokenVector:
    vectors = list(vectors)
    if not vectors:
        return {}
    if isinstance(vectors[0], dict):
        accumulator: Dict[str, float] = defaultdict(float)
        for vector in vectors:  # type: ignore[assignment]
            for term, value in vector.items():
                accumulator[term] += value
        mean = {term: value / len(vectors) for term, value in accumulator.items()}
        return _normalize(mean)

    length = len(vectors[0])
    sums = [0.0] * length
    for vector in vectors:
        for idx, value in enumerate(vector):
            sums[idx] += value
    mean = [value / len(vectors) for value in sums]
    return _normalize(mean)


@dataclass
class DocumentStore:
    """In-memory document store backed by dense vectors or TF-IDF."""

    documents: Sequence[Document]
    embedder: "TextEmbedder | None" = None
    chunk_size: int = 320
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("DocumentStore requires at least one document")

        expanded_docs: List[Document] = []
        for doc in self.documents:
            chunks = _chunk_text(doc.text, self.chunk_size, self.chunk_overlap) or [doc.text]
            for idx, chunk in enumerate(chunks):
                metadata = dict(doc.metadata) if doc.metadata else {}
                metadata.setdefault("source_doc_id", doc.doc_id)
                metadata["chunk_id"] = idx
                chunk_doc_id = doc.doc_id if len(chunks) == 1 else f"{doc.doc_id}::chunk-{idx}"
                expanded_docs.append(
                    Document(
                        doc_id=chunk_doc_id,
                        text=chunk,
                        metadata=metadata,
                    )
                )

        self.documents = expanded_docs

        self._concepts_by_index: List[Set[str]] = [
            _normalize_concepts(doc.metadata.get("concepts") if doc.metadata else None)
            for doc in self.documents
        ]
        self._concept_vocabulary: Set[str] = set().union(*self._concepts_by_index)
        self._has_concepts = any(self._concepts_by_index)
        self._id_to_index = {doc.doc_id: idx for idx, doc in enumerate(self.documents)}

        if self.embedder is None:
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
            self._mode = "tfidf"
        else:
            embeddings = self.embedder.encode([doc.text for doc in self.documents])
            self._doc_vectors = [_normalize(vector) for vector in embeddings]
            self._idf = {}
            self._default_idf = 0.0
            self._mode = "dense"

    def vectorize(self, text: str) -> TokenVector:
        if self.embedder is not None:
            return _normalize(self.embedder.encode([text])[0])
        tokens = _tokenize(text)
        counter = Counter(tokens)
        return _tfidf(counter, self._idf, self._default_idf)

    def extract_concepts(self, text: str) -> Set[str]:
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



class HuggingFaceEmbedder:
    """Embedding helper that uses a Hugging Face encoder model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer=None,
        model=None,
        batch_size: int = 32,
        force_cpu: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        # Load tokenizer with trust_remote_code=False to avoid optional API calls
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=False,
                )
            except Exception as e:
                # Fallback: try with use_fast=False if the above fails
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=False,
                )
        else:
            self.tokenizer = tokenizer
        self.batch_size = batch_size
        device = "cpu"
        device_map = None
        if not force_cpu and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device_map = "auto"
            else:
                device = "cuda:0"

        model_kwargs = {"device_map": device_map} if device_map else {}
        self.model = model or AutoModel.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.model = self.model.to(device)
        self.device = device
        self._device_map = device_map
        self._torch = torch

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> List[List[float]]:
        torch = self._torch
        texts_list = list(texts)
        if batch_size is None:
            batch_size = self.batch_size

        # Process in batches to avoid OOM errors
        all_embeddings = []
        for i in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[i:i + batch_size]

            encoded_inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            # Always move inputs to the same device as the model
            model_device = next(self.model.parameters()).device
            encoded_inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in encoded_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                hidden_states = outputs.last_hidden_state
                mask = encoded_inputs["attention_mask"].unsqueeze(-1)
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
                batch_embeddings = [vector.cpu().tolist() for vector in pooled]
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

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


class SemanticRetriever:
    """Retrieve the top-k documents most similar to a query using dense vectors."""

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


TfidfRetriever = SemanticRetriever


__all__ = [
    "DocumentStore",
    "SemanticRetriever",
    "TfidfRetriever",
    "HuggingFaceEmbedder",
    "TextEmbedder",
    "cosine_similarity",
    "mean_vector",
    "TokenVector",
]
