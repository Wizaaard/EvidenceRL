"""FAISS-based vector retrieval for efficient similarity search at scale."""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .documents import Document, RetrievedDocument
from .retrieval import TextEmbedder, _normalize_concepts


class FAISSDocumentStore:
    """Efficient document store using FAISS for fast approximate nearest neighbor search.

    Benefits over exhaustive search:
    - 10-100x faster retrieval for large document collections
    - Sub-linear search complexity (vs O(n) exhaustive)
    - Lower memory footprint with quantization options
    - Scales to millions/billions of documents
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embedder: TextEmbedder,
        index_type: str = "IVF",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
    ) -> None:
        """Initialize FAISS-based document store.

        Args:
            documents: Documents to index
            embedder: Text embedder for generating vectors
            index_type: FAISS index type - "Flat" (exact, slow), "IVF" (fast, approximate), "HNSW" (fastest)
            nlist: Number of clusters for IVF index (more = slower build, faster search)
            nprobe: Number of clusters to search (more = slower but more accurate)
            use_gpu: Whether to use GPU acceleration for FAISS
        """
        try:
            import faiss
        except ImportError:
            raise RuntimeError(
                "FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )

        self.documents = list(documents)
        self.embedder = embedder
        self.use_gpu = use_gpu
        self._faiss = faiss

        # Extract concepts
        self._concepts_by_index: List[set] = [
            _normalize_concepts(doc.metadata.get("concepts") if doc.metadata else None)
            for doc in self.documents
        ]
        self._concept_vocabulary = set().union(*self._concepts_by_index) if self._concepts_by_index else set()
        self._has_concepts = any(self._concepts_by_index)
        self._id_to_index = {doc.doc_id: idx for idx, doc in enumerate(self.documents)}

        # Generate embeddings in batches
        print(f"Generating embeddings for {len(self.documents)} documents...")
        embeddings = self.embedder.encode([doc.text for doc in self.documents])
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)

        self.dimension = embeddings_np.shape[1]
        self.num_docs = len(self.documents)

        # Build FAISS index
        print(f"Building FAISS index (type={index_type})...")
        if index_type == "Flat":
            # Exact search (slow but accurate)
            index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine after normalization
        elif index_type == "IVF":
            # Inverted file index (fast approximate search)
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # Train the index
            print(f"Training IVF index with {len(embeddings_np)} vectors...")
            index.train(embeddings_np)
            index.nprobe = nprobe
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast, good accuracy)
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type: {index_type}. Use 'Flat', 'IVF', or 'HNSW'")

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add vectors to index
        index.add(embeddings_np)
        self.index = index
        print(f"FAISS index built with {self.num_docs} documents")

    def vectorize(self, text: str) -> np.ndarray:
        """Convert text to normalized embedding vector."""
        embedding = self.embedder.encode([text])[0]
        embedding_np = np.array([embedding], dtype=np.float32)
        self._faiss.normalize_L2(embedding_np)
        return embedding_np

    def extract_concepts(self, text: str) -> set:
        """Extract concepts mentioned in text."""
        if not self._concept_vocabulary:
            return set()
        lowered = text.lower()
        return {concept for concept in self._concept_vocabulary if concept in lowered}

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Search for most similar documents using FAISS.

        This is much faster than exhaustive search, especially for large collections.
        Complexity: O(log n) for IVF/HNSW vs O(n) for exhaustive
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Get query vector
        query_vector = self.vectorize(query)

        # Handle concept filtering
        query_concepts = self.extract_concepts(query)
        if self._has_concepts and query_concepts:
            # Filter to documents with overlapping concepts
            valid_indices = [
                idx for idx in range(len(self.documents))
                if self._concepts_by_index[idx] and not self._concepts_by_index[idx].isdisjoint(query_concepts)
            ]
            if not valid_indices:
                return []

            # Search only valid documents (requires index reconstruction - expensive)
            # For efficiency, we do a larger search and filter afterward
            search_k = min(top_k * 10, self.num_docs)
            distances, indices = self.index.search(query_vector, search_k)

            # Filter and re-rank
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                if idx in valid_indices:
                    results.append((idx, float(dist)))
                if len(results) >= top_k:
                    break
        else:
            # No concept filtering - direct FAISS search
            distances, indices = self.index.search(query_vector, top_k)
            results = [(int(idx), float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx >= 0]

        # Convert to RetrievedDocument
        retrieved = [
            RetrievedDocument(self.documents[idx], score)
            for idx, score in results
        ]
        return retrieved[:top_k]

    def get_document_vector(self, doc_id: str) -> List[float]:
        """Get the embedding vector for a document by its ID.

        This method is required for reward calculation compatibility.
        """
        # Find document index by ID
        for idx, doc in enumerate(self.documents):
            if doc.doc_id == doc_id:
                # Reconstruct vector from FAISS index
                vector = self.index.reconstruct(idx)
                return vector.tolist()

        raise KeyError(f"Document with id '{doc_id}' not found")

    def document_vectors(self, doc_ids: List[str]) -> List[List[float]]:
        """Get embedding vectors for multiple documents.

        This method is required for reward calculation compatibility.
        """
        return [self.get_document_vector(doc_id) for doc_id in doc_ids]

    def as_retrieved(self, indices: List[int], scores: List[float]) -> List[RetrievedDocument]:
        """Convert indices and scores to RetrievedDocument objects.

        This method is required for compatibility with DocumentStore interface.
        """
        return [
            RetrievedDocument(self.documents[idx], score)
            for idx, score in zip(indices, scores)
            if 0 <= idx < len(self.documents)
        ]

    def save(self, directory: Union[str, Path]) -> None:
        """Save the FAISS index and documents to disk.

        Args:
            directory: Directory path to save the index and metadata.
                       Will be created if it doesn't exist.

        Files created:
            - index.faiss: The FAISS index
            - documents.pkl: Pickled document list
            - metadata.json: Index configuration and statistics
        """
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (convert from GPU if necessary)
        index_path = directory / "index.faiss"
        index_to_save = self.index
        if self.use_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            # Convert GPU index to CPU for saving
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(index_to_save, str(index_path))
        print(f"Saved FAISS index to {index_path}")

        # Save documents
        docs_path = directory / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Saved {len(self.documents)} documents to {docs_path}")

        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "num_docs": self.num_docs,
            "has_concepts": self._has_concepts,
            "use_gpu": self.use_gpu,
        }
        metadata_path = directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        embedder: TextEmbedder,
        use_gpu: bool = False,
    ) -> "FAISSDocumentStore":
        """Load a pre-built FAISS index from disk.

        Args:
            directory: Directory containing saved index files.
            embedder: Text embedder (needed for query encoding, not for index loading).
            use_gpu: Whether to move index to GPU after loading.

        Returns:
            FAISSDocumentStore instance with pre-built index.

        Raises:
            FileNotFoundError: If required files are missing.
            RuntimeError: If FAISS is not installed.
        """
        try:
            import faiss
        except ImportError:
            raise RuntimeError(
                "FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )

        directory = Path(directory)

        # Check required files exist
        index_path = directory / "index.faiss"
        docs_path = directory / "documents.pkl"
        metadata_path = directory / "metadata.json"

        for path in [index_path, docs_path, metadata_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        print(f"Loading FAISS index from {directory}...")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load documents
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
        print(f"Loaded {len(documents)} documents")

        # Load FAISS index
        index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index with dimension={metadata['dimension']}")

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.documents = documents
        instance.embedder = embedder
        instance.use_gpu = use_gpu
        instance._faiss = faiss
        instance.index = index
        instance.dimension = metadata["dimension"]
        instance.num_docs = metadata["num_docs"]

        # Rebuild concept structures
        instance._concepts_by_index = [
            _normalize_concepts(doc.metadata.get("concepts") if doc.metadata else None)
            for doc in documents
        ]
        instance._concept_vocabulary = (
            set().union(*instance._concepts_by_index) if instance._concepts_by_index else set()
        )
        instance._has_concepts = metadata.get("has_concepts", any(instance._concepts_by_index))
        instance._id_to_index = {doc.doc_id: idx for idx, doc in enumerate(documents)}

        print(f"FAISS index loaded successfully ({instance.num_docs} documents)")
        return instance

    @staticmethod
    def index_exists(directory: Union[str, Path]) -> bool:
        """Check if a pre-built index exists at the given directory.

        Args:
            directory: Directory to check.

        Returns:
            True if all required index files exist.
        """
        directory = Path(directory)
        required_files = ["index.faiss", "documents.pkl", "metadata.json"]
        return all((directory / f).exists() for f in required_files)


class FAISSRetriever:
    """Retriever using FAISS for fast similarity search."""

    def __init__(self, store: FAISSDocumentStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Retrieve top-k most similar documents using fast FAISS search."""
        return self.store.search(query, top_k)


# Performance comparison for reference:
#
# Exhaustive Search (current implementation):
#   - 50k docs: ~100-500ms per query
#   - 1M docs: ~2-10 seconds per query
#   - Complexity: O(n)
#
# FAISS IVF:
#   - 50k docs: ~5-20ms per query (10-50x faster)
#   - 1M docs: ~10-50ms per query (100-500x faster)
#   - Complexity: O(log n)
#
# FAISS HNSW:
#   - 50k docs: ~1-5ms per query (50-500x faster)
#   - 1M docs: ~2-10ms per query (500-5000x faster)
#   - Complexity: O(log n)
