"""Alternative vector store implementations for scalable retrieval.

This module provides multiple backend options for vector similarity search,
each with different trade-offs in speed, accuracy, and scalability.
"""

from __future__ import annotations

from typing import List, Sequence

from .documents import Document, RetrievedDocument
from .retrieval import TextEmbedder, _normalize_concepts


class ChromaDBStore:
    """ChromaDB-based vector store with persistent storage and filtering.

    Benefits:
    - Persistent storage (survives restarts)
    - Built-in metadata filtering
    - Easy to use, minimal setup
    - Good for moderate scale (100k-1M docs)

    Install: pip install chromadb
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embedder: TextEmbedder,
        collection_name: str = "evidence_rl",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("ChromaDB not installed. Install with: pip install chromadb")

        self.documents = list(documents)
        self.embedder = embedder

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Add documents
        print(f"Adding {len(documents)} documents to ChromaDB...")
        ids = [doc.doc_id for doc in self.documents]
        texts = [doc.text for doc in self.documents]
        metadatas = [doc.metadata or {} for doc in self.documents]

        # Generate embeddings
        embeddings = self.embedder.encode(texts)

        # Batch add to collection
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
        print(f"ChromaDB collection created with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5, concept_filter: List[str] | None = None) -> List[RetrievedDocument]:
        """Search with optional metadata filtering."""
        query_embedding = self.embedder.encode([query])[0]

        # Build metadata filter if concepts provided
        where = None
        if concept_filter:
            where = {"concepts": {"$in": concept_filter}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        # Convert to RetrievedDocument
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
                # Find document
                doc = next((d for d in self.documents if d.doc_id == doc_id), None)
                if doc:
                    score = 1.0 - distance  # Convert distance to similarity
                    retrieved.append(RetrievedDocument(doc, score))

        return retrieved


class QdrantStore:
    """Qdrant vector database - production-grade, highly scalable.

    Benefits:
    - Built for production (persistent, replicated)
    - Advanced filtering capabilities
    - Excellent performance at scale
    - Good for large scale (1M-100M+ docs)
    - Supports payload filtering

    Install: pip install qdrant-client
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embedder: TextEmbedder,
        collection_name: str = "evidence_rl",
        url: str = "localhost",
        port: int = 6333,
        in_memory: bool = True,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise RuntimeError("Qdrant not installed. Install with: pip install qdrant-client")

        self.documents = list(documents)
        self.embedder = embedder
        self.collection_name = collection_name

        # Initialize client
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=url, port=port)

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedder.encode(texts)

        vector_size = len(embeddings[0])

        # Create collection
        print(f"Creating Qdrant collection '{collection_name}'...")
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # Prepare points
        points = []
        for idx, (doc, embedding) in enumerate(zip(self.documents, embeddings)):
            payload = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata or {},
            }
            points.append(PointStruct(id=idx, vector=embedding, payload=payload))

        # Upload in batches
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i+batch_size],
            )
        print(f"Qdrant collection created with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Search for similar documents in Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_embedding = self.embedder.encode([query])[0]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        # Convert to RetrievedDocument
        retrieved = []
        for result in results:
            # Reconstruct document from payload
            payload = result.payload
            doc = Document(
                doc_id=payload["doc_id"],
                text=payload["text"],
                metadata=payload.get("metadata"),
            )
            retrieved.append(RetrievedDocument(doc, result.score))

        return retrieved


class PineconeStore:
    """Pinecone managed vector database - serverless, fully managed.

    Benefits:
    - Fully managed (no infrastructure)
    - Auto-scaling
    - Low latency globally
    - Good for any scale (handles billions)

    Requires: Pinecone API key
    Install: pip install pinecone-client
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embedder: TextEmbedder,
        index_name: str = "evidence-rl",
        api_key: str | None = None,
        environment: str = "gcp-starter",
    ) -> None:
        try:
            import pinecone
        except ImportError:
            raise RuntimeError("Pinecone not installed. Install with: pip install pinecone-client")

        import os
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY env var or pass api_key parameter")

        self.documents = list(documents)
        self.embedder = embedder
        self.index_name = index_name

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedder.encode(texts)

        dimension = len(embeddings[0])

        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            print(f"Creating Pinecone index '{index_name}'...")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
            )

        # Connect to index
        self.index = pinecone.Index(index_name)

        # Upsert vectors
        print(f"Uploading {len(documents)} vectors to Pinecone...")
        vectors = []
        for idx, (doc, embedding) in enumerate(zip(self.documents, embeddings)):
            metadata = {
                "doc_id": doc.doc_id,
                "text": doc.text[:1000],  # Pinecone has metadata size limits
            }
            if doc.metadata:
                metadata.update(doc.metadata)

            vectors.append((str(idx), embedding, metadata))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size])
        print(f"Pinecone index created with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Search for similar documents in Pinecone."""
        query_embedding = self.embedder.encode([query])[0]

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        # Convert to RetrievedDocument
        retrieved = []
        for match in results["matches"]:
            metadata = match["metadata"]
            doc = Document(
                doc_id=metadata["doc_id"],
                text=metadata["text"],
                metadata={k: v for k, v in metadata.items() if k not in ["doc_id", "text"]},
            )
            retrieved.append(RetrievedDocument(doc, match["score"]))

        return retrieved


# Performance & Scalability Comparison:
#
# | Solution    | Setup      | Scale         | Speed       | Cost       | Best For                    |
# |-------------|------------|---------------|-------------|------------|-----------------------------|
# | Exhaustive  | Easy       | <100k docs    | Slow O(n)   | Free       | Small datasets, prototyping |
# | FAISS       | Easy       | 1M-1B docs    | Fast O(log) | Free       | Self-hosted, high scale     |
# | ChromaDB    | Easy       | 100k-1M docs  | Fast        | Free       | Local dev, persistence      |
# | Qdrant      | Medium     | 1M-100M docs  | Very Fast   | Free/Paid  | Production, self-hosted     |
# | Pinecone    | Easy       | Any scale     | Very Fast   | Paid       | Serverless, managed         |
# | Weaviate    | Medium     | 1M-100M docs  | Very Fast   | Free/Paid  | Hybrid search, GraphQL      |
# | Milvus      | Hard       | 1B+ docs      | Very Fast   | Free       | Massive scale, distributed  |
