#!/usr/bin/env python3
"""Pre-build FAISS index for EvidenceRL pipeline.

This script creates a FAISS index from knowledge documents and saves it to disk.
The pre-built index can be loaded by the EvidenceRL pipeline to skip the
expensive embedding and index building step during generation.

Usage:
    python scripts/build_faiss_index.py --output-dir /path/to/index

    # With custom parameters:
    python scripts/build_faiss_index.py \
        --output-dir /path/to/index \
        --knowledge-dataset ilyassacha/cardiologyChunks \
        --max-records 500000 \
        --chunk-size 200 \
        --chunk-overlap 50

The script will create the following files in the output directory:
    - index.faiss: The FAISS index
    - documents.pkl: Pickled document list (chunked)
    - metadata.json: Index configuration and statistics
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Add src to path (script is in scripts/generation/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl import Document, load_documents_from_hf_dataset
from evidence_rl.retrieval import HuggingFaceEmbedder
from evidence_rl.faiss_retrieval import FAISSDocumentStore


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Chunk documents into smaller pieces for better retrieval.

    This replicates the chunking logic from EvidenceRLPipeline to ensure
    the pre-built index matches what the pipeline expects.

    Args:
        documents: Original documents to chunk.
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.

    Returns:
        List of chunked documents.
    """

    def chunk_text(text: str, chunk_sz: int, overlap: int) -> List[str]:
        tokens = text.split()
        if not tokens:
            return []
        chunks_list = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + chunk_sz)
            chunks_list.append(" ".join(tokens[start:end]))
            if end == len(tokens):
                break
            start = end - overlap
        return chunks_list

    chunked_docs = []
    for doc in documents:
        chunks = chunk_text(doc.text, chunk_size, chunk_overlap) or [doc.text]
        for idx, chunk in enumerate(chunks):
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata.setdefault("source_doc_id", doc.doc_id)
            metadata["chunk_id"] = idx
            chunk_doc_id = doc.doc_id if len(chunks) == 1 else f"{doc.doc_id}::chunk-{idx}"
            chunked_docs.append(
                Document(
                    doc_id=chunk_doc_id,
                    text=chunk,
                    metadata=metadata,
                )
            )

    return chunked_docs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-build FAISS index for EvidenceRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the FAISS index",
    )

    # Knowledge source
    parser.add_argument(
        "--knowledge-dataset",
        default="ilyassacha/cardiologyChunks",
        help="HuggingFace dataset name (default: ilyassacha/cardiologyChunks)",
    )
    parser.add_argument(
        "--knowledge-dataset-split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--knowledge-dataset-text-field",
        default="text",
        help="Name of text field in dataset (default: text)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500000,
        help="Maximum records to load (default: 500000)",
    )

    # Chunking parameters (must match pipeline settings)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Chunk size in tokens (default: 200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)",
    )

    # Embedding parameters
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64)",
    )

    # FAISS parameters
    parser.add_argument(
        "--index-type",
        choices=["Flat", "IVF", "HNSW"],
        default="HNSW",
        help="FAISS index type (default: HNSW)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    print(f"\n{'='*60}")
    print("EvidenceRL FAISS Index Builder")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Knowledge dataset: {args.knowledge_dataset}")
    print(f"Max records: {args.max_records}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Index type: {args.index_type}")
    print(f"{'='*60}\n")

    # Check if index already exists
    if FAISSDocumentStore.index_exists(output_dir):
        print(f"WARNING: Index already exists at {output_dir}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    start_time = time.time()

    # Step 1: Load documents
    print("\n[Step 1/4] Loading knowledge documents...")
    documents = load_documents_from_hf_dataset(
        args.knowledge_dataset,
        split=args.knowledge_dataset_split,
        text_field=args.knowledge_dataset_text_field,
        max_records=args.max_records,
    )
    print(f"Loaded {len(documents)} documents")

    # Step 2: Chunk documents
    print("\n[Step 2/4] Chunking documents...")
    chunked_docs = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")

    # Step 3: Build embedder and index
    print("\n[Step 3/4] Building FAISS index...")
    embedder = HuggingFaceEmbedder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )

    store = FAISSDocumentStore(
        documents=chunked_docs,
        embedder=embedder,
        index_type=args.index_type,
        use_gpu=False,  # Build on CPU for portability
    )

    # Step 4: Save index
    print("\n[Step 4/4] Saving index to disk...")
    store.save(output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Index built successfully in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"\nTo use this index, set the following in your pipeline:")
    print(f"  --faiss-index-dir {output_dir}")
    print(f"\nOr set environment variable:")
    print(f"  export EVIDENCERL_FAISS_INDEX={output_dir}")


if __name__ == "__main__":
    main()
