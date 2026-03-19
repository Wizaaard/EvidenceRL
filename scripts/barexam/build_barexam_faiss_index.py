#!/usr/bin/env python3
"""Build FAISS index over the BarExam legal passage corpus.

The BarExam dataset (reglab/barexam_qa) includes ~857K legal passages with
pre-assigned faiss_id fields. This script embeds them with all-MiniLM-L6-v2
(matching the medical pipeline's retrieval embedder) and builds a FAISS index.

Usage:
    python scripts/barexam/build_barexam_faiss_index.py \
        --barexam-data-path data/barexam/ \
        --split test \
        --output-dir data/barexam/faiss_index_test/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl import Document
from evidence_rl.barexam_data import load_barexam_passages
from evidence_rl.retrieval import HuggingFaceEmbedder
from evidence_rl.faiss_retrieval import FAISSDocumentStore


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for BarExam passages")
    parser.add_argument("--barexam-data-path", required=True, help="Path to barexam data dir")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--output-dir", required=True, help="Output directory for FAISS index")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--index-type", default="HNSW", choices=["Flat", "IVF", "HNSW"])
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("BarExam FAISS Index Builder")
    print(f"{'='*60}")
    print(f"  Data path: {args.barexam_data_path}")
    print(f"  Split: {args.split}")
    print(f"  Output: {output_dir}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Index type: {args.index_type}")
    print(f"{'='*60}\n")

    if FAISSDocumentStore.index_exists(output_dir):
        print(f"Index already exists at {output_dir}. Skipping.")
        return

    t0 = time.time()

    # Load passages
    print("[1/3] Loading passages...")
    raw_passages = load_barexam_passages(args.barexam_data_path, split=args.split)
    print(f"  Loaded {len(raw_passages)} passages")

    # Convert to Document objects (no chunking — passages are already passage-sized)
    documents = []
    for p in raw_passages:
        text = p.get("text", "").strip()
        if not text:
            continue
        documents.append(Document(
            doc_id=p.get("idx", p.get("faiss_id", "")),
            text=text,
            metadata={"source": p.get("source", ""), "faiss_id": p.get("faiss_id", "")},
        ))
    print(f"  {len(documents)} non-empty documents")

    # Build embedder + index
    print("[2/3] Embedding and building FAISS index...")
    embedder = HuggingFaceEmbedder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        force_cpu=args.force_cpu,
    )

    store = FAISSDocumentStore(
        documents=documents,
        embedder=embedder,
        index_type=args.index_type,
        use_gpu=False,
    )

    # Save
    print("[3/3] Saving index...")
    store.save(output_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes. Index at: {output_dir}")


if __name__ == "__main__":
    main()
