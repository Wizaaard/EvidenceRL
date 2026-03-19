#!/usr/bin/env python3
"""Entry point for running the EvidenceRL pipeline.

This script orchestrates the full EvidenceRL workflow:
1. Load patient cases from MIMIC-IV-Ext cardiac dataset
2. Load knowledge documents from HuggingFace dataset
3. For each patient:
   - Pre-retrieval: Retrieve evidence based on patient context
   - Generation: LLM produces 5 diagnoses with reasoning (JSON format)

Reward computation (grounding + precision) is done separately in the analysis phase.
See compute_evidencerl_metrics.py for reward computation.

Usage:
    python run_evidence_rl.py --model-name <model> --patient-data-path <path> ...

See run_evidence_rl.sh for a complete example with all arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl import (
    Document,
    load_documents_from_hf_dataset,
    load_documents_from_jsonl,
    load_patient_cases,
)
from evidence_rl.evidence_pipeline import (
    EvidenceRLPipeline,
    EvidenceRLResult,
    summarize_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EvidenceRL pipeline for cardiac diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evidence_rl.py --model-name <model> --patient-data-path <path> \\
        --max-patients 10 --json-output results.json
        """,
    )

    # Required paths
    parser.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model path/name for diagnosis generation",
    )
    parser.add_argument(
        "--patient-data-path",
        required=True,
        help="Path to MIMIC-IV-Ext cardiac disease dataset directory",
    )

    # Retrieval parameters
    parser.add_argument(
        "--top-k-pre",
        type=int,
        default=3,
        help="Number of documents to retrieve before generation (default: 3)",
    )

    # Processing parameters
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit number of patient cases to process",
    )
    parser.add_argument(
        "--patient-start-idx",
        type=int,
        default=0,
        help="Start index for patient subset (0-indexed, inclusive). Used for distributed processing.",
    )
    parser.add_argument(
        "--patient-end-idx",
        type=int,
        default=None,
        help="End index for patient subset (exclusive). If None, process until max-patients or end of data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )

    # Knowledge source options
    parser.add_argument(
        "--knowledge-dataset",
        default=None,
        help="HuggingFace dataset name for knowledge chunks (e.g., ilyassacha/cardiologyChunks)",
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
        "--knowledge-dataset-max-records",
        type=int,
        default=None,
        help="Maximum records to load from dataset",
    )
    parser.add_argument(
        "--knowledge-jsonl",
        default=None,
        help="Path to pre-computed JSONL knowledge file",
    )
    parser.add_argument(
        "--faiss-index-dir",
        default=None,
        help="Path to pre-built FAISS index directory. If provided, knowledge-dataset "
             "and knowledge-jsonl are ignored (much faster startup).",
    )

    # Chunking parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=320,
        help="Chunk size for document processing (default: 320)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between chunks (default: 64)",
    )

    # Embedding parameters
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation to avoid OOM errors (default: 32)",
    )
    parser.add_argument(
        "--use-llm-extraction",
        action="store_true",
        default=True,
        help="Use LLM to extract diagnoses when JSON parsing fails (default: True)",
    )
    parser.add_argument(
        "--no-llm-extraction",
        action="store_false",
        dest="use_llm_extraction",
        help="Disable LLM extraction, use only regex fallback",
    )

    # Embedding model
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)",
    )

    # Baseline (no-RAG) mode
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Run in baseline mode without retrieval (zero-shot generation). "
             "Skips document loading and embedding computation.",
    )

    # Output
    parser.add_argument(
        "--json-output",
        required=True,
        help="Path for JSON output file",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save individual result files per patient",
    )

    return parser


def load_documents(args) -> List[Document] | None:
    """Load knowledge documents from configured source.

    Returns None if faiss_index_dir is provided (documents are loaded from the index).
    Returns None if no_rag is True (baseline mode, no retrieval).
    """
    # If baseline mode (no-RAG), skip document loading entirely
    if args.no_rag:
        print("Baseline mode (--no-rag): Skipping document loading")
        return None

    # If using pre-built index, we don't need to load documents
    if args.faiss_index_dir:
        print(f"Using pre-built FAISS index from: {args.faiss_index_dir}")
        return None

    if args.knowledge_dataset:
        print(f"Loading knowledge from HuggingFace dataset: {args.knowledge_dataset}")
        return load_documents_from_hf_dataset(
            args.knowledge_dataset,
            split=args.knowledge_dataset_split,
            text_field=args.knowledge_dataset_text_field,
            max_records=args.knowledge_dataset_max_records,
        )

    if args.knowledge_jsonl:
        jsonl_path = Path(args.knowledge_jsonl)
        if jsonl_path.exists():
            print(f"Loading knowledge from JSONL: {jsonl_path}")
            return load_documents_from_jsonl(jsonl_path)

    # In baseline mode (--no-rag), we already returned None above
    # For RAG mode, we need a knowledge source
    if not args.no_rag:
        raise ValueError(
            "No knowledge source specified. Use --faiss-index-dir, --knowledge-dataset, or --knowledge-jsonl. "
            "Or use --no-rag for baseline mode without retrieval."
        )
    return None


def format_result_summary(result: EvidenceRLResult) -> str:
    """Format a single result for console output."""

    lines = [
        f"\n{'='*60}",
        f"Patient: hadm_id={result.hadm_id}, subject_id={result.subject_id}",
        f"{'='*60}",
        f"\nPre-evidence ({len(result.pre_evidence)} docs):",
    ]

    for idx, doc in enumerate(result.pre_evidence[:3], 1):
        text_preview = doc.document.text[:100].replace("\n", " ")
        lines.append(f"  {idx}. [{doc.score:.3f}] {text_preview}...")

    if result.structured_output:
        lines.append(f"\nGenerated Diagnoses (parse_success={result.structured_output.parse_success}):")
        for idx, diag in enumerate(result.structured_output.diagnoses, 1):
            reasoning_preview = diag.reasoning[:80].replace("\n", " ") if diag.reasoning else "(no reasoning)"
            lines.append(f"  {idx}. {diag.name}")
            lines.append(f"     Reasoning: {reasoning_preview}...")

    lines.append(f"\nGround Truth Diagnoses: {len(result.ground_truth_diagnoses)}")
    for idx, diag in enumerate(result.ground_truth_diagnoses[:5], 1):
        lines.append(f"  {idx}. {diag}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Load patient cases
    print(f"\nLoading patient cases from: {args.patient_data_path}")
    patient_cases = load_patient_cases(args.patient_data_path, limit=args.max_patients)
    total_loaded = len(patient_cases)

    # Apply patient range selection for distributed processing
    start_idx = args.patient_start_idx
    end_idx = args.patient_end_idx if args.patient_end_idx is not None else len(patient_cases)
    end_idx = min(end_idx, len(patient_cases))

    if start_idx > 0 or end_idx < len(patient_cases):
        patient_cases = patient_cases[start_idx:end_idx]
        print(f"Loaded {total_loaded} total, processing subset [{start_idx}:{end_idx}] = {len(patient_cases)} patients")
    else:
        print(f"Loaded {len(patient_cases)} patient cases")

    # Load knowledge documents (or use pre-built index)
    documents = load_documents(args)
    if documents is not None:
        print(f"Loaded {len(documents)} knowledge documents")

    # Initialize pipeline
    mode_str = "baseline (no-RAG)" if args.no_rag else "RAG"
    print(f"\nInitializing EvidenceRL pipeline in {mode_str} mode...")
    pipeline = EvidenceRLPipeline(
        documents=documents,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        top_k_pre=args.top_k_pre,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        use_llm_extraction=args.use_llm_extraction,
        faiss_index_dir=args.faiss_index_dir,
        baseline_mode=args.no_rag,
    )
    print("Pipeline initialized successfully")

    # Run pipeline
    print(f"\nProcessing {len(patient_cases)} patient cases...")
    results = pipeline.run_batch(
        patient_cases,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Print individual results
    for result in results:
        print(format_result_summary(result))

    # Compute and print summary
    summary = summarize_results(results)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for key, value in sorted(summary.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nNote: Reward computation is done in the analysis phase.")
    print("Run compute_evidencerl_metrics.py on this output to compute rewards.")

    # Save results
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "model_name": args.model_name,
            "embedding_model_name": args.embedding_model_name,
            "top_k_pre": args.top_k_pre,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "num_patients": len(patient_cases),
            "patient_start_idx": start_idx,
            "patient_end_idx": end_idx,
            "baseline_mode": args.no_rag,
        },
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path.resolve()}")

    # Optionally save individual files
    if args.save_individual:
        individual_dir = output_path.parent / f"{output_path.stem}_individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            result.save_json(individual_dir / f"{result.hadm_id}.json")
        print(f"Individual results saved to: {individual_dir}")


if __name__ == "__main__":
    main()
