#!/usr/bin/env python3
"""Entry point for running the EvidenceRL pipeline with vLLM.

This script provides 14-24x faster inference compared to the HuggingFace version.
It uses vLLM with tensor parallelism for efficient multi-GPU generation.

Usage:
    python run_evidence_rl_vllm.py --model-name <model> --patient-data-path <path> ...

The output format is identical to run_evidence_rl.py, so downstream tools
(metrics computation, revision, etc.) work with either version.

See scripts/generation_vllm/launch_generation_vllm.sh for a complete example.
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
from evidence_rl.vllm_pipeline import (
    VLLMEvidenceRLPipeline,
    summarize_results,
)
from evidence_rl.evidence_pipeline import EvidenceRLResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EvidenceRL pipeline with vLLM (14-24x faster)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evidence_rl_vllm.py --model-name <model> --patient-data-path <path> \\
        --max-patients 10 --json-output results.json --tensor-parallel-size 2

vLLM Benefits:
    - 14-24x faster than HuggingFace Transformers
    - PagedAttention for efficient memory management
    - Continuous batching for optimal throughput
    - Tensor parallelism for multi-GPU inference
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

    # vLLM-specific parameters
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (default: 0.90)",
    )
    parser.add_argument(
        "--guided-json",
        action="store_true",
        default=False,
        help="Enable guided JSON decoding to enforce valid JSON output schema (experimental)",
    )
    parser.add_argument(
        "--extractor-model",
        default=None,
        help="Path to a separate (larger) model for fallback extraction when JSON "
             "parsing fails. E.g., use a 12B model to extract from a 4B model's output.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completions per patient (for multi-sample generation, e.g., fdpo). "
             "Default: 1 (single completion).",
    )

    # LoRA adapter parameters (for SFT/GRPO-trained models)
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Path to LoRA adapter directory (for SFT/GRPO-trained models). "
             "If provided, vLLM loads the base model with LoRA enabled.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="Maximum LoRA rank to support (default: 64)",
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
        help="Batch size hint (vLLM handles batching internally, default: 4)",
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

    # Baseline mode (no-RAG)
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Run in baseline mode without retrieval (zero-shot generation). "
             "Skips document loading, FAISS index, and embedding computation.",
    )

    # Self-RAG mode
    parser.add_argument(
        "--self-rag",
        action="store_true",
        help="Run in self-RAG mode: generate → self-critique → conditionally retrieve → refine. "
             "Requires FAISS index or knowledge source for conditional retrieval.",
    )

    # Self-Consistency mode
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Run in self-consistency mode: generate N diverse samples → cluster → vote. "
             "Pure generation technique, no retrieval needed.",
    )
    parser.add_argument(
        "--sc-num-samples",
        type=int,
        default=10,
        help="Number of diverse samples per patient for self-consistency (default: 10)",
    )
    parser.add_argument(
        "--sc-temperature",
        type=float,
        default=0.9,
        help="Temperature for diverse sampling in self-consistency (default: 0.9)",
    )
    parser.add_argument(
        "--sc-similarity-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for diagnosis clustering (default: 0.85)",
    )
    parser.add_argument(
        "--sc-embedding-model",
        default=None,
        help="Embedding model for SC diagnosis clustering (default: FremyCompany/BioLORD-2023)",
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
        help="Batch size for embedding computation (default: 32)",
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
        help="HuggingFace model for embeddings / retrieval (default: sentence-transformers/all-MiniLM-L6-v2)",
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

    Returns None if:
    - faiss_index_dir is provided (documents are loaded from the index)
    - no_rag is True (baseline mode, no documents needed)
    """
    # Baseline / Self-Consistency mode: no documents needed
    if args.no_rag:
        print("Baseline mode (--no-rag): Skipping document loading")
        return None
    if args.self_consistency:
        print("Self-Consistency mode: Skipping document loading (pure generation)")
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

    raise ValueError(
        "No knowledge source specified. Use --no-rag, --faiss-index-dir, --knowledge-dataset, or --knowledge-jsonl"
    )


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

    # Validate mutually exclusive modes
    mode_flags = sum([args.self_rag, args.no_rag, args.self_consistency])
    if mode_flags > 1:
        print("[ERROR] --self-rag, --no-rag, and --self-consistency are mutually exclusive.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("EvidenceRL Generation Pipeline (vLLM Version)")
    print("14-24x faster than HuggingFace Transformers")
    if args.num_samples > 1:
        print(f"MULTI-SAMPLE MODE: {args.num_samples} completions per patient")
    if args.no_rag:
        print("MODE: BASELINE (no-RAG, zero-shot generation)")
    elif args.self_rag:
        print("MODE: SELF-RAG (adaptive retrieval: generate → critique → retrieve → refine)")
    elif args.self_consistency:
        print(f"MODE: SELF-CONSISTENCY (n={args.sc_num_samples}, T={args.sc_temperature}, "
              f"sim={args.sc_similarity_threshold})")
    else:
        print("MODE: RAG (top-k retrieval + generation)")
    print("=" * 60)

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

    # Initialize vLLM pipeline
    if args.self_consistency:
        from evidence_rl.self_consistency_pipeline import SelfConsistencyVLLMPipeline

        print("\nInitializing Self-Consistency vLLM pipeline...")
        pipeline = SelfConsistencyVLLMPipeline(
            model_name=args.model_name,
            embedding_model_name=args.sc_embedding_model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_samples=args.sc_num_samples,
            sc_temperature=args.sc_temperature,
            similarity_threshold=args.sc_similarity_threshold,
            use_guided_json=args.guided_json,
            extractor_model_name=args.extractor_model,
        )
    elif args.self_rag:
        from evidence_rl.self_rag_pipeline import SelfRAGVLLMPipeline

        print("\nInitializing Self-RAG vLLM pipeline...")
        pipeline = SelfRAGVLLMPipeline(
            documents=documents,
            model_name=args.model_name,
            embedding_model_name=args.embedding_model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            top_k_pre=args.top_k_pre,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.embedding_batch_size,
            use_llm_extraction=args.use_llm_extraction,
            faiss_index_dir=args.faiss_index_dir,
            use_guided_json=args.guided_json,
            extractor_model_name=args.extractor_model,
        )
    else:
        print("\nInitializing vLLM EvidenceRL pipeline...")
        pipeline = VLLMEvidenceRLPipeline(
            documents=documents,
            model_name=args.model_name,
            embedding_model_name=args.embedding_model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            top_k_pre=args.top_k_pre,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.embedding_batch_size,
            use_llm_extraction=args.use_llm_extraction,
            faiss_index_dir=args.faiss_index_dir if not args.no_rag else None,
            baseline_mode=args.no_rag,
            lora_path=args.lora_path,
            max_lora_rank=args.max_lora_rank,
            use_guided_json=args.guided_json,
            extractor_model_name=args.extractor_model,
            num_samples=args.num_samples,
        )
    print("Pipeline initialized successfully")

    # Run pipeline
    print(f"\nProcessing {len(patient_cases)} patient cases with vLLM...")
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

    # Serialize results (self-RAG and SC add metadata)
    if args.self_consistency:
        from evidence_rl.self_consistency_pipeline import sc_result_to_dict
        serialized_results = [sc_result_to_dict(r) for r in results]
    elif args.self_rag:
        from evidence_rl.self_rag_pipeline import selfrag_result_to_dict
        serialized_results = [selfrag_result_to_dict(r) for r in results]
    else:
        serialized_results = [r.to_dict() for r in results]

    config = {
        "model_name": args.model_name,
        "embedding_model_name": args.embedding_model_name,
        "top_k_pre": args.top_k_pre,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_patients": len(patient_cases),
        "patient_start_idx": start_idx,
        "patient_end_idx": end_idx,
        "inference_engine": "vllm",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "baseline_mode": args.no_rag,
        "self_rag_mode": args.self_rag,
        "self_consistency_mode": args.self_consistency,
        "guided_json": args.guided_json,
        "extractor_model": args.extractor_model,
        "num_samples": args.num_samples,
    }
    if args.self_consistency:
        config["sc_num_samples"] = args.sc_num_samples
        config["sc_temperature"] = args.sc_temperature
        config["sc_similarity_threshold"] = args.sc_similarity_threshold
        config["sc_embedding_model"] = args.sc_embedding_model or "FremyCompany/BioLORD-2023"

    payload = {
        "config": config,
        "summary": summary,
        "results": serialized_results,
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
