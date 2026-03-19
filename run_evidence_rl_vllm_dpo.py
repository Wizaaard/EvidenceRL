#!/usr/bin/env python3
"""Entry point for running EvidenceRL with DPO-trained models using vLLM.

This script runs inference with models fine-tuned using Direct Preference
Optimization (DPO). It loads the base model with the trained LoRA adapters.

Usage:
    python run_evidence_rl_vllm_dpo.py \
        --base-model-name /path/to/base/model \
        --lora-path /path/to/lora/adapter \
        --patient-data-path /path/to/mimic-cardiac \
        --json-output results.json

The output format is identical to run_evidence_rl_vllm.py, so downstream tools
(metrics computation, analysis, etc.) work with either version.
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
        description="Run EvidenceRL with DPO-trained models (vLLM + LoRA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with DPO-trained model (no RAG - baseline evaluation)
    python run_evidence_rl_vllm_dpo.py \\
        --base-model-name /path/to/gemma-3-4b-it \\
        --lora-path /path/to/dpo_trained_models/gemma-3-4b_v1.0/final_model \\
        --patient-data-path /path/to/mimic-cardiac \\
        --no-rag \\
        --json-output results.json

    # Run with DPO-trained model (with RAG)
    python run_evidence_rl_vllm_dpo.py \\
        --base-model-name /path/to/gemma-3-4b-it \\
        --lora-path /path/to/dpo_trained_models/gemma-3-4b_v1.0/final_model \\
        --patient-data-path /path/to/mimic-cardiac \\
        --faiss-index-dir /path/to/faiss_index \\
        --json-output results.json

DPO Training:
    - Models are fine-tuned using Direct Preference Optimization
    - LoRA adapters are loaded on top of the base model
    - The adapter path should contain adapter_config.json and adapter_model.safetensors
        """,
    )

    # Required paths
    parser.add_argument(
        "--base-model-name",
        required=True,
        help="Path to the base model (before DPO training)",
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Path to LoRA adapter directory (e.g., .../final_model with adapter_config.json)",
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
        help="Start index for patient subset (0-indexed, inclusive).",
    )
    parser.add_argument(
        "--patient-end-idx",
        type=int,
        default=None,
        help="End index for patient subset (exclusive).",
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
        help="HuggingFace dataset name for knowledge chunks",
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
        help="Path to pre-built FAISS index directory.",
    )

    # Baseline mode (no-RAG)
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Run in baseline mode without retrieval (zero-shot generation).",
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
        help="HuggingFace model for embeddings",
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
    """Load knowledge documents from configured source."""
    if args.no_rag:
        print("DPO Baseline mode (--no-rag): Skipping document loading")
        return None

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

    # Validate LoRA path
    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        print(f"Error: LoRA adapter path does not exist: {lora_path}")
        sys.exit(1)

    adapter_config = lora_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: adapter_config.json not found in {lora_path}")
        print("Make sure you're pointing to the correct LoRA adapter directory (e.g., final_model/)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("EvidenceRL DPO Generation Pipeline (vLLM + LoRA)")
    print("Post-training inference with DPO fine-tuned models")
    print("=" * 60)
    print(f"\nBase Model: {args.base_model_name}")
    print(f"LoRA Adapter: {args.lora_path}")
    if args.no_rag:
        print("MODE: DPO + BASELINE (no-RAG)")
    else:
        print("MODE: DPO + RAG")

    # Load patient cases
    print(f"\nLoading patient cases from: {args.patient_data_path}")
    patient_cases = load_patient_cases(args.patient_data_path, limit=args.max_patients)
    total_loaded = len(patient_cases)

    # Apply patient range selection
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

    # Initialize vLLM pipeline with LoRA
    print("\nInitializing vLLM DPO pipeline with LoRA adapter...")
    pipeline = VLLMEvidenceRLPipeline(
        documents=documents,
        model_name=args.base_model_name,
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
        lora_path=str(lora_path),
        max_lora_rank=args.max_lora_rank,
        use_guided_json=args.guided_json,
        extractor_model_name=args.extractor_model,
    )
    print("DPO Pipeline initialized successfully")

    # Run pipeline
    print(f"\nProcessing {len(patient_cases)} patient cases with DPO model...")
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
    print("SUMMARY STATISTICS (DPO Model)")
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
            "base_model_name": args.base_model_name,
            "lora_path": str(args.lora_path),
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
            "dpo_trained": True,
            "guided_json": args.guided_json,
            "extractor_model": args.extractor_model,
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
