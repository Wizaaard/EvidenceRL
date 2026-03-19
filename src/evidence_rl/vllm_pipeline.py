#!/usr/bin/env python3
"""vLLM-based EvidenceRL pipeline for high-performance diagnosis generation.

This module provides the full EvidenceRL pipeline using vLLM,
offering 14-24x faster inference compared to HuggingFace Transformers.

Key classes:
- VLLMEvidenceRLPipeline: End-to-end pipeline with vLLM generation

Usage:
    from evidence_rl.vllm_pipeline import VLLMEvidenceRLPipeline

    pipeline = VLLMEvidenceRLPipeline(
        documents=docs,
        model_name="path/to/model",
        tensor_parallel_size=2,
    )
    results = pipeline.run_batch(patient_cases, batch_size=32)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

from .documents import Document
from .baseline import PatientCase, clean_patient_information
from .retrieval import HuggingFaceEmbedder
from .evidence_pipeline import (
    EvidenceRLResult,
    summarize_results,
)
from .faiss_retrieval import FAISSDocumentStore, FAISSRetriever
from .vllm_generation import VLLMStructuredDiagnosisGenerator, check_vllm_available


class VLLMEvidenceRLPipeline:
    """End-to-end EvidenceRL pipeline using vLLM for high-performance generation.

    This class provides the same interface as EvidenceRLPipeline
    but uses vLLM for significantly faster inference.

    Key differences from HuggingFace version:
    - Uses VLLMStructuredDiagnosisGenerator instead of StructuredDiagnosisGenerator
    - Tensor parallelism for efficient multi-GPU usage
    - PagedAttention for efficient memory management
    - Continuous batching for optimal throughput
    - 14-24x faster inference
    """

    def __init__(
        self,
        documents: Optional[Sequence[Document]] = None,
        model_name: str = "",
        embedding_model_name: Optional[str] = None,
        tensor_parallel_size: int = 2,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.90,
        top_k_pre: int = 3,
        generation_kwargs: Optional[Mapping[str, Any]] = None,
        chunk_size: int = 320,
        chunk_overlap: int = 64,
        embedding_batch_size: int = 32,
        use_llm_extraction: bool = True,
        faiss_index_dir: Optional[str] = None,
        baseline_mode: bool = False,
        lora_path: Optional[str] = None,
        max_lora_rank: int = 64,
        use_guided_json: bool = False,
        extractor_model_name: Optional[str] = None,
        num_samples: int = 1,
    ) -> None:
        """Initialize the vLLM EvidenceRL pipeline.

        Args:
            documents: Knowledge documents to index. Can be None if faiss_index_dir is provided
                      or if baseline_mode is True.
            model_name: HuggingFace model path for generation (base model for LoRA).
            embedding_model_name: Model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            max_tokens: Maximum tokens to generate.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
            top_k_pre: Number of documents to retrieve per query.
            generation_kwargs: Additional kwargs for the generator.
            chunk_size: Chunk size in tokens for document splitting.
            chunk_overlap: Overlap between chunks in tokens.
            embedding_batch_size: Batch size for embedding computation.
            use_llm_extraction: Whether to use LLM for fallback parsing.
            faiss_index_dir: Path to pre-built FAISS index. If provided, documents are ignored
                            and the index is loaded from disk (much faster startup).
            baseline_mode: If True, skip retrieval entirely (zero-shot generation).
            lora_path: Path to LoRA adapter directory (for DPO-trained models).
                      If provided, the model will be loaded with LoRA enabled.
            max_lora_rank: Maximum LoRA rank to support (default: 64).
            use_guided_json: Whether to use guided JSON decoding for guaranteed valid output.
            extractor_model_name: Optional path to a separate (larger) model for fallback
                                 extraction when JSON parsing fails.
        """
        check_vllm_available()

        self.baseline_mode = baseline_mode
        self.top_k_pre = top_k_pre
        self.lora_path = lora_path

        # In baseline mode, skip embedder and retriever setup
        if baseline_mode:
            print("[vLLM Pipeline] BASELINE MODE: Skipping retrieval setup")
            self.embedder = None
            self.store = None
            self.retriever = None
        else:
            # Set up embedder
            self.embedder = HuggingFaceEmbedder(
                model_name=embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2",
            )

            # Set up retriever
            if faiss_index_dir:
                print(f"[vLLM Pipeline] Loading pre-built FAISS index from: {faiss_index_dir}")
                if FAISSDocumentStore.index_exists(faiss_index_dir):
                    self.store = FAISSDocumentStore.load(
                        faiss_index_dir,
                        embedder=self.embedder,
                    )
                    self.retriever = FAISSRetriever(self.store)
                else:
                    raise ValueError(
                        f"FAISS index not found at {faiss_index_dir}. "
                        "Please build the index first or provide documents."
                    )
            elif documents is not None:
                print(f"[vLLM Pipeline] Building FAISS index from {len(documents)} documents...")
                self.store = FAISSDocumentStore(
                    documents=documents,
                    embedder=self.embedder,
                )
                self.retriever = FAISSRetriever(self.store)
            else:
                raise ValueError(
                    "Either documents, faiss_index_dir, or baseline_mode=True must be provided"
                )

        # Set up vLLM generator
        print(f"[vLLM Pipeline] Initializing vLLM generator...")
        if extractor_model_name:
            print(f"[vLLM Pipeline] Using separate extractor model: {extractor_model_name}")
        self.num_samples = num_samples
        self.generator = VLLMStructuredDiagnosisGenerator(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            generation_kwargs=generation_kwargs,
            use_llm_extraction=use_llm_extraction,
            lora_path=lora_path,
            max_lora_rank=max_lora_rank,
            use_guided_json=use_guided_json,
            extractor_model_name=extractor_model_name,
            num_samples=num_samples,
        )

        # Build mode string
        mode_parts = []
        if baseline_mode:
            mode_parts.append("BASELINE (no-RAG)")
        else:
            mode_parts.append("RAG")
        if lora_path:
            mode_parts.append("DPO (with LoRA)")
        mode_str = " + ".join(mode_parts)
        print(f"[vLLM Pipeline] Initialization complete. Mode: {mode_str}")

    def run_single(self, patient_case: PatientCase) -> EvidenceRLResult:
        """Process a single patient case.

        Args:
            patient_case: Patient case with context and ground truth diagnoses.

        Returns:
            EvidenceRLResult with generation output (no reward computation).
        """
        # Clean patient context (consistent with HuggingFace version)
        cleaned_context = clean_patient_information(patient_case.context)

        # Pre-retrieval: Find relevant evidence based on cleaned patient context
        # In baseline mode, skip retrieval entirely
        if self.baseline_mode:
            pre_evidence = []
        else:
            pre_evidence = self.retriever.retrieve(
                query=cleaned_context,
                top_k=self.top_k_pre,
            )

        # Generation: Produce structured diagnoses with reasoning
        structured_output = self.generator.generate(
            patient_context=cleaned_context,
            pre_evidence=pre_evidence,
        )

        return EvidenceRLResult(
            hadm_id=patient_case.hadm_id,
            subject_id=patient_case.subject_id,
            patient_context=cleaned_context,
            ground_truth_diagnoses=list(patient_case.diagnoses),
            pre_evidence=pre_evidence,
            structured_output=structured_output,
            generation_prompt=self.generator.last_prompt or "",
        )

    def run_batch(
        self,
        patient_cases: Sequence[PatientCase],
        batch_size: int = 4,  # Kept for API compatibility
        show_progress: bool = True,
    ) -> List[EvidenceRLResult]:
        """Process multiple patient cases using vLLM batch inference.

        vLLM handles batching internally with continuous batching,
        so we process all cases together for optimal throughput.

        Args:
            patient_cases: List of patient cases.
            batch_size: Kept for API compatibility (ignored by vLLM).
            show_progress: Whether to show progress bars.

        Returns:
            List of EvidenceRLResults.
        """
        if not patient_cases:
            return []

        # Step 1: Clean patient contexts and pre-retrieval for all patients
        # In baseline mode, skip retrieval entirely
        if self.baseline_mode:
            print(f"[vLLM Pipeline] BASELINE MODE: Skipping retrieval for {len(patient_cases)} patients...")
            cleaned_contexts = []
            all_pre_evidence = []
            iterator = patient_cases
            if show_progress:
                iterator = tqdm(patient_cases, desc="Preprocessing (no-RAG)")

            for patient_case in iterator:
                cleaned = clean_patient_information(patient_case.context)
                cleaned_contexts.append(cleaned)
                all_pre_evidence.append([])  # Empty evidence list for baseline
        else:
            print(f"[vLLM Pipeline] Pre-retrieval for {len(patient_cases)} patients...")
            all_pre_evidence = []
            cleaned_contexts = []
            iterator = patient_cases
            if show_progress:
                iterator = tqdm(patient_cases, desc="Pre-retrieval")

            for patient_case in iterator:
                # Clean patient context (consistent with HuggingFace version)
                cleaned = clean_patient_information(patient_case.context)
                cleaned_contexts.append(cleaned)
                pre_evidence = self.retriever.retrieve(
                    query=cleaned,
                    top_k=self.top_k_pre,
                )
                all_pre_evidence.append(pre_evidence)

        # Step 2: Batch generation with vLLM
        print(f"[vLLM Pipeline] Generating diagnoses for {len(patient_cases)} patients...")
        contexts_and_evidence = [
            (cleaned_context, pre_evidence)
            for cleaned_context, pre_evidence in zip(cleaned_contexts, all_pre_evidence)
        ]

        structured_outputs = self.generator.generate_batch(
            contexts_and_evidence,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Step 3: Build results
        results = []
        for patient_case, cleaned_context, pre_evidence, structured_output in zip(
            patient_cases, cleaned_contexts, all_pre_evidence, structured_outputs
        ):
            results.append(EvidenceRLResult(
                hadm_id=patient_case.hadm_id,
                subject_id=patient_case.subject_id,
                patient_context=cleaned_context,
                ground_truth_diagnoses=list(patient_case.diagnoses),
                pre_evidence=pre_evidence,
                structured_output=structured_output,
                generation_prompt=self.generator.last_prompt or "",
            ))

        return results


__all__ = [
    "VLLMEvidenceRLPipeline",
    "summarize_results",
]
