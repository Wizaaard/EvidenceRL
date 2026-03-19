#!/usr/bin/env python3
"""Self-RAG pipeline for EvidenceRL.

Implements adaptive retrieval: the model first generates diagnoses zero-shot,
then critiques its own output to decide which diagnoses need clinical evidence,
retrieves evidence only for uncertain diagnoses, and finally refines its output.

Pipeline:
    1. Zero-shot generation (baseline prompt)
    2. Self-critique (confidence + needs_evidence per diagnosis)
    3. Conditional retrieval (FAISS, only for uncertain diagnoses)
    4. Refinement (regenerate with evidence, only if retrieval was triggered)

Usage:
    from evidence_rl.self_rag_pipeline import SelfRAGVLLMPipeline

    pipeline = SelfRAGVLLMPipeline(
        model_name="path/to/model",
        faiss_index_dir="path/to/faiss_index",
        tensor_parallel_size=2,
    )
    results = pipeline.run_batch(patient_cases)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

from .documents import Document
from .baseline import PatientCase, clean_patient_information
from .retrieval import HuggingFaceEmbedder
from .evidence_pipeline import (
    EvidenceRLResult,
    StructuredDiagnosisOutput,
    _build_structured_diagnosis_prompt,
    summarize_results,
)
from .vllm_generation import _strip_thinking_tokens
from .faiss_retrieval import FAISSDocumentStore, FAISSRetriever
from .vllm_generation import VLLMStructuredDiagnosisGenerator, check_vllm_available
from .self_rag_prompts import (
    SelfRAGCritique,
    SelfRAGMetadata,
    build_selfrag_critique_prompt,
    build_selfrag_refinement_prompt,
    build_retrieval_query,
    build_fallback_critique,
    parse_selfrag_critique,
)

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None


class SelfRAGVLLMPipeline:
    """Self-RAG pipeline: generate -> critique -> conditionally retrieve -> refine.

    Uses a single vLLM model instance for all three LLM calls (generation,
    critique, refinement). Retrieval is conditional based on the model's
    self-assessed confidence.
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
        use_guided_json: bool = False,
        extractor_model_name: Optional[str] = None,
    ) -> None:
        """Initialize the Self-RAG pipeline.

        Unlike baseline mode, Self-RAG always initializes the retriever
        (for conditional use) and the generator (for all 3 steps).

        Args:
            documents: Knowledge documents to index (or None if faiss_index_dir provided).
            model_name: Model path for generation, critique, and refinement.
            embedding_model_name: Model for embeddings.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            max_tokens: Maximum tokens to generate.
            gpu_memory_utilization: Fraction of GPU memory to use.
            top_k_pre: Number of documents to retrieve per query.
            generation_kwargs: Additional generation kwargs.
            chunk_size: Chunk size for document splitting.
            chunk_overlap: Overlap between chunks.
            embedding_batch_size: Batch size for embedding.
            use_llm_extraction: Whether to use LLM fallback for parsing.
            faiss_index_dir: Path to pre-built FAISS index.
            use_guided_json: Whether to use guided JSON decoding.
            extractor_model_name: Optional separate model for fallback extraction.
        """
        check_vllm_available()

        self.top_k_pre = top_k_pre
        self.model_name = model_name

        # Self-RAG always needs a retriever (for conditional use)
        print("[Self-RAG Pipeline] Initializing retrieval components...")
        self.embedder = HuggingFaceEmbedder(
            model_name=embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2",
        )

        if faiss_index_dir:
            print(f"[Self-RAG Pipeline] Loading pre-built FAISS index from: {faiss_index_dir}")
            if FAISSDocumentStore.index_exists(faiss_index_dir):
                self.store = FAISSDocumentStore.load(
                    faiss_index_dir,
                    embedder=self.embedder,
                )
                self.retriever = FAISSRetriever(self.store)
            else:
                raise ValueError(
                    f"FAISS index not found at {faiss_index_dir}. "
                    "Self-RAG requires a knowledge index for conditional retrieval."
                )
        elif documents is not None:
            print(f"[Self-RAG Pipeline] Building FAISS index from {len(documents)} documents...")
            self.store = FAISSDocumentStore(
                documents=documents,
                embedder=self.embedder,
            )
            self.retriever = FAISSRetriever(self.store)
        else:
            raise ValueError(
                "Self-RAG requires a knowledge source. "
                "Provide documents or faiss_index_dir."
            )

        # Initialize vLLM generator (reused for all 3 steps)
        print("[Self-RAG Pipeline] Initializing vLLM generator...")
        self.generator = VLLMStructuredDiagnosisGenerator(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            generation_kwargs=generation_kwargs,
            use_llm_extraction=use_llm_extraction,
            use_guided_json=use_guided_json,
            extractor_model_name=extractor_model_name,
        )

        # Critique uses shorter max_tokens and lower temperature for determinism
        gen_kwargs = dict(generation_kwargs or {})
        self._critique_sampling_params = SamplingParams(
            temperature=gen_kwargs.get("temperature", 0.3),
            top_p=0.9,
            max_tokens=1024,  # Critique output is shorter than diagnosis
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.15),
        )

        print("[Self-RAG Pipeline] Initialization complete. Mode: SELF-RAG (adaptive retrieval)")

    def run_batch(
        self,
        patient_cases: Sequence[PatientCase],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[EvidenceRLResult]:
        """Process patient cases with the self-RAG pipeline.

        Steps:
            1. Zero-shot batch generation (no evidence)
            2. Batch self-critique (confidence assessment)
            3. Targeted retrieval (only for uncertain diagnoses)
            4. Batch refinement (only patients with retrieved evidence)

        Args:
            patient_cases: Patient cases to process.
            batch_size: Kept for API compatibility.
            show_progress: Whether to show progress bars.

        Returns:
            List of EvidenceRLResults with self_rag_metadata.
        """
        if not patient_cases:
            return []

        # Preprocess: clean patient contexts
        print(f"[Self-RAG] Preprocessing {len(patient_cases)} patients...")
        cleaned_contexts = []
        iterator = patient_cases
        if show_progress:
            iterator = tqdm(patient_cases, desc="Preprocessing")
        for patient_case in iterator:
            cleaned_contexts.append(clean_patient_information(patient_case.context))

        # ─── STEP 1: Zero-shot generation (baseline prompt, no evidence) ───
        print(f"[Self-RAG] Step 1/3: Zero-shot generation for {len(patient_cases)} patients...")
        initial_outputs = self.generator.generate_batch(
            [(ctx, []) for ctx in cleaned_contexts],
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # ─── Clean up extractor if it stole GPU memory from the generator ───
        # When a separate extractor model is used, generate_batch() frees the
        # generator LLM and loads the extractor model.  We must free the
        # extractor before re-initializing the generator for critique/refinement.
        if (
            self.generator._llm is None
            and self.generator.llm_extractor is not None
            and hasattr(self.generator.llm_extractor, "_llm")
            and self.generator.llm_extractor._llm is not None
        ):
            import gc
            import torch
            print("[Self-RAG] Freeing extractor model to reclaim GPU memory...")
            del self.generator.llm_extractor._llm
            self.generator.llm_extractor._llm = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[Self-RAG] Extractor model freed.")

        # ─── STEP 2: Self-critique (batch) ───
        print(f"[Self-RAG] Step 2/3: Self-critique for {len(patient_cases)} patients...")
        self.generator._init_llm()  # Ensure LLM is initialized

        critique_conversations = []
        for ctx, output in zip(cleaned_contexts, initial_outputs):
            prompt = build_selfrag_critique_prompt(ctx, output)
            critique_conversations.append([{"role": "user", "content": prompt}])

        # Use the generator's LLM directly for critique (raw chat, not diagnosis parsing)
        critique_raw_outputs = self.generator._llm.chat(
            critique_conversations,
            sampling_params=self._critique_sampling_params,
        )

        # Parse critique outputs
        critiques: List[SelfRAGCritique] = []
        for i, raw_output in enumerate(critique_raw_outputs):
            if not raw_output.outputs:
                critiques.append(build_fallback_critique(initial_outputs[i]))
                continue

            raw_text = _strip_thinking_tokens(raw_output.outputs[0].text.strip())
            critique = parse_selfrag_critique(raw_text)

            if not critique.parse_success:
                print(f"[Self-RAG] Critique parse failed for patient {i}: {critique.parse_error}")
                print(f"[Self-RAG] Using fallback (all uncertain) for patient {i}")
                critique = build_fallback_critique(initial_outputs[i])

            critiques.append(critique)

        # Log critique statistics
        num_need_retrieval = sum(1 for c in critiques if c.needs_retrieval)
        total_uncertain = sum(c.num_uncertain for c in critiques)
        total_diagnoses = len(patient_cases) * 5
        print(f"[Self-RAG] Critique results: {num_need_retrieval}/{len(patient_cases)} patients need retrieval")
        print(f"[Self-RAG] {total_uncertain}/{total_diagnoses} diagnoses marked as uncertain")

        # ─── STEP 3: Targeted retrieval + refinement ───
        print(f"[Self-RAG] Step 3/3: Targeted retrieval and refinement...")

        # Retrieve evidence for patients with uncertain diagnoses
        all_evidence = []
        all_queries = []
        retrieval_iterator = range(len(patient_cases))
        if show_progress:
            retrieval_iterator = tqdm(retrieval_iterator, desc="Targeted retrieval")

        for i in retrieval_iterator:
            if critiques[i].needs_retrieval:
                query = build_retrieval_query(initial_outputs[i], critiques[i])
                evidence = self.retriever.retrieve(query=query, top_k=self.top_k_pre)
                all_evidence.append(evidence)
                all_queries.append(query)
            else:
                all_evidence.append([])
                all_queries.append("")

        # Refinement: only for patients that retrieved evidence
        needs_refinement = [i for i, ev in enumerate(all_evidence) if ev]
        final_outputs = list(initial_outputs)  # Copy; will replace refined ones

        if needs_refinement:
            print(f"[Self-RAG] Refining {len(needs_refinement)}/{len(patient_cases)} patients with evidence...")

            # Build refinement prompts
            refinement_inputs = []
            for i in needs_refinement:
                prompt = build_selfrag_refinement_prompt(
                    cleaned_contexts[i],
                    initial_outputs[i],
                    all_evidence[i],
                    critiques[i],
                )
                refinement_inputs.append((prompt, all_evidence[i]))

            # Generate refined diagnoses using the standard prompt path
            # We build conversations manually since refinement uses a custom prompt
            refinement_conversations = [
                [{"role": "user", "content": prompt}]
                for prompt, _ in refinement_inputs
            ]
            refinement_raw = self.generator._llm.chat(
                refinement_conversations,
                sampling_params=self.generator._sampling_params,
            )

            # Parse refinement outputs
            from .evidence_pipeline import _parse_structured_output
            for idx, (patient_idx, raw_output) in enumerate(zip(needs_refinement, refinement_raw)):
                if not raw_output.outputs:
                    print(f"[Self-RAG] Refinement produced no output for patient {patient_idx}, keeping initial")
                    continue

                raw_text = _strip_thinking_tokens(raw_output.outputs[0].text.strip())
                refined = _parse_structured_output(raw_text, llm_extractor=None, skip_fallback=True)

                if refined.parse_success and refined.diagnoses:
                    final_outputs[patient_idx] = refined
                else:
                    print(f"[Self-RAG] Refinement parse failed for patient {patient_idx}, keeping initial")
        else:
            print("[Self-RAG] No patients need refinement (all diagnoses high confidence)")

        # ─── Build results ───
        results = []
        for i, patient_case in enumerate(patient_cases):
            # Build self-RAG metadata
            metadata = SelfRAGMetadata(
                initial_diagnoses=[
                    {"name": d.name, "reasoning": d.reasoning}
                    for d in initial_outputs[i].diagnoses
                ],
                critique=critiques[i].to_dict(),
                retrieval_triggered=bool(all_evidence[i]),
                num_uncertain_diagnoses=critiques[i].num_uncertain,
                retrieval_query=all_queries[i],
            )

            result = EvidenceRLResult(
                hadm_id=patient_case.hadm_id,
                subject_id=patient_case.subject_id,
                patient_context=cleaned_contexts[i],
                ground_truth_diagnoses=list(patient_case.diagnoses),
                pre_evidence=all_evidence[i],
                structured_output=final_outputs[i],
                generation_prompt=self.generator.last_prompt or "",
            )

            # Attach metadata to the result for serialization
            result._self_rag_metadata = metadata

            results.append(result)

        # Print summary
        refined_count = len(needs_refinement)
        skipped_count = len(patient_cases) - refined_count
        print(f"[Self-RAG] Complete: {refined_count} refined with evidence, {skipped_count} kept zero-shot")

        return results


def selfrag_result_to_dict(result: EvidenceRLResult) -> Dict[str, Any]:
    """Serialize an EvidenceRLResult with self-RAG metadata.

    Extends the standard to_dict() with self_rag_metadata.
    """
    d = result.to_dict()
    if hasattr(result, "_self_rag_metadata") and result._self_rag_metadata is not None:
        d["self_rag_metadata"] = result._self_rag_metadata.to_dict()
    return d


__all__ = [
    "SelfRAGVLLMPipeline",
    "selfrag_result_to_dict",
    "summarize_results",
]
