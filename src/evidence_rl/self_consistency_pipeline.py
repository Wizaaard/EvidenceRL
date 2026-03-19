#!/usr/bin/env python3
"""Self-Consistency (SC) pipeline for EvidenceRL.

Implements Wang et al. (2023) Self-Consistency with semantic aggregation
for free-form medical diagnosis text.  Instead of exact-match majority
voting, uses embedding-based clustering to group semantically equivalent
diagnoses across N sampled reasoning paths.

Pipeline:
    1. Generate N diverse outputs per patient (higher temperature, vLLM n parameter)
    2. Parse each into StructuredDiagnosisOutput (5 diagnoses per sample)
    3. Pool all diagnosis names across N samples
    4. Cluster by embedding cosine similarity (>= threshold)
    5. Rank clusters by vote count (tiebreak: average position)
    6. For each top-5 cluster, select the best reasoning
    7. Emit final StructuredDiagnosisOutput

Usage:
    from evidence_rl.self_consistency_pipeline import SelfConsistencyVLLMPipeline

    pipeline = SelfConsistencyVLLMPipeline(
        model_name="path/to/model",
        num_samples=10,
        sc_temperature=0.9,
    )
    results = pipeline.run_batch(patient_cases)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from .baseline import PatientCase, clean_patient_information
from .evidence_pipeline import (
    DiagnosisWithReasoning,
    EvidenceRLResult,
    StructuredDiagnosisOutput,
    _build_structured_diagnosis_prompt,
    _parse_structured_output,
    summarize_results,
)
from .retrieval import HuggingFaceEmbedder
from .vllm_generation import (
    VLLMDiagnosisExtractor,
    VLLMStructuredDiagnosisGenerator,
    _strip_thinking_tokens,
    check_vllm_available,
)

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisVote:
    """A single diagnosis occurrence from one sample."""

    name: str
    reasoning: str
    sample_idx: int
    position: int  # 0-indexed rank within its sample's top-5


@dataclass
class DiagnosisCluster:
    """A cluster of semantically equivalent diagnoses across samples."""

    canonical_name: str
    votes: List[DiagnosisVote]
    vote_count: int          # Number of *distinct* samples containing this diagnosis
    avg_position: float      # Average position across all votes (lower = higher rank)
    best_reasoning: str = ""

    @property
    def score(self) -> float:
        """Composite ranking score: vote count primary, position tiebreak."""
        return self.vote_count * 100 - self.avg_position


@dataclass
class SCMetadata:
    """Metadata tracking the SC voting process for analysis."""

    num_samples: int
    temperature: float
    similarity_threshold: float
    num_parse_successes: int
    num_parse_failures: int
    total_diagnoses_pooled: int
    num_clusters: int
    cluster_details: List[Dict[str, Any]]
    per_sample_diagnoses: List[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "similarity_threshold": self.similarity_threshold,
            "num_parse_successes": self.num_parse_successes,
            "num_parse_failures": self.num_parse_failures,
            "total_diagnoses_pooled": self.total_diagnoses_pooled,
            "num_clusters": self.num_clusters,
            "cluster_details": self.cluster_details,
            "per_sample_diagnoses": self.per_sample_diagnoses,
        }


# ---------------------------------------------------------------------------
# Clustering algorithm
# ---------------------------------------------------------------------------

def cluster_diagnoses_by_embedding(
    votes: List[DiagnosisVote],
    embedder: HuggingFaceEmbedder,
    similarity_threshold: float = 0.85,
) -> List[DiagnosisCluster]:
    """Cluster diagnosis votes by embedding cosine similarity.

    Algorithm:
        1. Embed all unique diagnosis names.
        2. Greedy agglomerative clustering: process names sorted by frequency
           (most common first).  For each name, merge into the first existing
           cluster whose centroid has cosine similarity >= *threshold*.  If no
           match, start a new cluster.
        3. Build ``DiagnosisCluster`` objects with per-cluster statistics.

    Returns clusters sorted by ``score`` (vote_count desc, avg_position asc).
    """
    if not votes:
        return []

    # --- 1. Unique names & embeddings ---
    unique_names = list({v.name for v in votes})
    if not unique_names:
        return []

    embeddings = embedder.encode(unique_names)
    emb_matrix = np.array(embeddings, dtype=np.float32)

    # L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(unique_names)}

    # Pairwise cosine similarity
    sim_matrix = emb_matrix @ emb_matrix.T  # (U, U)

    # --- 2. Greedy agglomerative clustering ---
    # Process names by descending frequency so the most common name becomes
    # the centroid (and thus the canonical name) of its cluster.
    name_counts = Counter(v.name for v in votes)
    sorted_names = sorted(unique_names, key=lambda n: -name_counts[n])

    clusters_names: List[List[str]] = []   # each cluster is a list of names
    cluster_centroid_idx: List[int] = []   # embedding index of centroid
    assigned: set = set()

    for name in sorted_names:
        if name in assigned:
            continue

        idx = name_to_idx[name]

        # Try merging into existing cluster
        merged = False
        for ci, centroid_idx in enumerate(cluster_centroid_idx):
            if sim_matrix[idx, centroid_idx] >= similarity_threshold:
                clusters_names[ci].append(name)
                assigned.add(name)
                merged = True
                break

        if not merged:
            clusters_names.append([name])
            cluster_centroid_idx.append(idx)
            assigned.add(name)

    # --- 3. Build DiagnosisCluster objects ---
    result: List[DiagnosisCluster] = []
    for cluster_name_list in clusters_names:
        name_set = set(cluster_name_list)
        cluster_votes = [v for v in votes if v.name in name_set]

        # Canonical name: most frequent in the cluster
        freq = Counter(v.name for v in cluster_votes)
        canonical = freq.most_common(1)[0][0]

        # Vote count: number of *distinct* samples (not raw occurrences)
        distinct_samples = len({v.sample_idx for v in cluster_votes})

        # Average position across all votes
        avg_pos = sum(v.position for v in cluster_votes) / len(cluster_votes)

        # Best reasoning
        best_reasoning = _select_best_reasoning(cluster_votes)

        result.append(DiagnosisCluster(
            canonical_name=canonical,
            votes=cluster_votes,
            vote_count=distinct_samples,
            avg_position=avg_pos,
            best_reasoning=best_reasoning,
        ))

    # Sort by composite score (descending)
    result.sort(key=lambda c: c.score, reverse=True)
    return result


def _select_best_reasoning(votes: List[DiagnosisVote]) -> str:
    """Pick the best reasoning from a cluster of votes.

    Strategy: among votes ranked in the top-2 positions within their sample,
    pick the longest reasoning.  Falls back to all votes if none are top-2.
    """
    if not votes:
        return ""
    if len(votes) == 1:
        return votes[0].reasoning

    top_ranked = [v for v in votes if v.position <= 1]
    candidates = top_ranked if top_ranked else votes
    best = max(candidates, key=lambda v: len(v.reasoning))
    return best.reasoning


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SelfConsistencyVLLMPipeline:
    """Self-Consistency pipeline: generate N samples -> cluster -> vote.

    Uses vLLM with ``SamplingParams(n=num_samples)`` so that N completions
    are produced per prompt in a single forward pass.  No retrieval is
    performed — this is a pure generation-time technique.
    """

    def __init__(
        self,
        model_name: str = "",
        embedding_model_name: Optional[str] = None,
        tensor_parallel_size: int = 2,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.90,
        num_samples: int = 10,
        sc_temperature: float = 0.9,
        similarity_threshold: float = 0.85,
        generation_kwargs: Optional[Mapping[str, Any]] = None,
        use_guided_json: bool = False,
        lora_path: Optional[str] = None,
        max_lora_rank: int = 64,
        extractor_model_name: Optional[str] = None,
    ) -> None:
        check_vllm_available()

        self.model_name = model_name
        self.num_samples = num_samples
        self.sc_temperature = sc_temperature
        self.similarity_threshold = similarity_threshold
        self.extractor_model_name = extractor_model_name

        # Embedder for diagnosis-name clustering (lightweight, CPU-only).
        # force_cpu=True ensures the embedder stays off GPUs owned by vLLM.
        print("[SC Pipeline] Initializing embedder for diagnosis clustering...")
        self.embedder = HuggingFaceEmbedder(
            model_name=embedding_model_name or "FremyCompany/BioLORD-2023",
            force_cpu=True,
        )

        # Initialise the vLLM generator (reuse the existing class for model
        # loading, but we will drive generation with our own SamplingParams).
        print("[SC Pipeline] Initializing vLLM generator...")
        self.generator = VLLMStructuredDiagnosisGenerator(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            generation_kwargs=generation_kwargs,
            use_llm_extraction=False,  # extraction handled below if needed
            use_guided_json=use_guided_json,
            lora_path=lora_path,
            max_lora_rank=max_lora_rank,
        )

        # LLM extractor for rescuing failed JSON parses.
        # By default, the generator model itself is reused (zero extra cost).
        # When a separate extractor_model_name is provided, that model is
        # lazily loaded after generation (generator freed first to reclaim GPU).
        self._uses_separate_extractor = (
            extractor_model_name is not None
            and extractor_model_name != model_name
        )
        extractor_name = extractor_model_name or model_name

        if self._uses_separate_extractor:
            print(f"[SC Pipeline] Extractor model configured: {extractor_name}")
            print(f"[SC Pipeline] Extractor will be loaded on-demand after generation")
            self._llm_extractor: Optional[VLLMDiagnosisExtractor] = VLLMDiagnosisExtractor(
                model_name=extractor_name,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                llm_instance=None,
            )
        else:
            # Reuse generator LLM instance (set in run_batch after _init_llm)
            print(f"[SC Pipeline] Extractor reusing generator model for failed parse rescue")
            self._llm_extractor = VLLMDiagnosisExtractor(
                model_name=extractor_name,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                llm_instance=None,  # will be set in run_batch after _init_llm
            )

        # Build SC-specific sampling params (higher temperature, n > 1)
        gen_kwargs = dict(generation_kwargs or {})
        self._sc_sampling_params = SamplingParams(
            temperature=self.sc_temperature,
            top_p=gen_kwargs.get("top_p", 0.9),
            top_k=gen_kwargs.get("top_k", 50),
            max_tokens=max_tokens,
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.15),
            frequency_penalty=gen_kwargs.get("frequency_penalty", 0.1),
            n=self.num_samples,
        )

        self._lora_request = None
        self._lora_path = lora_path
        self._max_lora_rank = max_lora_rank

        print(
            f"[SC Pipeline] Initialization complete. "
            f"Mode: SELF-CONSISTENCY (n={num_samples}, T={sc_temperature}, "
            f"sim_thresh={similarity_threshold})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_batch(
        self,
        patient_cases: Sequence[PatientCase],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[EvidenceRLResult]:
        """Process patient cases with the Self-Consistency pipeline.

        Steps:
            1. Build baseline prompts (no evidence) for all patients.
            2. Generate N completions per prompt via vLLM ``n`` parameter.
            3. Parse each completion into StructuredDiagnosisOutput.
            4. Cluster & vote to select top-5 diagnoses per patient.
            5. Assemble EvidenceRLResult with SC metadata.
        """
        if not patient_cases:
            return []

        # ── Preprocessing ──
        print(f"[SC] Preprocessing {len(patient_cases)} patients...")
        cleaned_contexts: List[str] = []
        iterator: Any = patient_cases
        if show_progress:
            iterator = tqdm(patient_cases, desc="Preprocessing")
        for pc in iterator:
            cleaned_contexts.append(clean_patient_information(pc.context))

        # ── Step 1: Build baseline prompts (no evidence) ──
        conversations = [
            [{"role": "user", "content": _build_structured_diagnosis_prompt(ctx, [])}]
            for ctx in cleaned_contexts
        ]

        # ── Step 2: Generate N completions per prompt ──
        self.generator._init_llm()

        # If extractor reuses the generator model, share the LLM instance
        if self._llm_extractor and not self._uses_separate_extractor:
            self._llm_extractor._llm = self.generator._llm

        # Prepare LoRA request if needed
        if self._lora_path and self._lora_request is None:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest(
                lora_name="dpo_adapter",
                lora_int_id=1,
                lora_path=self._lora_path,
            )

        chat_kwargs: Dict[str, Any] = {"sampling_params": self._sc_sampling_params}
        if self._lora_request:
            chat_kwargs["lora_request"] = self._lora_request

        print(
            f"[SC] Generating {len(conversations)} x {self.num_samples} "
            f"completions (T={self.sc_temperature})..."
        )
        raw_outputs = self.generator._llm.chat(conversations, **chat_kwargs)

        # ── Step 3: Parse all N outputs per patient ──
        total_completions = sum(len(ro.outputs) for ro in raw_outputs)
        print(f"[SC] Parsing {total_completions} completions...")
        all_parsed: List[List[StructuredDiagnosisOutput]] = []

        parse_iterator: Any = enumerate(raw_outputs)
        if show_progress:
            parse_iterator = tqdm(
                parse_iterator, total=len(raw_outputs), desc="Parsing"
            )

        for _i, request_output in parse_iterator:
            patient_samples: List[StructuredDiagnosisOutput] = []
            for completion in request_output.outputs:
                text = _strip_thinking_tokens(completion.text.strip())
                parsed = _parse_structured_output(
                    text, llm_extractor=None, skip_fallback=True
                )
                patient_samples.append(parsed)
            all_parsed.append(patient_samples)

        # ── Step 3b: LLM extraction for failed parses ──
        if self._llm_extractor is not None:
            # Collect all failed parses across all patients with their indices
            failed_refs: List[tuple[int, int]] = []  # (patient_idx, sample_idx)
            failed_raw: List[str] = []
            for p_idx, patient_samples in enumerate(all_parsed):
                for s_idx, sample in enumerate(patient_samples):
                    if not sample.parse_success and sample.raw_output:
                        failed_refs.append((p_idx, s_idx))
                        failed_raw.append(sample.raw_output)

            if failed_raw:
                print(
                    f"[SC] Rescuing {len(failed_raw)} failed parses "
                    f"via extractor ({self.extractor_model_name or 'generator model'})..."
                )

                # If using a separate extractor, free the generator first
                if self._uses_separate_extractor:
                    self.generator._free_generator_llm()

                extracted_batches = self._llm_extractor.extract_batch(
                    failed_raw, show_progress=show_progress,
                )

                rescued = 0
                for (p_idx, s_idx), diagnoses in zip(failed_refs, extracted_batches):
                    sample = all_parsed[p_idx][s_idx]
                    sample.diagnoses = diagnoses
                    complete = sum(1 for d in diagnoses if d.name and d.reasoning)
                    if complete >= 3:
                        sample.parse_success = True
                        sample.parse_error = None
                        rescued += 1

                print(f"[SC] Extractor rescued {rescued}/{len(failed_raw)} failed parses")

        # ── Step 4: Cluster & vote ──
        print("[SC] Clustering and voting...")
        results: List[EvidenceRLResult] = []

        vote_iterator: Any = enumerate(zip(patient_cases, cleaned_contexts, all_parsed))
        if show_progress:
            vote_iterator = tqdm(
                vote_iterator, total=len(patient_cases), desc="Voting"
            )

        for idx, (pc, ctx, samples) in vote_iterator:
            final_output, metadata = self._aggregate_samples(samples)

            result = EvidenceRLResult(
                hadm_id=pc.hadm_id,
                subject_id=pc.subject_id,
                patient_context=ctx,
                ground_truth_diagnoses=list(pc.diagnoses),
                pre_evidence=[],  # No retrieval
                structured_output=final_output,
                generation_prompt=conversations[idx][0]["content"],
            )
            result._sc_metadata = metadata  # type: ignore[attr-defined]
            results.append(result)

        # Summary
        parse_rates = [
            m.num_parse_successes / max(m.num_parse_successes + m.num_parse_failures, 1)
            for r in results
            for m in [r._sc_metadata]  # type: ignore[attr-defined]
        ]
        avg_parse = sum(parse_rates) / len(parse_rates) if parse_rates else 0
        avg_clusters = sum(
            r._sc_metadata.num_clusters for r in results  # type: ignore[attr-defined]
        ) / max(len(results), 1)

        print(
            f"[SC] Complete: {len(results)} patients, "
            f"avg parse rate {avg_parse:.1%}, "
            f"avg {avg_clusters:.1f} clusters per patient"
        )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_samples(
        self,
        samples: List[StructuredDiagnosisOutput],
    ) -> tuple[StructuredDiagnosisOutput, SCMetadata]:
        """Cluster diagnoses across N samples and select top-5 by vote."""

        # Collect votes from successfully parsed samples
        votes: List[DiagnosisVote] = []
        num_successes = 0
        num_failures = 0
        per_sample_diags: List[List[str]] = []

        for sample_idx, sample in enumerate(samples):
            if not sample.parse_success or not sample.diagnoses:
                num_failures += 1
                per_sample_diags.append([])
                continue

            num_successes += 1
            sample_names: List[str] = []
            for pos, diag in enumerate(sample.diagnoses):
                name = diag.name.strip()
                if name:
                    votes.append(DiagnosisVote(
                        name=name,
                        reasoning=diag.reasoning,
                        sample_idx=sample_idx,
                        position=pos,
                    ))
                    sample_names.append(name)
            per_sample_diags.append(sample_names)

        # Edge case: nothing parsed at all — return empty
        if not votes:
            metadata = SCMetadata(
                num_samples=len(samples),
                temperature=self.sc_temperature,
                similarity_threshold=self.similarity_threshold,
                num_parse_successes=num_successes,
                num_parse_failures=num_failures,
                total_diagnoses_pooled=0,
                num_clusters=0,
                cluster_details=[],
                per_sample_diagnoses=per_sample_diags,
            )
            return (
                StructuredDiagnosisOutput(
                    raw_output="",
                    parse_success=False,
                    parse_error="SC: all samples failed to parse",
                ),
                metadata,
            )

        # Cluster
        clusters = cluster_diagnoses_by_embedding(
            votes, self.embedder, self.similarity_threshold
        )

        # Assemble top-5
        diagnoses: List[DiagnosisWithReasoning] = []
        for cluster in clusters[:5]:
            diagnoses.append(DiagnosisWithReasoning(
                name=cluster.canonical_name,
                reasoning=cluster.best_reasoning,
            ))

        # Pad to 5 if fewer unique clusters
        while len(diagnoses) < 5:
            diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))

        has_enough = sum(1 for d in diagnoses if d.name) >= 3
        output = StructuredDiagnosisOutput(
            diagnoses=diagnoses,
            raw_output="[self-consistency aggregated]",
            parse_success=has_enough,
            parse_error=None if has_enough else "SC: fewer than 3 unique diagnosis clusters",
        )

        # Build metadata
        cluster_details = [
            {
                "canonical_name": c.canonical_name,
                "vote_count": c.vote_count,
                "avg_position": round(c.avg_position, 3),
                "all_names": sorted({v.name for v in c.votes}),
            }
            for c in clusters
        ]

        metadata = SCMetadata(
            num_samples=len(samples),
            temperature=self.sc_temperature,
            similarity_threshold=self.similarity_threshold,
            num_parse_successes=num_successes,
            num_parse_failures=num_failures,
            total_diagnoses_pooled=len(votes),
            num_clusters=len(clusters),
            cluster_details=cluster_details,
            per_sample_diagnoses=per_sample_diags,
        )

        return output, metadata


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def sc_result_to_dict(result: EvidenceRLResult) -> Dict[str, Any]:
    """Serialise an EvidenceRLResult with SC metadata."""
    d = result.to_dict()
    if hasattr(result, "_sc_metadata") and result._sc_metadata is not None:
        d["sc_metadata"] = result._sc_metadata.to_dict()
    return d


__all__ = [
    "SelfConsistencyVLLMPipeline",
    "SCMetadata",
    "DiagnosisCluster",
    "DiagnosisVote",
    "cluster_diagnoses_by_embedding",
    "sc_result_to_dict",
    "summarize_results",
]
