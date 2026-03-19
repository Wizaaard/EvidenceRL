#!/usr/bin/env python3
"""Compute EvidenceRL metrics with vLLM-accelerated LLM judge.

This script uses vLLM for the LLM judge, offering 14-24x faster inference
compared to HuggingFace Transformers for precision/verdict computation.

Note: The NLI model (CrossEncoder) remains unchanged as it's already
optimized and doesn't benefit from vLLM.

Usage:
    python compute_evidencerl_metrics_vllm.py \
        --results-json <path> \
        --output-json <path> \
        --judge-model <path>

The output format is identical to compute_evidencerl_metrics.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

from tqdm.auto import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass
class EvidenceRLMetrics:
    """Metrics for a single patient case."""

    hadm_id: str
    subject_id: str

    # Ground truth
    ground_truth_diagnoses: List[str]

    # Predictions
    predicted_diagnoses: List[str]
    predicted_diagnoses_with_reasoning: List[Dict[str, str]]

    # Precision/Recall metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)

    # Per-diagnosis verdicts (True = correct, False = incorrect)
    verdicts: List[bool] = field(default_factory=list)

    # Per-diagnosis rewards
    per_diagnosis_rewards: List[Dict[str, float]] = field(default_factory=list)

    # Rewards@k (grounding + precision)
    reward_grounding_at_k: Dict[int, float] = field(default_factory=dict)
    reward_precision_at_k: Dict[int, float] = field(default_factory=dict)
    reward_combined_at_k: Dict[int, float] = field(default_factory=dict)

    # Evidence grounding analysis
    num_diagnoses_with_evidence: int = 0
    avg_reasoning_length: float = 0.0
    parse_success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hadm_id": self.hadm_id,
            "subject_id": self.subject_id,
            "ground_truth_diagnoses": self.ground_truth_diagnoses,
            "predicted_diagnoses": self.predicted_diagnoses,
            "predicted_diagnoses_with_reasoning": self.predicted_diagnoses_with_reasoning,
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "verdicts": self.verdicts,
            "per_diagnosis_rewards": self.per_diagnosis_rewards,
            "reward_grounding_at_k": {str(k): v for k, v in self.reward_grounding_at_k.items()},
            "reward_precision_at_k": {str(k): v for k, v in self.reward_precision_at_k.items()},
            "reward_combined_at_k": {str(k): v for k, v in self.reward_combined_at_k.items()},
            "num_diagnoses_with_evidence": self.num_diagnoses_with_evidence,
            "avg_reasoning_length": self.avg_reasoning_length,
            "parse_success": self.parse_success,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across patient cohort."""

    num_patients: int = 0

    # Precision/Recall averages
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)

    # Reward@k statistics (grounding + precision)
    avg_reward_grounding_at_k: Dict[int, float] = field(default_factory=dict)
    avg_reward_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_reward_combined_at_k: Dict[int, float] = field(default_factory=dict)

    # Evidence grounding statistics
    parse_success_rate: float = 0.0
    avg_diagnoses_per_patient: float = 0.0
    avg_diagnoses_with_evidence: float = 0.0
    avg_reasoning_length: float = 0.0

    # For hallucination analysis
    low_reward_cases: int = 0  # Cases with reward_combined < 0.3
    high_reward_cases: int = 0  # Cases with reward_combined >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_patients": self.num_patients,
            "avg_precision_at_k": {str(k): v for k, v in self.avg_precision_at_k.items()},
            "avg_recall_at_k": {str(k): v for k, v in self.avg_recall_at_k.items()},
            "avg_reward_grounding_at_k": {str(k): v for k, v in self.avg_reward_grounding_at_k.items()},
            "avg_reward_precision_at_k": {str(k): v for k, v in self.avg_reward_precision_at_k.items()},
            "avg_reward_combined_at_k": {str(k): v for k, v in self.avg_reward_combined_at_k.items()},
            "parse_success_rate": self.parse_success_rate,
            "avg_diagnoses_per_patient": self.avg_diagnoses_per_patient,
            "avg_diagnoses_with_evidence": self.avg_diagnoses_with_evidence,
            "avg_reasoning_length": self.avg_reasoning_length,
            "low_reward_cases": self.low_reward_cases,
            "high_reward_cases": self.high_reward_cases,
            "inference_engine": "vllm",
        }


def compute_reward_at_k(rewards: List[float], max_k: int = 5) -> Dict[int, float]:
    """Compute cumulative average reward at each k value."""
    result = {}
    for k in range(1, max_k + 1):
        considered = rewards[:k]
        if considered:
            result[k] = sum(considered) / len(considered)
        else:
            result[k] = 0.0
    return result


def compute_precision_recall_at_k(
    verdicts: List[bool],
    num_ground_truth: int,
    max_k: int = 5,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute precision@k and recall@k from verdicts."""
    precision_at_k = {}
    recall_at_k = {}

    for k in range(1, max_k + 1):
        considered = verdicts[:k]
        if not considered:
            precision_at_k[k] = 0.0
            recall_at_k[k] = 0.0
            continue

        hits = sum(considered)
        precision_at_k[k] = hits / len(considered)
        recall_at_k[k] = min(hits, num_ground_truth) / num_ground_truth if num_ground_truth > 0 else 0.0

    return precision_at_k, recall_at_k


def _extract_patient_data(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pre-extract patient data shared by both phases."""
    from evidence_rl.documents import Document, RetrievedDocument

    patients = []
    for result in results:
        structured_output = result.get("structured_output", {})
        diagnoses = structured_output.get("diagnoses", [])
        valid_diagnoses = [d for d in diagnoses if d.get("name", "").strip()]

        pre_evidence_data = result.get("pre_evidence", [])
        pre_evidence = []
        for ev in pre_evidence_data:
            doc = Document(
                doc_id=ev.get("doc_id", ""),
                text=ev.get("text", ""),
            )
            pre_evidence.append(RetrievedDocument(document=doc, score=ev.get("score", 0.0)))

        patients.append({
            "hadm_id": result["hadm_id"],
            "subject_id": result["subject_id"],
            "ground_truth": result["ground_truth_diagnoses"],
            "patient_context": result["patient_context"],
            "valid_diagnoses": valid_diagnoses,
            "predicted": [d["name"] for d in valid_diagnoses],
            "predicted_with_reasoning": [
                {"name": d["name"], "reasoning": d.get("reasoning", "")}
                for d in valid_diagnoses
            ],
            "pre_evidence": pre_evidence,
            "parse_success": structured_output.get("parse_success", False),
        })
    return patients


def _phase1_judge_accuracy(
    patients: List[Dict[str, Any]],
    judge,
) -> List[Dict[str, Any]]:
    """Phase 1: Compute accuracy verdicts using vLLM judge on GPU."""
    accuracy_results = []
    for patient in tqdm(patients, desc="Phase 1: Accuracy (vLLM judge)"):
        precision_rewards = []
        verdicts = []
        for diag_data in patient["valid_diagnoses"]:
            diagnosis_name = diag_data.get("name", "")
            is_correct = judge.is_correct(
                diagnosis_name, patient["ground_truth"]
            )
            precision_rewards.append(1.0 if is_correct else 0.0)
            verdicts.append(is_correct)
        accuracy_results.append({
            "precision_rewards": precision_rewards,
            "verdicts": verdicts,
        })
    return accuracy_results


def _phase2_nli_grounding(
    patients: List[Dict[str, Any]],
    nli_model,
) -> List[Dict[str, Any]]:
    """Phase 2: Compute NLI grounding scores (on GPU after vLLM is freed)."""
    from evidence_rl.reward import reward_grounding

    grounding_results = []
    for patient in tqdm(patients, desc="Phase 2: Grounding (NLI)"):
        grounding_rewards = []
        for diag_data in patient["valid_diagnoses"]:
            reasoning = diag_data.get("reasoning", "")
            if not reasoning.strip():
                grounding_rewards.append(0.0)
                continue
            g_max, _g_avg = reward_grounding(
                diagnosis_reasoning=reasoning,
                patient_context=patient["patient_context"],
                evidence_docs=patient["pre_evidence"],
                nli_model=nli_model,
            )
            grounding_rewards.append(g_max)
        grounding_results.append({"grounding_rewards": grounding_rewards})
    return grounding_results


def _merge_metrics(
    patients: List[Dict[str, Any]],
    accuracy_results: List[Dict[str, Any]],
    grounding_results: List[Dict[str, Any]],
    max_k: int = 5,
    reward_weight_grounding: float = 0.5,
) -> List[EvidenceRLMetrics]:
    """Merge accuracy and grounding results into final EvidenceRLMetrics."""
    all_metrics = []
    for patient, acc, grnd in zip(patients, accuracy_results, grounding_results):
        precision_rewards = acc["precision_rewards"]
        verdicts = acc["verdicts"]
        grounding_rewards = grnd["grounding_rewards"]
        valid_diagnoses = patient["valid_diagnoses"]

        combined_rewards = [
            reward_weight_grounding * g + (1.0 - reward_weight_grounding) * p
            for g, p in zip(grounding_rewards, precision_rewards)
        ]

        per_diagnosis_rewards = []
        for i, diag_data in enumerate(valid_diagnoses):
            per_diagnosis_rewards.append({
                "diagnosis_name": diag_data.get("name", ""),
                "grounding": float(grounding_rewards[i]),
                "precision": float(precision_rewards[i]),
                "combined": float(combined_rewards[i]),
            })

        reward_grounding_at_k = compute_reward_at_k(grounding_rewards, max_k)
        reward_precision_at_k = compute_reward_at_k(precision_rewards, max_k)
        reward_combined_at_k = compute_reward_at_k(combined_rewards, max_k)

        precision_at_k, recall_at_k = compute_precision_recall_at_k(
            verdicts, len(patient["ground_truth"]), max_k
        )

        num_with_evidence = sum(1 for d in valid_diagnoses if d.get("reasoning"))
        avg_reasoning_len = (
            sum(len(d.get("reasoning", "")) for d in valid_diagnoses) / len(valid_diagnoses)
            if valid_diagnoses else 0.0
        )

        metrics = EvidenceRLMetrics(
            hadm_id=patient["hadm_id"],
            subject_id=patient["subject_id"],
            ground_truth_diagnoses=patient["ground_truth"],
            predicted_diagnoses=patient["predicted"],
            predicted_diagnoses_with_reasoning=patient["predicted_with_reasoning"],
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            verdicts=verdicts,
            per_diagnosis_rewards=per_diagnosis_rewards,
            reward_grounding_at_k=reward_grounding_at_k,
            reward_precision_at_k=reward_precision_at_k,
            reward_combined_at_k=reward_combined_at_k,
            num_diagnoses_with_evidence=num_with_evidence,
            avg_reasoning_length=avg_reasoning_len,
            parse_success=patient["parse_success"],
        )
        all_metrics.append(metrics)

    return all_metrics


def compute_aggregate_metrics(patient_metrics: List[EvidenceRLMetrics]) -> AggregateMetrics:
    """Compute aggregate metrics across all patients."""

    num_patients = len(patient_metrics)
    if num_patients == 0:
        return AggregateMetrics()

    # Average precision/recall at k
    avg_precision: Dict[int, float] = {}
    avg_recall: Dict[int, float] = {}

    # Get all k values
    all_k = set()
    for m in patient_metrics:
        all_k.update(m.precision_at_k.keys())

    for k in sorted(all_k):
        precision_sum = sum(m.precision_at_k.get(k, 0.0) for m in patient_metrics)
        recall_sum = sum(m.recall_at_k.get(k, 0.0) for m in patient_metrics)
        avg_precision[k] = precision_sum / num_patients
        avg_recall[k] = recall_sum / num_patients

    # Average rewards@k (grounding + precision)
    all_k_values = set()
    for m in patient_metrics:
        all_k_values.update(m.reward_grounding_at_k.keys())

    avg_reward_grounding_at_k = {}
    avg_reward_precision_at_k = {}
    avg_reward_combined_at_k = {}

    for k in sorted(all_k_values):
        grounding_k = [m.reward_grounding_at_k.get(k, 0.0) for m in patient_metrics]
        precision_k = [m.reward_precision_at_k.get(k, 0.0) for m in patient_metrics]
        combined_k = [m.reward_combined_at_k.get(k, 0.0) for m in patient_metrics]

        avg_reward_grounding_at_k[k] = sum(grounding_k) / num_patients
        avg_reward_precision_at_k[k] = sum(precision_k) / num_patients
        avg_reward_combined_at_k[k] = sum(combined_k) / num_patients

    # Evidence grounding statistics
    parse_success_count = sum(1 for m in patient_metrics if m.parse_success)
    parse_success_rate = parse_success_count / num_patients

    avg_diagnoses = sum(len(m.predicted_diagnoses) for m in patient_metrics) / num_patients
    avg_diagnoses_with_evidence = sum(m.num_diagnoses_with_evidence for m in patient_metrics) / num_patients
    avg_reasoning_length = sum(m.avg_reasoning_length for m in patient_metrics) / num_patients

    # Hallucination analysis (use reward_combined@3 as representative)
    low_reward_cases = sum(1 for m in patient_metrics if m.reward_combined_at_k.get(3, 0.0) < 0.3)
    high_reward_cases = sum(1 for m in patient_metrics if m.reward_combined_at_k.get(3, 0.0) >= 0.7)

    return AggregateMetrics(
        num_patients=num_patients,
        avg_precision_at_k=avg_precision,
        avg_recall_at_k=avg_recall,
        avg_reward_grounding_at_k=avg_reward_grounding_at_k,
        avg_reward_precision_at_k=avg_reward_precision_at_k,
        avg_reward_combined_at_k=avg_reward_combined_at_k,
        parse_success_rate=parse_success_rate,
        avg_diagnoses_per_patient=avg_diagnoses,
        avg_diagnoses_with_evidence=avg_diagnoses_with_evidence,
        avg_reasoning_length=avg_reasoning_length,
        low_reward_cases=low_reward_cases,
        high_reward_cases=high_reward_cases,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for EvidenceRL results with vLLM-accelerated LLM judge"
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Path to EvidenceRL results JSON file",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save metrics JSON file",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="HuggingFace model for vLLM judge (precision computation)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (default: 0.90)",
    )
    parser.add_argument(
        "--nli-model",
        default="pritamdeka/PubMedBERT-MNLI-MedNLI",
        help="HuggingFace model for NLI (grounding computation)",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model for embeddings",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maximum k for precision@k and recall@k (default: 5)",
    )
    parser.add_argument(
        "--reward-weight-grounding",
        type=float,
        default=0.5,
        help="Weight for grounding in combined reward (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for NLI model (default: 8)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EvidenceRL Metrics Computation (vLLM Version)")
    print("14-24x faster LLM judge inference")
    print("=" * 60)

    # Load results
    print(f"\nLoading results from: {args.results_json}")
    with open(args.results_json, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Loaded {len(results)} patient cases")

    import gc
    import torch

    # Pre-extract patient data (shared by both phases)
    print(f"\nPre-extracting patient data for {len(results)} patients...")
    patients = _extract_patient_data(results)

    # =====================================================================
    # PHASE 1: Accuracy verdicts via vLLM judge (GPU)
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Computing accuracy verdicts (vLLM judge on GPU)")
    print(f"{'='*60}")

    print(f"\nInitializing vLLM judge: {args.judge_model}")
    from evidence_rl.vllm_evaluation import VLLMAnswerJudge
    judge = VLLMAnswerJudge(
        model_name=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=32,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("vLLM judge initialized successfully")

    accuracy_results = _phase1_judge_accuracy(patients, judge)
    print(f"Phase 1 complete: {len(accuracy_results)} patients scored")

    # Free vLLM judge and release GPU memory
    print("\nFreeing vLLM judge to release GPU memory...")
    if hasattr(judge, '_llm') and judge._llm is not None:
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except (ImportError, Exception) as e:
            print(f"  Note: destroy_model_parallel skipped ({e})")
        del judge._llm
    del judge
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory released")

    # =====================================================================
    # PHASE 2: NLI grounding scores (GPU, now available)
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Computing NLI grounding scores (GPU)")
    print(f"{'='*60}")

    print(f"\nInitializing NLI model: {args.nli_model}")
    from evidence_rl.evidence_pipeline import CrossEncoderNLI
    nli_model = CrossEncoderNLI(
        model_name=args.nli_model,
        batch_size=args.batch_size,
    )
    print("NLI model initialized successfully")

    grounding_results = _phase2_nli_grounding(patients, nli_model)
    print(f"Phase 2 complete: {len(grounding_results)} patients scored")

    del nli_model
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # MERGE: Combine accuracy + grounding into final metrics
    # =====================================================================
    print("\nMerging accuracy and grounding results...")
    patient_metrics = _merge_metrics(
        patients=patients,
        accuracy_results=accuracy_results,
        grounding_results=grounding_results,
        max_k=args.max_k,
        reward_weight_grounding=args.reward_weight_grounding,
    )

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    aggregate = compute_aggregate_metrics(patient_metrics)

    # Print summary
    print("\n" + "=" * 70)
    print("METRICS SUMMARY (vLLM)")
    print("=" * 70)
    print(f"Number of patients: {aggregate.num_patients}")
    print(f"Parse success rate: {aggregate.parse_success_rate:.1%}")
    print(f"\nAverage diagnoses per patient: {aggregate.avg_diagnoses_per_patient:.2f}")
    print(f"Average diagnoses with evidence: {aggregate.avg_diagnoses_with_evidence:.2f}")
    print(f"Average reasoning length: {aggregate.avg_reasoning_length:.1f} chars")

    print(f"\nPrecision@k:")
    for k in sorted(aggregate.avg_precision_at_k.keys()):
        print(f"  P@{k}: {aggregate.avg_precision_at_k[k]:.3f}")

    print(f"\nRecall@k:")
    for k in sorted(aggregate.avg_recall_at_k.keys()):
        print(f"  R@{k}: {aggregate.avg_recall_at_k[k]:.3f}")

    print(f"\nReward@k Statistics:")
    if aggregate.avg_reward_combined_at_k:
        for k in sorted(aggregate.avg_reward_combined_at_k.keys()):
            grounding_k = aggregate.avg_reward_grounding_at_k.get(k, 0.0)
            precision_k = aggregate.avg_reward_precision_at_k.get(k, 0.0)
            combined_k = aggregate.avg_reward_combined_at_k.get(k, 0.0)
            print(f"  @{k}: Grounding={grounding_k:.3f}, Precision={precision_k:.3f}, Combined={combined_k:.3f}")

    print(f"\nEvidence Consistency Analysis:")
    print(f"  Low reward cases (< 0.3): {aggregate.low_reward_cases} ({aggregate.low_reward_cases/aggregate.num_patients:.1%})")
    print(f"  High reward cases (>= 0.7): {aggregate.high_reward_cases} ({aggregate.high_reward_cases/aggregate.num_patients:.1%})")
    print("=" * 70)

    # Save results
    output_data = {
        "config": {
            "judge_model": args.judge_model,
            "nli_model": args.nli_model,
            "embedding_model": args.embedding_model,
            "max_k": args.max_k,
            "reward_weight_grounding": args.reward_weight_grounding,
            "inference_engine": "vllm",
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "aggregate_metrics": aggregate.to_dict(),
        "patient_metrics": [m.to_dict() for m in patient_metrics],
    }

    print(f"\nSaving metrics to: {args.output_json}")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
