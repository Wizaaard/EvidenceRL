#!/usr/bin/env python3
"""Compute precision@k, recall@k, and evidence consistency metrics for EvidenceRL results.

For the paper "EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models"

This script:
1. Loads EvidenceRL pipeline results (which contain generated diagnoses + reasoning)
2. Runs LLM judge in batches to compute precision (diagnosis correctness)
3. Runs NLI model to compute grounding (reasoning faithfulness)
4. Computes precision@k, recall@k, and reward@k metrics
5. Outputs metrics JSON for visualization
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


def compute_patient_metrics_batch(
    results: List[Dict[str, Any]],
    judge,
    nli_model,
    embedder,
    max_k: int = 5,
    reward_weight_grounding: float = 0.5,
    batch_size: int = 8,
) -> List[EvidenceRLMetrics]:
    """Compute metrics for all patients with batched LLM judge and NLI."""
    from evidence_rl.reward import combined_reward

    all_metrics = []

    for result in tqdm(results, desc="Computing rewards"):
        hadm_id = result["hadm_id"]
        subject_id = result["subject_id"]
        ground_truth = result["ground_truth_diagnoses"]
        patient_context = result["patient_context"]

        # Extract predicted diagnoses
        structured_output = result.get("structured_output", {})
        diagnoses = structured_output.get("diagnoses", [])

        predicted = [d["name"] for d in diagnoses if d.get("name")]
        predicted_with_reasoning = [
            {"name": d["name"], "reasoning": d.get("reasoning", "")}
            for d in diagnoses if d.get("name")
        ]

        # Extract pre-evidence for grounding computation
        pre_evidence_data = result.get("pre_evidence", [])

        # Create mock evidence documents for reward computation
        from evidence_rl.documents import Document, RetrievedDocument
        pre_evidence = []
        for ev in pre_evidence_data:
            doc = Document(
                doc_id=ev.get("doc_id", ""),
                text=ev.get("text", ""),
            )
            pre_evidence.append(RetrievedDocument(document=doc, score=ev.get("score", 0.0)))

        # Compute rewards for each diagnosis
        per_diagnosis_rewards = []
        grounding_rewards = []
        precision_rewards = []
        combined_rewards = []
        verdicts = []

        for diag_data in diagnoses:
            diagnosis_name = diag_data.get("name", "")
            diagnosis_reasoning = diag_data.get("reasoning", "")

            if not diagnosis_name.strip():
                continue

            # Compute combined reward (grounding + precision)
            r_combined, r_grounding, r_precision = combined_reward(
                diagnosis_name=diagnosis_name,
                diagnosis_reasoning=diagnosis_reasoning,
                patient_context=patient_context,
                evidence_docs=pre_evidence,
                ground_truth_diagnoses=ground_truth,
                nli_model=nli_model,
                judge=judge,
                embedder=embedder,
                weight_grounding=reward_weight_grounding,
            )

            per_diagnosis_rewards.append({
                "diagnosis_name": diagnosis_name,
                "grounding": float(r_grounding),
                "precision": float(r_precision),
                "combined": float(r_combined),
            })

            grounding_rewards.append(r_grounding)
            precision_rewards.append(r_precision)
            combined_rewards.append(r_combined)
            verdicts.append(r_precision == 1.0)

        # Compute reward@k
        reward_grounding_at_k = compute_reward_at_k(grounding_rewards, max_k)
        reward_precision_at_k = compute_reward_at_k(precision_rewards, max_k)
        reward_combined_at_k = compute_reward_at_k(combined_rewards, max_k)

        # Compute precision@k and recall@k
        precision_at_k, recall_at_k = compute_precision_recall_at_k(
            verdicts, len(ground_truth), max_k
        )

        # Evidence grounding analysis
        num_with_evidence = sum(1 for d in diagnoses if d.get("reasoning"))
        avg_reasoning_len = (
            sum(len(d.get("reasoning", "")) for d in diagnoses) / len(diagnoses)
            if diagnoses else 0.0
        )
        parse_success = structured_output.get("parse_success", False)

        metrics = EvidenceRLMetrics(
            hadm_id=hadm_id,
            subject_id=subject_id,
            ground_truth_diagnoses=ground_truth,
            predicted_diagnoses=predicted,
            predicted_diagnoses_with_reasoning=predicted_with_reasoning,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            verdicts=verdicts,
            per_diagnosis_rewards=per_diagnosis_rewards,
            reward_grounding_at_k=reward_grounding_at_k,
            reward_precision_at_k=reward_precision_at_k,
            reward_combined_at_k=reward_combined_at_k,
            num_diagnoses_with_evidence=num_with_evidence,
            avg_reasoning_length=avg_reasoning_len,
            parse_success=parse_success,
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
        description="Compute metrics for EvidenceRL results with LLM judge and NLI grounding"
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
        help="HuggingFace model for LLM judge (precision computation)",
    )
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
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

    # Load results
    print(f"Loading results from: {args.results_json}")
    with open(args.results_json, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Loaded {len(results)} patient cases")

    # Initialize LLM judge
    print(f"\nInitializing LLM judge: {args.judge_model}")
    from evidence_rl.evaluation import LLMAnswerJudge
    judge = LLMAnswerJudge(
        model_name=args.judge_model,
        generation_kwargs={"max_new_tokens": 32, "do_sample": False},
    )
    print("Judge initialized successfully")

    # Initialize NLI model
    print(f"\nInitializing NLI model: {args.nli_model}")
    from evidence_rl.evidence_pipeline import CrossEncoderNLI
    nli_model = CrossEncoderNLI(
        model_name=args.nli_model,
        batch_size=args.batch_size,
    )
    print("NLI model initialized successfully")

    # Initialize embedder for Focus-Then-Verify
    print(f"\nInitializing embedding model: {args.embedding_model}")
    from evidence_rl.retrieval import HuggingFaceEmbedder
    embedder = HuggingFaceEmbedder(model_name=args.embedding_model)
    print("Embedder initialized successfully")

    # Compute metrics for all patients with batched processing
    print(f"\nComputing metrics for {len(results)} patients...")
    patient_metrics = compute_patient_metrics_batch(
        results=results,
        judge=judge,
        nli_model=nli_model,
        embedder=embedder,
        max_k=args.max_k,
        reward_weight_grounding=args.reward_weight_grounding,
        batch_size=args.batch_size,
    )

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    aggregate = compute_aggregate_metrics(patient_metrics)

    # Print summary
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
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
    print("="*70)

    # Save results
    output_data = {
        "config": {
            "judge_model": args.judge_model,
            "nli_model": args.nli_model,
            "embedding_model": args.embedding_model,
            "max_k": args.max_k,
            "reward_weight_grounding": args.reward_weight_grounding,
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
