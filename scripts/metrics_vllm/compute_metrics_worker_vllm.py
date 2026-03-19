#!/usr/bin/env python3
"""Metrics worker using vLLM for the LLM judge.

This script processes a subset of patients for distributed metrics computation
using vLLM for the judge model.

Usage:
    python compute_metrics_worker_vllm.py \
        --generation-json <path> \
        --output-json <path> \
        --judge-model <path> \
        --patient-start <int> \
        --patient-end <int> \
        --worker-id <int>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

from tqdm.auto import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass
class EvidenceRLMetrics:
    """Metrics for a single patient case."""

    hadm_id: str
    subject_id: str
    ground_truth_diagnoses: List[str]
    predicted_diagnoses: List[str]
    predicted_diagnoses_with_reasoning: List[Dict[str, str]]
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    verdicts: List[bool] = field(default_factory=list)
    per_diagnosis_rewards: List[Dict[str, float]] = field(default_factory=list)
    reward_grounding_at_k: Dict[int, float] = field(default_factory=dict)
    reward_precision_at_k: Dict[int, float] = field(default_factory=dict)
    reward_combined_at_k: Dict[int, float] = field(default_factory=dict)
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


def _phase1_judge_accuracy(patients, judge, worker_id):
    """Phase 1: Compute accuracy verdicts using vLLM judge on GPU."""
    accuracy_results = []
    for patient in tqdm(patients, desc=f"[worker {worker_id}] Phase 1: Accuracy"):
        precision_rewards = []
        verdicts = []
        for diag_data in patient["valid_diagnoses"]:
            diagnosis_name = diag_data.get("name", "")
            is_correct = judge.is_correct(diagnosis_name, patient["ground_truth"])
            precision_rewards.append(1.0 if is_correct else 0.0)
            verdicts.append(is_correct)
        accuracy_results.append({
            "precision_rewards": precision_rewards,
            "verdicts": verdicts,
        })
    return accuracy_results


def _phase2_nli_grounding(patients, nli_model, worker_id):
    """Phase 2: Compute NLI grounding scores (on GPU after vLLM is freed)."""
    from evidence_rl.reward import reward_grounding

    grounding_results = []
    for patient in tqdm(patients, desc=f"[worker {worker_id}] Phase 2: Grounding"):
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


def _merge_metrics(patients, accuracy_results, grounding_results, max_k=5, reward_weight_grounding=0.5):
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


def main():
    parser = argparse.ArgumentParser(
        description="Metrics worker using vLLM for the judge"
    )
    parser.add_argument("--generation-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--reward-weight-grounding", type=float, default=0.5)
    parser.add_argument("--patient-start", type=int, required=True)
    parser.add_argument("--patient-end", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)

    args = parser.parse_args()

    print(f"[worker {args.worker_id}] vLLM Metrics Worker")
    print(f"[worker {args.worker_id}] Patient range: [{args.patient_start}:{args.patient_end}]")

    # Load results
    print(f"[worker {args.worker_id}] Loading generation results...")
    with open(args.generation_json, "r") as f:
        data = json.load(f)

    all_results = data.get("results", [])
    print(f"[worker {args.worker_id}] Total patients: {len(all_results)}")

    # Get this worker's subset
    results = all_results[args.patient_start:args.patient_end]
    print(f"[worker {args.worker_id}] Processing {len(results)} patients")

    import gc
    import torch

    # Pre-extract patient data
    print(f"[worker {args.worker_id}] Pre-extracting patient data...")
    patients = _extract_patient_data(results)

    # =====================================================================
    # PHASE 1: Accuracy verdicts via vLLM judge (GPU)
    # =====================================================================
    print(f"[worker {args.worker_id}] Phase 1: Accuracy (vLLM judge on GPU)")
    from evidence_rl.vllm_evaluation import VLLMAnswerJudge
    judge = VLLMAnswerJudge(
        model_name=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=32,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    accuracy_results = _phase1_judge_accuracy(patients, judge, args.worker_id)
    print(f"[worker {args.worker_id}] Phase 1 complete: {len(accuracy_results)} patients")

    # Free vLLM judge and release GPU memory
    print(f"[worker {args.worker_id}] Freeing vLLM judge...")
    if hasattr(judge, '_llm') and judge._llm is not None:
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except (ImportError, Exception):
            pass
        del judge._llm
    del judge
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 2: NLI grounding scores (GPU, now available)
    # =====================================================================
    print(f"[worker {args.worker_id}] Phase 2: Grounding (NLI on GPU)")
    from evidence_rl.evidence_pipeline import CrossEncoderNLI
    nli_model = CrossEncoderNLI(
        model_name=args.nli_model,
        batch_size=32,
    )

    grounding_results = _phase2_nli_grounding(patients, nli_model, args.worker_id)
    print(f"[worker {args.worker_id}] Phase 2 complete: {len(grounding_results)} patients")

    del nli_model
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # MERGE
    # =====================================================================
    print(f"[worker {args.worker_id}] Merging results...")
    patient_metrics = _merge_metrics(
        patients=patients,
        accuracy_results=accuracy_results,
        grounding_results=grounding_results,
        max_k=args.max_k,
        reward_weight_grounding=args.reward_weight_grounding,
    )

    # Save results
    output_data = {
        "config": {
            "judge_model": args.judge_model,
            "nli_model": args.nli_model,
            "embedding_model": args.embedding_model,
            "max_k": args.max_k,
            "reward_weight_grounding": args.reward_weight_grounding,
            "inference_engine": "vllm",
            "worker_id": args.worker_id,
            "patient_start": args.patient_start,
            "patient_end": args.patient_end,
        },
        "patient_metrics": [m.to_dict() for m in patient_metrics],
    }

    print(f"[worker {args.worker_id}] Saving metrics to: {args.output_json}")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[worker {args.worker_id}] Done! Processed {len(patient_metrics)} patients.")


if __name__ == "__main__":
    main()
