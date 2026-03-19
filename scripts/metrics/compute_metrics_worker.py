#!/usr/bin/env python3
"""Metrics computation worker for distributed EvidenceRL evaluation.

This worker processes a subset of patients from generation output and computes:
- Grounding reward (NLI-based)
- Precision reward (LLM judge-based, uses medgemma-27b-it as fixed judge)
- Combined reward
- Precision@k and Recall@k

Called by metrics_worker.sbatch with patient range arguments.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

from tqdm.auto import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass
class PatientMetrics:
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
) -> tuple:
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


def compute_patient_metrics(
    result: Dict[str, Any],
    judge,
    nli_model,
    embedder,
    max_k: int = 5,
    reward_weight_grounding: float = 0.5,
) -> PatientMetrics:
    """Compute metrics for a single patient using batched inference.

    Uses combined_reward_batch for efficient processing:
    - Patient section embeddings are cached and reused
    - NLI pairs are batched across all diagnoses
    - Judge prompts are batched across all diagnoses
    """
    from evidence_rl.reward import combined_reward_batch
    from evidence_rl.documents import Document, RetrievedDocument

    hadm_id = result["hadm_id"]
    subject_id = result["subject_id"]
    ground_truth = result["ground_truth_diagnoses"]
    patient_context = result["patient_context"]

    # Extract predicted diagnoses
    structured_output = result.get("structured_output", {})
    diagnoses = structured_output.get("diagnoses", [])

    # Filter to valid diagnoses (have a name)
    valid_diagnoses = [d for d in diagnoses if d.get("name", "").strip()]

    predicted = [d["name"] for d in valid_diagnoses]
    predicted_with_reasoning = [
        {"name": d["name"], "reasoning": d.get("reasoning", "")}
        for d in valid_diagnoses
    ]

    # Extract pre-evidence for grounding computation
    pre_evidence_data = result.get("pre_evidence", [])

    # Create evidence documents for reward computation
    pre_evidence = []
    for ev in pre_evidence_data:
        doc = Document(
            doc_id=ev.get("doc_id", ""),
            text=ev.get("text", ""),
        )
        pre_evidence.append(RetrievedDocument(document=doc, score=ev.get("score", 0.0)))

    # Compute rewards for all diagnoses in batch
    per_diagnosis_rewards = []
    grounding_rewards = []
    precision_rewards = []
    combined_rewards = []
    verdicts = []

    if valid_diagnoses:
        # Use batch function for efficient processing
        batch_results = combined_reward_batch(
            diagnoses=valid_diagnoses,
            patient_context=patient_context,
            evidence_docs=pre_evidence,
            ground_truth_diagnoses=ground_truth,
            nli_model=nli_model,
            judge=judge,
            embedder=embedder,
            weight_grounding=reward_weight_grounding,
        )

        for i, (r_combined, r_grounding_max, r_grounding_avg, r_precision) in enumerate(batch_results):
            diagnosis_name = valid_diagnoses[i].get("name", "")

            per_diagnosis_rewards.append({
                "diagnosis_name": diagnosis_name,
                "grounding_max": float(r_grounding_max),
                "grounding_avg": float(r_grounding_avg),
                "precision": float(r_precision),
                "combined": float(r_combined),
            })

            grounding_rewards.append(r_grounding_max)
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

    return PatientMetrics(
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


def compute_metrics_batch(
    results: List[Dict[str, Any]],
    judge,
    nli_model,
    embedder,
    max_k: int = 5,
    reward_weight_grounding: float = 0.5,
    judge_batch_size: int = 16,
    nli_batch_size: int = 32,
) -> List[PatientMetrics]:
    """Compute metrics for multiple patients with cross-patient batching.

    This optimized function collects all judge and NLI calls across ALL patients
    and runs them in large batches for maximum GPU efficiency.

    Args:
        judge_batch_size: Batch size for LLM judge (conservative for 2x H200)
        nli_batch_size: Batch size for NLI model (DeBERTa, can be larger)

    Phases:
    1. Pre-compute all patient data structures
    2. Collect ALL judge prompts across all patients
    3. Run batch judge inference (single call for all patients)
    4. Collect ALL NLI pairs across all patients
    5. Run batch NLI inference (single call for all patients)
    6. Distribute results back to each patient and compute metrics
    """
    from evidence_rl.documents import Document, RetrievedDocument

    if not results:
        return []

    total_patients = len(results)
    print(f"[Progress] Starting metrics computation for {total_patients} patients...")
    print(f"[Batch] Processing {total_patients} patients with cross-patient batching...")

    # === Phase 1: Pre-compute patient data structures ===
    patient_data = []
    for result in results:
        hadm_id = result["hadm_id"]
        subject_id = result["subject_id"]
        ground_truth = result["ground_truth_diagnoses"]
        patient_context = result["patient_context"]

        structured_output = result.get("structured_output", {})
        diagnoses = structured_output.get("diagnoses", [])
        valid_diagnoses = [d for d in diagnoses if d.get("name", "").strip()]

        # Pre-evidence
        pre_evidence_data = result.get("pre_evidence", [])
        pre_evidence = []
        for ev in pre_evidence_data:
            doc = Document(doc_id=ev.get("doc_id", ""), text=ev.get("text", ""))
            pre_evidence.append(RetrievedDocument(document=doc, score=ev.get("score", 0.0)))

        patient_data.append({
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            "ground_truth": ground_truth,
            "patient_context": patient_context,
            "valid_diagnoses": valid_diagnoses,
            "pre_evidence": pre_evidence,
            "parse_success": structured_output.get("parse_success", False),
            "all_diagnoses": diagnoses,
        })

    print(f"[Progress] Phase 1 complete: Pre-processed {len(patient_data)} patients")

    # === Phase 2: Collect ALL judge prompts ===
    all_judge_prompts = []
    judge_prompt_mapping = []  # (patient_idx, diag_idx)

    for patient_idx, pd in enumerate(patient_data):
        ground_truth_str = "\n- ".join(pd["ground_truth"]) if pd["ground_truth"] else ""
        for diag_idx, diag in enumerate(pd["valid_diagnoses"]):
            name = diag.get("name", "").strip()
            if name and pd["ground_truth"]:
                prompt = f"""You are evaluating the diagnosis prediction of a clinical model.

CANDIDATE ANSWER: "{name}"

ACCEPTED GROUND TRUTHS:
- {ground_truth_str}

TASK: Does the CANDIDATE ANSWER semantically match ANY of the ACCEPTED GROUND TRUTHS?
Respond 'TRUE' if the candidate refers to the same underlying clinical concept as any item in the list (allowing for synonyms, abbreviations, or minor wording differences).
Respond 'FALSE' if it represents a different clinical concept, severity, or anatomical location.

Verdict:"""
                all_judge_prompts.append(prompt)
                judge_prompt_mapping.append((patient_idx, diag_idx))

    # === Phase 3: Batch judge inference ===
    total_judge_prompts = len(all_judge_prompts)
    print(f"[Batch] Running judge inference on {total_judge_prompts} prompts...")
    all_judge_results = []
    if all_judge_prompts:
        try:
            # Process in chunks to avoid OOM
            num_batches = (total_judge_prompts + judge_batch_size - 1) // judge_batch_size
            for batch_idx, i in enumerate(range(0, total_judge_prompts, judge_batch_size)):
                batch = all_judge_prompts[i:i + judge_batch_size]
                batch_results = judge.is_correct_batch(batch)
                all_judge_results.extend(batch_results)
                processed = min(i + judge_batch_size, total_judge_prompts)
                print(f"[Progress] Judge: {processed}/{total_judge_prompts} prompts ({batch_idx+1}/{num_batches} batches)")
        except Exception as e:
            print(f"[Batch] Judge inference failed: {e}")
            all_judge_results = [False] * len(all_judge_prompts)

    # Map judge results back to patients
    judge_results_by_patient = {i: {} for i in range(len(patient_data))}
    for result_idx, (patient_idx, diag_idx) in enumerate(judge_prompt_mapping):
        judge_results_by_patient[patient_idx][diag_idx] = all_judge_results[result_idx]

    # === Phase 4: Collect ALL NLI pairs ===
    all_nli_pairs = []
    nli_pair_mapping = []  # (patient_idx, diag_idx)

    for patient_idx, pd in enumerate(patient_data):
        evidence_texts = [doc.document.text.strip() for doc in pd["pre_evidence"]] if pd["pre_evidence"] else [""]
        for diag_idx, diag in enumerate(pd["valid_diagnoses"]):
            reasoning = diag.get("reasoning", "").strip()
            if reasoning:
                # Use full patient context (simplified - no FTV for batch mode)
                for evidence_text in evidence_texts:
                    premise = f"{pd['patient_context']}\n\n{evidence_text}".strip()
                    if len(premise) > 2048:
                        premise = premise[:2048]
                    all_nli_pairs.append((premise, reasoning))
                    nli_pair_mapping.append((patient_idx, diag_idx, len(evidence_texts)))

    print(f"[Progress] Phase 3 complete: Judge inference done")

    # === Phase 5: Batch NLI inference ===
    total_nli_pairs = len(all_nli_pairs)
    print(f"[Batch] Running NLI inference on {total_nli_pairs} pairs...")
    all_nli_scores = []
    if all_nli_pairs:
        try:
            # Process in chunks
            num_batches = (total_nli_pairs + nli_batch_size - 1) // nli_batch_size
            for batch_idx, i in enumerate(range(0, total_nli_pairs, nli_batch_size)):
                batch = all_nli_pairs[i:i + nli_batch_size]
                predictions = nli_model.predict(batch)
                scores = [p.get('entailment', 0.0) - p.get('contradiction', 0.0) for p in predictions]
                all_nli_scores.extend(scores)
                processed = min(i + nli_batch_size, total_nli_pairs)
                print(f"[Progress] NLI: {processed}/{total_nli_pairs} pairs ({batch_idx+1}/{num_batches} batches)")
        except Exception as e:
            print(f"[Batch] NLI inference failed: {e}")
            all_nli_scores = [0.0] * len(all_nli_pairs)

    # Map NLI results back to patients (take max across evidence for each diagnosis)
    nli_results_by_patient = {i: {} for i in range(len(patient_data))}
    nli_idx = 0
    for patient_idx, diag_idx, num_evidence in nli_pair_mapping:
        if diag_idx not in nli_results_by_patient[patient_idx]:
            nli_results_by_patient[patient_idx][diag_idx] = []
        nli_results_by_patient[patient_idx][diag_idx].append(all_nli_scores[nli_idx])
        nli_idx += 1

    print(f"[Progress] Phase 5 complete: NLI inference done")

    # === Phase 6: Build PatientMetrics for each patient ===
    print("[Batch] Building patient metrics...")
    patient_metrics = []
    progress_interval = max(1, total_patients // 10)  # Report every 10%

    for patient_idx, pd in enumerate(patient_data):
        # Progress reporting
        if (patient_idx + 1) % progress_interval == 0 or patient_idx == total_patients - 1:
            pct = ((patient_idx + 1) / total_patients) * 100
            print(f"[Progress] Building metrics: {patient_idx + 1}/{total_patients} patients ({pct:.0f}%)")
        per_diagnosis_rewards = []
        grounding_rewards = []
        precision_rewards = []
        combined_rewards = []
        verdicts = []

        for diag_idx, diag in enumerate(pd["valid_diagnoses"]):
            # Get judge result (precision)
            is_correct = judge_results_by_patient[patient_idx].get(diag_idx, False)
            r_precision = 1.0 if is_correct else 0.0

            # Get NLI result (grounding)
            nli_scores = nli_results_by_patient[patient_idx].get(diag_idx, [])
            r_grounding = max(nli_scores) if nli_scores else 0.0

            # Combined
            r_combined = reward_weight_grounding * r_grounding + (1.0 - reward_weight_grounding) * r_precision

            per_diagnosis_rewards.append({
                "diagnosis_name": diag.get("name", ""),
                "grounding": float(r_grounding),
                "precision": float(r_precision),
                "combined": float(r_combined),
            })
            grounding_rewards.append(r_grounding)
            precision_rewards.append(r_precision)
            combined_rewards.append(r_combined)
            verdicts.append(is_correct)

        # Compute @k metrics
        reward_grounding_at_k = compute_reward_at_k(grounding_rewards, max_k)
        reward_precision_at_k = compute_reward_at_k(precision_rewards, max_k)
        reward_combined_at_k = compute_reward_at_k(combined_rewards, max_k)
        precision_at_k, recall_at_k = compute_precision_recall_at_k(
            verdicts, len(pd["ground_truth"]), max_k
        )

        # Evidence analysis
        num_with_evidence = sum(1 for d in pd["all_diagnoses"] if d.get("reasoning"))
        avg_reasoning_len = (
            sum(len(d.get("reasoning", "")) for d in pd["all_diagnoses"]) / len(pd["all_diagnoses"])
            if pd["all_diagnoses"] else 0.0
        )

        patient_metrics.append(PatientMetrics(
            hadm_id=pd["hadm_id"],
            subject_id=pd["subject_id"],
            ground_truth_diagnoses=pd["ground_truth"],
            predicted_diagnoses=[d["name"] for d in pd["valid_diagnoses"]],
            predicted_diagnoses_with_reasoning=[
                {"name": d["name"], "reasoning": d.get("reasoning", "")}
                for d in pd["valid_diagnoses"]
            ],
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            verdicts=verdicts,
            per_diagnosis_rewards=per_diagnosis_rewards,
            reward_grounding_at_k=reward_grounding_at_k,
            reward_precision_at_k=reward_precision_at_k,
            reward_combined_at_k=reward_combined_at_k,
            num_diagnoses_with_evidence=num_with_evidence,
            avg_reasoning_length=avg_reasoning_len,
            parse_success=pd["parse_success"],
        ))

    print(f"[Progress] Phase 6 complete: Built metrics for {len(patient_metrics)} patients")
    print(f"[Progress] ✓ All {total_patients} patients processed successfully")
    return patient_metrics


def compute_worker_aggregate(patient_metrics: List[PatientMetrics]) -> Dict[str, Any]:
    """Compute aggregate metrics for this worker's patient subset."""
    num_patients = len(patient_metrics)
    if num_patients == 0:
        return {"num_patients": 0}

    # Average precision/recall at k
    all_k = set()
    for m in patient_metrics:
        all_k.update(m.precision_at_k.keys())

    avg_precision = {}
    avg_recall = {}
    for k in sorted(all_k):
        precision_sum = sum(m.precision_at_k.get(k, 0.0) for m in patient_metrics)
        recall_sum = sum(m.recall_at_k.get(k, 0.0) for m in patient_metrics)
        avg_precision[k] = precision_sum / num_patients
        avg_recall[k] = recall_sum / num_patients

    # Average rewards@k
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

    return {
        "num_patients": num_patients,
        "avg_precision_at_k": {str(k): v for k, v in avg_precision.items()},
        "avg_recall_at_k": {str(k): v for k, v in avg_recall.items()},
        "avg_reward_grounding_at_k": {str(k): v for k, v in avg_reward_grounding_at_k.items()},
        "avg_reward_precision_at_k": {str(k): v for k, v in avg_reward_precision_at_k.items()},
        "avg_reward_combined_at_k": {str(k): v for k, v in avg_reward_combined_at_k.items()},
        "parse_success_rate": parse_success_rate,
        "avg_diagnoses_per_patient": avg_diagnoses,
        "avg_diagnoses_with_evidence": avg_diagnoses_with_evidence,
        "avg_reasoning_length": avg_reasoning_length,
        "low_reward_cases": low_reward_cases,
        "high_reward_cases": high_reward_cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute EvidenceRL metrics for a patient subset (worker)"
    )
    parser.add_argument(
        "--generation-file",
        required=True,
        help="Path to generation results JSON file",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save worker metrics JSON",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Path to LLM judge model",
    )
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model for grounding computation",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for section selection",
    )
    parser.add_argument(
        "--patient-start-idx",
        type=int,
        required=True,
        help="Start index (inclusive) for patient subset",
    )
    parser.add_argument(
        "--patient-end-idx",
        type=int,
        required=True,
        help="End index (exclusive) for patient subset",
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
        default=48,
        help="Batch size for judge model (default: 48 for medgemma-27b-it on 2x H200)",
    )

    args = parser.parse_args()

    start_time = time.time()

    # Load generation results
    print(f"[Worker] Loading generation results from: {args.generation_file}")
    with open(args.generation_file, "r") as f:
        data = json.load(f)

    all_results = data.get("results", [])
    total_patients = len(all_results)
    print(f"[Worker] Total patients in file: {total_patients}")

    # Extract patient subset
    start_idx = args.patient_start_idx
    end_idx = min(args.patient_end_idx, total_patients)
    results = all_results[start_idx:end_idx]
    print(f"[Worker] Processing patients [{start_idx}:{end_idx}] ({len(results)} patients)")

    if not results:
        print("[Worker] No patients to process. Saving empty output.")
        output_data = {
            "config": {
                "judge_model": args.judge_model,
                "nli_model": args.nli_model,
                "embedding_model": args.embedding_model,
                "max_k": args.max_k,
                "reward_weight_grounding": args.reward_weight_grounding,
                "patient_start_idx": start_idx,
                "patient_end_idx": end_idx,
            },
            "worker_aggregate": {"num_patients": 0},
            "patient_metrics": [],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        return

    # Initialize LLM judge
    print(f"\n[Worker] Initializing LLM judge: {args.judge_model}")
    from evidence_rl.evaluation import LLMAnswerJudge
    judge = LLMAnswerJudge(
        model_name=args.judge_model,
        generation_kwargs={"max_new_tokens": 32, "do_sample": False},
    )
    print("[Worker] Judge initialized successfully")

    # Initialize NLI model
    print(f"\n[Worker] Initializing NLI model: {args.nli_model}")
    from evidence_rl.evidence_pipeline import CrossEncoderNLI
    nli_model = CrossEncoderNLI(
        model_name=args.nli_model,
        batch_size=args.batch_size,
    )
    print("[Worker] NLI model initialized successfully")

    # Initialize embedder for Focus-Then-Verify
    print(f"\n[Worker] Initializing embedding model: {args.embedding_model}")
    from evidence_rl.retrieval import HuggingFaceEmbedder
    embedder = HuggingFaceEmbedder(model_name=args.embedding_model)
    print("[Worker] Embedder initialized successfully")

    model_init_time = time.time()
    print(f"[Worker] Model initialization took {model_init_time - start_time:.1f}s")

    # Compute metrics using cross-patient batching for efficiency
    # Batch sizes for 2x H200 GPUs (286GB total):
    # - Judge (medgemma-27b-it, 54GB): batch 48 (judge uses max_new_tokens=32, small KV cache)
    # - NLI (DeBERTa ~400MB): 192 is safe, model is small
    # Tested: batch=24 uses ~26% VRAM, batch=48 uses ~37% VRAM
    judge_batch = args.batch_size  # Default: 48 for medgemma-27b-it
    nli_batch = args.batch_size * 4  # Default: 192 (DeBERTa is small)
    print(f"\n[Worker] Computing metrics for {len(results)} patients with cross-patient batching...")
    print(f"[Worker] Batch sizes: judge={judge_batch}, NLI={nli_batch}")
    try:
        patient_metrics = compute_metrics_batch(
            results=results,
            judge=judge,
            nli_model=nli_model,
            embedder=embedder,
            max_k=args.max_k,
            reward_weight_grounding=args.reward_weight_grounding,
            judge_batch_size=judge_batch,
            nli_batch_size=nli_batch,
        )
    except Exception as e:
        print(f"[Worker] Batch processing failed: {e}, falling back to per-patient processing...")
        patient_metrics = []
        total_fallback = len(results)
        progress_interval = max(1, total_fallback // 10)  # Report every 10%
        for idx, result in enumerate(results):
            try:
                metrics = compute_patient_metrics(
                    result=result,
                    judge=judge,
                    nli_model=nli_model,
                    embedder=embedder,
                    max_k=args.max_k,
                    reward_weight_grounding=args.reward_weight_grounding,
                )
                patient_metrics.append(metrics)
                # Progress reporting
                if (idx + 1) % progress_interval == 0 or idx == total_fallback - 1:
                    pct = ((idx + 1) / total_fallback) * 100
                    print(f"[Progress] Processed {idx + 1}/{total_fallback} patients ({pct:.0f}%)")
            except Exception as e2:
                print(f"[Worker] Error processing patient {result.get('hadm_id', 'unknown')}: {e2}")
                patient_metrics.append(PatientMetrics(
                    hadm_id=result.get("hadm_id", "unknown"),
                    subject_id=result.get("subject_id", "unknown"),
                    ground_truth_diagnoses=result.get("ground_truth_diagnoses", []),
                    predicted_diagnoses=[],
                    predicted_diagnoses_with_reasoning=[],
                ))
        print(f"[Progress] ✓ All {total_fallback} patients processed (fallback mode)")

    compute_time = time.time()
    print(f"[Worker] Metrics computation took {compute_time - model_init_time:.1f}s")

    # Compute worker aggregate
    worker_aggregate = compute_worker_aggregate(patient_metrics)

    # Print summary
    print("\n" + "=" * 60)
    print(f"[Worker] SUMMARY for patients [{start_idx}:{end_idx}]")
    print("=" * 60)
    print(f"Patients processed: {worker_aggregate['num_patients']}")
    print(f"Parse success rate: {worker_aggregate.get('parse_success_rate', 0.0):.1%}")

    if worker_aggregate.get("avg_precision_at_k"):
        print("\nPrecision@k:")
        for k, v in sorted(worker_aggregate["avg_precision_at_k"].items(), key=lambda x: int(x[0])):
            print(f"  P@{k}: {v:.3f}")

    if worker_aggregate.get("avg_reward_combined_at_k"):
        print("\nReward@k (Combined):")
        for k, v in sorted(worker_aggregate["avg_reward_combined_at_k"].items(), key=lambda x: int(x[0])):
            print(f"  @{k}: {v:.3f}")

    print("=" * 60)

    # Save results
    output_data = {
        "config": {
            "judge_model": args.judge_model,
            "nli_model": args.nli_model,
            "embedding_model": args.embedding_model,
            "max_k": args.max_k,
            "reward_weight_grounding": args.reward_weight_grounding,
            "patient_start_idx": start_idx,
            "patient_end_idx": end_idx,
            "batch_size": args.batch_size,
        },
        "worker_aggregate": worker_aggregate,
        "patient_metrics": [m.to_dict() for m in patient_metrics],
    }

    print(f"\n[Worker] Saving metrics to: {args.output_json}")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    total_time = time.time() - start_time
    print(f"[Worker] Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("[Worker] Done!")


if __name__ == "__main__":
    main()
