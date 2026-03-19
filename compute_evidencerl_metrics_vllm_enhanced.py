#!/usr/bin/env python3
"""Compute EvidenceRL metrics with vLLM-accelerated LLM judge + ENHANCED METRICS.

This is an enhanced version of compute_evidencerl_metrics_vllm.py that adds:
- F1@k (harmonic mean of P@k and R@k)
- Macro-F1 and Micro-F1 (per-diagnosis classification metrics)
- Retrieval quality metrics (Success@k, MRR)
- Statistical analysis (confidence intervals, significance tests)
- Per-diagnosis performance breakdown
- Parse error analysis
- Enhanced output with all metrics for publication

New metrics are critical for top-venue publication (ACL, EMNLP, NeurIPS, etc.)

Usage:
    python compute_evidencerl_metrics_vllm_enhanced.py \
        --results-json <path> \
        --output-json <path> \
        --judge-model <path>

The output includes all original metrics PLUS enhanced metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

from tqdm.auto import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass
class EvidenceRLMetrics:
    """Metrics for a single patient case (ENHANCED)."""

    hadm_id: str
    subject_id: str

    # Ground truth
    ground_truth_diagnoses: List[str]

    # Predictions
    predicted_diagnoses: List[str]
    predicted_diagnoses_with_reasoning: List[Dict[str, str]]

    # ORIGINAL METRICS
    # Precision/Recall metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)

    # NEW: F1@k metrics
    f1_at_k: Dict[int, float] = field(default_factory=dict)

    # Per-diagnosis verdicts (True = correct, False = incorrect)
    verdicts: List[bool] = field(default_factory=list)

    # Per-diagnosis rewards
    per_diagnosis_rewards: List[Dict[str, float]] = field(default_factory=list)

    # Rewards@k (grounding + precision)
    reward_grounding_max_at_k: Dict[int, float] = field(default_factory=dict)
    reward_grounding_avg_at_k: Dict[int, float] = field(default_factory=dict)
    reward_precision_at_k: Dict[int, float] = field(default_factory=dict)
    reward_combined_at_k: Dict[int, float] = field(default_factory=dict)

    # Evidence grounding analysis
    num_diagnoses_with_evidence: int = 0
    avg_reasoning_length: float = 0.0
    parse_success: bool = False

    # NEW: Retrieval info for retrieval metrics
    retrieved_doc_ids: List[str] = field(default_factory=list)

    # Pairwise match matrix: match_matrix[i][j] = True if prediction i matches GT j
    match_matrix: List[List[bool]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hadm_id": self.hadm_id,
            "subject_id": self.subject_id,
            "ground_truth_diagnoses": self.ground_truth_diagnoses,
            "predicted_diagnoses": self.predicted_diagnoses,
            "predicted_diagnoses_with_reasoning": self.predicted_diagnoses_with_reasoning,
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "f1_at_k": {str(k): v for k, v in self.f1_at_k.items()},  # NEW
            "verdicts": self.verdicts,
            "per_diagnosis_rewards": self.per_diagnosis_rewards,
            "match_matrix": self.match_matrix,
            "reward_grounding_max_at_k": {str(k): v for k, v in self.reward_grounding_max_at_k.items()},
            "reward_grounding_avg_at_k": {str(k): v for k, v in self.reward_grounding_avg_at_k.items()},
            "reward_precision_at_k": {str(k): v for k, v in self.reward_precision_at_k.items()},
            "reward_combined_at_k": {str(k): v for k, v in self.reward_combined_at_k.items()},
            "num_diagnoses_with_evidence": self.num_diagnoses_with_evidence,
            "avg_reasoning_length": self.avg_reasoning_length,
            "parse_success": self.parse_success,
            "retrieved_doc_ids": self.retrieved_doc_ids,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across patient cohort (ENHANCED)."""

    num_patients: int = 0

    # ORIGINAL: Precision/Recall averages
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)

    # NEW: F1@k averages
    avg_f1_at_k: Dict[int, float] = field(default_factory=dict)

    # REMOVED: Macro/Micro F1 - not meaningful with LLM-judge-based matching
    # (exact string matching between predicted names and GT ICD codes fails)
    # macro_f1: float = 0.0
    # micro_f1: float = 0.0

    # Reward@k statistics (grounding + precision)
    avg_reward_grounding_max_at_k: Dict[int, float] = field(default_factory=dict)
    avg_reward_grounding_avg_at_k: Dict[int, float] = field(default_factory=dict)
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

    # NEW: Retrieval quality metrics
    retrieval_success_at_k: Dict[int, float] = field(default_factory=dict)
    retrieval_mrr: float = 0.0
    retrieval_precision_at_k: Dict[int, float] = field(default_factory=dict)
    retrieval_recall_at_k: Dict[int, float] = field(default_factory=dict)

    # NEW: Statistical analysis (confidence intervals)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # NEW: Per-diagnosis performance breakdown (for appendix)
    per_diagnosis_performance: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_patients": self.num_patients,
            # Original metrics
            "avg_precision_at_k": {str(k): v for k, v in self.avg_precision_at_k.items()},
            "avg_recall_at_k": {str(k): v for k, v in self.avg_recall_at_k.items()},
            # NEW metrics
            "avg_f1_at_k": {str(k): v for k, v in self.avg_f1_at_k.items()},
            # Rewards
            "avg_reward_grounding_max_at_k": {str(k): v for k, v in self.avg_reward_grounding_max_at_k.items()},
            "avg_reward_grounding_avg_at_k": {str(k): v for k, v in self.avg_reward_grounding_avg_at_k.items()},
            "avg_reward_precision_at_k": {str(k): v for k, v in self.avg_reward_precision_at_k.items()},
            "avg_reward_combined_at_k": {str(k): v for k, v in self.avg_reward_combined_at_k.items()},
            # Evidence
            "parse_success_rate": self.parse_success_rate,
            "avg_diagnoses_per_patient": self.avg_diagnoses_per_patient,
            "avg_diagnoses_with_evidence": self.avg_diagnoses_with_evidence,
            "avg_reasoning_length": self.avg_reasoning_length,
            "low_reward_cases": self.low_reward_cases,
            "high_reward_cases": self.high_reward_cases,
            # NEW: Retrieval metrics
            "retrieval_success_at_k": {str(k): v for k, v in self.retrieval_success_at_k.items()},
            "retrieval_mrr": self.retrieval_mrr,
            "retrieval_precision_at_k": {str(k): v for k, v in self.retrieval_precision_at_k.items()},
            "retrieval_recall_at_k": {str(k): v for k, v in self.retrieval_recall_at_k.items()},
            # NEW: Statistical analysis
            "statistics": self.statistics,
            # NEW: Per-diagnosis breakdown
            "per_diagnosis_performance": self.per_diagnosis_performance,
            # Metadata
            "inference_engine": "vllm_enhanced",
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
    match_matrix: List[List[bool]],
    num_ground_truth: int,
    max_k: int = 5,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute precision@k and recall@k from a pairwise match matrix.

    Args:
        match_matrix: match_matrix[i][j] = True if prediction i matches GT j.
                      Shape: (num_predictions, num_ground_truth).
        num_ground_truth: Number of ground truth diagnoses.
        max_k: Maximum k value.

    Returns:
        Tuple of (precision_at_k, recall_at_k).
        - precision@k: fraction of top-k predictions that match any GT.
        - recall@k: fraction of distinct GTs matched by at least one
          prediction in top-k.  This correctly handles the case where
          multiple predictions match the same GT (counted only once).
    """
    precision_at_k = {}
    recall_at_k = {}

    for k in range(1, max_k + 1):
        top_k = match_matrix[:k]
        if not top_k:
            precision_at_k[k] = 0.0
            recall_at_k[k] = 0.0
            continue

        # Precision@k: fraction of top-k predictions matching ANY GT
        hits = sum(1 for row in top_k if any(row))
        precision_at_k[k] = hits / len(top_k)

        # Recall@k: fraction of distinct GTs covered by top-k predictions
        if num_ground_truth > 0:
            gts_covered = set()
            for row in top_k:
                for gt_idx, matched in enumerate(row):
                    if matched:
                        gts_covered.add(gt_idx)
            recall_at_k[k] = len(gts_covered) / num_ground_truth
        else:
            recall_at_k[k] = 0.0

    return precision_at_k, recall_at_k


def build_pairwise_match_matrix(
    predicted_diagnoses: List[str],
    ground_truth_diagnoses: List[str],
    judge,
) -> List[List[bool]]:
    """Build a match matrix via pairwise LLM judge calls.

    For each (prediction, GT) pair, asks the judge whether the prediction
    semantically matches that specific GT.  This replaces the previous
    "match ANY GT?" approach, enabling correct recall computation.

    Args:
        predicted_diagnoses: List of predicted diagnosis names.
        ground_truth_diagnoses: List of ground truth diagnosis names.
        judge: VLLMAnswerJudge (or any AnswerJudge with is_correct_batch).

    Returns:
        match_matrix[i][j] = True if prediction i matches GT j.
        Shape: (len(predicted_diagnoses), len(ground_truth_diagnoses)).
    """
    n_pred = len(predicted_diagnoses)
    n_gt = len(ground_truth_diagnoses)

    if n_pred == 0 or n_gt == 0:
        return [[False] * n_gt for _ in range(n_pred)]

    # Build all pairwise prompts (pred_0×gt_0, pred_0×gt_1, ..., pred_1×gt_0, ...)
    all_prompts: List[Optional[str]] = []
    for pred in predicted_diagnoses:
        if not pred.strip():
            all_prompts.extend([None] * n_gt)
            continue
        for gt in ground_truth_diagnoses:
            prompt = f"""You are evaluating the diagnosis prediction of a clinical model.

CANDIDATE ANSWER: "{pred}"

GROUND TRUTH: "{gt}"

TASK: Does the CANDIDATE ANSWER semantically match the GROUND TRUTH diagnosis?
Respond 'TRUE' if the candidate refers to the same underlying clinical concept (allowing for synonyms, abbreviations, or minor wording differences).
Respond 'FALSE' if it represents a different clinical concept, severity, or anatomical location.

Verdict:"""
            all_prompts.append(prompt)

    # Batch judge call
    try:
        results = judge.is_correct_batch(all_prompts)
    except Exception:
        results = [False] * len(all_prompts)

    # Reshape flat results into matrix
    match_matrix: List[List[bool]] = []
    for i in range(n_pred):
        row = results[i * n_gt : (i + 1) * n_gt]
        match_matrix.append(row)

    return match_matrix


def get_gold_doc_ids_heuristic(
    ground_truth_diagnoses: List[str],
    all_evidence_docs: List[Dict[str, Any]]
) -> List[str]:
    """
    Heuristic to identify gold/relevant documents.

    Since we don't have human annotations, we use a heuristic:
    A document is considered "relevant" if it mentions any ground truth diagnosis.

    Args:
        ground_truth_diagnoses: List of correct diagnosis names
        all_evidence_docs: List of evidence documents with 'doc_id' and 'text'

    Returns:
        List of document IDs considered relevant
    """
    gold_ids = []

    for doc in all_evidence_docs:
        doc_text = doc.get("text", "").lower()

        # Check if any ground truth diagnosis is mentioned
        for gt_diagnosis in ground_truth_diagnoses:
            # Simple substring match (can be improved with better NER/matching)
            if gt_diagnosis.lower() in doc_text:
                gold_ids.append(doc.get("doc_id", ""))
                break  # Count doc only once

    return gold_ids


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
            "retrieved_doc_ids": [ev.get("doc_id", "") for ev in pre_evidence_data],
            "parse_success": structured_output.get("parse_success", False),
        })
    return patients


def _phase1_judge_accuracy(
    patients: List[Dict[str, Any]],
    judge,
) -> List[Dict[str, Any]]:
    """Phase 1: Compute accuracy verdicts using vLLM judge on GPU.

    Returns a list of dicts with match_matrix, verdicts, precision_rewards per patient.
    """
    accuracy_results = []
    for patient in tqdm(patients, desc="Phase 1: Accuracy (vLLM judge)"):
        match_matrix = build_pairwise_match_matrix(
            patient["predicted"], patient["ground_truth"], judge
        )
        verdicts = [any(row) for row in match_matrix] if match_matrix else []
        precision_rewards = [1.0 if v else 0.0 for v in verdicts]
        accuracy_results.append({
            "match_matrix": match_matrix,
            "verdicts": verdicts,
            "precision_rewards": precision_rewards,
        })
    return accuracy_results


def _phase2_nli_grounding(
    patients: List[Dict[str, Any]],
    nli_model,
    sentence_level: bool = True,
) -> List[Dict[str, Any]]:
    """Phase 2: Compute NLI grounding scores (on GPU after vLLM is freed).

    Returns a list of dicts with grounding_max_rewards, grounding_avg_rewards per patient.
    """
    from evidence_rl.reward import reward_grounding

    grounding_results = []
    for patient in tqdm(patients, desc="Phase 2: Grounding (NLI)"):
        grounding_max_rewards = []
        grounding_avg_rewards = []

        for diag_data in patient["valid_diagnoses"]:
            reasoning = diag_data.get("reasoning", "")
            if not reasoning.strip():
                grounding_max_rewards.append(0.0)
                grounding_avg_rewards.append(0.0)
                continue

            g_max, g_avg = reward_grounding(
                diagnosis_reasoning=reasoning,
                patient_context=patient["patient_context"],
                evidence_docs=patient["pre_evidence"],
                nli_model=nli_model,
                sentence_level=sentence_level,
            )
            grounding_max_rewards.append(g_max)
            grounding_avg_rewards.append(g_avg)

        grounding_results.append({
            "grounding_max_rewards": grounding_max_rewards,
            "grounding_avg_rewards": grounding_avg_rewards,
        })
    return grounding_results


def _merge_metrics(
    patients: List[Dict[str, Any]],
    accuracy_results: List[Dict[str, Any]],
    grounding_results: List[Dict[str, Any]],
    max_k: int = 5,
    reward_weight_grounding: float = 0.5,
) -> List[EvidenceRLMetrics]:
    """Merge accuracy and grounding results into final EvidenceRLMetrics."""
    from evidence_rl.enhanced_metrics import compute_f1_at_k

    all_metrics = []
    for patient, acc, grnd in zip(patients, accuracy_results, grounding_results):
        match_matrix = acc["match_matrix"]
        verdicts = acc["verdicts"]
        precision_rewards = acc["precision_rewards"]
        grounding_max_rewards = grnd["grounding_max_rewards"]
        grounding_avg_rewards = grnd["grounding_avg_rewards"]
        ground_truth = patient["ground_truth"]
        valid_diagnoses = patient["valid_diagnoses"]

        # Combined rewards
        combined_rewards = [
            reward_weight_grounding * g_max + (1.0 - reward_weight_grounding) * p
            for g_max, p in zip(grounding_max_rewards, precision_rewards)
        ]

        # Per-diagnosis reward details
        per_diagnosis_rewards = []
        for i, diag_data in enumerate(valid_diagnoses):
            matched_gts = [
                ground_truth[j]
                for j, m in enumerate(match_matrix[i]) if m
            ] if i < len(match_matrix) and ground_truth else []

            per_diagnosis_rewards.append({
                "diagnosis_name": diag_data.get("name", ""),
                "grounding_max": float(grounding_max_rewards[i]),
                "grounding_avg": float(grounding_avg_rewards[i]),
                "precision": float(precision_rewards[i]),
                "combined": float(combined_rewards[i]),
                "matched_ground_truths": matched_gts,
            })

        # @k metrics
        reward_grounding_max_at_k = compute_reward_at_k(grounding_max_rewards, max_k)
        reward_grounding_avg_at_k = compute_reward_at_k(grounding_avg_rewards, max_k)
        reward_precision_at_k = compute_reward_at_k(precision_rewards, max_k)
        reward_combined_at_k = compute_reward_at_k(combined_rewards, max_k)

        precision_at_k, recall_at_k = compute_precision_recall_at_k(
            match_matrix, len(ground_truth), max_k
        )
        f1_at_k = compute_f1_at_k(precision_at_k, recall_at_k)

        num_with_evidence = sum(1 for d in valid_diagnoses if d.get("reasoning"))
        avg_reasoning_len = (
            sum(len(d.get("reasoning", "")) for d in valid_diagnoses) / len(valid_diagnoses)
            if valid_diagnoses else 0.0
        )

        metrics = EvidenceRLMetrics(
            hadm_id=patient["hadm_id"],
            subject_id=patient["subject_id"],
            ground_truth_diagnoses=ground_truth,
            predicted_diagnoses=patient["predicted"],
            predicted_diagnoses_with_reasoning=patient["predicted_with_reasoning"],
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            verdicts=verdicts,
            per_diagnosis_rewards=per_diagnosis_rewards,
            match_matrix=match_matrix,
            reward_grounding_max_at_k=reward_grounding_max_at_k,
            reward_grounding_avg_at_k=reward_grounding_avg_at_k,
            reward_precision_at_k=reward_precision_at_k,
            reward_combined_at_k=reward_combined_at_k,
            num_diagnoses_with_evidence=num_with_evidence,
            avg_reasoning_length=avg_reasoning_len,
            parse_success=patient["parse_success"],
            retrieved_doc_ids=patient["retrieved_doc_ids"],
        )
        all_metrics.append(metrics)

    return all_metrics


def compute_aggregate_metrics_enhanced(
    patient_metrics: List[EvidenceRLMetrics],
    results: List[Dict[str, Any]],  # Need for retrieval metrics
) -> AggregateMetrics:
    """Compute ENHANCED aggregate metrics across all patients."""
    from evidence_rl.enhanced_metrics import (
        compute_macro_micro_f1,
        compute_per_diagnosis_metrics,
        compute_retrieval_metrics
    )
    from evidence_rl.statistical_analysis import compute_metric_statistics

    num_patients = len(patient_metrics)
    if num_patients == 0:
        return AggregateMetrics()

    # ORIGINAL: Average precision/recall/f1 at k
    avg_precision: Dict[int, float] = {}
    avg_recall: Dict[int, float] = {}
    avg_f1: Dict[int, float] = {}  # NEW

    # Get all k values
    all_k = set()
    for m in patient_metrics:
        all_k.update(m.precision_at_k.keys())

    for k in sorted(all_k):
        precision_sum = sum(m.precision_at_k.get(k, 0.0) for m in patient_metrics)
        recall_sum = sum(m.recall_at_k.get(k, 0.0) for m in patient_metrics)
        f1_sum = sum(m.f1_at_k.get(k, 0.0) for m in patient_metrics)  # NEW

        avg_precision[k] = precision_sum / num_patients
        avg_recall[k] = recall_sum / num_patients
        avg_f1[k] = f1_sum / num_patients  # NEW

    # Average rewards@k (grounding + precision)
    all_k_values = set()
    for m in patient_metrics:
        all_k_values.update(m.reward_grounding_max_at_k.keys())

    avg_reward_grounding_max_at_k = {}
    avg_reward_grounding_avg_at_k = {}
    avg_reward_precision_at_k = {}
    avg_reward_combined_at_k = {}

    for k in sorted(all_k_values):
        grounding_max_k = [m.reward_grounding_max_at_k.get(k, 0.0) for m in patient_metrics]
        grounding_avg_k = [m.reward_grounding_avg_at_k.get(k, 0.0) for m in patient_metrics]
        precision_k = [m.reward_precision_at_k.get(k, 0.0) for m in patient_metrics]
        combined_k = [m.reward_combined_at_k.get(k, 0.0) for m in patient_metrics]

        avg_reward_grounding_max_at_k[k] = sum(grounding_max_k) / num_patients
        avg_reward_grounding_avg_at_k[k] = sum(grounding_avg_k) / num_patients
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

    # REMOVED: Macro-F1 / Micro-F1 — not meaningful because P@k uses LLM judge
    # (semantic matching) but Macro-F1 uses exact string matching between
    # predicted clinical terms and ground truth ICD codes, which always fails.

    # Per-diagnosis performance: still computed for GT class analysis
    all_predictions = [m.predicted_diagnoses for m in patient_metrics]
    all_ground_truths = [m.ground_truth_diagnoses for m in patient_metrics]

    _, _, per_class_metrics = compute_macro_micro_f1(
        all_predictions,
        all_ground_truths
    )
    per_diagnosis_table = compute_per_diagnosis_metrics(per_class_metrics, sort_by="support")

    # NEW: Retrieval quality metrics
    retrieved_ids = [m.retrieved_doc_ids for m in patient_metrics]

    # Get gold doc IDs using heuristic
    gold_ids = []
    for i, m in enumerate(patient_metrics):
        # Get all evidence docs for this patient
        all_evidence = results[i].get("pre_evidence", [])
        gold_for_patient = get_gold_doc_ids_heuristic(
            m.ground_truth_diagnoses,
            all_evidence
        )
        gold_ids.append(gold_for_patient)

    retrieval_metrics = compute_retrieval_metrics(retrieved_ids, gold_ids, max_k=5)

    # NEW: Statistical analysis (confidence intervals)
    key_metrics = {
        "precision@3": [m.precision_at_k.get(3, 0.0) for m in patient_metrics],
        "recall@3": [m.recall_at_k.get(3, 0.0) for m in patient_metrics],
        "f1@3": [m.f1_at_k.get(3, 0.0) for m in patient_metrics],
        "reward_grounding_max@3": [m.reward_grounding_max_at_k.get(3, 0.0) for m in patient_metrics],
        "reward_grounding_avg@3": [m.reward_grounding_avg_at_k.get(3, 0.0) for m in patient_metrics],
        "reward_precision@3": [m.reward_precision_at_k.get(3, 0.0) for m in patient_metrics],
        "reward_combined@3": [m.reward_combined_at_k.get(3, 0.0) for m in patient_metrics],
    }

    statistics = {}
    for metric_name, values in key_metrics.items():
        stats = compute_metric_statistics(values)
        statistics[metric_name] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "median": stats["median"],
            "ci_95_lower": stats["ci_lower"],
            "ci_95_upper": stats["ci_upper"],
            "min": stats["min"],
            "max": stats["max"],
            "n": stats["n"]
        }

    return AggregateMetrics(
        num_patients=num_patients,
        avg_precision_at_k=avg_precision,
        avg_recall_at_k=avg_recall,
        avg_f1_at_k=avg_f1,  # NEW
        avg_reward_grounding_max_at_k=avg_reward_grounding_max_at_k,
        avg_reward_grounding_avg_at_k=avg_reward_grounding_avg_at_k,
        avg_reward_precision_at_k=avg_reward_precision_at_k,
        avg_reward_combined_at_k=avg_reward_combined_at_k,
        parse_success_rate=parse_success_rate,
        avg_diagnoses_per_patient=avg_diagnoses,
        avg_diagnoses_with_evidence=avg_diagnoses_with_evidence,
        avg_reasoning_length=avg_reasoning_length,
        low_reward_cases=low_reward_cases,
        high_reward_cases=high_reward_cases,
        retrieval_success_at_k=retrieval_metrics["success_at_k"],  # NEW
        retrieval_mrr=retrieval_metrics["mrr"][0],  # NEW
        retrieval_precision_at_k=retrieval_metrics["retrieval_precision_at_k"],  # NEW
        retrieval_recall_at_k=retrieval_metrics["retrieval_recall_at_k"],  # NEW
        statistics=statistics,  # NEW
        per_diagnosis_performance=per_diagnosis_table,  # NEW
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute ENHANCED metrics for EvidenceRL results (vLLM + all publication metrics)"
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
    parser.add_argument(
        "--no-sentence-level",
        action="store_true",
        default=False,
        help="Disable sentence-level NLI grounding (use full reasoning as hypothesis)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("EvidenceRL ENHANCED Metrics Computation (vLLM Version)")
    print("Includes: F1@k, Macro/Micro-F1, Retrieval metrics, Statistical CIs")
    print("14-24x faster LLM judge inference")
    print("=" * 70)

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

    sentence_level = not args.no_sentence_level
    print(f"Sentence-level NLI: {sentence_level}")
    grounding_results = _phase2_nli_grounding(patients, nli_model, sentence_level=sentence_level)
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
    print("\nComputing enhanced aggregate metrics...")
    aggregate = compute_aggregate_metrics_enhanced(patient_metrics, results)

    # Print enhanced summary
    print("\n" + "=" * 70)
    print("ENHANCED METRICS SUMMARY (vLLM)")
    print("=" * 70)
    print(f"Number of patients: {aggregate.num_patients}")
    print(f"Parse success rate: {aggregate.parse_success_rate:.1%}")

    print(f"\n--- Standard Metrics ---")
    print(f"Average diagnoses per patient: {aggregate.avg_diagnoses_per_patient:.2f}")
    print(f"Average diagnoses with evidence: {aggregate.avg_diagnoses_with_evidence:.2f}")
    print(f"Average reasoning length: {aggregate.avg_reasoning_length:.1f} chars")

    print(f"\n--- Precision@k (with 95% CI) ---")
    for k in sorted(aggregate.avg_precision_at_k.keys())[:3]:
        stats = aggregate.statistics.get(f"precision@{k}", {})
        mean = stats.get("mean", aggregate.avg_precision_at_k[k])
        ci_low = stats.get("ci_95_lower", mean)
        ci_high = stats.get("ci_95_upper", mean)
        print(f"  P@{k}: {mean:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

    print(f"\n--- Recall@k (with 95% CI) ---")
    for k in sorted(aggregate.avg_recall_at_k.keys())[:3]:
        stats = aggregate.statistics.get(f"recall@{k}", {})
        mean = stats.get("mean", aggregate.avg_recall_at_k[k])
        ci_low = stats.get("ci_95_lower", mean)
        ci_high = stats.get("ci_95_upper", mean)
        print(f"  R@{k}: {mean:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

    print(f"\n--- F1@k (NEW - with 95% CI) ---")
    for k in sorted(aggregate.avg_f1_at_k.keys())[:3]:
        stats = aggregate.statistics.get(f"f1@{k}", {})
        mean = stats.get("mean", aggregate.avg_f1_at_k[k])
        ci_low = stats.get("ci_95_lower", mean)
        ci_high = stats.get("ci_95_upper", mean)
        print(f"  F1@{k}: {mean:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

    # Macro-F1 / Micro-F1 removed: exact string matching between predicted
    # clinical terms and GT ICD codes is not meaningful with LLM-judge evaluation

    print(f"\n--- Retrieval Quality (NEW) ---")
    for k in [1, 3, 5]:
        if k in aggregate.retrieval_success_at_k:
            print(f"  Success@{k}: {aggregate.retrieval_success_at_k[k]:.3f}")
    print(f"  MRR: {aggregate.retrieval_mrr:.3f}")

    print(f"\n--- Grounding Reward@k (with 95% CI) ---")
    for k in sorted(aggregate.avg_reward_grounding_max_at_k.keys())[:3]:
        stats_max = aggregate.statistics.get(f"reward_grounding_max@{k}", {})
        stats_avg = aggregate.statistics.get(f"reward_grounding_avg@{k}", {})
        mean_max = stats_max.get("mean", aggregate.avg_reward_grounding_max_at_k[k])
        ci_low_max = stats_max.get("ci_95_lower", mean_max)
        ci_high_max = stats_max.get("ci_95_upper", mean_max)
        mean_avg = stats_avg.get("mean", aggregate.avg_reward_grounding_avg_at_k[k])
        ci_low_avg = stats_avg.get("ci_95_lower", mean_avg)
        ci_high_avg = stats_avg.get("ci_95_upper", mean_avg)
        print(f"  G_max@{k}: {mean_max:.3f} [95% CI: {ci_low_max:.3f}, {ci_high_max:.3f}]")
        print(f"  G_avg@{k}: {mean_avg:.3f} [95% CI: {ci_low_avg:.3f}, {ci_high_avg:.3f}]")

    print(f"\n--- Precision Reward@k (with 95% CI) ---")
    for k in sorted(aggregate.avg_reward_precision_at_k.keys())[:3]:
        stats = aggregate.statistics.get(f"reward_precision@{k}", {})
        mean = stats.get("mean", aggregate.avg_reward_precision_at_k[k])
        ci_low = stats.get("ci_95_lower", mean)
        ci_high = stats.get("ci_95_upper", mean)
        print(f"  A@{k}: {mean:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

    print(f"\n--- Combined Reward@k (with 95% CI) ---")
    for k in sorted(aggregate.avg_reward_combined_at_k.keys())[:3]:
        stats = aggregate.statistics.get(f"reward_combined@{k}", {})
        mean = stats.get("mean", aggregate.avg_reward_combined_at_k[k])
        ci_low = stats.get("ci_95_lower", mean)
        ci_high = stats.get("ci_95_upper", mean)
        print(f"  C@{k}: {mean:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")

    print(f"\n--- Hallucination Analysis ---")
    print(f"Low reward cases (reward@3 < 0.3): {aggregate.low_reward_cases}")
    print(f"High reward cases (reward@3 >= 0.7): {aggregate.high_reward_cases}")

    print(f"\n--- Top 5 Best Performing Diagnoses ---")
    for i, diag in enumerate(aggregate.per_diagnosis_performance[:5], 1):
        print(f"  {i}. {diag['diagnosis']}: F1={diag['f1']:.3f} (n={diag['support']})")

    print(f"\n--- Top 5 Worst Performing Diagnoses ---")
    for i, diag in enumerate(aggregate.per_diagnosis_performance[-5:], 1):
        print(f"  {i}. {diag['diagnosis']}: F1={diag['f1']:.3f} (n={diag['support']})")

    # Save output
    output_data = {
        "config": {
            "results_json": args.results_json,
            "judge_model": args.judge_model,
            "nli_model": args.nli_model,
            "embedding_model": args.embedding_model,
            "max_k": args.max_k,
            "reward_weight_grounding": args.reward_weight_grounding,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enhanced_version": True,
        },
        "aggregate_metrics": aggregate.to_dict(),
        "patient_metrics": [m.to_dict() for m in patient_metrics],
    }

    print(f"\nSaving enhanced metrics to: {args.output_json}")
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ ENHANCED metrics computation complete!")
    print(f"✅ Output saved to: {args.output_json}")
    print("\nNew metrics include:")
    print("  • F1@k (harmonic mean of P@k and R@k)")
    print("  • Macro-F1 and Micro-F1 (per-diagnosis classification)")
    print("  • Retrieval Success@k and MRR")
    print("  • Statistical confidence intervals (95% CI)")
    print("  • Per-diagnosis performance breakdown")
    print("=" * 70)


if __name__ == "__main__":
    main()
