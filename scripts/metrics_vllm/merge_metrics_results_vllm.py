#!/usr/bin/env python3
"""Merge metrics results from multiple vLLM workers into a single file.

This script:
1. Finds all worker metrics JSON files in input directory (metrics_worker{N}.json)
2. Validates structure and sorts by worker ID
3. Merges patient_metrics lists
4. Computes global aggregate metrics
5. Writes merged output JSON

Called by metrics_master_vllm.sbatch after all workers complete.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_worker_files(input_dir: Path, num_workers: int, model_id: str = "model", version: str = "1.0", metrics_type: str = "") -> List[Dict[str, Any]]:
    """Load and validate all worker JSON files.

    File naming convention:
    - {model_id}_{metrics_type}_metrics_vllm_worker{N}_v{version}.json (with metrics_type)
    - {model_id}_metrics_worker{N}_v{version}.json (without metrics_type, legacy)

    The function will try multiple patterns to find the worker files.
    """
    worker_data = []

    for worker_id in range(num_workers):
        # Try multiple file naming patterns
        patterns = []
        if metrics_type:
            # New naming: {model_id}_{type}_metrics_vllm_worker{N}_v{version}.json
            patterns.append(f"{model_id}_{metrics_type}_metrics_vllm_worker{worker_id}_v{version}.json")
        # Legacy naming: {model_id}_metrics_worker{N}_v{version}.json
        patterns.append(f"{model_id}_metrics_worker{worker_id}_v{version}.json")
        # Also try finding any file matching worker pattern
        patterns.append(f"*_worker{worker_id}_v{version}.json")

        metrics_file = None
        for pattern in patterns:
            if "*" in pattern:
                # Glob pattern
                matches = list(input_dir.glob(pattern))
                if matches:
                    metrics_file = matches[0]
                    break
            else:
                candidate = input_dir / pattern
                if candidate.exists():
                    metrics_file = candidate
                    break

        if metrics_file and metrics_file.exists():
            try:
                with open(metrics_file, "r") as fp:
                    data = json.load(fp)

                if "patient_metrics" not in data:
                    print(f"[Merge] WARNING: {metrics_file.name} missing 'patient_metrics', skipping")
                    continue

                worker_data.append({
                    "worker_id": worker_id,
                    "file": metrics_file.name,
                    "data": data,
                    "num_patients": len(data["patient_metrics"]),
                })
                print(f"[Merge] Loaded {metrics_file.name}: {len(data['patient_metrics'])} patients")

            except json.JSONDecodeError as e:
                print(f"[Merge] ERROR: Invalid JSON in {metrics_file.name}: {e}")
            except Exception as e:
                print(f"[Merge] ERROR: Failed to load {metrics_file.name}: {e}")
        else:
            print(f"[Merge] WARNING: Missing file: {metrics_file.name}")

    return worker_data


def compute_global_aggregate(patient_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics across all patients."""
    num_patients = len(patient_metrics)
    if num_patients == 0:
        return {"num_patients": 0}

    # Collect all k values
    all_k = set()
    for m in patient_metrics:
        if m.get("precision_at_k"):
            all_k.update(int(k) for k in m["precision_at_k"].keys())

    # Average precision/recall at k
    avg_precision = {}
    avg_recall = {}
    for k in sorted(all_k):
        precision_sum = sum(float(m.get("precision_at_k", {}).get(str(k), 0.0)) for m in patient_metrics)
        recall_sum = sum(float(m.get("recall_at_k", {}).get(str(k), 0.0)) for m in patient_metrics)
        avg_precision[k] = precision_sum / num_patients
        avg_recall[k] = recall_sum / num_patients

    # Collect reward k values
    all_reward_k = set()
    for m in patient_metrics:
        if m.get("reward_grounding_at_k"):
            all_reward_k.update(int(k) for k in m["reward_grounding_at_k"].keys())

    # Average rewards@k
    avg_reward_grounding_at_k = {}
    avg_reward_precision_at_k = {}
    avg_reward_combined_at_k = {}

    for k in sorted(all_reward_k):
        grounding_sum = sum(float(m.get("reward_grounding_at_k", {}).get(str(k), 0.0)) for m in patient_metrics)
        precision_sum = sum(float(m.get("reward_precision_at_k", {}).get(str(k), 0.0)) for m in patient_metrics)
        combined_sum = sum(float(m.get("reward_combined_at_k", {}).get(str(k), 0.0)) for m in patient_metrics)

        avg_reward_grounding_at_k[k] = grounding_sum / num_patients
        avg_reward_precision_at_k[k] = precision_sum / num_patients
        avg_reward_combined_at_k[k] = combined_sum / num_patients

    # Evidence grounding statistics
    parse_success_count = sum(1 for m in patient_metrics if m.get("parse_success", False))
    parse_success_rate = parse_success_count / num_patients

    total_diagnoses = sum(len(m.get("predicted_diagnoses", [])) for m in patient_metrics)
    avg_diagnoses = total_diagnoses / num_patients

    total_with_evidence = sum(m.get("num_diagnoses_with_evidence", 0) for m in patient_metrics)
    avg_diagnoses_with_evidence = total_with_evidence / num_patients

    total_reasoning_len = sum(m.get("avg_reasoning_length", 0.0) for m in patient_metrics)
    avg_reasoning_length = total_reasoning_len / num_patients

    # Hallucination analysis (use reward_combined@3 as representative)
    low_reward_cases = sum(
        1 for m in patient_metrics
        if float(m.get("reward_combined_at_k", {}).get("3", 0.0)) < 0.3
    )
    high_reward_cases = sum(
        1 for m in patient_metrics
        if float(m.get("reward_combined_at_k", {}).get("3", 0.0)) >= 0.7
    )

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
        "low_reward_pct": low_reward_cases / num_patients * 100,
        "high_reward_pct": high_reward_cases / num_patients * 100,
        "inference_engine": "vllm",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge metrics results from multiple vLLM workers"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing worker JSON files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save merged JSON file",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Number of workers to merge",
    )
    parser.add_argument(
        "--model-id",
        default="model",
        help="Model ID for file naming (default: model)",
    )
    parser.add_argument(
        "--version",
        default="1.0",
        help="Version for file naming (default: 1.0)",
    )
    parser.add_argument(
        "--metrics-type",
        default="",
        help="Metrics type prefix (e.g., 'baseline', 'baseline_rag', 'dpo')",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    print(f"[Merge] Input directory: {input_dir}")
    print(f"[Merge] Number of workers: {args.num_workers}")
    print(f"[Merge] Model ID: {args.model_id}")
    print(f"[Merge] Version: {args.version}")
    print(f"[Merge] Output: {output_path}")
    print(f"[Merge] Inference engine: vLLM")

    # Load all worker files
    worker_data = load_worker_files(input_dir, args.num_workers, args.model_id, args.version, args.metrics_type)

    if not worker_data:
        print("[Merge] ERROR: No valid worker files found")
        sys.exit(1)

    print(f"\n[Merge] Found {len(worker_data)} valid worker files")

    # Sort by worker_id
    worker_data.sort(key=lambda x: x["worker_id"])

    # Merge patient metrics
    all_patient_metrics = []
    seen_hadm_ids = set()

    for w in worker_data:
        for m in w["data"]["patient_metrics"]:
            hadm_id = m.get("hadm_id", "")
            if hadm_id in seen_hadm_ids:
                print(f"[Merge] WARNING: Duplicate hadm_id {hadm_id}, skipping")
                continue
            seen_hadm_ids.add(hadm_id)
            all_patient_metrics.append(m)

    print(f"[Merge] Total patients after merge: {len(all_patient_metrics)}")

    # Compute global aggregate
    aggregate_metrics = compute_global_aggregate(all_patient_metrics)

    # Get config from first worker (they should all be the same)
    config = worker_data[0]["data"].get("config", {}).copy() if worker_data else {}
    config["num_workers"] = len(worker_data)
    config["total_patients"] = len(all_patient_metrics)
    config["inference_engine"] = "vllm"
    # Remove worker-specific fields
    config.pop("patient_start_idx", None)
    config.pop("patient_end_idx", None)
    config.pop("worker_id", None)

    # Print summary
    print("\n" + "=" * 70)
    print("MERGED METRICS SUMMARY (vLLM)")
    print("=" * 70)
    print(f"Total patients: {aggregate_metrics['num_patients']}")
    print(f"Workers merged: {len(worker_data)}")
    print(f"Inference engine: vLLM")
    print(f"Parse success rate: {aggregate_metrics['parse_success_rate']:.1%}")

    print("\nPrecision@k:")
    for k in sorted(aggregate_metrics.get("avg_precision_at_k", {}).keys(), key=int):
        print(f"  P@{k}: {aggregate_metrics['avg_precision_at_k'][k]:.3f}")

    print("\nRecall@k:")
    for k in sorted(aggregate_metrics.get("avg_recall_at_k", {}).keys(), key=int):
        print(f"  R@{k}: {aggregate_metrics['avg_recall_at_k'][k]:.3f}")

    print("\nReward@k Statistics:")
    for k in sorted(aggregate_metrics.get("avg_reward_combined_at_k", {}).keys(), key=int):
        grounding_k = aggregate_metrics["avg_reward_grounding_at_k"].get(k, 0.0)
        precision_k = aggregate_metrics["avg_reward_precision_at_k"].get(k, 0.0)
        combined_k = aggregate_metrics["avg_reward_combined_at_k"].get(k, 0.0)
        print(f"  @{k}: Grounding={grounding_k:.3f}, Precision={precision_k:.3f}, Combined={combined_k:.3f}")

    print(f"\nEvidence Consistency Analysis:")
    print(f"  Low reward cases (< 0.3): {aggregate_metrics['low_reward_cases']} ({aggregate_metrics['low_reward_pct']:.1f}%)")
    print(f"  High reward cases (>= 0.7): {aggregate_metrics['high_reward_cases']} ({aggregate_metrics['high_reward_pct']:.1f}%)")
    print("=" * 70)

    # Build output
    output_data = {
        "config": config,
        "aggregate_metrics": aggregate_metrics,
        "patient_metrics": all_patient_metrics,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Merge] Saved merged metrics to: {output_path}")
    print("[Merge] Done!")


if __name__ == "__main__":
    main()
