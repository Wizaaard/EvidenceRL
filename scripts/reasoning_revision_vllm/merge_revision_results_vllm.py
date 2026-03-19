#!/usr/bin/env python3
"""Merge revision results from multiple vLLM workers into single files.

This script:
1. Finds all worker generation and metrics JSON files in input directory
2. Validates structure and sorts by patient index
3. Merges results lists
4. Computes global revision statistics
5. Writes merged output JSON files

Called by revision_master_vllm.sbatch after all workers complete.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_worker_files(input_dir: Path, num_workers: int) -> Tuple[List[Dict], List[Dict]]:
    """Load and validate all worker JSON files."""
    gen_data = []
    metrics_data = []

    for worker_id in range(num_workers):
        gen_file = input_dir / f"generation_worker{worker_id}.json"
        metrics_file = input_dir / f"metrics_worker{worker_id}.json"

        # Load generation file
        if gen_file.exists():
            try:
                with open(gen_file, "r") as fp:
                    data = json.load(fp)

                if "results" not in data:
                    print(f"[Merge] WARNING: {gen_file.name} missing 'results', skipping")
                    continue

                gen_data.append({
                    "worker_id": worker_id,
                    "file": gen_file.name,
                    "data": data,
                    "num_patients": len(data["results"]),
                })
                print(f"[Merge] Loaded {gen_file.name}: {len(data['results'])} patients")

            except json.JSONDecodeError as e:
                print(f"[Merge] ERROR: Invalid JSON in {gen_file.name}: {e}")
            except Exception as e:
                print(f"[Merge] ERROR: Failed to load {gen_file.name}: {e}")
        else:
            print(f"[Merge] WARNING: Missing file: {gen_file.name}")

        # Load metrics file
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as fp:
                    data = json.load(fp)

                if "patient_metrics" not in data:
                    print(f"[Merge] WARNING: {metrics_file.name} missing 'patient_metrics', skipping")
                    continue

                metrics_data.append({
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

    return gen_data, metrics_data


def compute_revision_stats(gen_data: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics about revisions."""
    total_patients = 0
    total_diagnoses = 0
    total_diagnoses_revised = 0
    total_case_a = 0
    total_case_b = 0
    total_skipped = 0

    for worker in gen_data:
        stats = worker["data"].get("revision_stats", {})
        total_patients += stats.get("total_patients", 0)
        total_diagnoses += stats.get("total_diagnoses", 0)
        total_diagnoses_revised += stats.get("total_diagnoses_revised", 0)
        total_case_a += stats.get("case_a_count", 0)
        total_case_b += stats.get("case_b_count", 0)
        total_skipped += stats.get("skipped_count", 0)

    return {
        "total_patients": total_patients,
        "total_diagnoses": total_diagnoses,
        "total_diagnoses_revised": total_diagnoses_revised,
        "case_a_count": total_case_a,
        "case_b_count": total_case_b,
        "skipped_count": total_skipped,
        "revision_rate": total_diagnoses_revised / max(total_diagnoses, 1),
        "case_a_percentage": total_case_a / max(total_case_a + total_case_b, 1) * 100,
        "case_b_percentage": total_case_b / max(total_case_a + total_case_b, 1) * 100,
        "inference_engine": "vllm",
    }


def merge_generation_results(gen_data: List[Dict]) -> Dict[str, Any]:
    """Merge generation results from all workers."""
    # Sort by worker_id to maintain order
    gen_data.sort(key=lambda x: x["worker_id"])

    # Merge results
    all_results = []
    seen_hadm_ids = set()

    for worker in gen_data:
        for result in worker["data"]["results"]:
            hadm_id = result.get("hadm_id", "")
            if hadm_id in seen_hadm_ids:
                print(f"[Merge] WARNING: Duplicate hadm_id {hadm_id}, skipping")
                continue
            seen_hadm_ids.add(hadm_id)
            all_results.append(result)

    # Get config from first worker (remove worker-specific fields)
    config = gen_data[0]["data"].get("config", {}).copy() if gen_data else {}
    config.pop("patient_start", None)
    config.pop("patient_end", None)
    config.pop("worker_id", None)
    config["num_workers"] = len(gen_data)
    config["total_patients"] = len(all_results)
    config["inference_engine"] = "vllm"

    # Compute revision stats
    revision_stats = compute_revision_stats(gen_data)

    return {
        "config": config,
        "revision_stats": revision_stats,
        "results": all_results,
    }


def merge_metrics_results(metrics_data: List[Dict]) -> Dict[str, Any]:
    """Merge metrics results from all workers."""
    # Sort by worker_id to maintain order
    metrics_data.sort(key=lambda x: x["worker_id"])

    # Merge patient_metrics
    all_patient_metrics = []
    seen_hadm_ids = set()

    for worker in metrics_data:
        for pm in worker["data"]["patient_metrics"]:
            hadm_id = pm.get("hadm_id", "")
            if hadm_id in seen_hadm_ids:
                print(f"[Merge] WARNING: Duplicate hadm_id in metrics {hadm_id}, skipping")
                continue
            seen_hadm_ids.add(hadm_id)
            all_patient_metrics.append(pm)

    # Get config from first worker
    config = metrics_data[0]["data"].get("config", {}).copy() if metrics_data else {}
    config.pop("patient_start", None)
    config.pop("patient_end", None)
    config.pop("worker_id", None)
    config["num_workers"] = len(metrics_data)
    config["total_patients"] = len(all_patient_metrics)
    config["inference_engine"] = "vllm"

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(all_patient_metrics)

    return {
        "config": config,
        "aggregate_metrics": aggregate,
        "patient_metrics": all_patient_metrics,
    }


def compute_aggregate_metrics(patient_metrics: List[Dict]) -> Dict[str, Any]:
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

    # Parse success rate
    parse_success_count = sum(1 for m in patient_metrics if m.get("parse_success", False))
    parse_success_rate = parse_success_count / num_patients

    return {
        "num_patients": num_patients,
        "avg_precision_at_k": {str(k): v for k, v in avg_precision.items()},
        "avg_recall_at_k": {str(k): v for k, v in avg_recall.items()},
        "avg_reward_grounding_at_k": {str(k): v for k, v in avg_reward_grounding_at_k.items()},
        "avg_reward_precision_at_k": {str(k): v for k, v in avg_reward_precision_at_k.items()},
        "avg_reward_combined_at_k": {str(k): v for k, v in avg_reward_combined_at_k.items()},
        "parse_success_rate": parse_success_rate,
        "inference_engine": "vllm",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge revision results from multiple vLLM workers"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing worker JSON files",
    )
    parser.add_argument(
        "--output-gen",
        required=True,
        help="Path to save merged generation JSON file",
    )
    parser.add_argument(
        "--output-metrics",
        required=True,
        help="Path to save merged metrics JSON file",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Number of workers to merge",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_gen = Path(args.output_gen)
    output_metrics = Path(args.output_metrics)

    print(f"[Merge] Input directory: {input_dir}")
    print(f"[Merge] Number of workers: {args.num_workers}")
    print(f"[Merge] Output generation: {output_gen}")
    print(f"[Merge] Output metrics: {output_metrics}")
    print(f"[Merge] Inference engine: vLLM")

    # Load all worker files
    gen_data, metrics_data = load_worker_files(input_dir, args.num_workers)

    if not gen_data:
        print("[Merge] ERROR: No valid generation files found")
        sys.exit(1)

    print(f"\n[Merge] Found {len(gen_data)} generation files, {len(metrics_data)} metrics files")

    # Merge generation results
    merged_gen = merge_generation_results(gen_data)

    # Merge metrics results (if available)
    merged_metrics = merge_metrics_results(metrics_data) if metrics_data else None

    # Print summary
    revision_stats = merged_gen.get("revision_stats", {})
    print("\n" + "=" * 70)
    print("MERGED REVISION SUMMARY (vLLM)")
    print("=" * 70)
    print(f"Total patients: {revision_stats.get('total_patients', 0)}")
    print(f"Workers merged: {len(gen_data)}")
    print(f"Inference engine: vLLM")
    print(f"Total diagnoses revised: {revision_stats.get('total_diagnoses_revised', 0)}")
    print(f"  Case A (well-grounded): {revision_stats.get('case_a_count', 0)} ({revision_stats.get('case_a_percentage', 0):.1f}%)")
    print(f"  Case B (contradictory): {revision_stats.get('case_b_count', 0)} ({revision_stats.get('case_b_percentage', 0):.1f}%)")
    print(f"Skipped (outside range): {revision_stats.get('skipped_count', 0)}")
    print(f"Revision rate: {revision_stats.get('revision_rate', 0):.1%}")

    if merged_metrics:
        aggregate = merged_metrics.get("aggregate_metrics", {})
        print(f"\nMetrics (after revision):")
        print(f"  Parse success rate: {aggregate.get('parse_success_rate', 0):.1%}")
        if aggregate.get("avg_reward_combined_at_k"):
            for k in sorted(aggregate["avg_reward_combined_at_k"].keys(), key=int):
                grounding = aggregate["avg_reward_grounding_at_k"].get(k, 0.0)
                precision = aggregate["avg_reward_precision_at_k"].get(k, 0.0)
                combined = aggregate["avg_reward_combined_at_k"].get(k, 0.0)
                print(f"  @{k}: Grounding={grounding:.3f}, Precision={precision:.3f}, Combined={combined:.3f}")

    print("=" * 70)

    # Save results
    output_gen.parent.mkdir(parents=True, exist_ok=True)
    with open(output_gen, "w") as f:
        json.dump(merged_gen, f, indent=2)
    print(f"\n[Merge] Saved merged generation to: {output_gen}")

    if merged_metrics:
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(output_metrics, "w") as f:
            json.dump(merged_metrics, f, indent=2)
        print(f"[Merge] Saved merged metrics to: {output_metrics}")

    print("[Merge] Done!")


if __name__ == "__main__":
    main()
