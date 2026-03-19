#!/usr/bin/env python3
"""Merge generation results from distributed worker jobs.

This script combines JSON output files from multiple worker jobs into a single
merged output file, preserving all results and aggregating statistics.

Usage:
    python merge_generation_results.py --input-dir <dir> --output <file>

The script will:
1. Find all worker output files matching the pattern *_worker*_v*.json
2. Load and validate each file
3. Merge all results sorted by patient_start_idx
4. Compute aggregate statistics
5. Write merged output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_worker_results(input_dir: Path, model_id: str | None = None) -> list[dict]:
    """Load all worker result files from input directory."""
    results = []

    # Find all worker output files
    pattern = "*_worker*_v*.json"
    worker_files = sorted(input_dir.glob(pattern))

    if not worker_files:
        print(f"[merge] Warning: No worker files found in {input_dir}")
        return results

    print(f"[merge] Found {len(worker_files)} worker files")

    for filepath in worker_files:
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate structure
            if "results" not in data or "config" not in data:
                print(f"[merge] Warning: Invalid structure in {filepath.name}, skipping")
                continue

            # Filter by model_id if specified
            if model_id and model_id not in filepath.name:
                continue

            # Get worker info from config
            config = data.get("config", {})
            start_idx = config.get("patient_start_idx", 0)

            results.append({
                "filepath": filepath,
                "start_idx": start_idx,
                "config": config,
                "summary": data.get("summary", {}),
                "results": data.get("results", []),
            })

            print(f"[merge] Loaded {filepath.name}: {len(data['results'])} patients")

        except json.JSONDecodeError as e:
            print(f"[merge] Error: Failed to parse {filepath.name}: {e}")
        except Exception as e:
            print(f"[merge] Error: Failed to load {filepath.name}: {e}")

    return results


def merge_results(worker_results: list[dict]) -> dict:
    """Merge worker results into a single output."""
    if not worker_results:
        return {
            "config": {},
            "summary": {"num_cases": 0, "error": "No worker results found"},
            "results": [],
        }

    # Sort by start index
    worker_results.sort(key=lambda x: x["start_idx"])

    # Merge results
    all_results = []
    seen_hadm_ids = set()

    for worker in worker_results:
        for result in worker["results"]:
            hadm_id = result.get("hadm_id")
            if hadm_id and hadm_id not in seen_hadm_ids:
                all_results.append(result)
                seen_hadm_ids.add(hadm_id)
            elif not hadm_id:
                # No hadm_id, add anyway
                all_results.append(result)

    # Use first worker's config as base, update patient counts
    base_config = worker_results[0]["config"].copy()
    base_config["num_patients"] = len(all_results)
    base_config["num_workers"] = len(worker_results)
    base_config["patient_start_idx"] = 0
    base_config["patient_end_idx"] = len(all_results)

    # Compute aggregate summary
    summary = compute_aggregate_summary(all_results)
    summary["num_cases"] = len(all_results)
    summary["num_workers"] = len(worker_results)

    return {
        "config": base_config,
        "summary": summary,
        "results": all_results,
    }


def compute_aggregate_summary(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate statistics from results."""
    if not results:
        return {}

    num_cases = len(results)

    # Parse success rate
    parse_successes = sum(
        1 for r in results
        if r.get("structured_output", {}).get("parse_success", False)
    )
    parse_success_rate = parse_successes / num_cases if num_cases > 0 else 0

    # Count diagnoses with reasoning
    total_diagnoses_with_reasoning = 0
    for r in results:
        structured = r.get("structured_output", {})
        diagnoses = structured.get("diagnoses", [])
        for diag in diagnoses:
            if diag.get("reasoning"):
                total_diagnoses_with_reasoning += 1

    avg_diagnoses_with_reasoning = total_diagnoses_with_reasoning / num_cases if num_cases > 0 else 0

    return {
        "parse_success_rate": parse_success_rate,
        "avg_diagnoses_with_reasoning": avg_diagnoses_with_reasoning,
        "total_diagnoses_with_reasoning": total_diagnoses_with_reasoning,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge distributed generation results"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing worker output files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for merged output file",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Filter by model ID (optional)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string (for logging only)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"[merge] Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"[merge] Input directory: {input_dir}")
    print(f"[merge] Output file: {output_path}")

    # Load worker results
    worker_results = load_worker_results(input_dir, args.model_id)

    if not worker_results:
        print("[merge] Error: No valid worker results found")
        sys.exit(1)

    # Merge
    merged = merge_results(worker_results)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"[merge] Merged {len(merged['results'])} results from {len(worker_results)} workers")
    print(f"[merge] Output saved to: {output_path}")

    # Print summary
    print("\n[merge] Summary:")
    for key, value in merged["summary"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
