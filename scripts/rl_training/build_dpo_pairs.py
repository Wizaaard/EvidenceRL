#!/usr/bin/env python3
"""
Build DPO (Direct Preference Optimization) training pairs from metrics output.

This script constructs preference pairs for DPO training by comparing diagnoses
within each patient based on:
1. Correctness (verdict from LLM judge)
2. Grounding score (NLI-based evidence faithfulness)

Pair Construction Strategy:
- Within each patient, compare all pairs of diagnoses
- If both correct: prefer higher grounding (requires gap >= threshold)
- If one correct, one incorrect: prefer correct one
- If both incorrect: skip (no clear preference signal)

Usage:
    python build_dpo_pairs.py --metrics-file <path> --output <path> [options]
    python build_dpo_pairs.py --metrics-file <path> --stats-only  # Just show statistics
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from itertools import combinations


@dataclass
class DPOPair:
    """A single DPO preference pair."""
    patient_id: str
    hadm_id: str

    # Chosen (preferred) response
    chosen_diagnosis: str
    chosen_reasoning: str
    chosen_grounding: float
    chosen_correct: bool

    # Rejected response
    rejected_diagnosis: str
    rejected_reasoning: str
    rejected_grounding: float
    rejected_correct: bool

    # Metadata
    pair_type: str  # "correctness" or "grounding"
    grounding_gap: float

    # Patient context (for training prompt)
    ground_truth_diagnoses: list


@dataclass
class PairStatistics:
    """Statistics about the generated pairs."""
    total_patients: int
    patients_with_pairs: int
    total_diagnoses: int
    total_pairs: int

    # By pair type
    correctness_pairs: int  # one correct, one incorrect
    grounding_pairs: int    # both correct, different grounding

    # Gap distribution for grounding pairs
    grounding_gap_mean: float
    grounding_gap_std: float
    grounding_gap_min: float
    grounding_gap_max: float

    # Correctness distribution
    correct_diagnoses: int
    incorrect_diagnoses: int

    # Grounding distribution
    grounding_mean: float
    grounding_std: float
    grounding_positive_count: int  # grounding > 0
    grounding_high_count: int      # grounding >= 0.5


def load_metrics(metrics_file: str) -> dict:
    """Load metrics JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def build_pairs_for_patient(
    patient: dict,
    grounding_gap_threshold: float = 0.2,
    require_chosen_correct: bool = True,
) -> list[DPOPair]:
    """
    Build all valid DPO pairs for a single patient.

    Args:
        patient: Patient metrics dict
        grounding_gap_threshold: Minimum grounding difference for grounding-based pairs
        require_chosen_correct: If True, chosen must always be correct

    Returns:
        List of DPOPair objects
    """
    pairs = []

    # Extract data
    hadm_id = patient.get('hadm_id', 'unknown')
    subject_id = patient.get('subject_id', 'unknown')
    ground_truth = patient.get('ground_truth_diagnoses', [])
    verdicts = patient.get('verdicts', [])
    per_diagnosis = patient.get('per_diagnosis_rewards', [])
    reasoning_data = patient.get('predicted_diagnoses_with_reasoning', [])

    # Build lookup for reasoning
    reasoning_lookup = {d['name']: d['reasoning'] for d in reasoning_data}

    # Combine into list of diagnosis info
    diagnoses = []
    for i, diag_reward in enumerate(per_diagnosis):
        name = diag_reward['diagnosis_name']
        diagnoses.append({
            'name': name,
            'reasoning': reasoning_lookup.get(name, ''),
            'grounding': diag_reward['grounding'],
            'correct': verdicts[i] if i < len(verdicts) else False,
        })

    # Generate all pairs
    for i, j in combinations(range(len(diagnoses)), 2):
        d1, d2 = diagnoses[i], diagnoses[j]

        # Skip if either has no reasoning
        if not d1['reasoning'] or not d2['reasoning']:
            continue

        # Determine pair type and preference
        pair_type = None
        chosen, rejected = None, None

        if d1['correct'] and d2['correct']:
            # Both correct: prefer higher grounding (if gap sufficient)
            gap = abs(d1['grounding'] - d2['grounding'])
            if gap >= grounding_gap_threshold:
                pair_type = "grounding"
                if d1['grounding'] > d2['grounding']:
                    chosen, rejected = d1, d2
                else:
                    chosen, rejected = d2, d1

        elif d1['correct'] and not d2['correct']:
            # d1 correct, d2 incorrect: prefer d1
            pair_type = "correctness"
            chosen, rejected = d1, d2

        elif not d1['correct'] and d2['correct']:
            # d2 correct, d1 incorrect: prefer d2
            pair_type = "correctness"
            chosen, rejected = d2, d1

        else:
            # Both incorrect: skip (no clear preference)
            continue

        # Skip if no valid pair was created
        if pair_type is None or chosen is None or rejected is None:
            continue

        # Apply require_chosen_correct filter
        if require_chosen_correct and not chosen['correct']:
            continue

        # Create pair
        grounding_gap = chosen['grounding'] - rejected['grounding']

        pair = DPOPair(
            patient_id=subject_id,
            hadm_id=hadm_id,
            chosen_diagnosis=chosen['name'],
            chosen_reasoning=chosen['reasoning'],
            chosen_grounding=chosen['grounding'],
            chosen_correct=chosen['correct'],
            rejected_diagnosis=rejected['name'],
            rejected_reasoning=rejected['reasoning'],
            rejected_grounding=rejected['grounding'],
            rejected_correct=rejected['correct'],
            pair_type=pair_type,
            grounding_gap=grounding_gap,
            ground_truth_diagnoses=ground_truth,
        )
        pairs.append(pair)

    return pairs


def compute_statistics(pairs: list[DPOPair], metrics_data: dict) -> PairStatistics:
    """Compute statistics about the generated pairs."""
    import numpy as np

    patient_metrics = metrics_data.get('patient_metrics', [])

    # Collect all diagnoses info
    all_groundings = []
    correct_count = 0
    incorrect_count = 0

    for patient in patient_metrics:
        verdicts = patient.get('verdicts', [])
        per_diagnosis = patient.get('per_diagnosis_rewards', [])

        for i, diag in enumerate(per_diagnosis):
            all_groundings.append(diag['grounding'])
            if i < len(verdicts):
                if verdicts[i]:
                    correct_count += 1
                else:
                    incorrect_count += 1

    # Pair statistics
    correctness_pairs = [p for p in pairs if p.pair_type == "correctness"]
    grounding_pairs = [p for p in pairs if p.pair_type == "grounding"]

    grounding_gaps = [p.grounding_gap for p in grounding_pairs]

    # Count patients with pairs
    patients_with_pairs = len(set(p.hadm_id for p in pairs))

    groundings_arr = np.array(all_groundings) if all_groundings else np.array([0.0])
    gaps_arr = np.array(grounding_gaps) if grounding_gaps else np.array([0.0])

    return PairStatistics(
        total_patients=len(patient_metrics),
        patients_with_pairs=patients_with_pairs,
        total_diagnoses=len(all_groundings),
        total_pairs=len(pairs),
        correctness_pairs=len(correctness_pairs),
        grounding_pairs=len(grounding_pairs),
        grounding_gap_mean=float(np.mean(gaps_arr)) if grounding_gaps else 0.0,
        grounding_gap_std=float(np.std(gaps_arr)) if grounding_gaps else 0.0,
        grounding_gap_min=float(np.min(gaps_arr)) if grounding_gaps else 0.0,
        grounding_gap_max=float(np.max(gaps_arr)) if grounding_gaps else 0.0,
        correct_diagnoses=correct_count,
        incorrect_diagnoses=incorrect_count,
        grounding_mean=float(np.mean(groundings_arr)),
        grounding_std=float(np.std(groundings_arr)),
        grounding_positive_count=int(np.sum(groundings_arr > 0)),
        grounding_high_count=int(np.sum(groundings_arr >= 0.5)),
    )


def print_statistics(stats: PairStatistics, grounding_gap_threshold: float):
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("DPO PAIR CONSTRUCTION STATISTICS")
    print("=" * 70)

    print(f"\n{'Dataset Overview':^70}")
    print("-" * 70)
    print(f"  Total patients:              {stats.total_patients:,}")
    print(f"  Patients with valid pairs:   {stats.patients_with_pairs:,} ({100*stats.patients_with_pairs/stats.total_patients:.1f}%)")
    print(f"  Total diagnoses:             {stats.total_diagnoses:,}")
    print(f"  Avg diagnoses per patient:   {stats.total_diagnoses/stats.total_patients:.1f}")

    print(f"\n{'Correctness Distribution':^70}")
    print("-" * 70)
    print(f"  Correct diagnoses:           {stats.correct_diagnoses:,} ({100*stats.correct_diagnoses/stats.total_diagnoses:.1f}%)")
    print(f"  Incorrect diagnoses:         {stats.incorrect_diagnoses:,} ({100*stats.incorrect_diagnoses/stats.total_diagnoses:.1f}%)")

    print(f"\n{'Grounding Distribution':^70}")
    print("-" * 70)
    print(f"  Mean grounding:              {stats.grounding_mean:.3f}")
    print(f"  Std grounding:               {stats.grounding_std:.3f}")
    print(f"  Positive grounding (>0):     {stats.grounding_positive_count:,} ({100*stats.grounding_positive_count/stats.total_diagnoses:.1f}%)")
    print(f"  High grounding (>=0.5):      {stats.grounding_high_count:,} ({100*stats.grounding_high_count/stats.total_diagnoses:.1f}%)")

    print(f"\n{'DPO Pairs Generated':^70}")
    print("-" * 70)
    print(f"  TOTAL PAIRS:                 {stats.total_pairs:,}")
    print(f"  Correctness pairs:           {stats.correctness_pairs:,} (correct vs incorrect)")
    print(f"  Grounding pairs:             {stats.grounding_pairs:,} (both correct, gap >= {grounding_gap_threshold})")

    if stats.grounding_pairs > 0:
        print(f"\n{'Grounding Gap Distribution (for grounding pairs)':^70}")
        print("-" * 70)
        print(f"  Mean gap:                    {stats.grounding_gap_mean:.3f}")
        print(f"  Std gap:                     {stats.grounding_gap_std:.3f}")
        print(f"  Min gap:                     {stats.grounding_gap_min:.3f}")
        print(f"  Max gap:                     {stats.grounding_gap_max:.3f}")

    print("\n" + "=" * 70)

    # Assessment
    print(f"\n{'Assessment':^70}")
    print("-" * 70)
    if stats.total_pairs < 500:
        print("  WARNING: Low pair count (<500). Consider:")
        print("    - Reducing grounding_gap_threshold")
        print("    - Using cross-model pairs (multiple model outputs)")
        print("    - Generating more data")
    elif stats.total_pairs < 1000:
        print("  MODERATE: Pair count (500-1000) may be sufficient for LoRA.")
        print("    - Consider augmentation strategies for better results")
    else:
        print("  GOOD: Sufficient pairs (>1000) for DPO training.")

    print("=" * 70 + "\n")


def format_for_training(pairs: list[DPOPair], include_context: bool = True) -> list[dict]:
    """
    Format pairs for DPO training.

    Returns list of dicts with:
    - prompt: The input context
    - chosen: The preferred response
    - rejected: The rejected response
    """
    training_data = []

    for pair in pairs:
        # Build prompt (can be customized based on your training setup)
        if include_context:
            prompt = f"""Based on the patient's clinical data, provide a diagnosis with supporting reasoning.

Ground truth diagnoses for reference: {', '.join(pair.ground_truth_diagnoses)}

Provide your diagnosis and explain your reasoning based on the available evidence."""
        else:
            prompt = "Provide a diagnosis with supporting reasoning based on the patient's clinical data."

        # Format chosen response
        chosen = f"""Diagnosis: {pair.chosen_diagnosis}

Reasoning: {pair.chosen_reasoning}"""

        # Format rejected response
        rejected = f"""Diagnosis: {pair.rejected_diagnosis}

Reasoning: {pair.rejected_reasoning}"""

        training_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'metadata': {
                'patient_id': pair.patient_id,
                'hadm_id': pair.hadm_id,
                'pair_type': pair.pair_type,
                'chosen_grounding': pair.chosen_grounding,
                'rejected_grounding': pair.rejected_grounding,
                'grounding_gap': pair.grounding_gap,
                'chosen_correct': pair.chosen_correct,
                'rejected_correct': pair.rejected_correct,
            }
        })

    return training_data


def analyze_gap_thresholds(metrics_data: dict) -> dict:
    """Analyze how many pairs we get at different gap thresholds."""
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}

    for threshold in thresholds:
        all_pairs = []
        for patient in metrics_data.get('patient_metrics', []):
            pairs = build_pairs_for_patient(
                patient,
                grounding_gap_threshold=threshold,
                require_chosen_correct=True
            )
            all_pairs.extend(pairs)

        correctness_pairs = len([p for p in all_pairs if p.pair_type == "correctness"])
        grounding_pairs = len([p for p in all_pairs if p.pair_type == "grounding"])

        results[threshold] = {
            'total': len(all_pairs),
            'correctness': correctness_pairs,
            'grounding': grounding_pairs,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build DPO training pairs from metrics output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to metrics JSON file (e.g., gemma-3-12b_metrics-v1.0.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for DPO pairs JSON"
    )
    parser.add_argument(
        "--grounding-gap-threshold",
        type=float,
        default=0.2,
        help="Minimum grounding gap for grounding-based pairs (default: 0.2)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't save pairs"
    )
    parser.add_argument(
        "--analyze-thresholds",
        action="store_true",
        help="Analyze pair counts at different gap thresholds"
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        default=True,
        help="Include patient context in training prompts"
    )

    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from: {args.metrics_file}")
    metrics_data = load_metrics(args.metrics_file)

    config = metrics_data.get('config', {})
    print(f"Model: {config.get('model_id', 'unknown')}")
    print(f"Total patients: {len(metrics_data.get('patient_metrics', []))}")

    # Analyze thresholds if requested
    if args.analyze_thresholds:
        print("\n" + "=" * 70)
        print("GAP THRESHOLD ANALYSIS")
        print("=" * 70)
        print(f"{'Threshold':<12} {'Total':<10} {'Correctness':<15} {'Grounding':<12}")
        print("-" * 70)

        threshold_results = analyze_gap_thresholds(metrics_data)
        for threshold, counts in sorted(threshold_results.items()):
            print(f"{threshold:<12.1f} {counts['total']:<10} {counts['correctness']:<15} {counts['grounding']:<12}")

        print("=" * 70)
        print("\nNote: 'Correctness' pairs compare correct vs incorrect diagnoses.")
        print("      'Grounding' pairs compare two correct diagnoses by grounding score.")
        print("      Lower threshold = more grounding pairs but weaker signal.")
        print()

    # Build pairs
    print(f"\nBuilding pairs with grounding_gap_threshold={args.grounding_gap_threshold}...")

    all_pairs = []
    for patient in metrics_data.get('patient_metrics', []):
        pairs = build_pairs_for_patient(
            patient,
            grounding_gap_threshold=args.grounding_gap_threshold,
            require_chosen_correct=True
        )
        all_pairs.extend(pairs)

    # Compute and print statistics
    stats = compute_statistics(all_pairs, metrics_data)
    print_statistics(stats, args.grounding_gap_threshold)

    # Save if not stats-only
    if not args.stats_only and args.output:
        # Format for training
        training_data = format_for_training(all_pairs, include_context=args.include_context)

        output_data = {
            'config': {
                'source_metrics_file': args.metrics_file,
                'model_id': config.get('model_id', 'unknown'),
                'grounding_gap_threshold': args.grounding_gap_threshold,
                'require_chosen_correct': True,
            },
            'statistics': asdict(stats),
            'pairs': training_data,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved {len(training_data)} pairs to: {args.output}")
    elif not args.stats_only and not args.output:
        print("No output path specified. Use --output to save pairs or --stats-only for stats only.")


if __name__ == "__main__":
    main()
