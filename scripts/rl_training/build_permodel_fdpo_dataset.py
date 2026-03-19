#!/usr/bin/env python3
"""
Build per-model Faithfulness-DPO datasets from multi-sample scored outputs.

Strategy: Within-model preference pairs (no cross-model distillation).
  For each patient, within a single model's N completions:
    chosen  = completion with highest avg grounding across 5 diagnoses
    rejected = completion with lowest avg grounding across 5 diagnoses

This ensures each model's fdpo training data comes ONLY from its own
generations, eliminating the implicit distillation present in cross-model fdpo.

Input: scored_samples_{model_id}.json (from generate_permodel_fdpo_samples.py)
Output: training_data/dpo/faithfulness_dpo_{model_id}.json (TRL DPO format)

Usage:
    # Build for one model:
    python build_permodel_fdpo_dataset.py \
        --input permodel_fdpo_samples/scored_samples_gemma-3-4b.json \
        --output training_data/dpo/faithfulness_dpo_gemma-3-4b.json

    # Build for all models:
    python build_permodel_fdpo_dataset.py --build-all

    # Analyze gap thresholds:
    python build_permodel_fdpo_dataset.py \
        --input permodel_fdpo_samples/scored_samples_gemma-3-4b.json \
        --analyze-thresholds
"""

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # evidenceRL/
SAMPLES_DIR = PROJECT_ROOT / "permodel_fdpo_samples"
OUTPUT_DIR = PROJECT_ROOT / "training_data" / "dpo"

ALL_MODELS = [
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3-27b",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "gpt-oss-20b",
]


def _build_norag_prompt(patient_context: str) -> str:
    """Reconstruct the no-RAG diagnosis prompt (identical to GRPO training prompt)."""
    return f'''You are an expert cardiology clinical assistant. Based on the patient information below, provide exactly 5 cardiac diagnoses ranked from most to least likely.

For EACH diagnosis, you MUST provide:
1. The diagnosis name (concise clinical term)
2. A reasoning paragraph following these "Clinical Synthesis" rules:
   - Pathophysiological Link: Explain how the specific symptoms (e.g., dyspnea) are directly explained by the clinical findings (e.g., the mitral regurgitation seen on ultrasound).
   - Evidence Integration: Use exact values from the Physical Exam (BP, RR) and Imaging (LVEF, PA pressures) as "anchors" for your argument.
   - Avoid generic summaries: Do not just list facts; explain the "why" behind the diagnosis using the patient's unique data.

IMPORTANT: Your response MUST be valid JSON in exactly this format:
{{
  "diagnoses": [
    {{"name": "Diagnosis 1 name", "reasoning": "Detailed reasoning for diagnosis 1..."}},
    {{"name": "Diagnosis 2 name", "reasoning": "Detailed reasoning for diagnosis 2..."}},
    {{"name": "Diagnosis 3 name", "reasoning": "Detailed reasoning for diagnosis 3..."}},
    {{"name": "Diagnosis 4 name", "reasoning": "Detailed reasoning for diagnosis 4..."}},
    {{"name": "Diagnosis 5 name", "reasoning": "Detailed reasoning for diagnosis 5..."}}
  ]
}}

Patient Information:
{patient_context}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''


@dataclass
class PerModelFDPOPair:
    hadm_id: str
    ground_truth_diagnoses: list
    prompt_text: str
    chosen_raw_output: str
    rejected_raw_output: str
    chosen_grounding: float
    rejected_grounding: float
    grounding_gap: float
    model_id: str


def build_pairs_from_samples(
    samples_data: dict,
    gap_threshold: float = 0.2,
    min_chosen_grounding: float = -0.5,
    max_rejected_grounding: float = 0.5,
) -> list[PerModelFDPOPair]:
    """Build within-model DPO pairs from scored multi-sample data."""
    model_id = samples_data['config']['model_id']
    patients = samples_data['patients']
    pairs = []

    for patient in patients:
        hadm_id = patient['hadm_id']
        patient_context = patient['patient_context']
        gt = patient['ground_truth_diagnoses']

        # Filter to successfully parsed samples
        valid_samples = [
            s for s in patient['samples']
            if s.get('parse_success', False)
        ]

        if len(valid_samples) < 2:
            continue

        # Sort by grounding_max (descending)
        valid_samples.sort(key=lambda s: s['grounding_max'], reverse=True)

        chosen = valid_samples[0]
        rejected = valid_samples[-1]

        gap = chosen['grounding_max'] - rejected['grounding_max']

        if gap < gap_threshold:
            continue
        if chosen['grounding_max'] < min_chosen_grounding:
            continue
        if rejected['grounding_max'] > max_rejected_grounding:
            continue

        prompt_text = _build_norag_prompt(patient_context)

        pairs.append(PerModelFDPOPair(
            hadm_id=hadm_id,
            ground_truth_diagnoses=gt,
            prompt_text=prompt_text,
            chosen_raw_output=chosen['raw_output'],
            rejected_raw_output=rejected['raw_output'],
            chosen_grounding=chosen['grounding_max'],
            rejected_grounding=rejected['grounding_max'],
            grounding_gap=gap,
            model_id=model_id,
        ))

    return pairs


def format_for_trl(pairs: list[PerModelFDPOPair]) -> list[dict]:
    """Format as TRL DPOTrainer chat format."""
    trl_data = []
    for pair in pairs:
        trl_data.append({
            "prompt": [{"role": "user", "content": pair.prompt_text}],
            "chosen": [{"role": "assistant", "content": pair.chosen_raw_output}],
            "rejected": [{"role": "assistant", "content": pair.rejected_raw_output}],
            "metadata": {
                "hadm_id": pair.hadm_id,
                "ground_truth_diagnoses": pair.ground_truth_diagnoses,
                "model_id": pair.model_id,
                "chosen_grounding": pair.chosen_grounding,
                "rejected_grounding": pair.rejected_grounding,
                "grounding_gap": pair.grounding_gap,
            },
        })
    return trl_data


def print_statistics(pairs: list[PerModelFDPOPair], model_id: str) -> None:
    if not pairs:
        print(f"\n  {model_id}: NO PAIRS generated!")
        return

    gaps = [p.grounding_gap for p in pairs]
    chosen_scores = [p.chosen_grounding for p in pairs]
    rejected_scores = [p.rejected_grounding for p in pairs]

    print(f"\n{'='*70}")
    print(f"PER-MODEL FDPO DATASET: {model_id}")
    print(f"{'='*70}")
    print(f"  Total pairs:        {len(pairs):,}")
    print(f"  Unique patients:    {len(set(p.hadm_id for p in pairs)):,}")
    print(f"\n  Grounding gap:")
    print(f"    Mean: {np.mean(gaps):.3f}  Std: {np.std(gaps):.3f}")
    print(f"    Min:  {np.min(gaps):.3f}  Max: {np.max(gaps):.3f}")
    print(f"\n  Chosen grounding:")
    print(f"    Mean: {np.mean(chosen_scores):.3f}  Min: {np.min(chosen_scores):.3f}")
    print(f"\n  Rejected grounding:")
    print(f"    Mean: {np.mean(rejected_scores):.3f}  Max: {np.max(rejected_scores):.3f}")
    print(f"{'='*70}")


def build_single(input_path: Path, output_path: Path, gap_threshold: float,
                 min_chosen: float, max_rejected: float) -> int:
    """Build dataset for one model. Returns pair count."""
    with open(input_path) as f:
        samples_data = json.load(f)

    model_id = samples_data['config']['model_id']
    pairs = build_pairs_from_samples(
        samples_data,
        gap_threshold=gap_threshold,
        min_chosen_grounding=min_chosen,
        max_rejected_grounding=max_rejected,
    )

    print_statistics(pairs, model_id)

    if not pairs:
        print(f"  Skipping save (no pairs).")
        return 0

    trl_data = format_for_trl(pairs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "strategy": "per-model within-model pairs (no cross-model distillation)",
            "model_id": model_id,
            "gap_threshold": gap_threshold,
            "min_chosen_grounding": min_chosen,
            "max_rejected_grounding": max_rejected,
            "source": str(input_path),
        },
        "statistics": {
            "total_pairs": len(pairs),
            "unique_patients": len(set(p.hadm_id for p in pairs)),
            "mean_grounding_gap": float(np.mean([p.grounding_gap for p in pairs])),
            "mean_chosen_grounding": float(np.mean([p.chosen_grounding for p in pairs])),
            "mean_rejected_grounding": float(np.mean([p.rejected_grounding for p in pairs])),
        },
        "pairs": trl_data,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(trl_data):,} pairs to: {output_path}")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Build per-model Faithfulness-DPO datasets",
    )
    parser.add_argument("--input", type=str, help="Input scored samples JSON")
    parser.add_argument("--output", type=str, help="Output DPO dataset JSON")
    parser.add_argument("--build-all", action="store_true",
                        help="Build datasets for all 6 models")
    parser.add_argument("--gap-threshold", type=float, default=0.2,
                        help="Min grounding gap between chosen and rejected (default: 0.2)")
    parser.add_argument("--min-chosen-grounding", type=float, default=-0.5,
                        help="Min avg grounding for chosen (default: -0.5)")
    parser.add_argument("--max-rejected-grounding", type=float, default=0.5,
                        help="Max avg grounding for rejected (default: 0.5)")
    parser.add_argument("--analyze-thresholds", action="store_true",
                        help="Show pair counts across gap thresholds")
    parser.add_argument("--samples-dir", type=str, default=str(SAMPLES_DIR),
                        help="Directory with scored sample files")
    args = parser.parse_args()

    if args.analyze_thresholds:
        if not args.input:
            print("ERROR: --analyze-thresholds requires --input")
            return

        with open(args.input) as f:
            samples_data = json.load(f)

        model_id = samples_data['config']['model_id']
        print(f"\nGAP THRESHOLD ANALYSIS for {model_id}")
        print(f"{'Gap':>6}  {'Pairs':>8}  {'Patients':>10}  {'MeanGap':>10}  {'MeanChosen':>12}")
        print("-" * 55)
        for gap in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            pairs = build_pairs_from_samples(
                samples_data,
                gap_threshold=gap,
                min_chosen_grounding=-1.0,
                max_rejected_grounding=1.0,
            )
            if pairs:
                n_patients = len(set(p.hadm_id for p in pairs))
                mean_gap = np.mean([p.grounding_gap for p in pairs])
                mean_chosen = np.mean([p.chosen_grounding for p in pairs])
                print(f"  {gap:.2f}  {len(pairs):>8,}  {n_patients:>10,}  {mean_gap:>10.3f}  {mean_chosen:>12.3f}")
            else:
                print(f"  {gap:.2f}         0           0")
        return

    if args.build_all:
        samples_dir = Path(args.samples_dir)
        print(f"\n{'='*70}")
        print(f"BUILDING PER-MODEL FDPO DATASETS")
        print(f"  Samples dir: {samples_dir}")
        print(f"  Output dir:  {OUTPUT_DIR}")
        print(f"  Gap threshold: {args.gap_threshold}")
        print(f"{'='*70}")

        total_pairs = 0
        for model_id in ALL_MODELS:
            input_path = samples_dir / f"scored_samples_{model_id}.json"
            if not input_path.exists():
                # Try worker-merged file
                input_path = samples_dir / f"{model_id}_merged.json"
            if not input_path.exists():
                print(f"\n  {model_id}: SKIPPED (no input file)")
                continue

            output_path = OUTPUT_DIR / f"faithfulness_dpo_{model_id}.json"
            n = build_single(
                input_path, output_path,
                args.gap_threshold,
                args.min_chosen_grounding,
                args.max_rejected_grounding,
            )
            total_pairs += n

        print(f"\n{'='*70}")
        print(f"TOTAL: {total_pairs:,} pairs across all models")
        print(f"{'='*70}")
        return

    # Single model
    if not args.input:
        parser.error("--input is required (or use --build-all)")
    if not args.output:
        with open(args.input) as f:
            d = json.load(f)
        model_id = d['config']['model_id']
        args.output = str(OUTPUT_DIR / f"faithfulness_dpo_{model_id}.json")

    build_single(
        Path(args.input), Path(args.output),
        args.gap_threshold,
        args.min_chosen_grounding,
        args.max_rejected_grounding,
    )


if __name__ == "__main__":
    main()
