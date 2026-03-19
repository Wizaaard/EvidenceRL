#!/usr/bin/env python3
"""Score NLI grounding for multi-sample generation outputs.

This is step 2 of the per-model fdpo pipeline:
  1. generate_multisample  → merged JSON with N completions per patient
  2. score_grounding       → scored_samples_{model_id}.json (this script)
  3. build_permodel_fdpo   → faithfulness_dpo_{model_id}.json

For each patient, scores all N completions' grounding against patient context
using CrossEncoderNLI (sentence-level). No RAG evidence docs — grounding is
measured against patient context sections only.

Usage:
    python score_multisample_grounding.py \
        --input permodel_fdpo_samples/gemma-3-4b/gemma-3-4b_multisample_3x_3700patients_v1.0.json \
        --output permodel_fdpo_samples/scored_samples_gemma-3-4b.json

    # Or score all models:
    python score_multisample_grounding.py --score-all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # evidenceRL/code/
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EVIDENCERL_ROOT = PROJECT_ROOT.parent  # evidenceRL/
SAMPLES_DIR = EVIDENCERL_ROOT / "permodel_fdpo_samples"

ALL_MODELS = [
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3-27b",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "gpt-oss-20b",
]


def load_nli_model(nli_model_name: str = "pritamdeka/PubMedBERT-MNLI-MedNLI",
                   force_cpu: bool = False):
    """Load CrossEncoderNLI model (same default as existing metrics pipeline)."""
    from evidence_rl.evidence_pipeline import CrossEncoderNLI
    nli_model = CrossEncoderNLI(
        model_name=nli_model_name,
        force_cpu=force_cpu,
    )
    return nli_model


def score_samples(
    input_path: Path,
    output_path: Path,
    nli_model,
    sentence_level: bool = False,
) -> int:
    """Score all multi-sample completions for grounding.

    Returns the number of patients scored.
    """
    from evidence_rl.reward import reward_grounding

    with open(input_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    model_name = config.get("model_name", "unknown")
    num_samples = config.get("num_samples", 1)
    results = data.get("results", [])

    # Extract model_id from model_name path
    model_id = Path(model_name).name if "/" in model_name else model_name

    print(f"\n{'='*70}")
    print(f"SCORING GROUNDING: {model_id}")
    print(f"{'='*70}")
    print(f"  Input:        {input_path}")
    print(f"  Patients:     {len(results)}")
    print(f"  Num samples:  {num_samples}")
    print(f"  Sentence-level: {sentence_level}")

    scored_patients = []

    for patient in tqdm(results, desc=f"Scoring {model_id}"):
        hadm_id = patient.get("hadm_id", "")
        patient_context = patient.get("patient_context", "")
        gt_diagnoses = patient.get("ground_truth_diagnoses", [])
        structured_output = patient.get("structured_output", {})

        if not structured_output:
            continue

        # Get all samples (multi-sample mode stores them in all_samples)
        all_samples = structured_output.get("all_samples", None)
        if all_samples is None:
            # Single-sample fallback: treat the main output as the only sample
            all_samples = [structured_output]

        scored_samples = []
        for sample in all_samples:
            diagnoses = sample.get("diagnoses", [])
            raw_output = sample.get("raw_output", "")
            parse_success = sample.get("parse_success", False)

            if not parse_success or not diagnoses:
                scored_samples.append({
                    "raw_output": raw_output,
                    "parse_success": False,
                    "diagnoses": diagnoses,
                    "grounding_max": 0.0,
                    "grounding_avg": 0.0,
                    "per_diagnosis_grounding": [],
                })
                continue

            # Score each diagnosis's reasoning
            per_diag_grounding = []
            for diag in diagnoses:
                reasoning = diag.get("reasoning", "")
                if not reasoning.strip():
                    per_diag_grounding.append({"max": 0.0, "avg": 0.0})
                    continue

                g_max, g_avg = reward_grounding(
                    diagnosis_reasoning=reasoning,
                    patient_context=patient_context,
                    evidence_docs=[],  # no-RAG: no evidence docs
                    nli_model=nli_model,
                    sentence_level=sentence_level,
                )
                per_diag_grounding.append({"max": float(g_max), "avg": float(g_avg)})

            # Aggregate: mean grounding across all 5 diagnoses
            valid_maxes = [g["max"] for g in per_diag_grounding]
            valid_avgs = [g["avg"] for g in per_diag_grounding]
            agg_max = float(np.mean(valid_maxes)) if valid_maxes else 0.0
            agg_avg = float(np.mean(valid_avgs)) if valid_avgs else 0.0

            scored_samples.append({
                "raw_output": raw_output,
                "parse_success": True,
                "diagnoses": diagnoses,
                "grounding_max": agg_max,
                "grounding_avg": agg_avg,
                "per_diagnosis_grounding": per_diag_grounding,
            })

        scored_patients.append({
            "hadm_id": hadm_id,
            "patient_context": patient_context,
            "ground_truth_diagnoses": gt_diagnoses,
            "samples": scored_samples,
        })

    # Save scored output
    output_data = {
        "config": {
            "model_id": model_id,
            "model_name": model_name,
            "num_samples": num_samples,
            "sentence_level": sentence_level,
            "source_file": str(input_path),
        },
        "patients": scored_patients,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print statistics
    all_maxes = []
    for p in scored_patients:
        for s in p["samples"]:
            if s["parse_success"]:
                all_maxes.append(s["grounding_max"])

    if all_maxes:
        print(f"\n  Scored {len(scored_patients)} patients, {len(all_maxes)} valid samples")
        print(f"  Grounding max:  mean={np.mean(all_maxes):.3f}  std={np.std(all_maxes):.3f}")
        print(f"  Grounding max:  min={np.min(all_maxes):.3f}   max={np.max(all_maxes):.3f}")
    else:
        print(f"\n  WARNING: No valid samples scored!")

    print(f"  Saved to: {output_path}")
    return len(scored_patients)


def find_merged_file(model_id: str) -> Path | None:
    """Find the merged multi-sample generation file for a model."""
    model_dir = SAMPLES_DIR / model_id
    if not model_dir.exists():
        return None

    # Look for merged files
    candidates = list(model_dir.glob(f"{model_id}_multisample_*.json"))
    if not candidates:
        return None

    # Return most recent
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Score NLI grounding for multi-sample outputs",
    )
    parser.add_argument("--input", type=str, help="Input merged multi-sample JSON")
    parser.add_argument("--output", type=str, help="Output scored samples JSON")
    parser.add_argument("--score-all", action="store_true",
                        help="Score all 6 models")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force NLI model to CPU (slower but avoids GPU conflicts)")
    parser.add_argument("--nli-model", type=str,
                        default="pritamdeka/PubMedBERT-MNLI-MedNLI",
                        help="NLI model (default: same as existing metrics pipeline)")
    parser.add_argument("--sentence-level", action="store_true", default=False,
                        help="Use sentence-level grounding (default: False = full_reasoning)")
    parser.add_argument("--no-sentence-level", action="store_false",
                        dest="sentence_level",
                        help="Use full-reasoning grounding (default)")
    args = parser.parse_args()

    # Load NLI model (same as existing metrics pipeline)
    print(f"Loading NLI model: {args.nli_model}")
    nli_model = load_nli_model(nli_model_name=args.nli_model,
                               force_cpu=args.force_cpu)

    if args.score_all:
        total = 0
        for model_id in ALL_MODELS:
            input_path = find_merged_file(model_id)
            if input_path is None:
                print(f"\n  {model_id}: SKIPPED (no merged file found)")
                continue

            output_path = SAMPLES_DIR / f"scored_samples_{model_id}.json"
            n = score_samples(input_path, output_path, nli_model,
                              sentence_level=args.sentence_level)
            total += n

        print(f"\n{'='*70}")
        print(f"TOTAL: Scored {total} patients across all models")
        print(f"{'='*70}")
        return

    # Single model
    if not args.input:
        parser.error("--input is required (or use --score-all)")

    input_path = Path(args.input)
    if not args.output:
        # Auto-generate output path
        with open(input_path) as f:
            d = json.load(f)
        model_name = d.get("config", {}).get("model_name", "unknown")
        model_id = Path(model_name).name if "/" in model_name else model_name
        args.output = str(SAMPLES_DIR / f"scored_samples_{model_id}.json")

    score_samples(input_path, Path(args.output), nli_model,
                  sentence_level=args.sentence_level)


if __name__ == "__main__":
    main()
