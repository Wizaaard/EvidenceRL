#!/usr/bin/env python3
"""
Build Faithfulness-DPO training dataset from existing generation + metrics outputs.

Strategy: Cross-model preference pairs.
  - For each patient, all 8 model backbones have generated outputs scored by NLI grounding.
  - chosen  = full RAW output from the model with highest avg grounding across 5 diagnoses
  - rejected = full RAW output from the model with lowest avg grounding across 5 diagnoses
  - prompt  = no-RAG chat prompt (patient_context only — identical to GRPO training distribution)

This is a PURE faithfulness signal: no correctness requirement on chosen.
The model learns to generate evidence-grounded reasoning regardless of accuracy.

Data sources (per model):
  generation_baseline_output/{model}_v1.0-TRAIN/{model}_3700-v1.0-TRAIN.json
    → patient_context, structured_output.raw_output

  train_metrics_output/no-rag/{model}_v1.0-TRAIN_llama70/enhanced_metrics.json
    → per_diagnosis_rewards[i].grounding_max  (the NLI score per diagnosis)

Usage:
    python build_faithfulness_dpo_dataset.py --output training_data/dpo/faithfulness_dpo.json
    python build_faithfulness_dpo_dataset.py --stats-only
    python build_faithfulness_dpo_dataset.py --output ... --top-k 2 --bottom-k 2
"""

import argparse
import json
import re
from pathlib import Path
from itertools import product
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # evidenceRL/

GEN_NORAG_DIR = PROJECT_ROOT / "generation_baseline_output"
METRICS_NORAG_DIR = PROJECT_ROOT / "train_metrics_output" / "no-rag"

# All 8 model backbone identifiers (must match directory name prefixes)
ALL_MODELS = [
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "Llama-3.3-70B",
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3-27b",
    "gpt-oss-20b",
    "gpt-oss-120b",
]


# ── Prompt template (identical to build_grpo_dataset.py) ─────────────────────

def _clean_evidence_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&quot;', '"').replace('&#xb7;', '\u00b7')
    return text.strip()


def _build_norag_prompt(patient_context: str) -> str:
    """Reconstruct the no-RAG diagnosis prompt (identical to GRPO training prompt).

    Matches _build_norag_prompt() in build_grpo_dataset.py exactly so that
    Faithfulness-DPO trains on the same input distribution as EvidenceRL GRPO.
    """
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


# ── Data loading ──────────────────────────────────────────────────────────────

def _find_gen_file(model_id: str) -> Optional[Path]:
    """Find the TRAIN no-RAG generation JSON for this model."""
    model_dir = GEN_NORAG_DIR / f"{model_id}_v1.0-TRAIN"
    if not model_dir.exists():
        return None
    candidates = [f for f in model_dir.glob("*.json") if f.parent == model_dir]
    return candidates[0] if candidates else None


def _find_metrics_file(model_id: str) -> Optional[Path]:
    """Find the TRAIN no-RAG metrics JSON for this model."""
    # Pattern: {model_id}_v1.0-TRAIN_llama70
    for d in METRICS_NORAG_DIR.iterdir():
        if d.is_dir() and d.name.startswith(model_id):
            f = d / "enhanced_metrics.json"
            if f.exists():
                return f
    return None


def load_generation_data(model_id: str) -> dict[str, dict]:
    """Load generation output: {hadm_id -> {patient_context, pre_evidence, raw_output, parse_success}}"""
    gen_file = _find_gen_file(model_id)
    if gen_file is None:
        print(f"  [WARNING] No generation file found for {model_id}")
        return {}

    with open(gen_file) as f:
        data = json.load(f)

    result = {}
    for r in data.get('results', []):
        hadm_id = str(r['hadm_id'])
        so = r.get('structured_output', {})
        raw_output = so.get('raw_output', '') if isinstance(so, dict) else ''
        parse_success = so.get('parse_success', False) if isinstance(so, dict) else False

        if not raw_output or not parse_success:
            continue  # skip failed parses — can't use as DPO target

        result[hadm_id] = {
            'patient_context': r.get('patient_context', ''),
            'pre_evidence': r.get('pre_evidence', []),
            'ground_truth_diagnoses': r.get('ground_truth_diagnoses', []),
            'raw_output': raw_output,
        }

    print(f"  {model_id}: {len(result)} valid generation outputs loaded")
    return result


def load_metrics_data(model_id: str) -> dict[str, float]:
    """Load metrics: {hadm_id -> avg_grounding_across_5_diagnoses}"""
    metrics_file = _find_metrics_file(model_id)
    if metrics_file is None:
        print(f"  [WARNING] No metrics file found for {model_id}")
        return {}

    with open(metrics_file) as f:
        data = json.load(f)

    result = {}
    for p in data.get('patient_metrics', []):
        hadm_id = str(p['hadm_id'])
        rewards = p.get('per_diagnosis_rewards', [])
        if not rewards:
            continue
        # Average grounding_max across all 5 diagnoses (range [-1, 1])
        scores = [r['grounding_max'] for r in rewards if 'grounding_max' in r]
        if scores:
            result[hadm_id] = float(np.mean(scores))

    print(f"  {model_id}: {len(result)} patients with grounding scores loaded")
    return result


# ── Pair construction ─────────────────────────────────────────────────────────

@dataclass
class FaithfulnessDPOPair:
    hadm_id: str
    ground_truth_diagnoses: list
    prompt_text: str           # raw RAG prompt string
    chosen_raw_output: str     # full JSON output string from chosen model
    rejected_raw_output: str   # full JSON output string from rejected model
    chosen_model: str
    rejected_model: str
    chosen_grounding: float
    rejected_grounding: float
    grounding_gap: float


def build_pairs_for_patient(
    hadm_id: str,
    model_scores: dict[str, tuple[float, dict]],  # {model_id -> (avg_grounding, gen_data)}
    gap_threshold: float,
    top_k: int,
    bottom_k: int,
    min_chosen_grounding: float,
    max_rejected_grounding: float,
) -> list[FaithfulnessDPOPair]:
    """Build preference pairs for one patient from all available model outputs."""
    if len(model_scores) < 2:
        return []

    # Sort models by grounding score (descending)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1][0], reverse=True)

    chosen_candidates = sorted_models[:top_k]
    rejected_candidates = sorted_models[-bottom_k:]

    # Avoid overlap (a model can't be both chosen and rejected)
    chosen_ids = {m for m, _ in chosen_candidates}
    rejected_candidates = [(m, v) for m, v in rejected_candidates if m not in chosen_ids]

    if not rejected_candidates:
        return []

    pairs = []
    for (c_model, (c_score, c_gen)), (r_model, (r_score, r_gen)) in product(chosen_candidates, rejected_candidates):
        gap = c_score - r_score
        if gap < gap_threshold:
            continue
        if c_score < min_chosen_grounding:
            continue
        if r_score > max_rejected_grounding:
            continue

        patient_context = c_gen['patient_context']
        gt = c_gen['ground_truth_diagnoses']

        prompt_text = _build_norag_prompt(patient_context)

        pairs.append(FaithfulnessDPOPair(
            hadm_id=hadm_id,
            ground_truth_diagnoses=gt,
            prompt_text=prompt_text,
            chosen_raw_output=c_gen['raw_output'],
            rejected_raw_output=r_gen['raw_output'],
            chosen_model=c_model,
            rejected_model=r_model,
            chosen_grounding=c_score,
            rejected_grounding=r_score,
            grounding_gap=gap,
        ))

    return pairs


def build_dataset(
    gap_threshold: float = 0.4,
    top_k: int = 1,
    bottom_k: int = 1,
    min_chosen_grounding: float = 0.1,
    max_rejected_grounding: float = -0.1,
    models: Optional[list[str]] = None,
) -> list[FaithfulnessDPOPair]:
    """Build the full Faithfulness-DPO dataset."""
    if models is None:
        models = ALL_MODELS

    print("\nLoading generation outputs...")
    gen_data: dict[str, dict[str, dict]] = {}   # {model_id -> {hadm_id -> gen_entry}}
    for model_id in models:
        gen_data[model_id] = load_generation_data(model_id)

    print("\nLoading grounding scores...")
    metrics_data: dict[str, dict[str, float]] = {}  # {model_id -> {hadm_id -> avg_grounding}}
    for model_id in models:
        metrics_data[model_id] = load_metrics_data(model_id)

    # Collect all hadm_ids that appear in at least 2 models
    all_hadm_ids: set[str] = set()
    for model_id in models:
        if gen_data[model_id] and metrics_data[model_id]:
            common = set(gen_data[model_id].keys()) & set(metrics_data[model_id].keys())
            all_hadm_ids.update(common)

    print(f"\nTotal unique patients with at least one valid output: {len(all_hadm_ids)}")

    # Build per-patient model scores
    all_pairs: list[FaithfulnessDPOPair] = []
    patients_with_pairs = 0

    for hadm_id in sorted(all_hadm_ids):
        # Collect all models that have BOTH generation + metrics for this patient
        model_scores: dict[str, tuple[float, dict]] = {}
        for model_id in models:
            gen = gen_data[model_id].get(hadm_id)
            score = metrics_data[model_id].get(hadm_id)
            if gen is not None and score is not None:
                model_scores[model_id] = (score, gen)

        if len(model_scores) < 2:
            continue

        pairs = build_pairs_for_patient(
            hadm_id=hadm_id,
            model_scores=model_scores,
            gap_threshold=gap_threshold,
            top_k=top_k,
            bottom_k=bottom_k,
            min_chosen_grounding=min_chosen_grounding,
            max_rejected_grounding=max_rejected_grounding,
        )

        if pairs:
            patients_with_pairs += 1
            all_pairs.extend(pairs)

    print(f"Patients with ≥1 valid pair: {patients_with_pairs} / {len(all_hadm_ids)}")
    print(f"Total pairs generated:       {len(all_pairs)}")

    return all_pairs


# ── Output formatting ─────────────────────────────────────────────────────────

def format_for_trl(pairs: list[FaithfulnessDPOPair]) -> list[dict]:
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
                "chosen_model": pair.chosen_model,
                "rejected_model": pair.rejected_model,
                "chosen_grounding": pair.chosen_grounding,
                "rejected_grounding": pair.rejected_grounding,
                "grounding_gap": pair.grounding_gap,
            },
        })
    return trl_data


def print_statistics(pairs: list[FaithfulnessDPOPair], config: dict) -> None:
    gaps = [p.grounding_gap for p in pairs]
    chosen_scores = [p.chosen_grounding for p in pairs]
    rejected_scores = [p.rejected_grounding for p in pairs]
    chosen_models = {}
    rejected_models = {}
    for p in pairs:
        chosen_models[p.chosen_model] = chosen_models.get(p.chosen_model, 0) + 1
        rejected_models[p.rejected_model] = rejected_models.get(p.rejected_model, 0) + 1

    print("\n" + "=" * 70)
    print("FAITHFULNESS-DPO DATASET STATISTICS")
    print("=" * 70)
    print(f"  Total pairs:              {len(pairs):,}")
    print(f"  Unique patients:          {len(set(p.hadm_id for p in pairs)):,}")
    print(f"\n  Grounding gap:")
    print(f"    Mean:  {np.mean(gaps):.3f}")
    print(f"    Std:   {np.std(gaps):.3f}")
    print(f"    Min:   {np.min(gaps):.3f}")
    print(f"    Max:   {np.max(gaps):.3f}")
    print(f"\n  Chosen grounding (avg_across_5_diags):")
    print(f"    Mean:  {np.mean(chosen_scores):.3f}   (range [-1, 1])")
    print(f"    Min:   {np.min(chosen_scores):.3f}")
    print(f"\n  Rejected grounding (avg_across_5_diags):")
    print(f"    Mean:  {np.mean(rejected_scores):.3f}")
    print(f"    Max:   {np.max(rejected_scores):.3f}")
    print(f"\n  Top chosen models: {sorted(chosen_models.items(), key=lambda x: -x[1])[:4]}")
    print(f"  Top rejected models: {sorted(rejected_models.items(), key=lambda x: -x[1])[:4]}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Faithfulness-DPO dataset from RAG generation + metrics outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", type=str,
                        help="Output path for DPO dataset JSON")
    parser.add_argument("--gap-threshold", type=float, default=0.4,
                        help="Min grounding gap between chosen and rejected (default: 0.4 on [-1,1] scale)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Top-k highest-grounding outputs per patient used as chosen candidates (default: 1)")
    parser.add_argument("--bottom-k", type=int, default=1,
                        help="Bottom-k lowest-grounding outputs per patient used as rejected candidates (default: 1)")
    parser.add_argument("--min-chosen-grounding", type=float, default=0.1,
                        help="Minimum avg grounding score for chosen output (default: 0.1)")
    parser.add_argument("--max-rejected-grounding", type=float, default=-0.1,
                        help="Maximum avg grounding score for rejected output (default: -0.1)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model IDs to use (default: all 8)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics without saving")
    parser.add_argument("--analyze-thresholds", action="store_true",
                        help="Show pair counts across gap thresholds 0.1..0.7")

    args = parser.parse_args()

    if args.analyze_thresholds:
        print("\nGAP THRESHOLD ANALYSIS")
        print(f"{'Gap':>6}  {'Pairs':>8}  {'Patients':>10}")
        print("-" * 30)
        for gap in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pairs = build_dataset(
                gap_threshold=gap,
                top_k=args.top_k,
                bottom_k=args.bottom_k,
                min_chosen_grounding=-1.0,   # no filter for analysis
                max_rejected_grounding=1.0,
                models=args.models,
            )
            n_patients = len(set(p.hadm_id for p in pairs))
            print(f"  {gap:.1f}  {len(pairs):>8,}  {n_patients:>10,}")
        return

    config = dict(
        gap_threshold=args.gap_threshold,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        min_chosen_grounding=args.min_chosen_grounding,
        max_rejected_grounding=args.max_rejected_grounding,
        models=args.models or ALL_MODELS,
    )

    print("\n" + "=" * 70)
    print("FAITHFULNESS-DPO DATASET CONSTRUCTION")
    print("=" * 70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    pairs = build_dataset(**config)
    print_statistics(pairs, config)

    if args.stats_only or not args.output:
        if not args.output:
            print("\nNo --output specified. Use --output to save. Exiting.")
        return

    trl_data = format_for_trl(pairs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": config,
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

    print(f"\nSaved {len(trl_data):,} pairs to: {output_path}")


if __name__ == "__main__":
    main()
