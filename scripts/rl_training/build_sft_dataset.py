#!/usr/bin/env python3
"""
Build SFT (Supervised Fine-Tuning) dataset from Evidence-Based model outputs.

This script constructs training data for SFT by selecting high-quality outputs
where models produce predominantly Evidence-Based differential diagnoses.

Filter: >= N Evidence-Based diagnoses in top-3 (default N=2)
  Evidence-Based := Correct (LLM judge verdict=True) AND Grounded (r_g^max > 0.5)

Deduplication: Cap at K outputs per patient (default K=2), selecting by:
  1. Highest EB count in top-3 (descending)
  2. Highest mean r_g^max across top-3 (tie-break)

Produces separate datasets for RAG and no-RAG modes.

Usage:
    # Build both RAG and no-RAG datasets
    python build_sft_dataset.py --output-dir sft_data/

    # Stats only (no output)
    python build_sft_dataset.py --stats-only

    # Custom thresholds
    python build_sft_dataset.py --min-eb 3 --max-per-patient 1 --output-dir sft_data/
"""

import argparse
import json
import re
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # evidenceRL/

GENERATION_RAG_DIR = PROJECT_ROOT / "generation_output"
GENERATION_NORAG_DIR = PROJECT_ROOT / "generation_baseline_output"
METRICS_RAG_DIR = PROJECT_ROOT / "train_metrics_output" / "rag"
METRICS_NORAG_DIR = PROJECT_ROOT / "train_metrics_output" / "no-rag"

# Model configurations: (generation_dir_suffix, metrics_dir_suffix, display_name)
MODELS = [
    ("gemma-3-4b_v1.0-TRAIN",   "gemma-3-4b_v1.0-TRAIN_llama70",   "gemma-3-4b"),
    ("gemma-3-12b_v1.0-TRAIN",  "gemma-3-12b_v1.0-TRAIN_llama70",  "gemma-3-12b"),
    ("gemma-3-27b_v1.0-TRAIN",  "gemma-3-27b_v1.0-TRAIN_llama70",  "gemma-3-27b"),
    ("Llama-3.2-3B_v1.0-TRAIN", "Llama-3.2-3B_v1.0-TRAIN_llama70", "Llama-3.2-3B"),
    ("Llama-3.1-8B_v1.0-TRAIN", "Llama-3.1-8B_v1.0-TRAIN_llama70", "Llama-3.1-8B"),
    ("Llama-3.3-70B_v1.0-TRAIN","Llama-3.3-70B_v1.0-TRAIN_llama70","Llama-3.3-70B"),
    ("gpt-oss-20b_v1.0-TRAIN",  "gpt-oss-20b_v1.0-TRAIN_llama70",  "gpt-oss-20b"),
    ("gpt-oss-120b_v1.0-TRAIN", "gpt-oss-120b_v1.0-TRAIN_llama70", "gpt-oss-120b"),
]

# Grounding threshold for Evidence-Based classification
GROUNDING_THRESHOLD = 0.5


@dataclass
class SFTExample:
    """A single SFT training example (full 5-diagnosis output)."""
    hadm_id: str
    subject_id: str
    model_name: str
    mode: str  # "rag" or "no-rag"

    # Quality metrics (for selection/ranking)
    eb_count_top3: int          # Number of EB diagnoses in top-3
    mean_rg_max_top3: float     # Mean r_g^max across top-3
    per_diagnosis_eb: list      # [bool, bool, bool, bool, bool]

    # Training data
    patient_context: str
    pre_evidence: list          # Retrieved docs (empty for no-RAG)
    structured_diagnoses: list  # Parsed [{name, reasoning}, ...]
    ground_truth_diagnoses: list


@dataclass
class DatasetStats:
    """Statistics for an SFT dataset."""
    mode: str
    total_patient_model_pairs: int
    qualifying_pairs: int
    unique_patients_qualifying: int
    after_dedup: int
    unique_patients_after_dedup: int

    # Per-model breakdown
    per_model: dict = field(default_factory=dict)

    # Quality distribution
    eb3_count: int = 0  # pairs with 3/3 EB in top-3
    eb2_count: int = 0  # pairs with 2/3 EB in top-3


def _clean_evidence_text(text: str) -> str:
    """Clean XML/HTML tags from evidence text (mirrors evidence_pipeline.py)."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&amp;', '&').replace('&quot;', '"')
    text = text.replace('&#xb7;', '\u00b7')
    return text.strip()


def _build_rag_prompt(patient_context: str, pre_evidence: list) -> str:
    """Reconstruct the RAG prompt from saved generation data."""
    if pre_evidence:
        evidence_lines = []
        for idx, item in enumerate(pre_evidence, start=1):
            cleaned_text = _clean_evidence_text(item['text'])
            evidence_lines.append(f"Evidence {idx}:\n{cleaned_text}")
        evidence_text = "\n\n".join(evidence_lines)
    else:
        evidence_text = "No external clinical evidence available."

    prompt = f'''You are an expert cardiology clinical assistant. Based on the patient information and retrieved clinical evidence, provide exactly 5 cardiac diagnoses ranked from most to least likely.

For EACH diagnosis, you MUST provide:
1. The diagnosis name (concise clinical term)
2. A reasoning paragraph following these "Clinical Synthesis" rules:
   - Pathophysiological Link: Explain how the specific symptoms (e.g., dyspnea) are directly explained by the clinical findings (e.g., the mitral regurgitation seen on ultrasound).
   - Evidence Integration: Use exact values from the Physical Exam (BP, RR) and Imaging (LVEF, PA pressures) as "anchors" for your argument.
   - Guideline Alignment: Explicitly state which criteria from the Retrieved Clinical Evidence are met by this specific patient's data.
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

Retrieved Clinical Evidence:
{evidence_text}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''
    return prompt


def _build_norag_prompt(patient_context: str) -> str:
    """Reconstruct the no-RAG prompt from saved generation data."""
    prompt = f'''You are an expert cardiology clinical assistant. Based on the patient information below, provide exactly 5 cardiac diagnoses ranked from most to least likely.

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
    return prompt



def load_generation_data(gen_dir: Path, model_gen_suffix: str, is_baseline: bool) -> Optional[dict]:
    """Load generation JSON for a model. Returns {hadm_id -> result} mapping."""
    model_dir = gen_dir / model_gen_suffix
    if not model_dir.exists():
        return None

    # Find the JSON file (pattern: {model_name}[_baseline]_{N}-v1.0-TRAIN.json)
    json_files = list(model_dir.glob("*.json"))
    # Filter to the main results file (not in subdirs)
    json_files = [f for f in json_files if f.parent == model_dir]
    if not json_files:
        return None

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    # Index by hadm_id for fast lookup
    results_by_hadm = {}
    for result in data.get('results', []):
        hadm_id = str(result['hadm_id'])
        results_by_hadm[hadm_id] = result

    return results_by_hadm


def load_metrics_data(metrics_dir: Path, model_metrics_suffix: str) -> Optional[dict]:
    """Load enhanced metrics JSON. Returns {hadm_id -> patient_metrics} mapping."""
    metrics_file = metrics_dir / model_metrics_suffix / "enhanced_metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    metrics_by_hadm = {}
    for patient in data.get('patient_metrics', []):
        hadm_id = str(patient['hadm_id'])
        metrics_by_hadm[hadm_id] = patient

    return metrics_by_hadm


def classify_diagnoses(patient_metrics: dict, grounding_threshold: float = GROUNDING_THRESHOLD) -> list[dict]:
    """Classify each diagnosis as Evidence-Based or not.

    Returns list of dicts with: name, correct, grounding_max, is_eb
    """
    verdicts = patient_metrics.get('verdicts', [])
    per_diag = patient_metrics.get('per_diagnosis_rewards', [])

    classified = []
    for i, diag_reward in enumerate(per_diag):
        correct = verdicts[i] if i < len(verdicts) else False
        grounding_max = diag_reward.get('grounding_max', 0.0)
        is_eb = correct and grounding_max > grounding_threshold

        classified.append({
            'name': diag_reward['diagnosis_name'],
            'correct': correct,
            'grounding_max': grounding_max,
            'is_eb': is_eb,
        })

    return classified


def build_sft_examples(
    mode: str,
    gen_dir: Path,
    metrics_dir: Path,
    min_eb: int = 2,
    grounding_threshold: float = GROUNDING_THRESHOLD,
) -> list[SFTExample]:
    """Build all qualifying SFT examples for a given mode (rag/no-rag)."""
    is_baseline = (mode == "no-rag")
    examples = []

    for model_gen_suffix, model_metrics_suffix, model_name in MODELS:
        # Load data
        gen_data = load_generation_data(gen_dir, model_gen_suffix, is_baseline)
        if gen_data is None:
            print(f"  WARNING: No generation data for {model_name} ({mode}), skipping")
            continue

        metrics_data = load_metrics_data(metrics_dir, model_metrics_suffix)
        if metrics_data is None:
            print(f"  WARNING: No metrics data for {model_name} ({mode}), skipping")
            continue

        # Match generation + metrics by hadm_id
        common_hadm_ids = set(gen_data.keys()) & set(metrics_data.keys())

        model_count = 0
        for hadm_id in common_hadm_ids:
            gen_result = gen_data[hadm_id]
            patient_metrics = metrics_data[hadm_id]

            # Skip if parse failed
            structured_output = gen_result.get('structured_output', {})
            if not structured_output.get('parse_success', False):
                continue

            diagnoses = structured_output.get('diagnoses', [])
            if len(diagnoses) != 5:
                continue

            # Ensure every diagnosis has non-empty name and reasoning
            if not all(d.get('name', '').strip() and d.get('reasoning', '').strip() for d in diagnoses):
                continue

            # Classify top-3 diagnoses
            classified = classify_diagnoses(patient_metrics, grounding_threshold)
            if len(classified) < 3:
                continue

            top3_classified = classified[:3]
            eb_flags = [d['is_eb'] for d in top3_classified]
            eb_count = sum(eb_flags)

            # Apply filter: >= min_eb Evidence-Based in top-3
            if eb_count < min_eb:
                continue

            # Compute mean r_g^max across top-3 (for ranking)
            mean_rg_max = sum(d['grounding_max'] for d in top3_classified) / 3

            # All 5 diagnoses EB flags
            all_eb = [d['is_eb'] for d in classified]

            example = SFTExample(
                hadm_id=hadm_id,
                subject_id=str(gen_result.get('subject_id', '')),
                model_name=model_name,
                mode=mode,
                eb_count_top3=eb_count,
                mean_rg_max_top3=mean_rg_max,
                per_diagnosis_eb=all_eb[:5],
                patient_context=gen_result.get('patient_context', ''),
                pre_evidence=gen_result.get('pre_evidence', []),
                structured_diagnoses=diagnoses,
                ground_truth_diagnoses=gen_result.get('ground_truth_diagnoses', []),
            )
            examples.append(example)
            model_count += 1

        print(f"  {model_name}: {model_count} qualifying examples")

    return examples


def deduplicate(
    examples: list[SFTExample],
    max_per_patient: int = 2,
) -> list[SFTExample]:
    """Cap at max_per_patient examples per patient (hadm_id).

    Selection criteria (descending priority):
    1. Highest EB count in top-3
    2. Highest mean r_g^max across top-3
    """
    # Group by hadm_id
    by_patient: dict[str, list[SFTExample]] = {}
    for ex in examples:
        by_patient.setdefault(ex.hadm_id, []).append(ex)

    deduped = []
    for hadm_id, patient_examples in by_patient.items():
        # Sort by (eb_count desc, mean_rg_max desc)
        patient_examples.sort(
            key=lambda x: (x.eb_count_top3, x.mean_rg_max_top3),
            reverse=True,
        )
        deduped.extend(patient_examples[:max_per_patient])

    return deduped


def format_for_training(examples: list[SFTExample]) -> list[dict]:
    """Format examples as chat-style training data.

    Each example becomes:
    {
        "messages": [
            {"role": "user", "content": <prompt>},
            {"role": "assistant", "content": <cleaned_model_output>}
        ],
        "metadata": { ... }
    }
    """
    training_data = []

    for ex in examples:
        # Build prompt (matching the original inference prompt)
        if ex.mode == "rag":
            prompt = _build_rag_prompt(ex.patient_context, ex.pre_evidence)
        else:
            prompt = _build_norag_prompt(ex.patient_context)

        # Reconstruct clean JSON from parsed diagnoses (not raw_output)
        # This ensures the SFT target is always valid, parseable JSON,
        # teaching the model to produce well-formed output.
        response = json.dumps(
            {"diagnoses": ex.structured_diagnoses},
            indent=2,
            ensure_ascii=False,
        )

        training_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "hadm_id": ex.hadm_id,
                "subject_id": ex.subject_id,
                "model_name": ex.model_name,
                "mode": ex.mode,
                "eb_count_top3": ex.eb_count_top3,
                "mean_rg_max_top3": round(ex.mean_rg_max_top3, 4),
                "per_diagnosis_eb": ex.per_diagnosis_eb,
                "ground_truth_diagnoses": ex.ground_truth_diagnoses,
            },
        })

    return training_data


def compute_stats(
    all_examples: list[SFTExample],
    deduped_examples: list[SFTExample],
    mode: str,
) -> DatasetStats:
    """Compute dataset statistics."""
    stats = DatasetStats(
        mode=mode,
        total_patient_model_pairs=0,  # Set externally
        qualifying_pairs=len(all_examples),
        unique_patients_qualifying=len(set(ex.hadm_id for ex in all_examples)),
        after_dedup=len(deduped_examples),
        unique_patients_after_dedup=len(set(ex.hadm_id for ex in deduped_examples)),
    )

    # Per-model breakdown
    for _, _, model_name in MODELS:
        model_qualifying = [ex for ex in all_examples if ex.model_name == model_name]
        model_deduped = [ex for ex in deduped_examples if ex.model_name == model_name]
        stats.per_model[model_name] = {
            'qualifying': len(model_qualifying),
            'after_dedup': len(model_deduped),
        }

    # Quality distribution (after dedup)
    stats.eb3_count = sum(1 for ex in deduped_examples if ex.eb_count_top3 == 3)
    stats.eb2_count = sum(1 for ex in deduped_examples if ex.eb_count_top3 == 2)

    return stats


def print_stats(stats: DatasetStats, min_eb: int, max_per_patient: int, grounding_threshold: float = GROUNDING_THRESHOLD):
    """Print formatted statistics."""
    print(f"\n{'=' * 70}")
    print(f"SFT DATASET STATISTICS — {stats.mode.upper()} MODE")
    print(f"{'=' * 70}")

    print(f"\n  Filter:                      >= {min_eb} EB in top-3")
    print(f"  Dedup cap:                   {max_per_patient} per patient")
    print(f"  Grounding threshold:         > {grounding_threshold}")

    print(f"\n{'Pipeline':^70}")
    print(f"{'-' * 70}")
    print(f"  Qualifying patient-model pairs:  {stats.qualifying_pairs:,}")
    print(f"  Unique patients (qualifying):    {stats.unique_patients_qualifying:,}")
    print(f"  After dedup (cap={max_per_patient}):            {stats.after_dedup:,}")
    print(f"  Unique patients (final):         {stats.unique_patients_after_dedup:,}")

    print(f"\n{'Quality Distribution (after dedup)':^70}")
    print(f"{'-' * 70}")
    print(f"  3/3 EB in top-3:             {stats.eb3_count:,} ({100*stats.eb3_count/max(stats.after_dedup,1):.1f}%)")
    print(f"  2/3 EB in top-3:             {stats.eb2_count:,} ({100*stats.eb2_count/max(stats.after_dedup,1):.1f}%)")

    print(f"\n{'Per-Model Breakdown':^70}")
    print(f"{'-' * 70}")
    print(f"  {'Model':<20} {'Qualifying':>12} {'After Dedup':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for model_name, counts in stats.per_model.items():
        print(f"  {model_name:<20} {counts['qualifying']:>12,} {counts['after_dedup']:>12,}")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT training dataset from Evidence-Based model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for SFT datasets",
    )
    parser.add_argument(
        "--min-eb",
        type=int,
        default=2,
        help="Minimum Evidence-Based diagnoses in top-3 (default: 2)",
    )
    parser.add_argument(
        "--max-per-patient",
        type=int,
        default=2,
        help="Max examples per patient after dedup (default: 2)",
    )
    parser.add_argument(
        "--grounding-threshold",
        type=float,
        default=GROUNDING_THRESHOLD,
        help=f"Grounding threshold for EB classification (default: {GROUNDING_THRESHOLD})",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't save datasets",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["rag", "no-rag"],
        default=["rag", "no-rag"],
        help="Which modes to build datasets for (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["messages", "jsonl"],
        default="messages",
        help="Output format: 'messages' (single JSON with messages array) or 'jsonl' (one example per line)",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    grounding_threshold = args.grounding_threshold

    for mode in args.modes:
        print(f"\n{'#' * 70}")
        print(f"# Building SFT dataset: {mode.upper()} mode")
        print(f"{'#' * 70}")

        if mode == "rag":
            gen_dir = GENERATION_RAG_DIR
            metrics_dir = METRICS_RAG_DIR
        else:
            gen_dir = GENERATION_NORAG_DIR
            metrics_dir = METRICS_NORAG_DIR

        # Step 1: Build qualifying examples
        print(f"\nStep 1: Filtering (>= {args.min_eb} EB in top-3, grounding > {grounding_threshold})...")
        all_examples = build_sft_examples(
            mode=mode,
            gen_dir=gen_dir,
            metrics_dir=metrics_dir,
            min_eb=args.min_eb,
            grounding_threshold=grounding_threshold,
        )
        print(f"  Total qualifying: {len(all_examples)}")

        # Step 2: Deduplicate
        print(f"\nStep 2: Deduplication (cap={args.max_per_patient} per patient)...")
        deduped = deduplicate(all_examples, max_per_patient=args.max_per_patient)
        print(f"  After dedup: {len(deduped)}")

        # Step 3: Compute and print stats
        stats = compute_stats(all_examples, deduped, mode)
        print_stats(stats, args.min_eb, args.max_per_patient, grounding_threshold)

        # Step 4: Format and save
        if not args.stats_only and args.output_dir:
            # Shuffle deterministically
            random.shuffle(deduped)

            training_data = format_for_training(deduped)

            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            filename = f"sft_{mode.replace('-', '_')}_eb{args.min_eb}_cap{args.max_per_patient}"

            if args.format == "jsonl":
                out_file = output_path / f"{filename}.jsonl"
                with open(out_file, 'w') as f:
                    for example in training_data:
                        f.write(json.dumps(example) + '\n')
            else:
                out_file = output_path / f"{filename}.json"
                output_data = {
                    'config': {
                        'mode': mode,
                        'min_eb_top3': args.min_eb,
                        'max_per_patient': args.max_per_patient,
                        'grounding_threshold': grounding_threshold,
                        'seed': args.seed,
                        'total_qualifying': len(all_examples),
                        'after_dedup': len(deduped),
                        'unique_patients': stats.unique_patients_after_dedup,
                    },
                    'statistics': asdict(stats),
                    'examples': training_data,
                }
                with open(out_file, 'w') as f:
                    json.dump(output_data, f, indent=2)

            print(f"  Saved {len(training_data)} examples to: {out_file}")

        elif not args.stats_only and not args.output_dir:
            print("  No --output-dir specified. Use --output-dir to save or --stats-only for stats only.")


if __name__ == "__main__":
    main()
