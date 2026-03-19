#!/usr/bin/env python3
"""
Build GRPO training dataset from existing generation outputs.

The GRPO dataset is just prompts + metadata (not model outputs). During training,
the model generates its own completions and they are scored by reward functions.

Each entry contains:
  - prompt: The patient prompt in chat format
  - ground_truth_diagnoses: For accuracy reward scoring
  - patient_context: For grounding reward (NLI needs raw sections)
  - pre_evidence: Retrieved evidence docs (RAG only, empty for no-RAG)

Source: Extracts patient data from existing generation JSONs
        (any model's file works — same 3700 patients).

Usage:
    python build_grpo_dataset.py --output-dir training_data/grpo/ --modes no-rag
    python build_grpo_dataset.py --output-dir training_data/grpo/ --modes rag no-rag
    python build_grpo_dataset.py --stats-only
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import Optional


# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # evidenceRL/

GENERATION_RAG_DIR = PROJECT_ROOT / "generation_output"
GENERATION_NORAG_DIR = PROJECT_ROOT / "generation_baseline_output"

# Preferred source models (high parse success, available for both modes)
# We just need one model's generation file per mode to get all 3700 patients
PREFERRED_SOURCES = [
    "gemma-3-12b",
    "Llama-3.1-8B",
    "gemma-3-27b",
    "gemma-3-4b",
    "Llama-3.2-3B",
    "Llama-3.3-70B",
    "gpt-oss-20b",
    "gpt-oss-120b",
]


# ── Prompt templates (same as build_sft_dataset.py) ───────────────────────

def _clean_evidence_text(text: str) -> str:
    """Clean XML/HTML tags from evidence text."""
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


def find_generation_json(gen_dir: Path, mode: str) -> Optional[Path]:
    """Find a TRAIN generation JSON from the preferred source models."""
    for model_name in PREFERRED_SOURCES:
        model_dir = gen_dir / f"{model_name}_v1.0-TRAIN"
        if not model_dir.exists():
            continue

        json_files = [
            f for f in model_dir.glob("*.json")
            if f.parent == model_dir
        ]
        if json_files:
            print(f"  Using source: {json_files[0].name} from {model_name}")
            return json_files[0]

    return None


def build_grpo_prompts(gen_json_path: Path, mode: str) -> list[dict]:
    """Build GRPO prompt entries from a generation JSON."""
    with open(gen_json_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    prompts = []
    seen_hadm_ids = set()

    for result in results:
        hadm_id = str(result['hadm_id'])

        # Deduplicate by hadm_id
        if hadm_id in seen_hadm_ids:
            continue
        seen_hadm_ids.add(hadm_id)

        patient_context = result.get('patient_context', '')
        ground_truth = result.get('ground_truth_diagnoses', [])
        pre_evidence = result.get('pre_evidence', [])

        if not patient_context.strip() or not ground_truth:
            continue

        # Build the prompt
        if mode == "rag":
            prompt_text = _build_rag_prompt(patient_context, pre_evidence)
        else:
            prompt_text = _build_norag_prompt(patient_context)

        prompts.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth_diagnoses": ground_truth,
            "patient_context": patient_context,
            "pre_evidence": pre_evidence if mode == "rag" else [],
            "hadm_id": hadm_id,
            "subject_id": str(result.get('subject_id', '')),
        })

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Build GRPO training dataset from existing generation outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for GRPO datasets",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["rag", "no-rag"],
        default=["no-rag"],
        help="Which modes to build datasets for (default: no-rag)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't save datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    for mode in args.modes:
        print(f"\n{'#' * 70}")
        print(f"# Building GRPO dataset: {mode.upper()} mode")
        print(f"{'#' * 70}")

        if mode == "rag":
            gen_dir = GENERATION_RAG_DIR
        else:
            gen_dir = GENERATION_NORAG_DIR

        # Find a source generation JSON
        gen_json = find_generation_json(gen_dir, mode)
        if gen_json is None:
            print(f"  ERROR: No TRAIN generation JSON found in {gen_dir}")
            continue

        # Build prompts
        print(f"\nExtracting patient prompts...")
        prompts = build_grpo_prompts(gen_json, mode)
        print(f"  Total unique patients: {len(prompts)}")

        # Stats
        gt_counts = [len(p['ground_truth_diagnoses']) for p in prompts]
        avg_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 0
        print(f"  Avg ground truth diagnoses per patient: {avg_gt:.1f}")

        if mode == "rag":
            evidence_counts = [len(p['pre_evidence']) for p in prompts]
            avg_ev = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0
            print(f"  Avg evidence documents per patient: {avg_ev:.1f}")

        prompt_lengths = [len(p['prompt'][0]['content']) for p in prompts]
        avg_len = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        print(f"  Avg prompt length (chars): {avg_len:.0f}")

        # Save
        if not args.stats_only and args.output_dir:
            random.shuffle(prompts)

            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            mode_tag = mode.replace('-', '_')
            out_file = output_path / f"grpo_{mode_tag}_prompts.json"

            output_data = {
                'config': {
                    'mode': mode,
                    'source_file': str(gen_json),
                    'num_prompts': len(prompts),
                    'seed': args.seed,
                },
                'prompts': prompts,
            }

            with open(out_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"\n  Saved {len(prompts)} prompts to: {out_file}")

        elif not args.stats_only and not args.output_dir:
            print("\n  No --output-dir specified. Use --output-dir to save.")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
