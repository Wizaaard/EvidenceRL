#!/usr/bin/env python3
"""
Reasoning Revision Pipeline for EvidenceRL.

This script revises reasoning for diagnoses based on their correctness and grounding scores.

Cases:
  - Case A (Correct diagnosis, ambiguous grounding [-0.25, 0.25]):
    Regenerate well-grounded reasoning that properly integrates evidence.

  - Case B (Incorrect diagnosis, ambiguous grounding [-0.25, 0.25]):
    Regenerate contradictory reasoning (50% ignore key evidence, 50% fabricate evidence).

The script:
1. Loads generation JSON and metrics JSON
2. Filters diagnoses based on verdict AND grounding criteria
3. Uses medgemma-27b (or specified model) to regenerate reasoning
4. Saves new generation JSON and metrics JSON to output directory
5. Optionally runs NLI to recalculate grounding scores

Usage:
    python revise_reasoning.py \
        --generation-json <path> \
        --metrics-json <path> \
        --output-dir <path> \
        --model-name <model_path>
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


@dataclass
class DiagnosisToRevise:
    """A diagnosis that needs reasoning revision."""
    patient_idx: int
    diagnosis_idx: int
    diagnosis_name: str
    original_reasoning: str
    is_correct: bool
    grounding: float
    patient_context: str
    evidence: List[Dict[str, Any]]
    ground_truth_diagnoses: List[str]


def load_generation_json(path: str) -> Dict[str, Any]:
    """Load generation JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_metrics_json(path: str) -> Dict[str, Any]:
    """Load metrics JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def filter_diagnoses_for_revision(
    generation_data: Dict[str, Any],
    metrics_data: Dict[str, Any],
    grounding_min: float = -0.25,
    grounding_max: float = 0.25,
) -> Tuple[List[DiagnosisToRevise], List[DiagnosisToRevise]]:
    """
    Filter diagnoses that need revision based on criteria.

    Args:
        generation_data: Generation JSON data
        metrics_data: Metrics JSON data
        grounding_min: Minimum grounding score for ambiguous range
        grounding_max: Maximum grounding score for ambiguous range

    Returns:
        Tuple of (case_a_diagnoses, case_b_diagnoses) where:
        - case_a: Correct diagnoses with ambiguous grounding (to make well-grounded)
        - case_b: Incorrect diagnoses with ambiguous grounding (to make contradictory)
    """
    case_a = []  # Correct, ambiguous grounding -> well-grounded
    case_b = []  # Incorrect, ambiguous grounding -> contradictory

    results = generation_data.get('results', [])
    patient_metrics = metrics_data.get('patient_metrics', [])

    # Build lookup from hadm_id to metrics
    metrics_lookup = {pm['hadm_id']: pm for pm in patient_metrics}

    for patient_idx, patient in enumerate(results):
        hadm_id = patient.get('hadm_id')
        metrics = metrics_lookup.get(hadm_id)

        if metrics is None:
            continue

        patient_context = patient.get('patient_context', '')
        evidence = patient.get('pre_evidence', [])
        ground_truth = patient.get('ground_truth_diagnoses', [])

        structured_output = patient.get('structured_output', {})
        diagnoses = structured_output.get('diagnoses', [])

        verdicts = metrics.get('verdicts', [])
        per_diagnosis_rewards = metrics.get('per_diagnosis_rewards', [])

        for diag_idx, diagnosis in enumerate(diagnoses):
            if diag_idx >= len(verdicts) or diag_idx >= len(per_diagnosis_rewards):
                continue

            diag_name = diagnosis.get('name', '')
            diag_reasoning = diagnosis.get('reasoning', '')
            is_correct = verdicts[diag_idx]
            grounding = per_diagnosis_rewards[diag_idx].get('grounding', 0.0)

            # Skip if no reasoning or empty diagnosis
            if not diag_name or not diag_reasoning:
                continue

            # Check if grounding is in ambiguous range
            if not (grounding_min <= grounding <= grounding_max):
                continue

            diagnosis_to_revise = DiagnosisToRevise(
                patient_idx=patient_idx,
                diagnosis_idx=diag_idx,
                diagnosis_name=diag_name,
                original_reasoning=diag_reasoning,
                is_correct=is_correct,
                grounding=grounding,
                patient_context=patient_context,
                evidence=evidence,
                ground_truth_diagnoses=ground_truth,
            )

            if is_correct:
                case_a.append(diagnosis_to_revise)
            else:
                case_b.append(diagnosis_to_revise)

    return case_a, case_b


def build_well_grounded_prompt(diagnosis: DiagnosisToRevise) -> str:
    """
    Build prompt for generating well-grounded reasoning (Case A).

    The reasoning should properly integrate evidence and patient data.
    """
    evidence_text = ""
    if diagnosis.evidence:
        evidence_lines = []
        for idx, ev in enumerate(diagnosis.evidence, start=1):
            text = ev.get('text', '')
            if text:
                evidence_lines.append(f"Evidence {idx}: {text}")
        evidence_text = "\n\n".join(evidence_lines)
    else:
        evidence_text = "No external clinical evidence available."

    prompt = f'''You are an expert cardiology clinical assistant. Your task is to write a well-grounded reasoning paragraph for a diagnosis.

The reasoning MUST:
1. Explicitly reference specific findings from the patient data (vital signs, lab values, imaging results)
2. Connect patient findings to the clinical guidelines/evidence provided
3. Explain the pathophysiological mechanism linking symptoms to the diagnosis
4. Use exact values and measurements as evidence anchors

Patient Information:
{diagnosis.patient_context}

Clinical Evidence/Guidelines:
{evidence_text}

Diagnosis to explain: {diagnosis.diagnosis_name}

Write a detailed reasoning paragraph (3-5 sentences) that is well-grounded in both the patient data and clinical evidence. The reasoning should demonstrate clear logical connections between the evidence and the diagnosis.

Reasoning:'''

    return prompt


def build_contradictory_prompt(
    diagnosis: DiagnosisToRevise,
    contradiction_type: str,
) -> str:
    """
    Build prompt for generating ungrounded reasoning (Case B).

    For INCORRECT diagnoses with ambiguous grounding, we generate reasoning that
    is NOT properly grounded in the patient context. This creates negative training
    examples where the NLI model should score the reasoning as contradictory/neutral.

    Args:
        diagnosis: The diagnosis to revise
        contradiction_type: Either "ignore" (generic reasoning) or "fabricate" (mismatched claims)
    """
    evidence_text = ""
    if diagnosis.evidence:
        evidence_lines = []
        for idx, ev in enumerate(diagnosis.evidence, start=1):
            text = ev.get('text', '')
            if text:
                evidence_lines.append(f"Evidence {idx}: {text}")
        evidence_text = "\n\n".join(evidence_lines)
    else:
        evidence_text = "No external clinical evidence available."

    if contradiction_type == "ignore":
        instruction = """Write GENERIC reasoning that discusses this diagnosis in general clinical terms.
The reasoning should:
- Discuss typical presentation of this condition WITHOUT citing THIS patient's specific findings
- Use general medical knowledge rather than the actual values from the patient data
- Sound like reasoning from someone who hasn't reviewed the specific patient chart
- Avoid mentioning any specific lab values, vital signs, or test results from above"""
    else:  # fabricate
        instruction = """Write reasoning that references clinical findings NOT PRESENT in the patient data.
The reasoning should:
- Mention specific values or findings that differ from what's in the patient information
- Reference symptoms or test results that are not documented above
- Create a plausible-sounding clinical narrative that doesn't match the actual data
- Sound detailed but describe a different clinical picture than what's presented"""

    prompt = f'''You are helping create diverse training examples for a medical AI system that learns to evaluate reasoning quality.

For this exercise, write reasoning that is UNGROUNDED - not properly anchored to the actual patient evidence provided. This helps train the system to distinguish well-grounded from poorly-grounded reasoning.

{instruction}

Patient Information:
{diagnosis.patient_context}

Clinical Evidence/Guidelines:
{evidence_text}

Diagnosis to explain: {diagnosis.diagnosis_name}

Write a reasoning paragraph (3-5 sentences) that sounds clinical but is not grounded in the specific patient data above.

Reasoning:'''

    return prompt


class ReasoningReviser:
    """Handles reasoning revision using an LLM."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        max_new_tokens: int = 512,
    ):
        """
        Initialize the reasoning reviser.

        Args:
            model_name: HuggingFace model path or name
            batch_size: Batch size for generation
            max_new_tokens: Maximum new tokens to generate
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _init_pipeline(self):
        """Initialize the HuggingFace pipeline lazily."""
        if self._pipeline is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

        print(f"Loading model: {self.model_name}...")

        model_kwargs = {}
        pipeline_kwargs = {}

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                model_kwargs["device_map"] = "auto"
            else:
                pipeline_kwargs["device"] = 0

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )

        if hasattr(model, "config") and model.config is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if not model_kwargs.get("device_map") and pipeline_kwargs.get("device") == 0:
            model.to("cuda:0")

        self._pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **pipeline_kwargs,
        )

        print(f"Model loaded successfully.")

    def revise_batch(
        self,
        prompts: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate revised reasoning for a batch of prompts.

        Args:
            prompts: List of prompts for reasoning generation
            show_progress: Whether to show progress bar

        Returns:
            List of generated reasoning texts
        """
        self._init_pipeline()

        results = []

        generation_kwargs = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
            "pad_token_id": self._pipeline.tokenizer.pad_token_id,
        }

        iterator = range(0, len(prompts), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Generating revised reasoning",
                total=(len(prompts) + self.batch_size - 1) // self.batch_size,
            )

        for start in iterator:
            batch_prompts = prompts[start:start + self.batch_size]

            try:
                outputs = self._pipeline(batch_prompts, **generation_kwargs)

                for prompt, output in zip(batch_prompts, outputs):
                    if not output:
                        results.append("")
                        continue

                    out_item = output[0] if isinstance(output, list) else output
                    generated = out_item.get("generated_text") or out_item.get("text") or ""

                    # Clean up the generated text
                    generated = generated.strip()
                    # Remove any trailing incomplete sentences
                    if generated and not generated[-1] in '.!?':
                        last_period = max(
                            generated.rfind('.'),
                            generated.rfind('!'),
                            generated.rfind('?')
                        )
                        if last_period > 0:
                            generated = generated[:last_period + 1]

                    results.append(generated)

            except Exception as e:
                print(f"Error in batch generation: {e}")
                # Fallback to individual generation
                for prompt in batch_prompts:
                    try:
                        output = self._pipeline(prompt, **generation_kwargs)
                        out_item = output[0] if output else {}
                        generated = out_item.get("generated_text") or ""
                        results.append(generated.strip())
                    except Exception:
                        results.append("")

        return results


def revise_diagnoses(
    case_a_diagnoses: List[DiagnosisToRevise],
    case_b_diagnoses: List[DiagnosisToRevise],
    reviser: ReasoningReviser,
    show_progress: bool = True,
) -> Dict[Tuple[int, int], str]:
    """
    Revise reasoning for all filtered diagnoses.

    Args:
        case_a_diagnoses: Correct diagnoses needing well-grounded reasoning
        case_b_diagnoses: Incorrect diagnoses needing contradictory reasoning
        reviser: The reasoning reviser instance
        show_progress: Whether to show progress

    Returns:
        Dict mapping (patient_idx, diagnosis_idx) to revised reasoning
    """
    revised_reasoning = {}

    # Build prompts for Case A (well-grounded)
    case_a_prompts = []
    case_a_keys = []
    for diag in case_a_diagnoses:
        prompt = build_well_grounded_prompt(diag)
        case_a_prompts.append(prompt)
        case_a_keys.append((diag.patient_idx, diag.diagnosis_idx))

    # Build prompts for Case B (contradictory - 50% ignore, 50% fabricate)
    case_b_prompts = []
    case_b_keys = []
    for diag in case_b_diagnoses:
        # Randomly choose contradiction type
        contradiction_type = random.choice(["ignore", "fabricate"])
        prompt = build_contradictory_prompt(diag, contradiction_type)
        case_b_prompts.append(prompt)
        case_b_keys.append((diag.patient_idx, diag.diagnosis_idx))

    # Generate revised reasoning
    if case_a_prompts:
        print(f"Generating well-grounded reasoning for {len(case_a_prompts)} correct diagnoses...")
        case_a_results = reviser.revise_batch(case_a_prompts, show_progress=show_progress)
        for key, reasoning in zip(case_a_keys, case_a_results):
            revised_reasoning[key] = reasoning

    if case_b_prompts:
        print(f"Generating contradictory reasoning for {len(case_b_prompts)} incorrect diagnoses...")
        case_b_results = reviser.revise_batch(case_b_prompts, show_progress=show_progress)
        for key, reasoning in zip(case_b_keys, case_b_results):
            revised_reasoning[key] = reasoning

    return revised_reasoning


def apply_revisions(
    generation_data: Dict[str, Any],
    metrics_data: Dict[str, Any],
    revised_reasoning: Dict[Tuple[int, int], str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply revised reasoning to the generation and metrics data.

    Returns new copies of generation_data and metrics_data with revisions applied.
    """
    import copy

    # Deep copy to avoid modifying originals
    new_generation = copy.deepcopy(generation_data)
    new_metrics = copy.deepcopy(metrics_data)

    # Apply revisions to generation data
    results = new_generation.get('results', [])
    for (patient_idx, diagnosis_idx), new_reasoning in revised_reasoning.items():
        if patient_idx < len(results):
            patient = results[patient_idx]
            structured_output = patient.get('structured_output', {})
            diagnoses = structured_output.get('diagnoses', [])
            if diagnosis_idx < len(diagnoses):
                diagnoses[diagnosis_idx]['reasoning'] = new_reasoning

    # Apply revisions to metrics data (predicted_diagnoses_with_reasoning)
    patient_metrics = new_metrics.get('patient_metrics', [])

    # Build lookup from hadm_id to patient_idx
    hadm_to_idx = {}
    for idx, result in enumerate(results):
        hadm_to_idx[result.get('hadm_id')] = idx

    for pm in patient_metrics:
        hadm_id = pm.get('hadm_id')
        patient_idx = hadm_to_idx.get(hadm_id)
        if patient_idx is None:
            continue

        reasoning_list = pm.get('predicted_diagnoses_with_reasoning', [])
        for diag_idx, diag in enumerate(reasoning_list):
            key = (patient_idx, diag_idx)
            if key in revised_reasoning:
                diag['reasoning'] = revised_reasoning[key]

    # Add metadata about revision
    new_generation['revision_info'] = {
        'source_file': generation_data.get('config', {}).get('model_id', 'unknown'),
        'num_revisions': len(revised_reasoning),
        'revision_type': 'reasoning_revision_v1',
    }

    new_metrics['revision_info'] = {
        'source_file': metrics_data.get('config', {}).get('model_id', 'unknown'),
        'num_revisions': len(revised_reasoning),
        'revision_type': 'reasoning_revision_v1',
        'grounding_needs_recalculation': True,
    }

    return new_generation, new_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Revise reasoning for diagnoses based on correctness and grounding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--generation-json",
        type=str,
        required=True,
        help="Path to generation JSON file"
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        required=True,
        help="Path to metrics JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for revised JSON files"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_BASE_DIR", "//storage/ice-shared/bmed-sp-wang/Models") + "/medgemma-27b-it",
        help="Model to use for reasoning revision (default: $MODEL_BASE_DIR/medgemma-27b-it)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for generation (default: 64)"
    )
    parser.add_argument(
        "--grounding-min",
        type=float,
        default=-0.25,
        help="Minimum grounding score for ambiguous range (default: -0.25)"
    )
    parser.add_argument(
        "--grounding-max",
        type=float,
        default=0.25,
        help="Maximum grounding score for ambiguous range (default: 0.25)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--run-nli",
        action="store_true",
        help="Run NLI grounding recalculation after revision"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading generation data from: {args.generation_json}")
    generation_data = load_generation_json(args.generation_json)

    print(f"Loading metrics data from: {args.metrics_json}")
    metrics_data = load_metrics_json(args.metrics_json)

    # Filter diagnoses for revision
    print(f"\nFiltering diagnoses with grounding in [{args.grounding_min}, {args.grounding_max}]...")
    case_a, case_b = filter_diagnoses_for_revision(
        generation_data,
        metrics_data,
        grounding_min=args.grounding_min,
        grounding_max=args.grounding_max,
    )

    print(f"  Case A (correct, ambiguous grounding): {len(case_a)} diagnoses")
    print(f"  Case B (incorrect, ambiguous grounding): {len(case_b)} diagnoses")
    print(f"  Total diagnoses to revise: {len(case_a) + len(case_b)}")

    if not case_a and not case_b:
        print("\nNo diagnoses match the revision criteria. Exiting.")
        sys.exit(0)

    # Initialize reviser
    reviser = ReasoningReviser(
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    # Revise reasoning
    print("\nRevising reasoning...")
    revised_reasoning = revise_diagnoses(
        case_a,
        case_b,
        reviser,
        show_progress=True,
    )

    print(f"\nSuccessfully revised {len(revised_reasoning)} diagnoses.")

    # Apply revisions
    print("\nApplying revisions to data...")
    new_generation, new_metrics = apply_revisions(
        generation_data,
        metrics_data,
        revised_reasoning,
    )

    # Save revised data
    gen_filename = Path(args.generation_json).stem + "_revised.json"
    metrics_filename = Path(args.metrics_json).stem + "_revised.json"

    gen_output_path = output_dir / gen_filename
    metrics_output_path = output_dir / metrics_filename

    print(f"\nSaving revised generation data to: {gen_output_path}")
    with open(gen_output_path, 'w') as f:
        json.dump(new_generation, f, indent=2)

    print(f"Saving revised metrics data to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        json.dump(new_metrics, f, indent=2)

    # Optionally run NLI recalculation
    if args.run_nli:
        print("\nRunning NLI grounding recalculation...")
        nli_script = Path(__file__).parent / "compute_revised_grounding.py"
        if nli_script.exists():
            import subprocess
            result = subprocess.run([
                sys.executable,
                str(nli_script),
                "--generation-json", str(gen_output_path),
                "--metrics-json", str(metrics_output_path),
                "--output-dir", str(output_dir),
            ])
            if result.returncode != 0:
                print("Warning: NLI recalculation failed")
        else:
            print(f"Warning: NLI script not found at {nli_script}")

    print("\n" + "=" * 60)
    print("Reasoning Revision Complete")
    print("=" * 60)
    print(f"Revised generation: {gen_output_path}")
    print(f"Revised metrics: {metrics_output_path}")
    print(f"Case A revisions (well-grounded): {len(case_a)}")
    print(f"Case B revisions (contradictory): {len(case_b)}")
    if not args.run_nli:
        print("\nNote: Run compute_revised_grounding.py to recalculate grounding scores.")
    print("=" * 60)


if __name__ == "__main__":
    main()
