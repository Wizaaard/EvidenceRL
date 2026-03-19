#!/usr/bin/env python3
"""
Reasoning Revision Worker (vLLM Version) - High-performance batch inference.

This script uses vLLM for 14-24x faster inference compared to HuggingFace pipeline.
It processes a subset of patients for distributed revision.

Key differences from HuggingFace version:
- Uses vLLM's LLM class with tensor parallelism for multi-GPU
- PagedAttention for efficient memory management
- Continuous batching for optimal throughput

Usage:
    python revision_worker_vllm.py \
        --generation-json <path> \
        --metrics-json <path> \
        --output-dir <path> \
        --patient-start <int> \
        --patient-end <int> \
        --worker-id <int> \
        --model-name <path>
"""

import argparse
import json
import random
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

# Add src to path for evidence_rl imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Check vLLM availability early
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[ERROR] vLLM not available. Install with: pip install vllm")
    sys.exit(1)


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
    hadm_id: str


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def filter_diagnoses_for_revision(
    generation_data: Dict[str, Any],
    metrics_data: Dict[str, Any],
    patient_start: int,
    patient_end: int,
    grounding_min: float = -0.25,
    grounding_max: float = 0.25,
) -> Tuple[List[DiagnosisToRevise], List[DiagnosisToRevise]]:
    """
    Filter diagnoses that need revision for a subset of patients.

    Args:
        generation_data: Generation JSON data
        metrics_data: Metrics JSON data
        patient_start: Start patient index (inclusive)
        patient_end: End patient index (exclusive)
        grounding_min: Minimum grounding score for ambiguous range
        grounding_max: Maximum grounding score for ambiguous range

    Returns:
        Tuple of (case_a_diagnoses, case_b_diagnoses)
    """
    case_a = []  # Correct, ambiguous grounding -> well-grounded
    case_b = []  # Incorrect, ambiguous grounding -> contradictory

    results = generation_data.get('results', [])
    patient_metrics = metrics_data.get('patient_metrics', [])

    # Build lookup from hadm_id to metrics
    metrics_lookup = {pm['hadm_id']: pm for pm in patient_metrics}

    # Only process patients in the specified range
    for patient_idx in range(patient_start, min(patient_end, len(results))):
        patient = results[patient_idx]
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
                hadm_id=hadm_id,
            )

            if is_correct:
                case_a.append(diagnosis_to_revise)
            else:
                case_b.append(diagnosis_to_revise)

    return case_a, case_b


def build_well_grounded_prompt(diagnosis: DiagnosisToRevise) -> str:
    """Build prompt for generating well-grounded reasoning (Case A).

    For CORRECT diagnoses with ambiguous grounding, we generate improved reasoning
    that is properly grounded in the patient context and aligned with clinical evidence.
    This creates positive training examples:
    - Correct diagnosis + well-grounded reasoning → should get high reward
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
    """Build prompt for generating ungrounded reasoning (Case B).

    For INCORRECT diagnoses with ambiguous grounding, we generate reasoning that
    is NOT properly grounded in the patient context. This creates negative training
    examples where the NLI model should score the reasoning as contradictory/neutral.

    The reasoning is ungrounded by either:
    1. Being GENERIC (discussing the diagnosis without citing specific patient data)
    2. Being MISMATCHED (referencing findings not present in the patient data)

    Both approaches create reasoning that the NLI model should score as
    contradictory or neutral (low grounding), because the claims don't
    match the actual patient evidence.
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


class VLLMReasoningReviser:
    """Handles reasoning revision using vLLM for high-performance inference."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 2,
        max_tokens: int = 256,
        gpu_memory_utilization: float = 0.90,
    ):
        """
        Initialize the vLLM reasoning reviser.

        Args:
            model_name: HuggingFace model path or name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_tokens: Maximum new tokens to generate
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._sampling_params = None

    def _init_llm(self):
        """Initialize the vLLM LLM instance lazily."""
        if self._llm is not None:
            return

        print(f"[vLLM] Loading model: {self.model_name}")
        print(f"[vLLM] Tensor parallel size: {self.tensor_parallel_size}")
        print(f"[vLLM] GPU memory utilization: {self.gpu_memory_utilization}")

        # Detect available GPUs
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"[vLLM] Available GPUs: {gpu_count}")

        if gpu_count < self.tensor_parallel_size:
            print(f"[vLLM] Warning: Requested {self.tensor_parallel_size} GPUs but only {gpu_count} available")
            self.tensor_parallel_size = gpu_count

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

        self._sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.max_tokens,
            stop=["\n\n\n"],  # Stop on triple newline
        )

        print(f"[vLLM] Model loaded successfully.")

    def revise_batch(
        self,
        prompts: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate revised reasoning for all prompts using vLLM.

        vLLM handles batching internally with continuous batching,
        so we pass all prompts at once for optimal throughput.

        Args:
            prompts: List of prompts for reasoning generation
            show_progress: Unused (vLLM shows its own progress)

        Returns:
            List of generated reasoning texts
        """
        self._init_llm()

        if not prompts:
            return []

        print(f"[vLLM] Generating {len(prompts)} responses...")

        # vLLM handles all batching internally
        outputs = self._llm.generate(prompts, self._sampling_params)

        results = []
        for output in outputs:
            if output.outputs:
                generated = output.outputs[0].text.strip()

                # Clean up: truncate at incomplete sentence if needed
                if generated and generated[-1] not in '.!?':
                    last_period = max(
                        generated.rfind('.'),
                        generated.rfind('!'),
                        generated.rfind('?')
                    )
                    if last_period > 0:
                        generated = generated[:last_period + 1]

                results.append(generated)
            else:
                results.append("")

        print(f"[vLLM] Generated {len(results)} responses.")
        return results


def revise_diagnoses(
    case_a_diagnoses: List[DiagnosisToRevise],
    case_b_diagnoses: List[DiagnosisToRevise],
    reviser: VLLMReasoningReviser,
    show_progress: bool = True,
) -> Dict[Tuple[int, int], str]:
    """Revise reasoning for all filtered diagnoses."""
    revised_reasoning = {}

    # Build all prompts together for better batching
    all_prompts = []
    all_keys = []
    prompt_types = []  # Track which type each prompt is

    # Case A prompts (well-grounded)
    for diag in case_a_diagnoses:
        prompt = build_well_grounded_prompt(diag)
        all_prompts.append(prompt)
        all_keys.append((diag.patient_idx, diag.diagnosis_idx))
        prompt_types.append('A')

    # Case B prompts (simulated flawed reasoning - 50% ignore, 50% fabricate)
    for diag in case_b_diagnoses:
        contradiction_type = random.choice(["ignore", "fabricate"])
        prompt = build_contradictory_prompt(diag, contradiction_type)
        all_prompts.append(prompt)
        all_keys.append((diag.patient_idx, diag.diagnosis_idx))
        prompt_types.append('B')

    # Generate all at once (vLLM handles batching optimally)
    if all_prompts:
        print(f"Generating reasoning for {len(case_a_diagnoses)} Case A + {len(case_b_diagnoses)} Case B = {len(all_prompts)} total...")
        all_results = reviser.revise_batch(all_prompts, show_progress=show_progress)

        for key, reasoning in zip(all_keys, all_results):
            revised_reasoning[key] = reasoning

    return revised_reasoning


def compute_reward_at_k(rewards: List[float], max_k: int = 5) -> Dict[int, float]:
    """Compute cumulative average reward at each k value."""
    result = {}
    for k in range(1, max_k + 1):
        considered = rewards[:k]
        if considered:
            result[k] = sum(considered) / len(considered)
        else:
            result[k] = 0.0
    return result


def compute_precision_recall_at_k(
    verdicts: List[bool],
    num_ground_truth: int,
    max_k: int = 5,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute precision@k and recall@k from verdicts."""
    precision_at_k = {}
    recall_at_k = {}

    for k in range(1, max_k + 1):
        considered = verdicts[:k]
        if not considered:
            precision_at_k[k] = 0.0
            recall_at_k[k] = 0.0
            continue

        hits = sum(considered)
        precision_at_k[k] = hits / len(considered)
        recall_at_k[k] = min(hits, num_ground_truth) / num_ground_truth if num_ground_truth > 0 else 0.0

    return precision_at_k, recall_at_k


def recalculate_grounding_scores(
    revised_results: List[Dict[str, Any]],
    revised_patient_metrics: List[Dict[str, Any]],
    revised_keys: set,
    nli_model,
    embedder,
    reward_weight_grounding: float = 0.5,
    max_k: int = 5,
    worker_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Recalculate NLI grounding scores for revised diagnoses.

    Args:
        revised_results: List of revised patient results (generation data)
        revised_patient_metrics: List of revised patient metrics
        revised_keys: Set of (patient_idx, diag_idx) tuples that were revised
        nli_model: CrossEncoderNLI model for grounding calculation
        embedder: HuggingFaceEmbedder for Focus-Then-Verify
        reward_weight_grounding: Weight for grounding in combined reward
        max_k: Maximum k for reward@k metrics
        worker_id: Worker ID for logging

    Returns:
        Updated patient_metrics with recalculated grounding scores
    """
    from evidence_rl.reward import reward_grounding
    from evidence_rl.documents import Document, RetrievedDocument

    print(f"[worker {worker_id}] Recalculating NLI grounding scores for {len(revised_keys)} revised diagnoses...")

    # Build hadm_id to result lookup
    hadm_to_result = {r['hadm_id']: r for r in revised_results}

    for pm in tqdm(revised_patient_metrics, desc=f"[worker {worker_id}] NLI recalc"):
        hadm_id = pm['hadm_id']
        result = hadm_to_result.get(hadm_id)
        if result is None:
            continue

        patient_context = result.get('patient_context', '')
        pre_evidence_data = result.get('pre_evidence', [])

        # Convert evidence to RetrievedDocument objects
        pre_evidence = []
        for ev in pre_evidence_data:
            doc = Document(
                doc_id=ev.get('doc_id', ''),
                text=ev.get('text', ''),
            )
            pre_evidence.append(RetrievedDocument(document=doc, score=ev.get('score', 0.0)))

        structured_output = result.get('structured_output', {})
        diagnoses = structured_output.get('diagnoses', [])

        # Get patient index from result (stored during revision)
        patient_idx = result.get('_original_patient_idx')

        per_diagnosis_rewards = pm.get('per_diagnosis_rewards', [])
        grounding_rewards = []
        precision_rewards = []
        combined_rewards = []

        for diag_idx, diag_data in enumerate(diagnoses):
            if diag_idx >= len(per_diagnosis_rewards):
                continue

            diagnosis_name = diag_data.get('name', '')
            diagnosis_reasoning = diag_data.get('reasoning', '')

            # Get original reward entry
            reward_entry = per_diagnosis_rewards[diag_idx]
            original_precision = reward_entry.get('precision', 0.0)

            # Check if this diagnosis was revised
            key = (patient_idx, diag_idx) if patient_idx is not None else None
            was_revised = key in revised_keys if key else False

            if was_revised and diagnosis_reasoning:
                # Store original grounding for plotting (before updating)
                original_grounding = reward_entry.get('grounding', 0.0)

                # Recalculate grounding for revised reasoning
                new_grounding_max, new_grounding_avg = reward_grounding(
                    diagnosis_reasoning=diagnosis_reasoning,
                    patient_context=patient_context,
                    evidence_docs=pre_evidence,
                    nli_model=nli_model,
                    embedder=embedder,
                )

                # Update reward entry (combined uses grounding_max)
                new_combined = float(
                    reward_weight_grounding * new_grounding_max +
                    (1 - reward_weight_grounding) * original_precision
                )
                reward_entry['grounding_old'] = float(original_grounding)  # For plotting
                reward_entry['grounding_max'] = float(new_grounding_max)
                reward_entry['grounding_avg'] = float(new_grounding_avg)
                reward_entry['combined'] = new_combined
                reward_entry['revised'] = True

                grounding_rewards.append(new_grounding_max)
                combined_rewards.append(new_combined)
            else:
                # Keep original grounding
                grounding_rewards.append(reward_entry.get('grounding', 0.0))
                combined_rewards.append(reward_entry.get('combined', 0.0))

            precision_rewards.append(original_precision)

        # Recompute reward@k metrics
        pm['reward_grounding_at_k'] = {
            str(k): v for k, v in compute_reward_at_k(grounding_rewards, max_k).items()
        }
        pm['reward_precision_at_k'] = {
            str(k): v for k, v in compute_reward_at_k(precision_rewards, max_k).items()
        }
        pm['reward_combined_at_k'] = {
            str(k): v for k, v in compute_reward_at_k(combined_rewards, max_k).items()
        }

    print(f"[worker {worker_id}] NLI grounding recalculation complete.")
    return revised_patient_metrics


def create_worker_output(
    generation_data: Dict[str, Any],
    metrics_data: Dict[str, Any],
    revised_reasoning: Dict[Tuple[int, int], str],
    patient_start: int,
    patient_end: int,
    worker_id: int,
    nli_model=None,
    embedder=None,
    reward_weight_grounding: float = 0.5,
    max_k: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create worker output with revised data for the processed patient range.

    Returns subset of generation and metrics data with revisions applied.
    If nli_model and embedder are provided, recalculates grounding scores.
    """
    import copy

    results = generation_data.get('results', [])
    patient_metrics = metrics_data.get('patient_metrics', [])

    # Build lookup from hadm_id to metrics index
    hadm_to_metrics_idx = {pm['hadm_id']: idx for idx, pm in enumerate(patient_metrics)}

    # Extract and revise the subset of patients
    revised_results = []
    revised_patient_metrics = []

    for patient_idx in range(patient_start, min(patient_end, len(results))):
        patient = copy.deepcopy(results[patient_idx])
        hadm_id = patient.get('hadm_id')

        # Store original patient index for NLI recalculation
        patient['_original_patient_idx'] = patient_idx

        # Apply revisions to generation data
        structured_output = patient.get('structured_output', {})
        diagnoses = structured_output.get('diagnoses', [])

        for diag_idx, diagnosis in enumerate(diagnoses):
            key = (patient_idx, diag_idx)
            if key in revised_reasoning:
                diagnosis['reasoning'] = revised_reasoning[key]

        revised_results.append(patient)

        # Get corresponding metrics
        metrics_idx = hadm_to_metrics_idx.get(hadm_id)
        if metrics_idx is not None:
            pm = copy.deepcopy(patient_metrics[metrics_idx])

            # Apply revisions to metrics (predicted_diagnoses_with_reasoning)
            reasoning_list = pm.get('predicted_diagnoses_with_reasoning', [])
            for diag_idx, diag in enumerate(reasoning_list):
                key = (patient_idx, diag_idx)
                if key in revised_reasoning:
                    diag['reasoning'] = revised_reasoning[key]

            revised_patient_metrics.append(pm)

    # Recalculate grounding scores if NLI model is available
    if nli_model is not None and embedder is not None and revised_reasoning:
        revised_patient_metrics = recalculate_grounding_scores(
            revised_results=revised_results,
            revised_patient_metrics=revised_patient_metrics,
            revised_keys=set(revised_reasoning.keys()),
            nli_model=nli_model,
            embedder=embedder,
            reward_weight_grounding=reward_weight_grounding,
            max_k=max_k,
            worker_id=worker_id,
        )

    # Build worker output
    worker_generation = {
        'config': {
            **generation_data.get('config', {}),
            'worker_id': worker_id,
            'patient_start': patient_start,
            'patient_end': patient_end,
            'inference_engine': 'vllm',
        },
        'results': revised_results,
    }

    worker_metrics = {
        'config': {
            **metrics_data.get('config', {}),
            'worker_id': worker_id,
            'patient_start': patient_start,
            'patient_end': patient_end,
            'inference_engine': 'vllm',
        },
        'patient_metrics': revised_patient_metrics,
    }

    return worker_generation, worker_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning revision worker (vLLM) - high-performance batch inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Output directory for worker results"
    )
    parser.add_argument(
        "--patient-start",
        type=int,
        required=True,
        help="Start patient index (inclusive)"
    )
    parser.add_argument(
        "--patient-end",
        type=int,
        required=True,
        help="End patient index (exclusive)"
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Worker ID"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_BASE_DIR", "//storage/ice-shared/bmed-sp-wang/Models") + "/medgemma-27b-it",
        help="Model to use for reasoning revision (default: $MODEL_BASE_DIR/medgemma-27b-it)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (default: 0.90)"
    )
    parser.add_argument(
        "--grounding-min",
        type=float,
        default=-0.25,
        help="Minimum grounding score for ambiguous range"
    )
    parser.add_argument(
        "--grounding-max",
        type=float,
        default=0.25,
        help="Maximum grounding score for ambiguous range"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model for grounding recalculation (default: cross-encoder/nli-deberta-v3-base)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for Focus-Then-Verify (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--reward-weight-grounding",
        type=float,
        default=0.5,
        help="Weight for grounding in combined reward (default: 0.5)"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maximum k for reward@k metrics (default: 5)"
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip NLI grounding recalculation (faster but grounding scores won't reflect revised reasoning)"
    )

    args = parser.parse_args()

    # Set random seed (add worker_id for different randomness per worker)
    random.seed(args.seed + args.worker_id)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worker {args.worker_id}] vLLM Reasoning Revision Worker")
    print(f"[worker {args.worker_id}] Loading data...")
    print(f"[worker {args.worker_id}] Patient range: [{args.patient_start}:{args.patient_end}]")

    # Load data
    generation_data = load_json(args.generation_json)
    metrics_data = load_json(args.metrics_json)

    total_patients = len(generation_data.get('results', []))
    print(f"[worker {args.worker_id}] Total patients in dataset: {total_patients}")

    # Filter diagnoses for this worker's patient range
    print(f"[worker {args.worker_id}] Filtering diagnoses with grounding in [{args.grounding_min}, {args.grounding_max}]...")
    case_a, case_b = filter_diagnoses_for_revision(
        generation_data,
        metrics_data,
        patient_start=args.patient_start,
        patient_end=args.patient_end,
        grounding_min=args.grounding_min,
        grounding_max=args.grounding_max,
    )

    print(f"[worker {args.worker_id}] Case A (correct, ambiguous): {len(case_a)} diagnoses")
    print(f"[worker {args.worker_id}] Case B (incorrect, ambiguous): {len(case_b)} diagnoses")
    print(f"[worker {args.worker_id}] Total to revise: {len(case_a) + len(case_b)}")

    revised_reasoning = {}

    if case_a or case_b:
        # Initialize vLLM reviser
        reviser = VLLMReasoningReviser(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # Revise reasoning
        print(f"[worker {args.worker_id}] Revising reasoning with vLLM...")
        revised_reasoning = revise_diagnoses(
            case_a,
            case_b,
            reviser,
            show_progress=True,
        )

        print(f"[worker {args.worker_id}] Successfully revised {len(revised_reasoning)} diagnoses.")
    else:
        print(f"[worker {args.worker_id}] No diagnoses to revise in this range.")

    # Initialize NLI model and embedder for grounding recalculation (unless skipped)
    nli_model = None
    embedder = None

    if not args.skip_nli and revised_reasoning:
        print(f"[worker {args.worker_id}] Initializing NLI model for grounding recalculation: {args.nli_model}")
        from evidence_rl.evidence_pipeline import CrossEncoderNLI
        from evidence_rl.retrieval import HuggingFaceEmbedder

        nli_model = CrossEncoderNLI(
            model_name=args.nli_model,
            batch_size=32,
            force_cpu=True,
        )

        print(f"[worker {args.worker_id}] Initializing embedder: {args.embedding_model}")
        embedder = HuggingFaceEmbedder(model_name=args.embedding_model, force_cpu=True)
    elif args.skip_nli:
        print(f"[worker {args.worker_id}] Skipping NLI grounding recalculation (--skip-nli flag)")

    # Create worker output
    print(f"[worker {args.worker_id}] Creating output files...")
    worker_generation, worker_metrics = create_worker_output(
        generation_data,
        metrics_data,
        revised_reasoning,
        args.patient_start,
        args.patient_end,
        args.worker_id,
        nli_model=nli_model,
        embedder=embedder,
        reward_weight_grounding=args.reward_weight_grounding,
        max_k=args.max_k,
    )

    # Calculate total diagnoses processed vs revised
    results = generation_data.get('results', [])
    total_diagnoses = 0
    for patient_idx in range(args.patient_start, min(args.patient_end, len(results))):
        patient = results[patient_idx]
        structured_output = patient.get('structured_output', {})
        diagnoses = structured_output.get('diagnoses', [])
        total_diagnoses += len(diagnoses)

    skipped = total_diagnoses - len(case_a) - len(case_b)
    num_patients = min(args.patient_end, len(results)) - args.patient_start

    # Add revision statistics
    revision_stats = {
        'total_patients': num_patients,
        'total_diagnoses': total_diagnoses,
        'total_diagnoses_revised': len(revised_reasoning),
        'case_a_count': len(case_a),
        'case_b_count': len(case_b),
        'skipped_count': skipped,
        'inference_engine': 'vllm',
        'nli_recalculated': not args.skip_nli and len(revised_reasoning) > 0,
        'nli_model': args.nli_model if not args.skip_nli else None,
        'reward_weight_grounding': args.reward_weight_grounding,
    }
    worker_generation['revision_stats'] = revision_stats.copy()
    worker_metrics['revision_stats'] = revision_stats.copy()

    # Save outputs
    gen_output = output_dir / f"generation_worker{args.worker_id}.json"
    metrics_output = output_dir / f"metrics_worker{args.worker_id}.json"

    print(f"[worker {args.worker_id}] Saving generation output to: {gen_output}")
    save_json(worker_generation, str(gen_output))

    print(f"[worker {args.worker_id}] Saving metrics output to: {metrics_output}")
    save_json(worker_metrics, str(metrics_output))

    print(f"[worker {args.worker_id}] Done!")


if __name__ == "__main__":
    main()
