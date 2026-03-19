#!/usr/bin/env python3
"""
Compute NLI-based grounding scores for revised reasoning.

This script recalculates grounding scores after reasoning revision,
updating the metrics JSON with new grounding values.

The grounding score measures how well the reasoning is supported by:
- Patient clinical data (context)
- Retrieved clinical evidence

Score range: [-1, 1]
- 1.0: Strongly entailed by evidence
- 0.0: Neutral (neither supported nor contradicted)
- -1.0: Contradicts evidence

Usage:
    python compute_revised_grounding.py \
        --generation-json <path> \
        --metrics-json <path> \
        --output-dir <path>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_rl.documents import Document, RetrievedDocument
from src.evidence_rl.evidence_pipeline import CrossEncoderNLI
from src.evidence_rl.retrieval import HuggingFaceEmbedder
from src.evidence_rl.reward import reward_grounding_cached, PatientContextCache


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def reconstruct_evidence_docs(evidence_list: List[Dict[str, Any]]) -> List[RetrievedDocument]:
    """Reconstruct RetrievedDocument objects from serialized evidence."""
    docs = []
    for ev in evidence_list:
        doc = Document(
            doc_id=ev.get('doc_id', ''),
            text=ev.get('text', ''),
        )
        retrieved = RetrievedDocument(
            document=doc,
            score=ev.get('score', 0.0),
        )
        docs.append(retrieved)
    return docs


def compute_grounding_for_patient(
    patient_data: Dict[str, Any],
    patient_metrics: Dict[str, Any],
    nli_model: "CrossEncoderNLI",
    embedder: "HuggingFaceEmbedder",
) -> List[float]:
    """
    Compute grounding scores for all diagnoses in a patient.

    Args:
        patient_data: Patient data from generation JSON
        patient_metrics: Patient metrics data
        nli_model: NLI model for entailment checking
        embedder: Embedder for section selection

    Returns:
        List of grounding scores, one per diagnosis
    """
    patient_context = patient_data.get('patient_context', '')
    evidence_list = patient_data.get('pre_evidence', [])

    # Reconstruct evidence documents
    evidence_docs = reconstruct_evidence_docs(evidence_list)

    # Create patient context cache for efficiency
    cache = PatientContextCache(patient_context, embedder)

    # Get diagnoses
    structured_output = patient_data.get('structured_output', {})
    diagnoses = structured_output.get('diagnoses', [])

    grounding_scores = []

    for diagnosis in diagnoses:
        reasoning = diagnosis.get('reasoning', '')

        if not reasoning.strip():
            grounding_scores.append(0.0)
            continue

        # Compute grounding using cached context
        grounding_max, grounding_avg = reward_grounding_cached(
            diagnosis_reasoning=reasoning,
            cache=cache,
            evidence_docs=evidence_docs,
            nli_model=nli_model,
            embedder=embedder,
        )

        grounding_scores.append(grounding_max)

    return grounding_scores


def compute_all_grounding(
    generation_data: Dict[str, Any],
    metrics_data: Dict[str, Any],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 8,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute grounding scores for all patients and update metrics.

    Args:
        generation_data: Generation JSON data
        metrics_data: Metrics JSON data to update
        nli_model_name: NLI model name
        embedding_model_name: Embedding model name
        batch_size: Batch size for NLI
        show_progress: Whether to show progress bar

    Returns:
        Updated metrics data with new grounding scores
    """
    import copy

    # Deep copy metrics to avoid modifying original
    updated_metrics = copy.deepcopy(metrics_data)

    # Initialize models
    print(f"Loading NLI model: {nli_model_name}...")
    nli_model = CrossEncoderNLI(model_name=nli_model_name, batch_size=batch_size)

    print(f"Loading embedding model: {embedding_model_name}...")
    embedder = HuggingFaceEmbedder(model_name=embedding_model_name, batch_size=32)

    results = generation_data.get('results', [])
    patient_metrics = updated_metrics.get('patient_metrics', [])

    # Build lookup from hadm_id to patient data
    patient_lookup = {p.get('hadm_id'): p for p in results}

    # Track statistics
    total_diagnoses = 0
    grounding_changes = []

    iterator = patient_metrics
    if show_progress:
        iterator = tqdm(iterator, desc="Computing grounding scores")

    for pm in iterator:
        hadm_id = pm.get('hadm_id')
        patient_data = patient_lookup.get(hadm_id)

        if patient_data is None:
            continue

        # Compute new grounding scores
        new_groundings = compute_grounding_for_patient(
            patient_data,
            pm,
            nli_model,
            embedder,
        )

        # Update per_diagnosis_rewards
        per_diagnosis_rewards = pm.get('per_diagnosis_rewards', [])
        for i, (diag_reward, new_grounding) in enumerate(zip(per_diagnosis_rewards, new_groundings)):
            old_grounding = diag_reward.get('grounding', 0.0)
            diag_reward['grounding'] = new_grounding
            diag_reward['grounding_old'] = old_grounding  # Keep old value for reference

            grounding_changes.append(new_grounding - old_grounding)
            total_diagnoses += 1

        # Update combined rewards
        verdicts = pm.get('verdicts', [])
        for i, diag_reward in enumerate(per_diagnosis_rewards):
            if i < len(new_groundings):
                grounding = new_groundings[i]
                is_correct = verdicts[i] if i < len(verdicts) else False
                precision = 1.0 if is_correct else 0.0
                # Combined reward with default weight 0.5
                diag_reward['combined'] = 0.5 * grounding + 0.5 * precision

    # Add statistics to metrics
    if grounding_changes:
        changes = np.array(grounding_changes)
        updated_metrics['grounding_recalculation'] = {
            'total_diagnoses': total_diagnoses,
            'mean_change': float(np.mean(changes)),
            'std_change': float(np.std(changes)),
            'min_change': float(np.min(changes)),
            'max_change': float(np.max(changes)),
            'nli_model': nli_model_name,
            'embedding_model': embedding_model_name,
        }

    # Mark grounding as recalculated
    if 'revision_info' in updated_metrics:
        updated_metrics['revision_info']['grounding_needs_recalculation'] = False
        updated_metrics['revision_info']['grounding_recalculated'] = True

    return updated_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute NLI-based grounding scores for revised reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--generation-json",
        type=str,
        required=True,
        help="Path to (revised) generation JSON file"
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        required=True,
        help="Path to (revised) metrics JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for updated metrics"
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model name (default: cross-encoder/nli-deberta-v3-base)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for NLI inference (default: 8)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading generation data from: {args.generation_json}")
    generation_data = load_json(args.generation_json)

    print(f"Loading metrics data from: {args.metrics_json}")
    metrics_data = load_json(args.metrics_json)

    # Compute grounding
    print("\nComputing grounding scores...")
    updated_metrics = compute_all_grounding(
        generation_data,
        metrics_data,
        nli_model_name=args.nli_model,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Save updated metrics
    metrics_filename = Path(args.metrics_json).stem
    if not metrics_filename.endswith('_grounded'):
        metrics_filename += '_grounded'
    output_path = output_dir / f"{metrics_filename}.json"

    print(f"\nSaving updated metrics to: {output_path}")
    save_json(updated_metrics, str(output_path))

    # Print summary
    recalc_info = updated_metrics.get('grounding_recalculation', {})
    print("\n" + "=" * 60)
    print("Grounding Recalculation Complete")
    print("=" * 60)
    print(f"Total diagnoses processed: {recalc_info.get('total_diagnoses', 0)}")
    print(f"Mean grounding change: {recalc_info.get('mean_change', 0):.4f}")
    print(f"Std grounding change: {recalc_info.get('std_change', 0):.4f}")
    print(f"Min change: {recalc_info.get('min_change', 0):.4f}")
    print(f"Max change: {recalc_info.get('max_change', 0):.4f}")
    print(f"\nOutput saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
