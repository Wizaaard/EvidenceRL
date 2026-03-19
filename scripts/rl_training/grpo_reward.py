#!/usr/bin/env python3
"""
GRPO reward functions for EvidenceRL.

Three reward components compatible with TRL GRPOTrainer's reward_funcs interface:
  1. R_format:    Valid JSON with 5 named+reasoned diagnoses → {0, 1}
  2. R_accuracy:  Clinical embedding similarity to ground truth → [0, 1]
  3. R_grounding: NLI entailment score for reasoning → [0, 1]

TRL reward function signature:
    fn(completions, prompts, **dataset_columns) -> list[float]

All auxiliary models (NLI, embedder) run on CPU to avoid GPU conflicts with vLLM.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # code/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_rl.evidence_pipeline import CrossEncoderNLI
from src.evidence_rl.reward import PatientContextCache, reward_grounding_cached


# ── Singleton holders for models (loaded once, reused across calls) ──────

_nli_model: Optional[CrossEncoderNLI] = None
_embedder_model = None
_embedder_model_name: Optional[str] = None
_use_sentence_level: bool = True  # True=g_avg (default), False=g_max


def set_sentence_level(enabled: bool):
    """Set whether grounding reward uses sentence-level avg (True) or max (False)."""
    global _use_sentence_level
    _use_sentence_level = enabled
    print(f"[grpo_reward] Grounding reward: {'sentence-level avg (g_avg)' if enabled else 'diagnosis-level max (g_max)'}")


def _get_nli_model() -> CrossEncoderNLI:
    """Get or create the singleton NLI model (GPU).

    Runs on the current CUDA device (set by accelerate per process).
    Safe during reward computation because vLLM is in sleep mode.
    """
    global _nli_model
    if _nli_model is None:
        import torch
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        print(f"[grpo_reward] Loading NLI model ({device})...")
        _nli_model = CrossEncoderNLI(
            model_name="pritamdeka/PubMedBERT-MNLI-MedNLI",
            batch_size=64,
            device=device,
        )
    return _nli_model


def _get_embedder():
    """Get or create the singleton clinical embedder (GPU).

    Runs on the current CUDA device (set by accelerate per process).
    Safe during reward computation because vLLM is in sleep mode.
    """
    global _embedder_model, _embedder_model_name
    if _embedder_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        model_name = "FremyCompany/BioLORD-2023"
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        print(f"[grpo_reward] Loading clinical embedder: {model_name} ({device})...")
        _embedder_model = SentenceTransformer(model_name, device=device)
        _embedder_model_name = model_name
    return _embedder_model


def _extract_completion_text(completion) -> str:
    """Extract raw text from a TRL completion (handles all formats).

    TRL conversational format: completion = [{"role": "assistant", "content": "..."}]
    TRL standard format: completion = "raw text"
    TRL dict format: completion = {"content": "..."}
    """
    if isinstance(completion, list):
        # Conversational: [{"role": "assistant", "content": "..."}]
        return completion[0].get("content", "") if completion else ""
    elif isinstance(completion, dict):
        return completion.get("content", "") or completion.get("text", "")
    else:
        return str(completion)


def _parse_diagnoses(completion_text: str) -> Optional[list[dict]]:
    """Parse diagnoses from a model completion (JSON format).

    Returns list of {"name": str, "reasoning": str} dicts, or None if parse fails.
    """
    text = completion_text.strip()

    # Try to find JSON in the text (model may add preamble)
    # Look for the first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    diagnoses = data.get('diagnoses', [])
    if not isinstance(diagnoses, list):
        return None

    return diagnoses


# ── Reward Function 1: Format Compliance ─────────────────────────────────

def format_reward(completions: list[str], prompts: list, **kwargs) -> list[float]:
    """Check if each completion is valid JSON with 5 named+reasoned diagnoses.

    Returns 1.0 for valid format, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        diagnoses = _parse_diagnoses(text)

        if diagnoses is None:
            rewards.append(0.0)
            continue

        # Check exactly 5 diagnoses
        if len(diagnoses) != 5:
            rewards.append(0.0)
            continue

        # Check each diagnosis has non-empty name and reasoning
        valid = all(
            isinstance(d, dict)
            and d.get('name', '').strip()
            and d.get('reasoning', '').strip()
            for d in diagnoses
        )

        rewards.append(1.0 if valid else 0.0)

    return rewards


# ── Reward Function 2: Diagnosis Accuracy ────────────────────────────────

def accuracy_reward(
    completions: list[str],
    prompts: list,
    ground_truth_diagnoses: list = None,
    **kwargs,
) -> list[float]:
    """Score diagnosis accuracy using clinical embedding similarity (BioLORD-2023).

    For each completion:
      1. Parse top-3 diagnosis names
      2. Compute cosine similarity against each ground truth diagnosis
      3. A predicted diagnosis is "correct" if max similarity > threshold
      4. Return fraction correct in top-3

    Returns [0, 1] score per completion.
    """
    if ground_truth_diagnoses is None:
        return [0.0] * len(completions)

    embedder = _get_embedder()
    similarity_threshold = 0.80
    rewards = []

    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        diagnoses = _parse_diagnoses(text)
        if diagnoses is None or len(diagnoses) < 3:
            rewards.append(0.0)
            continue

        # Get ground truth for this sample
        gt = ground_truth_diagnoses[i] if i < len(ground_truth_diagnoses) else []
        if not gt:
            rewards.append(0.0)
            continue

        # Extract top-3 predicted diagnosis names
        pred_names = [d.get('name', '').strip() for d in diagnoses[:3]]
        pred_names = [n for n in pred_names if n]

        if not pred_names:
            rewards.append(0.0)
            continue

        # Compute embeddings
        all_texts = pred_names + list(gt)
        embeddings = embedder.encode(all_texts, convert_to_numpy=True)

        pred_embs = embeddings[:len(pred_names)]
        gt_embs = embeddings[len(pred_names):]

        # Compute similarity matrix
        # Normalize for cosine similarity
        pred_norms = pred_embs / (np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-8)
        gt_norms = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-8)
        sim_matrix = pred_norms @ gt_norms.T  # (num_pred, num_gt)

        # For each predicted, check if max similarity exceeds threshold
        correct_count = 0
        for j in range(len(pred_names)):
            max_sim = sim_matrix[j].max()
            if max_sim > similarity_threshold:
                correct_count += 1

        rewards.append(correct_count / len(pred_names))

    return rewards


# ── Reward Function 3: Evidence Grounding ────────────────────────────────

def grounding_reward(
    completions: list[str],
    prompts: list,
    patient_context: list = None,
    pre_evidence: list = None,
    **kwargs,
) -> list[float]:
    """Score evidence grounding using NLI (CrossEncoderNLI on CPU).

    For each completion:
      1. Parse top-3 diagnoses
      2. Compute grounding score using existing reward_grounding_cached()
      3. Average grounding_max across top-3
      4. Normalize from [-1, 1] to [0, 1]

    Returns [0, 1] score per completion.
    """
    if patient_context is None:
        return [0.0] * len(completions)

    nli = _get_nli_model()
    rewards = []

    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        diagnoses = _parse_diagnoses(text)
        if diagnoses is None or len(diagnoses) < 3:
            rewards.append(0.0)
            continue

        # Get patient context for this sample
        ctx = patient_context[i] if i < len(patient_context) else ""
        evidence = (pre_evidence[i] if pre_evidence and i < len(pre_evidence) else []) or []

        if not ctx:
            rewards.append(0.0)
            continue

        # Build patient context cache
        cache = PatientContextCache(ctx, embedder=None)

        # Build pseudo-RetrievedDocument objects for evidence
        evidence_docs = _make_evidence_docs(evidence)

        # Compute grounding for top-3 diagnoses
        grounding_scores = []
        for diag in diagnoses[:3]:
            reasoning = diag.get('reasoning', '').strip()
            if not reasoning:
                grounding_scores.append(0.0)
                continue

            g_max, g_avg = reward_grounding_cached(
                diagnosis_reasoning=reasoning,
                cache=cache,
                evidence_docs=evidence_docs,
                nli_model=nli,
                embedder=None,
                sentence_level=_use_sentence_level,
            )
            grounding_scores.append(g_avg if _use_sentence_level else g_max)

        # Average grounding score across top-3
        avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0

        # Normalize from [-1, 1] to [0, 1]
        normalized = (avg_grounding + 1.0) / 2.0
        rewards.append(normalized)

    return rewards


# ── Helper: Create pseudo-RetrievedDocument objects ──────────────────────

class _PseudoDocument:
    """Minimal document object compatible with reward.py's RetrievedDocument."""
    def __init__(self, text: str):
        self.text = text


class _PseudoRetrievedDocument:
    """Minimal wrapper compatible with reward.py's Sequence[RetrievedDocument]."""
    def __init__(self, text: str):
        self.document = _PseudoDocument(text)


def _make_evidence_docs(pre_evidence: list) -> list:
    """Convert pre_evidence dicts to pseudo-RetrievedDocument objects."""
    docs = []
    for item in pre_evidence:
        text = item.get('text', '') if isinstance(item, dict) else str(item)
        if text.strip():
            docs.append(_PseudoRetrievedDocument(text.strip()))
    return docs


# ── Convenience: Initialize all models eagerly ──────────────────────────

def initialize_reward_models():
    """Pre-load all reward models on GPU. Call this at training start to avoid
    lazy-loading during the first reward computation."""
    print("[grpo_reward] Pre-loading reward models (GPU)...")
    _get_nli_model()
    _get_embedder()
    print("[grpo_reward] All reward models loaded on GPU.")
