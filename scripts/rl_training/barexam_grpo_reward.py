#!/usr/bin/env python3
"""
GRPO reward functions for BarExam QA.

Three reward components compatible with TRL GRPOTrainer's reward_funcs interface:
  1. R_format:    Valid JSON with answer (A/B/C/D) + reasoning → {0, 1}
  2. R_accuracy:  Exact match on answer letter → {0, 1}
  3. R_grounding: NLI entailment score for reasoning vs gold passage → [0, 1]

TRL reward function signature:
    fn(completions, prompts, **dataset_columns) -> list[float]

NLI model (DeBERTa-v3-large) runs on GPU during reward computation
while vLLM is in sleep mode.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # code/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_rl.evidence_pipeline import CrossEncoderNLI


# ── Singleton holders (loaded once, reused across calls) ──────────────────

_nli_model: Optional[CrossEncoderNLI] = None
_nli_tokenizer = None


def _get_nli_model() -> CrossEncoderNLI:
    """Get or create the singleton NLI model (GPU).

    Safe during reward computation because vLLM is in sleep mode.
    """
    global _nli_model, _nli_tokenizer
    if _nli_model is None:
        import torch
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        print(f"[barexam_grpo_reward] Loading NLI model ({device})...")
        _nli_model = CrossEncoderNLI(
            model_name="cross-encoder/nli-deberta-v3-large",
            batch_size=64,
            device=device,
        )
        _nli_tokenizer = _nli_model.model.tokenizer
    return _nli_model


def _get_nli_tokenizer():
    """Get the NLI tokenizer (loads model if not already loaded)."""
    global _nli_tokenizer
    if _nli_tokenizer is None:
        _get_nli_model()
    return _nli_tokenizer


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_completion_text(completion) -> str:
    """Extract raw text from a TRL completion (handles all formats)."""
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    elif isinstance(completion, dict):
        return completion.get("content", "") or completion.get("text", "")
    else:
        return str(completion)


def _parse_barexam_output(text: str) -> Optional[dict]:
    """Parse BarExam JSON output: {"answer": "X", "reasoning": "..."}."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    answer = str(data.get("answer", "")).strip().upper()
    reasoning = str(data.get("reasoning", "")).strip()

    if answer not in ("A", "B", "C", "D"):
        return None

    return {"answer": answer, "reasoning": reasoning}


def _split_sentences(text: str) -> list[str]:
    """Split reasoning into sentences for sentence-level NLI."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _truncate_premise_for_hypothesis(
    premise: str,
    hypothesis: str,
    tokenizer,
    max_tokens: int = 512,
    special_tokens: int = 3,
) -> str:
    """Truncate premise to guarantee hypothesis fits within max_tokens.

    DeBERTa packs [CLS] premise [SEP] hypothesis [SEP] into max_tokens.
    We fix the hypothesis tokens and truncate only the premise.
    """
    hyp_tokens = tokenizer.encode(hypothesis, add_special_tokens=False)
    max_premise_tokens = max_tokens - len(hyp_tokens) - special_tokens
    if max_premise_tokens <= 0:
        return premise[:100]

    prem_tokens = tokenizer.encode(premise, add_special_tokens=False)
    if len(prem_tokens) <= max_premise_tokens:
        return premise

    truncated_tokens = prem_tokens[:max_premise_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


# ── Reward Function 1: Format Compliance ─────────────────────────────────

def format_reward(completions: list[str], prompts: list, **kwargs) -> list[float]:
    """Check if completion is valid JSON with answer (A/B/C/D) + non-empty reasoning.

    Returns 1.0 for valid format, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        text = _extract_completion_text(completion)
        parsed = _parse_barexam_output(text)

        if parsed is None:
            rewards.append(0.0)
        elif not parsed["reasoning"]:
            rewards.append(0.0)
        else:
            rewards.append(1.0)

    return rewards


# ── Reward Function 2: Answer Accuracy ───────────────────────────────────

def accuracy_reward(
    completions: list[str],
    prompts: list,
    gold_answer: list = None,
    **kwargs,
) -> list[float]:
    """Exact match on answer letter. Returns 1.0 (correct) or 0.0 (incorrect)."""
    if gold_answer is None:
        return [0.0] * len(completions)

    rewards = []
    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        parsed = _parse_barexam_output(text)

        if parsed is None:
            rewards.append(0.0)
            continue

        gt = gold_answer[i] if i < len(gold_answer) else ""
        if parsed["answer"] == gt.strip().upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# ── Reward Function 3: Evidence Grounding ────────────────────────────────

def grounding_reward(
    completions: list[str],
    prompts: list,
    gold_passage: list = None,
    question_text: list = None,
    **kwargs,
) -> list[float]:
    """Score evidence grounding using sentence-level NLI.

    For each completion:
      1. Parse reasoning text
      2. Split reasoning into sentences (hypotheses)
      3. Build premises: question_text + gold_passage
      4. Smart-truncate each premise to preserve hypothesis tokens
      5. NLI score = entailment - contradiction per (premise, hypothesis)
      6. Per-sentence: max score by abs across premises
      7. Final: mean of per-sentence scores
      8. Normalize from [-1, 1] to [0, 1]

    Returns [0, 1] score per completion.
    """
    if gold_passage is None or question_text is None:
        return [0.0] * len(completions)

    nli = _get_nli_model()
    tokenizer = _get_nli_tokenizer()

    # Collect all NLI pairs across the batch for batched inference
    all_pairs: list[tuple[str, str]] = []
    # Map: (completion_idx, sentence_idx, premise_idx)
    pair_map: list[tuple[int, int, int]] = []
    # Track which completions have valid reasoning
    completion_info: list[Optional[list[str]]] = []  # sentences per completion

    for i, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        parsed = _parse_barexam_output(text)

        if parsed is None or not parsed["reasoning"]:
            completion_info.append(None)
            continue

        sentences = _split_sentences(parsed["reasoning"])
        if not sentences:
            completion_info.append(None)
            continue

        completion_info.append(sentences)

        # Build premises
        qt = question_text[i] if i < len(question_text) else ""
        gp = gold_passage[i] if i < len(gold_passage) else ""

        raw_premises = []
        if qt and gp:
            raw_premises.append(f"{qt}\n\n{gp}".strip())
        elif qt:
            raw_premises.append(qt)
        elif gp:
            raw_premises.append(gp)
        else:
            completion_info[-1] = None
            continue

        # Build (premise, hypothesis) pairs with smart truncation
        for si, sent in enumerate(sentences):
            for pi, raw_premise in enumerate(raw_premises):
                premise = _truncate_premise_for_hypothesis(
                    raw_premise, sent, tokenizer
                )
                pair_map.append((i, si, pi))
                all_pairs.append((premise, sent))

    # Batch NLI inference
    if all_pairs:
        nli_results = nli.predict(all_pairs)
        all_scores = [
            r.get("entailment", 0.0) - r.get("contradiction", 0.0)
            for r in nli_results
        ]
    else:
        all_scores = []

    # Aggregate scores per completion
    rewards = []
    score_idx = 0

    for i in range(len(completions)):
        sentences = completion_info[i] if i < len(completion_info) else None
        if sentences is None:
            rewards.append(0.0)
            continue

        # Collect per-sentence scores
        per_sentence: dict[int, list[float]] = {}
        while score_idx < len(all_scores):
            ci, si, pi = pair_map[score_idx]
            if ci != i:
                break
            per_sentence.setdefault(si, []).append(all_scores[score_idx])
            score_idx += 1

        if not per_sentence:
            rewards.append(0.0)
            continue

        # Per-sentence: max by abs across premises
        per_sentence_scores = []
        for si in sorted(per_sentence.keys()):
            scores = per_sentence[si]
            best = max(scores, key=abs)
            per_sentence_scores.append(best)

        # Average across sentences
        g_avg = sum(per_sentence_scores) / len(per_sentence_scores)

        # Normalize from [-1, 1] to [0, 1]
        normalized = (g_avg + 1.0) / 2.0
        rewards.append(normalized)

    return rewards


# ── Convenience: Initialize all models eagerly ───────────────────────────

def initialize_reward_models():
    """Pre-load reward models on GPU. Call at training start."""
    print("[barexam_grpo_reward] Pre-loading reward models (GPU)...")
    _get_nli_model()
    print("[barexam_grpo_reward] NLI model loaded on GPU.")
    print("[barexam_grpo_reward] (No embedder needed — BarExam uses exact match accuracy)")
