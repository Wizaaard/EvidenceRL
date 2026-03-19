#!/usr/bin/env python3
"""Compute BarExam QA metrics: accuracy + NLI grounding (matching medical pipeline).

Two-phase pipeline:
  Phase 1: Accuracy (exact match on answer letter) — CPU only
  Phase 2: NLI grounding (reasoning vs premises) — GPU

Grounding design (mirrors medical domain's Focus-Then-Verify):
  Anchor = fact pattern (question text)
  Sections = retrieved legal passages (parsed from prompt) + gold passage
  Premise_i = anchor + section_i  (one per passage)

  For each reasoning sentence:
    score_i = entailment - contradiction  (range [-1, 1])
    sentence_score = max(score_i across all premises, by abs)
  grounding_max = max(per-sentence scores, by abs)  (strongest sentence)
  grounding_avg = mean(per-sentence scores)          (overall consistency)

Usage:
    python compute_barexam_metrics.py \
        --results-json barexam_output/test/llama8b_rag-v1.2/results.json \
        --output-dir barexam_output/test/llama8b_rag-v1.2/metrics/

    # Without NLI grounding (accuracy only):
    python compute_barexam_metrics.py --results-json ... --no-grounding
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl.domains import get_domain


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _parse_passages_from_prompt(prompt: str) -> list[str]:
    """Extract individual Legal Authority passages from prompt text."""
    parts = re.split(r'\[Legal Authority \d+\]:\n', prompt)
    if len(parts) <= 1:
        return []
    passages = []
    for part in parts[1:]:
        # Each authority ends at the next authority marker or prompt instruction
        text = re.split(r'\n\nCRITICAL|\n\nIMPORTANT|\n\nQuestion:', part)[0].strip()
        if text:
            passages.append(text)
    return passages


def _truncate_premise_for_hypothesis(
    premise: str,
    hypothesis: str,
    tokenizer,
    max_tokens: int = 512,
    special_tokens: int = 3,  # [CLS] ... [SEP] ... [SEP]
) -> str:
    """Truncate premise to guarantee hypothesis fits within max_tokens.

    DeBERTa packs [CLS] premise [SEP] hypothesis [SEP] into max_tokens.
    Default tokenizer truncation clips from the right of the combined input,
    which can destroy the hypothesis. Instead, we fix the hypothesis tokens
    and truncate only the premise.
    """
    hyp_tokens = tokenizer.encode(hypothesis, add_special_tokens=False)
    max_premise_tokens = max_tokens - len(hyp_tokens) - special_tokens
    if max_premise_tokens <= 0:
        return premise[:100]  # degenerate case: hypothesis alone exceeds limit

    prem_tokens = tokenizer.encode(premise, add_special_tokens=False)
    if len(prem_tokens) <= max_premise_tokens:
        return premise  # no truncation needed

    # Truncate premise tokens and decode back to text
    truncated_tokens = prem_tokens[:max_premise_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def compute_grounding(
    results: list[dict],
    nli_model_name: str,
    force_cpu: bool = False,
) -> list[dict]:
    """Compute NLI grounding using anchor+section pattern (matching medical pipeline).

    Premise construction (one per evidence source):
        anchor (fact pattern) + ONE legal passage  →  hypothesis (reasoning sentence)

    Smart truncation: premises are truncated in token-space to guarantee
    the hypothesis (reasoning sentence) is never clipped by the tokenizer.

    Scoring:
        score = P(entailment) - P(contradiction)   range: [-1, 1]
        Per-sentence: max(scores across premises, by abs)
        grounding_max = max(per-sentence scores, by abs)
        grounding_avg = mean(per-sentence scores)

    Returns list of per-case grounding dicts.
    """
    from evidence_rl.evidence_pipeline import CrossEncoderNLI

    nli = CrossEncoderNLI(model_name=nli_model_name, force_cpu=force_cpu)

    # Get the tokenizer from the underlying model for smart truncation
    tokenizer = nli.model.tokenizer

    # Pre-compute all NLI pairs for batched inference
    all_pairs: list[tuple[str, str]] = []
    # Track: (case_idx, sentence_idx, premise_idx) for each pair
    pair_map: list[tuple[int, int, int]] = []

    case_meta: list[dict] = []

    for ci, r in enumerate(results):
        reasoning = ""
        if r.get("structured_output"):
            reasoning = r["structured_output"].get("reasoning", "")

        anchor = r.get("question", "").strip()  # fact pattern = anchor
        gold_passage = (r.get("gold_passage", "") or "").strip()

        # Parse retrieved passages from prompt (may differ from gold)
        retrieved_passages = _parse_passages_from_prompt(r.get("prompt", ""))

        sentences = _split_sentences(reasoning) if reasoning.strip() else []
        case_meta.append({
            "case_id": r["case_id"],
            "num_sentences": len(sentences),
            "has_reasoning": bool(reasoning.strip()),
        })

        if not reasoning.strip():
            continue

        # Build premises: anchor + each section (matching medical pattern)
        raw_premises = []

        # Add retrieved passages as sections
        for passage in retrieved_passages:
            raw_premises.append(f"{anchor}\n\n{passage}".strip())

        # Add gold passage if not already in retrieved
        if gold_passage and gold_passage not in retrieved_passages:
            raw_premises.append(f"{anchor}\n\n{gold_passage}".strip())

        # Fallback: anchor alone if no passages
        if not raw_premises:
            if anchor:
                raw_premises.append(anchor)
            else:
                continue

        # Build all (premise, hypothesis) pairs — cartesian product
        # Smart truncation: truncate premise in token-space per hypothesis
        for si, sent in enumerate(sentences):
            for pi, raw_premise in enumerate(raw_premises):
                premise = _truncate_premise_for_hypothesis(
                    raw_premise, sent, tokenizer
                )
                pair_map.append((ci, si, pi))
                all_pairs.append((premise, sent))

    # Report truncation stats
    n_truncated = 0
    for premise, hyp in all_pairs:
        combined_len = len(tokenizer.encode(premise, add_special_tokens=False)) + \
                       len(tokenizer.encode(hyp, add_special_tokens=False)) + 3
        if combined_len > 512:
            n_truncated += 1
    if all_pairs:
        print(f"  Pairs exceeding 512 tokens after smart truncation: {n_truncated}/{len(all_pairs)} ({n_truncated/len(all_pairs):.1%})")

    # Batch NLI inference
    print(f"  Running NLI on {len(all_pairs)} premise-hypothesis pairs...")
    if all_pairs:
        nli_results = nli.predict(all_pairs)
    else:
        nli_results = []

    # Score = entailment - contradiction (range [-1, 1], matching medical pipeline)
    all_scores = [
        r.get("entailment", 0.0) - r.get("contradiction", 0.0)
        for r in nli_results
    ]

    # Aggregate: per-sentence max-abs across premises, then grounding_max / grounding_avg
    # Build nested structure: case -> sentence -> [scores across premises]
    case_sentence_scores: dict[int, dict[int, list[float]]] = {}
    for (ci, si, pi), score in zip(pair_map, all_scores):
        if ci not in case_sentence_scores:
            case_sentence_scores[ci] = {}
        if si not in case_sentence_scores[ci]:
            case_sentence_scores[ci][si] = []
        case_sentence_scores[ci][si].append(score)

    # Build grounding results
    grounding_results = []
    for ci, meta in enumerate(case_meta):
        gr = {
            "case_id": meta["case_id"],
            "num_sentences": meta["num_sentences"],
        }

        if ci in case_sentence_scores:
            per_sentence = case_sentence_scores[ci]
            # Per-sentence: max across premises by absolute value
            per_sentence_scores = []
            for si in sorted(per_sentence.keys()):
                premise_scores = per_sentence[si]
                best = max(premise_scores, key=abs)
                per_sentence_scores.append(best)

            gr["grounding_max"] = max(per_sentence_scores, key=abs)
            gr["grounding_avg"] = sum(per_sentence_scores) / len(per_sentence_scores)
            gr["per_sentence_scores"] = per_sentence_scores
        else:
            gr["grounding_max"] = None
            gr["grounding_avg"] = None
            gr["per_sentence_scores"] = []

        grounding_results.append(gr)

    return grounding_results


def main():
    parser = argparse.ArgumentParser(description="Compute BarExam metrics")
    parser.add_argument("--results-json", required=True, help="Path to generation results JSON")
    parser.add_argument("--output-dir", default=None, help="Output directory for metrics")
    parser.add_argument("--no-grounding", action="store_true", help="Skip NLI grounding (accuracy only)")
    parser.add_argument("--nli-model", default=None,
                        help="Override NLI model (default: domain default)")
    parser.add_argument("--force-cpu", action="store_true", help="Force NLI model to CPU")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    with results_path.open("r") as f:
        data = json.load(f)

    results = data["results"]
    config = data.get("config", {})
    domain = get_domain("barexam")

    print(f"Loaded {len(results)} results from {results_path}")

    # ── Phase 1: Accuracy ─────────────────────────────────────────────────
    print("\n=== Phase 1: Accuracy ===")

    n_parsed = sum(1 for r in results if r.get("parse_success"))
    n_correct = sum(1 for r in results if r.get("correct", False))
    n_total = len(results)

    # Answer distribution
    answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "unparsed": 0}
    for r in results:
        so = r.get("structured_output")
        if so and so.get("answer"):
            answer_dist[so["answer"]] = answer_dist.get(so["answer"], 0) + 1
        else:
            answer_dist["unparsed"] += 1

    # Per-subject accuracy
    subject_stats = {}
    for r in results:
        subj = r.get("subject", "unknown") or "unknown"
        if not subj.strip():
            subj = "unknown"
        if subj not in subject_stats:
            subject_stats[subj] = {"total": 0, "correct": 0}
        subject_stats[subj]["total"] += 1
        if r.get("correct", False):
            subject_stats[subj]["correct"] += 1

    print(f"  Parse rate: {n_parsed}/{n_total} ({n_parsed/n_total:.1%})")
    print(f"  Accuracy:   {n_correct}/{n_total} ({n_correct/n_total:.1%})")
    print(f"  Answer distribution: {answer_dist}")
    print(f"  Per-subject accuracy:")
    for subj, stats in sorted(subject_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {subj}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    metrics = {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "parse_rate": n_parsed / n_total if n_total else 0.0,
        "total_cases": n_total,
        "correct": n_correct,
        "parsed": n_parsed,
        "answer_distribution": answer_dist,
        "per_subject": {
            subj: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                **stats,
            }
            for subj, stats in sorted(subject_stats.items())
        },
    }

    # ── Phase 2: NLI Grounding ────────────────────────────────────────────
    if not args.no_grounding:
        print("\n=== Phase 2: NLI Grounding ===")
        t0 = time.time()

        nli_model = args.nli_model or domain.nli_model_name
        print(f"  NLI model: {nli_model}")
        print(f"  Scoring: entailment - contradiction (range [-1, 1])")

        grounding_results = compute_grounding(
            results,
            nli_model_name=nli_model,
            force_cpu=args.force_cpu,
        )

        elapsed = time.time() - t0

        # Aggregate grounding_max and grounding_avg across cases
        g_max_scores = [g["grounding_max"] for g in grounding_results if g["grounding_max"] is not None]
        g_avg_scores = [g["grounding_avg"] for g in grounding_results if g["grounding_avg"] is not None]

        grounding_summary = {
            "grounding_max": {
                "mean": round(sum(g_max_scores) / len(g_max_scores), 4) if g_max_scores else 0.0,
                "n_cases": len(g_max_scores),
            },
            "grounding_avg": {
                "mean": round(sum(g_avg_scores) / len(g_avg_scores), 4) if g_avg_scores else 0.0,
                "n_cases": len(g_avg_scores),
            },
            "grounding_time_sec": elapsed,
        }

        print(f"  grounding_max (mean): {grounding_summary['grounding_max']['mean']:.4f}  ({len(g_max_scores)} cases)")
        print(f"  grounding_avg (mean): {grounding_summary['grounding_avg']['mean']:.4f}  ({len(g_avg_scores)} cases)")
        print(f"  Grounding time: {elapsed:.1f}s")

        metrics["grounding"] = grounding_summary

        # Merge grounding into per-case results
        grounding_by_id = {g["case_id"]: g for g in grounding_results}
        for r in results:
            g = grounding_by_id.get(r["case_id"], {})
            r["num_sentences"] = g.get("num_sentences", 0)
            r["grounding_max"] = g.get("grounding_max")
            r["grounding_avg"] = g.get("grounding_avg")

    # ── Save ──────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "enhanced_metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"config": config, "metrics": metrics, "per_case": results}, f, indent=2)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_dir}")
    print(f"  Summary: {summary_path}")
    print(f"  Full:    {metrics_path}")


if __name__ == "__main__":
    main()
