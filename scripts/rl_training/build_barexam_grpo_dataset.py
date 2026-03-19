#!/usr/bin/env python3
"""
Build GRPO training dataset for BarExam QA from the train split.

Each entry contains:
  - prompt: The question prompt in chat format (RAG with gold passage)
  - gold_answer: Correct answer letter (A/B/C/D)
  - gold_passage: Gold legal passage for grounding reward
  - question_text: Full question text (for NLI premise construction)

Source: Loads directly from BarExam CSV data (no generation outputs needed).

Usage:
    python build_barexam_grpo_dataset.py --output-dir training_data/barexam_grpo/
    python build_barexam_grpo_dataset.py --output-dir training_data/barexam_grpo/ --mode no-rag
    python build_barexam_grpo_dataset.py --stats-only
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # evidenceRL/
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.evidence_rl.barexam_data import load_barexam_cases
from src.evidence_rl.domains.barexam import BarExamDomain

BAREXAM_DATA_PATH = PROJECT_ROOT / "data" / "barexam"
DOMAIN = BarExamDomain()


def _format_question_with_choices(case) -> str:
    """Format question with preamble and choices (same as run_barexam_vllm.py)."""
    parts = []
    if case.prompt and str(case.prompt).strip() and str(case.prompt).lower() != "nan":
        parts.append(str(case.prompt).strip())
    parts.append(case.question.strip())
    question_text = "\n\n".join(parts)

    choices_text = "\n".join([
        f"(A) {case.choices['A']}",
        f"(B) {case.choices['B']}",
        f"(C) {case.choices['C']}",
        f"(D) {case.choices['D']}",
    ])

    return f"{question_text}\n\n{choices_text}"


def build_barexam_grpo_prompts(split: str, mode: str) -> list[dict]:
    """Build GRPO prompt entries from BarExam data."""
    cases = load_barexam_cases(str(BAREXAM_DATA_PATH), split=split)

    prompts = []
    for case in cases:
        context = _format_question_with_choices(case)
        gold_passage = (case.gold_passage or "").strip()

        if mode == "rag":
            if not gold_passage:
                # Skip cases without gold passage for RAG mode
                continue
            prompt_text = DOMAIN.build_rag_prompt(context, [gold_passage])
        else:
            prompt_text = DOMAIN.build_norag_prompt(context)

        prompts.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "gold_answer": case.answer.strip().upper(),
            "gold_passage": gold_passage,
            "question_text": context,
        })

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Build GRPO training dataset for BarExam QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--mode", choices=["rag", "no-rag"], default="rag",
                        help="Prompt mode (default: rag with gold passage)")
    parser.add_argument("--split", default="train",
                        help="Data split: train, validation, or train+val (default: train)")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    print(f"{'#' * 70}")
    print(f"# Building BarExam GRPO dataset: {args.mode.upper()} mode, split={args.split}")
    print(f"{'#' * 70}")

    # Load prompts from specified split(s)
    if args.split == "train+val":
        prompts_train = build_barexam_grpo_prompts("train", args.mode)
        prompts_val = build_barexam_grpo_prompts("validation", args.mode)
        prompts = prompts_train + prompts_val
        print(f"  Train: {len(prompts_train)}, Validation: {len(prompts_val)}")
    else:
        prompts = build_barexam_grpo_prompts(args.split, args.mode)

    print(f"  Total prompts: {len(prompts)}")

    # Stats
    answer_dist = {}
    for p in prompts:
        a = p["gold_answer"]
        answer_dist[a] = answer_dist.get(a, 0) + 1
    print(f"  Answer distribution: {dict(sorted(answer_dist.items()))}")

    has_passage = sum(1 for p in prompts if p["gold_passage"])
    print(f"  With gold passage: {has_passage}/{len(prompts)}")

    prompt_lengths = [len(p["prompt"][0]["content"]) for p in prompts]
    avg_len = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    print(f"  Avg prompt length (chars): {avg_len:.0f}")
    print(f"  Min/max prompt length: {min(prompt_lengths)}/{max(prompt_lengths)}")

    # Save
    if not args.stats_only and args.output_dir:
        random.shuffle(prompts)

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mode_tag = args.mode.replace("-", "_")
        out_file = output_path / f"barexam_grpo_{mode_tag}_{args.split}_prompts.json"

        output_data = {
            "config": {
                "domain": "barexam",
                "mode": args.mode,
                "split": args.split,
                "num_prompts": len(prompts),
                "seed": args.seed,
            },
            "prompts": prompts,
        }

        with open(out_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n  Saved {len(prompts)} prompts to: {out_file}")

    elif not args.stats_only:
        print("\n  No --output-dir specified.")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
