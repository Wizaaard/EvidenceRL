"""Example entry point for running the EvidenceRL pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl import Document, RagRlPipeline

SAMPLE_DOCUMENTS = [
    Document(
        doc_id="preeclampsia-prevention",
        text="Begin low-dose aspirin at 12 weeks gestation in patients at high risk of preeclampsia.",
    ),
    Document(
        doc_id="htn-background",
        text="Chronic hypertension complicates up to two percent of pregnancies and increases adverse outcomes.",
    ),
    Document(
        doc_id="htn-first-line",
        text="ACOG guidelines recommend labetalol as a first-line oral agent for chronic hypertension in pregnancy.",
    ),
    Document(
        doc_id="htn-dosing",
        text="Typical labetalol dosing starts at 100 mg twice daily with titration every few days.",
    ),
    Document(
        doc_id="htn-second-line",
        text="Nifedipine extended release is an effective alternative when labetalol is contraindicated.",
    ),
    Document(
        doc_id="htn-severe",
        text="Severe-range blood pressures require intravenous therapy such as labetalol or hydralazine.",
    ),
]

SAMPLE_CASE = {
    "query": "What medication is first-line for treating chronic hypertension during pregnancy?",
    "ground_truth": "Labetalol is recommended as first-line therapy for chronic hypertension in pregnancy.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EvidenceRL demo pipeline")
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Optional Hugging Face model identifier to use for answer generation. "
            "Defaults to a small GPT-2 checkpoint."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        pipeline = RagRlPipeline(SAMPLE_DOCUMENTS, top_k=2, model_name=args.model_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - user feedback path
        if exc.name == "transformers":
            print("Install the 'transformers' package to run the demo (e.g., pip install transformers).")
            return
        raise
    result = pipeline.run(SAMPLE_CASE["query"], ground_truth=SAMPLE_CASE["ground_truth"])

    print(f"Query: {result.query}")
    print(f"Ground truth: {SAMPLE_CASE['ground_truth']}")
    print("Pre-generation evidence:")
    for item in result.pre_evidence:
        print(f"  - {item.document.doc_id}: score={item.score:.3f}")
    print(f"Generated answer: {result.generated_answer}")
    print("Post-generation evidence:")
    for item in result.post_evidence:
        print(f"  - {item.document.doc_id}: score={item.score:.3f}")
    print(f"Alignment score: {result.alignment_score:.3f}")
    print(f"Correct answer: {result.is_correct}")
    print(f"Reward: {result.reward:.3f}")


if __name__ == "__main__":
    main()
