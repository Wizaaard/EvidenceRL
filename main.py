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
        doc_id="doc1",
        text=(
            "Reinforcement learning optimizes policies via trial and error. "
            "In retrieval-augmented systems, policy improvements can encourage "
            "models to ground answers in external knowledge bases."
        ),
    ),
    Document(
        doc_id="doc2",
        text=(
            "Retrieval-augmented generation (RAG) pipelines first retrieve "
            "relevant passages and then generate answers conditioned on that "
            "evidence.  High-quality retrieval is essential for factual outputs."
        ),
    ),
    Document(
        doc_id="doc3",
        text=(
            "Reward models can evaluate alignment between generated answers and "
            "ground-truth evidence.  Similarity-based rewards provide a cheap "
            "signal compared to manual labeling."
        ),
    ),
]


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
    query = "How can reinforcement learning encourage better evidence usage in RAG?"
    result = pipeline.run(query)

    print(f"Query: {result.query}")
    print("Pre-generation evidence:")
    for item in result.pre_evidence:
        print(f"  - {item.document.doc_id}: score={item.score:.3f}")
    print(f"Generated answer: {result.generated_answer}")
    print("Post-generation evidence:")
    for item in result.post_evidence:
        print(f"  - {item.document.doc_id}: score={item.score:.3f}")
    print(f"Reward: {result.reward:.3f}")


if __name__ == "__main__":
    main()
