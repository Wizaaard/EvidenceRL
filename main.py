"""Example entry point for running the EvidenceRL pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl import Document, RagRlPipeline, RagRlResult  # noqa: E402

SAMPLE_DOCUMENTS: List[Document] = [
    Document(
        doc_id="htn-overview",
        text="Chronic hypertension complicates roughly one to two percent of pregnancies.",
    ),
    Document(
        doc_id="htn-therapy",
        text="Labetalol is recommended as a first-line oral agent for chronic hypertension in pregnancy.",
    ),
    Document(
        doc_id="htn-dosing",
        text="Initial labetalol dosing is commonly 100 mg twice daily with titration as needed.",
    ),
    Document(
        doc_id="htn-alt",
        text="Nifedipine extended release can be used when labetalol is contraindicated.",
    ),
    Document(
        doc_id="htn-severe",
        text="Severe hypertension with end-organ symptoms warrants intravenous therapy.",
    ),
    Document(
        doc_id="preeclampsia-ppx",
        text="Low-dose aspirin beginning in the late first trimester reduces preeclampsia risk.",
    ),
    Document(
        doc_id="preeclampsia-signs",
        text="Persistent headache, visual changes, and right upper quadrant pain suggest preeclampsia.",
    ),
    Document(
        doc_id="proteinuria",
        text="Proteinuria above 300 mg in 24 hours fulfills diagnostic criteria for preeclampsia.",
    ),
    Document(
        doc_id="gest-diabetes",
        text="Screen for gestational diabetes between 24 and 28 weeks with a glucose challenge test.",
    ),
    Document(
        doc_id="gest-diabetes-diet",
        text="Dietary modification and glucose monitoring are first-line for gestational diabetes.",
    ),
]

SAMPLE_CASES = [
    {
        "query": "What medication is first-line for treating chronic hypertension during pregnancy?",
        "ground_truth": "Labetalol is recommended as a first-line oral agent for chronic hypertension in pregnancy.",
    },
    {
        "query": "Which medication prevents preeclampsia in high-risk pregnancies?",
        "ground_truth": "Low-dose aspirin beginning in the late first trimester reduces preeclampsia risk.",
    },
    {
        "query": "What symptoms raise suspicion for preeclampsia?",
        "ground_truth": "Persistent headache, visual changes, and right upper quadrant pain suggest preeclampsia.",
    },
    {
        "query": "How is gestational diabetes initially managed?",
        "ground_truth": "Dietary modification and glucose monitoring are first-line for gestational diabetes.",
    },
    {
        "query": "Which oral agent can replace labetalol when contraindicated?",
        "ground_truth": "Nifedipine extended release can be used when labetalol is contraindicated.",
    },
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
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory where reward/alignment distribution plots will be written.",
    )
    return parser


def format_case_output(result: RagRlResult, ground_truth: str) -> str:
    pre_lines = "\n".join(
        f"  - {item.document.doc_id}: score={item.score:.3f}" for item in result.pre_evidence
    )
    post_lines = "\n".join(
        f"  - {item.document.doc_id}: score={item.score:.3f}" for item in result.post_evidence
    )
    return (
        f"Query: {result.query}\n"
        f"Ground truth: {ground_truth}\n"
        "Pre-generation evidence:\n"
        f"{pre_lines if pre_lines else '  (none)'}\n"
        f"Generated answer: {result.generated_answer}\n"
        "Post-generation evidence:\n"
        f"{post_lines if post_lines else '  (none)'}\n"
        f"Alignment score: {result.alignment_score:.3f}\n"
        f"Correct answer: {result.is_correct}\n"
        f"Reward: {result.reward:.3f}\n"
    )


def run_cases(pipeline: RagRlPipeline, cases: Iterable[dict[str, str]]) -> List[RagRlResult]:
    results: List[RagRlResult] = []
    for case in cases:
        result = pipeline.run(case["query"], ground_truth=case.get("ground_truth"))
        results.append(result)
        print(format_case_output(result, case.get("ground_truth", "")))
    return results


def plot_distributions(results: Iterable[RagRlResult], output_dir: Path | str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install matplotlib to enable plotting support.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filtered = [
        (
            1 if result.is_correct else 0,
            result.reward,
            result.alignment_score,
        )
        for result in results
        if result.is_correct is not None
    ]
    if not filtered:
        return

    correctness = [item[0] for item in filtered]
    rewards = [item[1] for item in filtered]
    alignments = [item[2] for item in filtered]

    def _scatter(values: List[float], title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        jitter = [(-0.05 if idx % 2 == 0 else 0.05) for idx in range(len(values))]
        xs = [c + j for c, j in zip(correctness, jitter)]
        ax.scatter(xs, values, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Correctness")
        ax.set_ylabel("Score")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"])
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path / filename, bbox_inches="tight")
        plt.close(fig)

    _scatter(rewards, "Reward vs correctness", "reward_vs_correctness.png")
    _scatter(alignments, "Alignment vs correctness", "alignment_vs_correctness.png")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        pipeline = RagRlPipeline(SAMPLE_DOCUMENTS, top_k=3, model_name=args.model_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - user feedback path
        if exc.name == "transformers":
            print("Install the 'transformers' package to run the demo (e.g., pip install transformers).")
            return
        raise

    results = run_cases(pipeline, SAMPLE_CASES)

    if args.plot_dir:
        plot_distributions(results, args.plot_dir)
        print(f"Saved plots to {Path(args.plot_dir).resolve()}")


if __name__ == "__main__":
    main()
