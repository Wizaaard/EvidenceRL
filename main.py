"""Example entry point for running the EvidenceRL pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evidence_rl import (
    Document,
    PromptingPredictor,
    RagRlPipeline,
    RagRlResult,
    load_cardiac_icd_documents,
    load_patient_cases,
    summarise_predictions,
)  # noqa: E402

CARDIAC_DOCUMENTS: List[Document] = [
    Document(
        doc_id="icd-i20-i25",
        text=(
            "Ischemic heart disease (ICD I20-I25) includes stable angina, unstable angina, and acute myocardial "
            "infarction. Chest pressure radiating to the left arm that improves with nitroglycerin suggests angina."
            " ST-elevation myocardial infarction requires immediate reperfusion therapy and antiplatelet agents."
        ),
        metadata={"concepts": {"ischemic heart disease", "angina", "stemi", "icd i20-i25"}},
    ),
    Document(
        doc_id="icd-i21-stemi-care",
        text=(
            "For STEMI cases, administer aspirin, load a P2Y12 inhibitor, and arrange emergent catheterization. "
            "Elevated troponin with ST elevations and reciprocal depressions indicate transmural ischemia."
        ),
        metadata={"concepts": {"stemi", "aspirin", "p2y12", "cath", "icd i21"}},
    ),
    Document(
        doc_id="icd-i50-heart-failure",
        text=(
            "Heart failure syndromes (ICD I50) present with dyspnea, orthopnea, elevated BNP, and pulmonary edema. "
            "Guideline-directed therapy includes ACE inhibitors, beta blockers, and diuretics for volume overload."
        ),
        metadata={"concepts": {"heart failure", "ace inhibitor", "beta blocker", "diuretic", "icd i50"}},
    ),
    Document(
        doc_id="icd-i47-arrhythmia",
        text=(
            "Arrhythmias (ICD I47) such as ventricular tachycardia cause palpitations and syncope. "
            "Unstable VT requires immediate synchronized cardioversion while stable VT can receive amiodarone."
        ),
        metadata={"concepts": {"ventricular tachycardia", "amiodarone", "cardioversion", "icd i47"}},
    ),
    Document(
        doc_id="icd-i30-pericarditis",
        text=(
            "Acute pericarditis (ICD I30) presents with sharp pleuritic chest pain improved by sitting forward. "
            "Diffuse ST elevations with PR depressions on ECG and a pericardial friction rub support the diagnosis."
        ),
        metadata={"concepts": {"pericarditis", "friction rub", "st elevation", "icd i30"}},
    ),
    Document(
        doc_id="icd-i25-complication",
        text=(
            "Post-myocardial infarction mechanical complications include papillary muscle rupture causing acute "
            "severe mitral regurgitation and heart failure. New holosystolic murmur with pulmonary edema warrants "
            "urgent surgical consultation."
        ),
        metadata={"concepts": {"papillary muscle rupture", "mitral regurgitation", "heart failure", "icd i25"}},
    ),
    Document(
        doc_id="icd-i42-cardiomyopathy",
        text=(
            "Dilated cardiomyopathy (ICD I42) leads to reduced ejection fraction and ventricular dilation. "
            "Alcohol excess, viral myocarditis, and genetic mutations are common causes; treat with standard HF "
            "therapy and consider ICD placement when EF remains low."
        ),
        metadata={"concepts": {"cardiomyopathy", "heart failure", "icd i42"}},
    ),
    Document(
        doc_id="icd-i48-atrial-fibrillation",
        text=(
            "Atrial fibrillation (ICD I48) presents with irregularly irregular rhythm and absent P waves. "
            "Rate control with beta blockers or calcium channel blockers and anticoagulation guided by CHA2DS2-VASc "
            "score reduce stroke risk."
        ),
        metadata={"concepts": {"atrial fibrillation", "anticoagulation", "beta blocker", "icd i48"}},
    ),
]

SAMPLE_CASES = [
    {
        "query": "A patient with crushing chest pain and ST elevations should receive which immediate therapy?",
        "ground_truth": "Administer aspirin, load a P2Y12 inhibitor, and send the patient for emergent catheterization.",
    },
    {
        "query": "How do we manage acute decompensated heart failure with pulmonary edema?",
        "ground_truth": "Use IV diuretics along with guideline-directed medical therapy for heart failure.",
    },
    {
        "query": "Irregularly irregular rhythm without P waves suggests what diagnosis and initial management?",
        "ground_truth": "Atrial fibrillation managed with rate control and anticoagulation guided by CHA2DS2-VASc.",
    },
    {
        "query": "Sharp chest pain relieved by leaning forward with diffuse ST elevations indicates which ICD chapter?",
        "ground_truth": "Acute pericarditis in ICD I30 should be considered.",
    },
    {
        "query": "What rhythm problem causes syncope and may need amiodarone when stable?",
        "ground_truth": "Ventricular tachycardia can cause syncope and stable cases can receive amiodarone.",
    },
    {
        "query": "Which ICD block covers stable angina and myocardial infarction?",
        "ground_truth": "ICD I20-I25 describes ischemic heart disease including angina and MI.",
    },
    {
        "query": "What complication should be suspected after MI with acute severe mitral regurgitation?",
        "ground_truth": "Papillary muscle rupture leading to heart failure is a post-MI complication.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EvidenceRL demo pipeline")
    parser.add_argument(
        "--data-path",
        default=None,
        help=(
            "Optional path to the MIMIC-IV-Ext cardiac dataset directory. When provided, the demo "
            "uses real ICD diagnosis chapters from the CSVs instead of the built-in toy corpus."
        ),
    )
    parser.add_argument(
        "--patient-data-path",
        default=None,
        help=(
            "Optional path to the MIMIC-IV-Ext cardiac dataset directory to run the prompt-only baseline "
            "for diagnoses and procedures."
        ),
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit the number of patient cases processed in baseline mode.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Optional Hugging Face model identifier to use for answer generation. "
            "Defaults to a small GPT-2 checkpoint."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for prompt-only baseline generation (for GPU efficiency).",
    )
    parser.add_argument(
        "--judge-model-name",
        default=None,
        help=(
            "Optional Hugging Face model identifier for the LLM answer judge. "
            "Defaults to the same checkpoint as the generator."
        ),
    )
    parser.add_argument(
        "--embedding-model-name",
        default=None,
        help=(
            "Optional Hugging Face model identifier for dense retrieval embeddings. "
            "Defaults to sentence-transformers/all-MiniLM-L6-v2."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory where reward/alignment distribution plots will be written.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path where pipeline results will be written as JSON (file or directory).",
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


def run_cases(
    pipeline: RagRlPipeline,
    cases: Iterable[dict[str, str]],
    save_dir: Path | None = None,
) -> List[RagRlResult]:
    results: List[RagRlResult] = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, start=1):
        json_path = None
        if save_dir is not None:
            json_path = save_dir / f"case_{index:02d}.json"
        result = pipeline.run(
            case["query"],
            ground_truth=case.get("ground_truth"),
            save_path=json_path,
        )
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

    if args.patient_data_path:
        try:
            cases = load_patient_cases(args.patient_data_path, limit=args.max_patients)
        except FileNotFoundError as exc:  # pragma: no cover - user feedback path
            print(str(exc))
            return

        predictor = PromptingPredictor(
            model_name=args.model_name,
            judge_model_name=args.judge_model_name,
        )
        predictions = predictor.predict_many(cases, batch_size=args.batch_size)
        summary = summarise_predictions(predictions)

        for pred in predictions:
            print(f"HADM {pred.hadm_id} predicted diagnoses: {pred.predicted_diagnoses}")
            print(f"HADM {pred.hadm_id} predicted procedures: {pred.predicted_procedures}\n")

        print("Averaged precision/recall:")
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value:.3f}")

        if args.json_output:
            output_path = Path(args.json_output)
            payload = {
                "model_name": getattr(predictor.generator, "model_name", args.model_name),
                "judge_model_name": getattr(predictor.judge, "model_name", args.judge_model_name),
                "predictions": [pred.to_dict() for pred in predictions],
                "summary": summary,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            print(f"Saved JSON results to {output_path.resolve()}")
        return

    documents: List[Document]
    if args.data_path:
        documents = load_cardiac_icd_documents(args.data_path)
    else:
        documents = CARDIAC_DOCUMENTS

    try:
        pipeline = RagRlPipeline(
            documents,
            top_k=3,
            model_name=args.model_name,
            judge_model_name=args.judge_model_name,
            embedding_model_name=args.embedding_model_name,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - user feedback path
        if exc.name == "transformers":
            print("Install the 'transformers' package to run the demo (e.g., pip install transformers).")
            return
        raise

    per_case_dir: Path | None = None
    summary_path: Path | None = None
    if args.json_output:
        output_path = Path(args.json_output)
        if output_path.suffix.lower() == ".json":
            summary_path = output_path
        else:
            per_case_dir = output_path
            summary_path = output_path / "results.json"

    results = run_cases(pipeline, SAMPLE_CASES, save_dir=per_case_dir)

    if args.plot_dir:
        plot_distributions(results, args.plot_dir)
        print(f"Saved plots to {Path(args.plot_dir).resolve()}")

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": getattr(pipeline.generator, "model_name", args.model_name),
            "judge_model_name": getattr(pipeline.judge, "model_name", args.judge_model_name),
            "results": [result.to_dict() for result in results],
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved JSON results to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
