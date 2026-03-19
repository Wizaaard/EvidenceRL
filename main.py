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
    RAGPredictor,
    RagRlPipeline,
    RagRlResult,
    export_documents_jsonl,
    load_cardiac_icd_documents,
    load_documents_from_jsonl,
    load_documents_from_hf_dataset,
    load_patient_cases,
    load_pdf_knowledge_documents,
    summarise_predictions,
)  # noqa: E402



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
            "Optional path to the MIMIC-IV-Ext cardiac dataset directory to run patient-focused pipelines "
            "for diagnoses and procedures."
        ),
    )
    parser.add_argument(
        "--patient-pipeline",
        choices=["baseline", "rag"],
        default="baseline",
        help=(
            "Choose whether to run the prompt-only patient baseline or the RAG pipeline when "
            "--patient-data-path is provided."
        ),
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit the number of patient cases processed when using patient data.",
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
        default=8,
        help="Batch size for prompt-only baseline generation (for GPU efficiency).",
    )
    parser.add_argument(
        "--knowledge-path",
        default=None,
        help=(
            "Optional directory containing clinical guideline PDFs (or .txt files). "
            "When provided, the RAG demo will chunk these into retrievable documents."
        ),
    )
    parser.add_argument(
        "--knowledge-dataset",
        default=None,
        help=(
            "Optional Hugging Face dataset name to load guideline chunks from, e.g. "
            "ilyassacha/cardiologyChunks. If provided, RAG retrieval will source documents "
            "directly from the dataset split instead of local PDFs/JSONL."
        ),
    )
    parser.add_argument(
        "--knowledge-dataset-split",
        default="train",
        help="Dataset split to load when using --knowledge-dataset (default: train).",
    )
    parser.add_argument(
        "--knowledge-dataset-text-field",
        default="text",
        help="Name of the text field containing chunks in the Hugging Face dataset (default: text).",
    )
    parser.add_argument(
        "--knowledge-dataset-max-records",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of records to load from the Hugging Face dataset to "
            "avoid pulling millions of chunks when experimenting."
        ),
    )
    parser.add_argument(
        "--knowledge-jsonl",
        default=None,
        help=(
            "Optional JSONL path for knowledge chunks. If the file exists it will be loaded; "
            "otherwise the processed chunks from --knowledge-path will be exported there."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Token-ish chunk size for guideline ingestion when using --knowledge-path.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="Overlap between chunks for guideline ingestion when using --knowledge-path.",
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


def load_documents_from_args(args) -> List[Document]:
    if args.knowledge_dataset:
        return load_documents_from_hf_dataset(
            args.knowledge_dataset,
            split=args.knowledge_dataset_split,
            text_field=args.knowledge_dataset_text_field,
            max_records=args.knowledge_dataset_max_records,
        )

    knowledge_jsonl_path = Path(args.knowledge_jsonl) if args.knowledge_jsonl else None
    if knowledge_jsonl_path and knowledge_jsonl_path.exists():
        return load_documents_from_jsonl(knowledge_jsonl_path)

    if args.knowledge_path:
        documents = load_pdf_knowledge_documents(
            args.knowledge_path,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        if knowledge_jsonl_path:
            export_documents_jsonl(documents, knowledge_jsonl_path)
            print(f"Exported chunked knowledge to {knowledge_jsonl_path.resolve()}")
        return documents

    if args.data_path:
        return load_cardiac_icd_documents(args.data_path)

    raise ValueError("No documents to load")


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
            case_id = case.get("case_id") or f"case_{index:02d}"
            json_path = save_dir / f"{case_id}.json"
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

    patient_cases = None
    if args.patient_data_path:
        try:
            patient_cases = load_patient_cases(args.patient_data_path, limit=args.max_patients)
        except FileNotFoundError as exc:  # pragma: no cover - user feedback path
            print(str(exc))
            return

    if patient_cases and args.patient_pipeline == "baseline":
        predictor = PromptingPredictor(
            model_name=args.model_name,
            judge_model_name=args.judge_model_name,
        )
        predictions = predictor.predict_many(patient_cases, batch_size=args.batch_size)
        summary = summarise_predictions(predictions)

        print("Averaged precision/recall:")
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value:.3f}")

        if args.json_output:
            output_path = Path(args.json_output)
            judge_output_path = (
                output_path.with_stem(output_path.stem + "_judge_details")
                if output_path.suffix
                else output_path / "judge_details.json"
            )
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

            judge_payload = {
                "model_name": getattr(predictor.generator, "model_name", args.model_name),
                "judge_model_name": getattr(predictor.judge, "model_name", args.judge_model_name),
                "cases": [
                    {
                        "hadm_id": pred.hadm_id,
                        "diagnoses": [detail.to_dict() for detail in pred.diagnoses_judge_details],
                        "procedures": [detail.to_dict() for detail in pred.procedures_judge_details],
                    }
                    for pred in predictions
                ],
            }
            judge_output_path.parent.mkdir(parents=True, exist_ok=True)
            with judge_output_path.open("w", encoding="utf-8") as handle:
                json.dump(judge_payload, handle, indent=2, ensure_ascii=False)
            print(f"Saved judge details to {judge_output_path.resolve()}")
        return

    documents = load_documents_from_args(args)
    
    if patient_cases and args.patient_pipeline == "rag":
        predictor = RAGPredictor(
            model_name=args.model_name,
            judge_model_name=args.judge_model_name,
            embedding_model_name=args.embedding_model_name,
            documents=documents
        )
        predictions = predictor.predict_many(patient_cases, batch_size=args.batch_size)
        summary = summarise_predictions(predictions)

        print("Averaged precision/recall:")
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value:.3f}")

        if args.json_output:
            output_path = Path(args.json_output)
            judge_output_path = (
                output_path.with_stem(output_path.stem + "_judge_details")
                if output_path.suffix
                else output_path / "judge_details.json"
            )
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

            judge_payload = {
                "model_name": getattr(predictor.generator, "model_name", args.model_name),
                "judge_model_name": getattr(predictor.judge, "model_name", args.judge_model_name),
                "cases": [
                    {
                        "hadm_id": pred.hadm_id,
                        "diagnoses": [detail.to_dict() for detail in pred.diagnoses_judge_details],
                        "procedures": [detail.to_dict() for detail in pred.procedures_judge_details],
                    }
                    for pred in predictions
                ],
            }
            judge_output_path.parent.mkdir(parents=True, exist_ok=True)
            with judge_output_path.open("w", encoding="utf-8") as handle:
                json.dump(judge_payload, handle, indent=2, ensure_ascii=False)
            print(f"Saved judge details to {judge_output_path.resolve()}")
        return


if __name__ == "__main__":
    main()
