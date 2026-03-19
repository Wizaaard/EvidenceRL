"""Data loading utilities for the BarExam QA benchmark.

Dataset: reglab/barexam_qa (Stanford RegLab)
  - 1,195 MBE (Multistate Bar Examination) questions
  - 857K legal passage corpus for retrieval
  - Each question has a gold_passage for grounding evaluation

Format:
  QA CSV columns: idx, dataset, source, subject, prompt (preamble),
    question, choice_a/b/c/d, answer (A/B/C/D), gold_passage, gold_idx
  Passages TSV columns: idx, source, faiss_id, text, ...
"""

from __future__ import annotations

import csv
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BarExamCase:
    """A single BarExam QA question."""
    case_id: str
    prompt: str           # preamble / fact pattern
    question: str         # the actual question
    choices: dict         # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str           # correct choice letter (A/B/C/D)
    gold_passage: str     # gold evidence passage
    gold_idx: str         # passage index
    source: str = ""
    subject: str = ""

    @property
    def full_question(self) -> str:
        """Combine preamble + question into one text."""
        parts = []
        if self.prompt and str(self.prompt).strip() and str(self.prompt) != "nan":
            parts.append(str(self.prompt).strip())
        parts.append(self.question.strip())
        return "\n\n".join(parts)


def load_barexam_cases(
    data_path: str | Path,
    split: str = "test",
    max_cases: int | None = None,
) -> list[BarExamCase]:
    """Load BarExam QA cases from CSV files.

    Args:
        data_path: Path to barexam data directory (contains data/qa/ subdirectory)
        split: One of "train", "validation", "test", or "all"
        max_cases: Maximum number of cases to load (None = all)

    Returns:
        List of BarExamCase instances
    """
    data_path = Path(data_path)

    if split == "all":
        csv_path = data_path / "data" / "qa" / "qa.csv"
    else:
        csv_path = data_path / "data" / "qa" / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"BarExam QA file not found: {csv_path}\n"
            f"Download with: python -c \"from evidence_rl.barexam_data import download_barexam; download_barexam('{data_path}')\""
        )

    cases = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = BarExamCase(
                case_id=row["idx"],
                prompt=row.get("prompt", ""),
                question=row["question"],
                choices={
                    "A": row["choice_a"],
                    "B": row["choice_b"],
                    "C": row["choice_c"],
                    "D": row["choice_d"],
                },
                answer=row["answer"],
                gold_passage=row.get("gold_passage", ""),
                gold_idx=row.get("gold_idx", ""),
                source=row.get("source", ""),
                subject=row.get("subject", ""),
            )
            cases.append(case)

            if max_cases and len(cases) >= max_cases:
                break

    return cases


def load_barexam_passages(
    data_path: str | Path,
    split: str = "test",
) -> list[dict]:
    """Load the passage corpus for retrieval.

    Args:
        data_path: Path to barexam data directory
        split: Which passage split to load ("train", "validation", "test", or "all")

    Returns:
        List of dicts with 'idx', 'text', 'faiss_id', etc.
    """
    data_path = Path(data_path)
    passages_dir = data_path / "data" / "passages"

    if split == "all":
        tsv_path = passages_dir / "passages.tsv"
        zip_path = passages_dir / "passages.tsv.zip"
    else:
        tsv_path = passages_dir / f"{split}.tsv"
        zip_path = passages_dir / f"{split}.tsv.zip"

    # Extract from zip if needed
    if not tsv_path.exists() and zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(passages_dir)

    if not tsv_path.exists():
        raise FileNotFoundError(f"Passages file not found: {tsv_path}")

    passages = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            passages.append({
                "idx": row.get("idx", ""),
                "text": row.get("text", ""),
                "faiss_id": row.get("faiss_id", ""),
                "source": row.get("source", ""),
            })

    return passages


def download_barexam(output_dir: str | Path) -> None:
    """Download BarExam QA dataset from HuggingFace.

    Downloads QA CSV files and passage TSV zips from reglab/barexam_qa.
    """
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "reglab/barexam_qa"

    # QA files
    qa_files = [
        "data/qa/qa.csv",
        "data/qa/test.csv",
        "data/qa/train.csv",
        "data/qa/validation.csv",
    ]
    # Passage files (zip)
    passage_files = [
        "data/passages/passages.tsv.zip",
        "data/passages/test.tsv.zip",
        "data/passages/train.tsv.zip",
        "data/passages/validation.tsv.zip",
    ]

    all_files = qa_files + passage_files
    for f in all_files:
        target = output_dir / f
        if target.exists():
            print(f"  Already exists: {f}")
            continue
        print(f"  Downloading: {f}")
        hf_hub_download(repo_id, f, repo_type="dataset", local_dir=str(output_dir))

    print(f"BarExam QA data ready at: {output_dir}")
