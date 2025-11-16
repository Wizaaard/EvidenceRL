"""Prompt-only baseline for cardiac diagnoses and procedures.

The module provides a light-weight alternative to the retrieval-heavy pipeline
by directly textualising patient encounters from the MIMIC-IV-Ext cardiac
extracts and prompting an LLM for the top diagnoses and procedures.  The
predictions are evaluated against the ground-truth ICD labels using
precision/recall at different cut-offs.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping

from .generation import DEFAULT_MODEL_NAME, PromptOnlyGenerator


_PATIENT_SECTIONS = (
    ("Chief complaint", "chief_complaint"),
    ("History of present illness", "HPI"),
    ("Physical exam", "physical_exam"),
    ("Invasions", "invasions"),
    ("X-ray", "X-ray"),
    ("CT", "CT"),
    ("Ultrasound", "Ultrasound"),
    ("CATH", "CATH"),
    ("ECG", "ECG"),
    ("MRI", "MRI"),
    ("ECG machine report", "reports"),
)


def _load_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items() if key}


def _textualise_patient(row: Mapping[str, str]) -> str:
    segments: List[str] = []
    prefix = []
    for label in ("note_id", "subject_id", "hadm_id"):
        value = row.get(label, "")
        if value:
            prefix.append(f"{label}: {value}")
    if prefix:
        segments.append("; ".join(prefix))

    for label, key in _PATIENT_SECTIONS:
        value = row.get(key, "")
        if not value:
            continue
        cleaned = value.replace("|", "; ")
        segments.append(f"{label}: {cleaned}")

    if not segments:
        return "Patient details unavailable."
    return "\n".join(segments)


def _parse_ranked_items(text: str, heading: str) -> List[str]:
    pattern = re.compile(r"^\s*(\d+)[\).:-]?\s*(.+)$")
    items: List[str] = []
    capturing = False
    for line in text.splitlines():
        if heading.lower() in line.lower():
            capturing = True
            continue
        if capturing:
            lower_line = line.lower().strip()
            if lower_line.startswith("diagnoses") or lower_line.startswith("procedures"):
                break
            match = pattern.match(line)
            if match:
                items.append(match.group(2).strip())
            elif line.strip():
                items.append(line.strip("- "))
            if len(items) >= 5:
                break
    return items


def _precision_recall_at_k(predicted: List[str], truth: List[str], k: int) -> tuple[float, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    preds = predicted[:k]
    truth_set = {item.lower() for item in truth if item}
    if not preds:
        return 0.0, 0.0
    hits = sum(1 for item in preds if item.lower() in truth_set)
    precision = hits / len(preds)
    recall = hits / len(truth_set) if truth_set else 0.0
    return precision, recall


@dataclass
class PatientCase:
    hadm_id: str
    subject_id: str
    note_id: str
    context: str
    diagnoses: List[str]
    procedures: List[str]


@dataclass
class PromptPrediction:
    hadm_id: str
    generated_text: str
    prompt: str
    predicted_diagnoses: List[str]
    predicted_procedures: List[str]
    diagnoses_precision_at_k: dict[int, float]
    diagnoses_recall_at_k: dict[int, float]
    procedures_precision_at_k: dict[int, float]
    procedures_recall_at_k: dict[int, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "hadm_id": self.hadm_id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "predicted_diagnoses": self.predicted_diagnoses,
            "predicted_procedures": self.predicted_procedures,
            "diagnoses_precision_at_k": self.diagnoses_precision_at_k,
            "diagnoses_recall_at_k": self.diagnoses_recall_at_k,
            "procedures_precision_at_k": self.procedures_precision_at_k,
            "procedures_recall_at_k": self.procedures_recall_at_k,
        }


def load_patient_cases(data_path: str | Path, limit: int | None = None) -> List[PatientCase]:
    base = Path(data_path)
    note_path = base / "heart_diagnoses.csv"
    diag_path = base / "heart_diagnoses_all.csv"
    proc_path = base / "heart_procedures.csv"

    if not (note_path.exists() and diag_path.exists() and proc_path.exists()):
        missing = [path.name for path in (note_path, diag_path, proc_path) if not path.exists()]
        raise FileNotFoundError(f"Missing required files under {base}: {', '.join(missing)}")

    diagnoses_by_hadm: MutableMapping[str, List[str]] = {}
    for row in _load_rows(diag_path):
        hadm_id = row.get("hadm_id", "")
        if not hadm_id:
            continue
        bucket = diagnoses_by_hadm.setdefault(hadm_id, [])
        bucket.append(row.get("long_title") or row.get("icd_code") or "")

    procedures_by_hadm: MutableMapping[str, List[str]] = {}
    for row in _load_rows(proc_path):
        hadm_id = row.get("hadm_id", "")
        if not hadm_id:
            continue
        bucket = procedures_by_hadm.setdefault(hadm_id, [])
        bucket.append(row.get("long_title") or row.get("icd_code") or "")

    cases: List[PatientCase] = []
    for row in _load_rows(note_path):
        hadm_id = row.get("hadm_id", "")
        subject_id = row.get("subject_id", "")
        note_id = row.get("note_id", "")
        if not hadm_id:
            continue
        diagnoses = diagnoses_by_hadm.get(hadm_id, [])
        procedures = procedures_by_hadm.get(hadm_id, [])
        if not diagnoses and not procedures:
            continue
        context = _textualise_patient(row)
        cases.append(
            PatientCase(
                hadm_id=hadm_id,
                subject_id=subject_id,
                note_id=note_id,
                context=context,
                diagnoses=diagnoses,
                procedures=procedures,
            )
        )
        if limit is not None and len(cases) >= limit:
            break

    if not cases:
        raise ValueError("No patient cases with ground-truth diagnoses or procedures were found.")
    return cases


class PromptingPredictor:
    """Prompt an LLM for diagnoses and procedures using textualised patient data."""

    def __init__(
        self,
        model_name: str | None = None,
        generation_kwargs: Mapping[str, object] | None = None,
        text_pipeline=None,
    ) -> None:
        self.generator = PromptOnlyGenerator(
            model_name=model_name or DEFAULT_MODEL_NAME,
            generation_kwargs=generation_kwargs,
            text_pipeline=text_pipeline,
        )

    def _build_prompt(self, case: PatientCase) -> str:
        return (
            "You are a clinical assistant. Read the patient information and propose the most likely diagnoses "
            "and procedures. List the top 5 diagnoses followed by the top 5 procedures. Use concise bullet points.\n\n"
            f"Patient information:\n{case.context}\n\n"
            "Format:\n"
            "Diagnoses:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Procedures:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
        )

    def predict(self, case: PatientCase) -> PromptPrediction:
        prompt = self._build_prompt(case)
        generated = self.generator.generate(prompt)
        predicted_diags = _parse_ranked_items(generated, "Diagnoses")
        predicted_procs = _parse_ranked_items(generated, "Procedures")

        diag_precision: dict[int, float] = {}
        diag_recall: dict[int, float] = {}
        proc_precision: dict[int, float] = {}
        proc_recall: dict[int, float] = {}
        for k in range(1, 6):
            p, r = _precision_recall_at_k(predicted_diags, case.diagnoses, k)
            diag_precision[k] = p
            diag_recall[k] = r
            p2, r2 = _precision_recall_at_k(predicted_procs, case.procedures, k)
            proc_precision[k] = p2
            proc_recall[k] = r2

        return PromptPrediction(
            hadm_id=case.hadm_id,
            generated_text=generated,
            prompt=prompt,
            predicted_diagnoses=predicted_diags,
            predicted_procedures=predicted_procs,
            diagnoses_precision_at_k=diag_precision,
            diagnoses_recall_at_k=diag_recall,
            procedures_precision_at_k=proc_precision,
            procedures_recall_at_k=proc_recall,
        )


def summarise_predictions(predictions: Iterable[PromptPrediction]) -> dict[str, float]:
    totals: MutableMapping[str, float] = {}
    count = 0
    for pred in predictions:
        count += 1
        for k, value in pred.diagnoses_precision_at_k.items():
            totals[f"diagnoses_precision@{k}"] = totals.get(f"diagnoses_precision@{k}", 0.0) + value
        for k, value in pred.diagnoses_recall_at_k.items():
            totals[f"diagnoses_recall@{k}"] = totals.get(f"diagnoses_recall@{k}", 0.0) + value
        for k, value in pred.procedures_precision_at_k.items():
            totals[f"procedures_precision@{k}"] = totals.get(f"procedures_precision@{k}", 0.0) + value
        for k, value in pred.procedures_recall_at_k.items():
            totals[f"procedures_recall@{k}"] = totals.get(f"procedures_recall@{k}", 0.0) + value

    if count == 0:
        return {}

    return {key: value / count for key, value in totals.items()}


__all__ = [
    "PatientCase",
    "PromptPrediction",
    "PromptingPredictor",
    "load_patient_cases",
    "summarise_predictions",
]
