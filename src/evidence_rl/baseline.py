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
from typing import Iterable, List, Mapping, MutableMapping, Protocol, Sequence
from abc import ABC, abstractmethod

from .generation import DEFAULT_MODEL_NAME, PromptOnlyGenerator
from .evaluation import LLMAnswerJudge, JudgeVerdictDetail
from .documents import Document
from .retrieval import DocumentStore, HuggingFaceEmbedder, SemanticRetriever


class AnswerJudge(Protocol):
    """Protocol for answer judges used by the baseline predictor."""

    def is_correct(self, prompt: str) -> bool:
        """Determine if the answer is correct based on the prompt."""
        ...


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
    """String-equality precision/recall at *k* (legacy helper)."""

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


def _judge_precision_recall_at_k(
    predicted: Sequence[str],
    truth: Sequence[str],
    k: int,
    judge: AnswerJudge,
    patient_context: str,
    label: str,
) -> tuple[float, float]:
    """LLM-judged precision/recall at *k* for diagnoses or procedures."""

    precision_by_k, recall_by_k, _, _ = _judge_precision_recall(
        predicted=predicted,
        truth=truth,
        judge=judge,
        patient_context=patient_context,
        label=label,
        max_k=k,
    )
    return precision_by_k.get(k, 0.0), recall_by_k.get(k, 0.0)


def _judge_precision_recall(
    predicted: Sequence[str],
    truth: Sequence[str],
    judge: AnswerJudge,
    patient_context: str,
    label: str,
    max_k: int = 5,
) -> tuple[dict[int, float], dict[int, float], list[bool], list[JudgeVerdictDetail]]:
    """LLM-judged precision/recall up to *max_k* using batched scoring when available.

    Returns precision/recall dictionaries and a verdict list indicating whether each
    predicted item (up to *max_k*) matched at least one ground-truth label.
    """

    if max_k <= 0:
        raise ValueError("max_k must be positive")

    preds, truths, tasks = _build_judge_tasks(predicted, truth, patient_context, label, max_k)

    if not preds or not truths:
        zeros = {k: 0.0 for k in range(1, max_k + 1)}
        return zeros, zeros, [False for _ in preds], []

    results, details = _run_judge_tasks(judge, tasks)
    precision_by_k, recall_by_k, verdicts = _precision_recall_from_results(preds, truths, results, max_k)
    return precision_by_k, recall_by_k, verdicts, details

def _build_judge_tasks(
    predicted: Sequence[str],
    truth: Sequence[str],
    patient_context: str,
    label: str,
    max_k: int,
) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """
    Prepare judge queries for top-k predictions against the full set of truths.
    Optimized to avoid N*M complexity by checking 1 pred vs ALL truths in one prompt.
    """

    # Take only the top K predictions
    preds = list(predicted[:max_k])
    
    # Keep ALL truths (order doesn't matter for the set of correct answers)
    truths = [item for item in truth if item]
    
    if not preds or not truths:
        return preds, truths, []

    # Format the list of truths into a single string for the prompt
    truth_list_str = "\n".join([f"- {t}" for t in truths])

    tasks: list[tuple[str, str, str]] = []
    
    for pred in preds:
        # We now create ONE task per prediction
        prompt = (
            f"You are evaluating the {label} prediction of a clinical model.\n\n"
            f"CANDIDATE ANSWER: \"{pred}\"\n\n"
            f"ACCEPTED GROUND TRUTHS:\n{truth_list_str}\n\n"
            f"TASK: Does the CANDIDATE ANSWER semantically match ANY of the ACCEPTED GROUND TRUTHS?\n"
            "Respond 'TRUE' if the candidate refers to the same underlying clinical concept as "
            "any item in the list (allowing for synonyms, abbreviations, or minor wording differences).\n"
            "Respond 'FALSE' if it represents a different clinical concept, severity, or anatomical location.\n\n"
            "Verdict:"
        )
        
        # We only need the prompt and the single prediction we are judging
        tasks.append((prompt, pred, truth_list_str))
            
    return preds, truths, tasks

def _run_judge_tasks(
    judge: AnswerJudge, tasks: Sequence[tuple[str, str, str]]
) -> tuple[list[bool], list[JudgeVerdictDetail]]:
    if not tasks:
        return [], []

    queries, answers, ground_truths = zip(*tasks)
    prompts: list[str | None] = []
    # prompt_builder = getattr(judge, "_build_prompt", None)
    # if callable(prompt_builder):
    #     prompts = [prompt_builder(q, a, g) for q, a, g in tasks]
    # else:
    prompts = [q for q in queries]

    results: list[bool]
    verdict_texts: list[str] = []
    raw_responses: list[str | None] = []

    if hasattr(judge, "is_correct_batch"):
        results = list(judge.is_correct_batch(prompts))
        if isinstance(judge, LLMAnswerJudge):
            verdict_texts = list(getattr(judge, "last_verdicts", []) or [])
            raw_responses = list(getattr(judge, "last_outputs", []) or [])
    else:
        results = []
        for p in prompts:
            results.append(judge.is_correct(prompt=p))
            if isinstance(judge, LLMAnswerJudge):
                verdict_texts.append(getattr(judge, "last_answer", "True" if results[-1] else "False"))
                raw_outputs = getattr(judge, "last_outputs", None)
                raw_responses.append(raw_outputs[-1] if raw_outputs else None)

    while len(verdict_texts) < len(results):
        verdict_texts.append("True" if results[len(verdict_texts)] else "False")
    while len(raw_responses) < len(results):
        raw_responses.append(None)

    details = [
        JudgeVerdictDetail(
            query=q,
            answer=a,
            ground_truth=g,
            prompt=prompts[idx] if idx < len(prompts) else None,
            verdict_text=verdict_texts[idx],
            is_correct=results[idx],
            raw_response=raw_responses[idx],
        )
        for idx, (q, a, g) in enumerate(tasks)
    ]

    return results, details


def _precision_recall_from_results(
    preds: Sequence[str],
    truths: Sequence[str],
    results: Sequence[bool],
    max_k: int,
) -> tuple[dict[int, float], dict[int, float], list[bool]]:
    """
    Compute Precision@k and Recall@k based on the judge's boolean verdicts.
    
    Args:
        preds: The list of candidate predictions.
        truths: The list of ground truths (used only for the count).
        results: A list of booleans where results[i] is True if preds[i] is a match.
        max_k: The maximum k to compute metrics for.
    """
    
    # 1. Handle edge cases
    if not preds or not truths:
        zeros = {k: 0.0 for k in range(1, max_k + 1)}
        return zeros, zeros, [False] * len(preds)

    # In the single-loop approach, the verdicts are just the results passed in.
    # We cast to list to ensure we can slice it easily.
    verdicts = list(results)
    total_truths = len(truths)
    
    precision_by_k: dict[int, float] = {}
    recall_by_k: dict[int, float] = {}

    for k in range(1, max_k + 1):
        # 2. Get the results for the top k items
        # If k > len(preds), Python's list slicing safely returns the whole list
        current_k_verdicts = verdicts[:k]
        
        # 3. Determine the denominator for Precision: we don't penalize if len(preds) < k
        considered = len(current_k_verdicts)
        
        if considered == 0:
            precision_by_k[k] = 0.0
            recall_by_k[k] = 0.0
            continue

        # 4. Count the hits (True values) in the top k
        hits = sum(current_k_verdicts)

        # 5. Compute Metrics
        # Precision: "How many of my K guesses were right?"
        precision_by_k[k] = hits / considered
        
        # Recall: "How many of the Total Truths did I find in my K guesses?"
        recall_by_k[k] = hits / total_truths

    return precision_by_k, recall_by_k, verdicts

def _batch_judge_scores(
    cases: Sequence[PatientCase],
    predicted_diags_list: Sequence[Sequence[str]],
    predicted_procs_list: Sequence[Sequence[str]],
    judge: AnswerJudge,
    max_k: int = 5,
) -> tuple[
    list[dict[int, float]],
    list[dict[int, float]],
    list[list[bool]],
    list[list[JudgeVerdictDetail]],
    list[dict[int, float]],
    list[dict[int, float]],
    list[list[bool]],
    list[list[JudgeVerdictDetail]],
]:
    """Batch LLM-judged precision/recall for diagnoses and procedures."""

    num_cases = len(cases)
    diag_precision = [{k: 0.0 for k in range(1, max_k + 1)} for _ in range(num_cases)]
    diag_recall = [{k: 0.0 for k in range(1, max_k + 1)} for _ in range(num_cases)]
    diag_verdicts: list[list[bool]] = [[] for _ in range(num_cases)]
    diag_details: list[list[JudgeVerdictDetail]] = [[] for _ in range(num_cases)]
    proc_precision = [{k: 0.0 for k in range(1, max_k + 1)} for _ in range(num_cases)]
    proc_recall = [{k: 0.0 for k in range(1, max_k + 1)} for _ in range(num_cases)]
    proc_verdicts: list[list[bool]] = [[] for _ in range(num_cases)]
    proc_details: list[list[JudgeVerdictDetail]] = [[] for _ in range(num_cases)]

    tasks: list[tuple[str, str, str]] = []
    specs: list[tuple[str, int, list[str], list[str], int, int]] = []
    cursor = 0

    for idx, case in enumerate(cases):
        diag_preds, diag_truths, diag_tasks = _build_judge_tasks(
            predicted_diags_list[idx], case.diagnoses, case.context, "diagnosis", max_k
        )
        start = cursor
        cursor += len(diag_tasks)
        tasks.extend(diag_tasks)
        specs.append(("diagnosis", idx, diag_preds, diag_truths, start, cursor))

        proc_preds, proc_truths, proc_tasks = _build_judge_tasks(
            predicted_procs_list[idx], case.procedures, case.context, "procedure", max_k
        )
        start = cursor
        cursor += len(proc_tasks)
        tasks.extend(proc_tasks)
        specs.append(("procedure", idx, proc_preds, proc_truths, start, cursor))

    results, details = _run_judge_tasks(judge, tasks) if tasks else ([], [])

    for label, case_idx, preds, truths, start, end in specs:
        slice_results = results[start:end] if end > start else []
        slice_details = details[start:end] if end > start else []
        precision_by_k, recall_by_k, verdicts = _precision_recall_from_results(
            preds, truths, slice_results, max_k
        )

        if label == "diagnosis":
            diag_precision[case_idx] = precision_by_k
            diag_recall[case_idx] = recall_by_k
            diag_verdicts[case_idx] = verdicts
            diag_details[case_idx] = slice_details
        else:
            proc_precision[case_idx] = precision_by_k
            proc_recall[case_idx] = recall_by_k
            proc_verdicts[case_idx] = verdicts
            proc_details[case_idx] = slice_details

    return (
        diag_precision,
        diag_recall,
        diag_verdicts,
        diag_details,
        proc_precision,
        proc_recall,
        proc_verdicts,
        proc_details,
    )


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
    subject_id: str
    hadm_id: str
    generated_text: str
    prompt: str
    predicted_diagnoses: List[str]
    predicted_procedures: List[str]
    ground_truth_diagnoses: List[str]
    ground_truth_procedures: List[str]
    diagnoses_precision_at_k: dict[int, float]
    diagnoses_recall_at_k: dict[int, float]
    procedures_precision_at_k: dict[int, float]
    procedures_recall_at_k: dict[int, float]
    diagnoses_judge_verdicts: List[bool]
    procedures_judge_verdicts: List[bool]
    diagnoses_judge_details: List[JudgeVerdictDetail]
    procedures_judge_details: List[JudgeVerdictDetail]

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "predicted_diagnoses": self.predicted_diagnoses,
            "predicted_procedures": self.predicted_procedures,
            "ground_truth_diagnoses": self.ground_truth_diagnoses,
            "ground_truth_procedures": self.ground_truth_procedures,
            "diagnoses_precision_at_k": self.diagnoses_precision_at_k,
            "diagnoses_recall_at_k": self.diagnoses_recall_at_k,
            "procedures_precision_at_k": self.procedures_precision_at_k,
            "procedures_recall_at_k": self.procedures_recall_at_k,
            "diagnoses_judge_verdicts": self.diagnoses_judge_verdicts,
            "procedures_judge_verdicts": self.procedures_judge_verdicts,
            "diagnoses_judge_details": [detail.to_dict() for detail in self.diagnoses_judge_details],
            "procedures_judge_details": [detail.to_dict() for detail in self.procedures_judge_details],
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
        icd_code = (row.get("icd_code") or "").strip()
        
        # Only include diagnoses with ICD codes starting with "I"
        if not icd_code.upper().startswith("I"):
            continue
        bucket = diagnoses_by_hadm.setdefault(hadm_id, [])
        bucket.append(row.get("long_title") or icd_code)

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

def _normalize_newlines(text: str) -> str:
    """Turn escaped \\n into real newlines and normalize CRLF."""
    # If the string comes from a JSON / DB dump with literal "\n"
    text = text.replace("\\n", "\n")
    # Normalize Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _remove_placeholders_and_list_artifacts(text: str) -> str:
    """
    Remove '___' placeholders and simple list artifacts like ['...'], ": ___, etc.
    This is intentionally conservative so we don't eat real clinical content.
    """
    # Remove "___" placeholders, leaving a single space
    text = re.sub(r"\b_+\b", " ", text)
    
    # Remove leading list wrappers like [' ... '] on single lines
    # e.g. "X-ray: [': ___\nRight upper lobe...']" -> "X-ray:\nRight upper lobe..."
    text = text.replace("[': ___", "")
    text = text.replace("': ___", "")
    text = text.replace('": ___', "")
    text = text.replace("['", "")
    text = text.replace("']", "")
    text = text.replace('["', "")
    text = text.replace('"]', "")

    # Some weird combos like "___- PCI)" -> " PCI)"
    text = re.sub(r"_+\s*-\s*", "", text)

    return text

def _remove_ids_and_diagnostic_sections(text: str) -> str:
    """
    Remove:
    - ID header lines (note_id / subject_id / hadm_id)
    - IMPRESSION: sections (any case)
    - FINAL DIAGNOSIS: sections (any case)

    Sections are removed from the header line through the next blank line
    or end of text, to account for multi-line content.
    """

    # Drop any line that contains note_id / subject_id / hadm_id
    text = re.sub(
        r"(?im)^.*\b(note_id|subject_id|hadm_id)\s*:.*\n?",
        "",
        text,
    )

    # Remove IMPRESSION: and FINAL DIAGNOSIS: sections (multi-line)
    # Handles variations like:
    # "IMPRESSION:", "Impression :", "Final Diagnosis:", "FINAL DIAGNOSIS :"
    section_pattern = re.compile(
        r"^\s*(impression|final\s+diagnosis)\s*:.*?(?=\n\s*\n|^\S|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    text = section_pattern.sub("", text)

    return text

def _cleanup_whitespace(text: str) -> str:
    """Collapse multiple blank lines and excess internal spaces."""
    # Strip trailing spaces at end of lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _clean_ecg_machine_report_block(block: str) -> str:
    """
    Clean the 'ECG machine report:' line:
    - split on ';'
    - trim whitespace
    - deduplicate phrases while keeping order
    - drop empty items
    """
    # Get everything after the label
    prefix = "ECG machine report:"
    _, _, rest = block.partition(prefix)
    rest = rest.strip()

    # Split on semicolons that separate phrases
    parts = [p.strip() for p in rest.split(";")]

    seen = set()
    unique_parts = []
    for p in parts:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        unique_parts.append(p)

    cleaned = prefix + " " + "; ".join(unique_parts) + "."
    return cleaned


def _clean_ecg_machine_report(text: str) -> str:
    """
    Find 'ECG machine report:' and clean that line only.
    Assumes the whole ECG report is on a single long logical line.
    """
    pattern = r"ECG machine report:[^\n]*"

    def repl(match: re.Match) -> str:
        block = match.group(0)
        return _clean_ecg_machine_report_block(block)

    return re.sub(pattern, repl, text)


def clean_patient_information(raw: str) -> str:
    """
    Main entry point to clean raw patient information notes
    (like your example) into a more readable, LLM-friendly form.
    """
    text = raw

    # 1) Normalize newlines
    text = _normalize_newlines(text)

    # 2) Remove placeholders and list artifacts
    text = _remove_placeholders_and_list_artifacts(text)

    # 3) Remove IDs and diagnostic sections (IMPRESSION / FINAL DIAGNOSIS)
    text = _remove_ids_and_diagnostic_sections(text)

    # 4) Clean ECG machine report (deduplicate phrases)
    text = _clean_ecg_machine_report(text)

    # 5) Normalize whitespace
    text = _cleanup_whitespace(text)

    return text

class BasePredictor(ABC):
    """Base predictor for patient cases using an LLM and an AnswerJudge."""

    def __init__(
        self,
        model_name: str | None = None,
        generation_kwargs: Mapping[str, object] | None = None,
        text_pipeline=None,
        answer_judge: AnswerJudge | None = None,
        judge_model_name: str | None = None,
        judge_generation_kwargs: Mapping[str, object] | None = None,
        judge_pipeline=None,
    ) -> None:
        self.generator = PromptOnlyGenerator(
            model_name=model_name or DEFAULT_MODEL_NAME,
            generation_kwargs=generation_kwargs,
            text_pipeline=text_pipeline,
        )
        self.judge: AnswerJudge = answer_judge or LLMAnswerJudge(
            model_name=judge_model_name or model_name or DEFAULT_MODEL_NAME,
            generation_kwargs=judge_generation_kwargs,
            text_pipeline=judge_pipeline,
        )

    @abstractmethod
    def _build_prompt(self, case: PatientCase) -> str:
        """Construct the LLM prompt for a given patient case."""
        raise NotImplementedError

    def _postprocess_generated(self, raw: str) -> str:
        """
        Optional hook to normalize the raw generation into a string that
        `_parse_ranked_items` can consume. Default is identity.
        """
        return raw

    def _score_predictions(
        self,
        case: PatientCase,
        predicted_diags: Sequence[str],
        predicted_procs: Sequence[str],
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
        """Compute precision/recall@k for diagnoses and procedures."""
        diag_precision, diag_recall, diag_verdicts, diag_details = _judge_precision_recall(
            predicted_diags,
            case.diagnoses,
            judge=self.judge,
            patient_context=case.context,
            label="diagnosis",
            max_k=5,
        )
        proc_precision, proc_recall, proc_verdicts, proc_details = _judge_precision_recall(
            predicted_procs,
            case.procedures,
            judge=self.judge,
            patient_context=case.context,
            label="procedure",
            max_k=5,
        )

        return diag_precision, diag_recall, diag_verdicts, diag_details, proc_precision, proc_recall, proc_verdicts, proc_details

    def predict(self, case: PatientCase) -> PromptPrediction:
        prompt = self._build_prompt(case)
        generated_raw = self.generator.generate(prompt)
        generated = self._postprocess_generated(generated_raw)

        predicted_diags = _parse_ranked_items(generated, "Diagnoses")
        predicted_procs = _parse_ranked_items(generated, "Procedures")

        diag_precision, diag_recall, diag_verdicts, diag_details, proc_precision, proc_recall, proc_verdicts, proc_details = self._score_predictions(
            case, predicted_diags, predicted_procs
        )
       

        return PromptPrediction(
            subject_id=getattr(case, "subject_id", None),
            hadm_id=case.hadm_id,
            generated_text=generated,
            prompt=prompt,
            predicted_diagnoses=predicted_diags,
            predicted_procedures=predicted_procs,
            ground_truth_diagnoses=list(case.diagnoses),
            ground_truth_procedures=list(case.procedures),
            diagnoses_precision_at_k=diag_precision,
            diagnoses_recall_at_k=diag_recall,
            procedures_precision_at_k=proc_precision,
            procedures_recall_at_k=proc_recall,
            diagnoses_judge_verdicts=diag_verdicts,
            procedures_judge_verdicts=proc_verdicts,
            diagnoses_judge_details=diag_details,
            procedures_judge_details=proc_details,
        )

    def predict_many(self, cases: Sequence[PatientCase], batch_size: int | None = None) -> List[PromptPrediction]:
        """Predict diagnoses and procedures for multiple cases with batched LLM calls."""

        if not cases:
            return []

        prompts = [self._build_prompt(case) for case in cases]
        generations = self.generator.generate_batch(prompts, batch_size=batch_size)

        predicted_diags_list: list[list[str]] = []
        predicted_procs_list: list[list[str]] = []
        generations_list = list(generations)
        for generated in generations_list:
            generated = self._postprocess_generated(generated)
            predicted_diags_list.append(_parse_ranked_items(generated, "Diagnoses"))
            predicted_procs_list.append(_parse_ranked_items(generated, "Procedures"))

        (
            diag_precision_list,
            diag_recall_list,
            diag_verdicts_list,
            diag_details_list,
            proc_precision_list,
            proc_recall_list,
            proc_verdicts_list,
            proc_details_list,
        ) = _batch_judge_scores(cases, predicted_diags_list, predicted_procs_list, self.judge, max_k=5)

        predictions: List[PromptPrediction] = []
        for idx, (case, prompt, generated) in enumerate(zip(cases, prompts, generations_list)):
            predicted_diags = predicted_diags_list[idx]
            predicted_procs = predicted_procs_list[idx]
            predictions.append(
                PromptPrediction(
                    subject_id=getattr(case, "subject_id", None),
                    hadm_id=case.hadm_id,
                    generated_text=generated,
                    prompt=prompt,
                    predicted_diagnoses=predicted_diags,
                    predicted_procedures=predicted_procs,
                    ground_truth_diagnoses=list(case.diagnoses),
                    ground_truth_procedures=list(case.procedures),
                    diagnoses_precision_at_k=diag_precision_list[idx],
                    diagnoses_recall_at_k=diag_recall_list[idx],
                    procedures_precision_at_k=proc_precision_list[idx],
                    procedures_recall_at_k=proc_recall_list[idx],
                    diagnoses_judge_verdicts=diag_verdicts_list[idx],
                    procedures_judge_verdicts=proc_verdicts_list[idx],
                    diagnoses_judge_details=diag_details_list[idx],
                    procedures_judge_details=proc_details_list[idx],
                )
            )
        return predictions

class PromptingPredictor(BasePredictor):
    """Prompt an LLM for cardiology diagnoses and procedures using textualised patient data."""

    def _build_prompt(self, case: PatientCase) -> str:
        cleaned_context = clean_patient_information(case.context)
        return (
            "You are an expert cardiology clinical assistant. You specialize in diagnosing and managing "
            "cardiovascular disease using current evidence-based guidelines.\n\n"
            "Read the patient information and identify the most likely *cardiac* diagnoses and the most "
            "appropriate *cardiology-related* procedures.\n"
            "- Prioritize cardiovascular conditions (ischemic heart disease, arrhythmias, heart failure, "
            "valvular disease, cardiomyopathies, pericardial disease, pulmonary hypertension, etc.).\n"
            "- Rank diagnoses from most to least likely based on the clinical data.\n"
            "- Focus on concise, guideline-aligned procedure recommendations (e.g., ECG, troponins, "
            "echocardiography, stress testing, cath/PCI, advanced imaging).\n"
            "- Use short, clinical phrases only. Do not add explanations or extra sections.\n\n"
            f"Patient information:\n{cleaned_context}\n\n"
            "Format (exactly):\n"
            "Diagnoses:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Procedures:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Diagnoses:\n1."
        )

    def _postprocess_generated(self, raw: str) -> str:
        if not raw.strip().startswith("Diagnoses:"):
            return "Diagnoses:\n1. " + raw.lstrip()
        return raw

class RAGPredictor(BasePredictor):
    """Patient-note predictor that augments prompts with retrieved clinical knowledge."""

    def __init__(
        self,
        documents: Sequence[Document],
        top_k: int = 3,
        model_name: str | None = None,
        generation_kwargs: Mapping[str, object] | None = None,
        text_pipeline=None,
        answer_judge: AnswerJudge | None = None,
        judge_model_name: str | None = None,
        judge_generation_kwargs: Mapping[str, object] | None = None,
        judge_pipeline=None,
        embedder: HuggingFaceEmbedder | None = None,
        embedding_model_name: str | None = None,
        chunk_size: int = 320,
        chunk_overlap: int = 64,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if embedder is None and embedding_model_name is not None:
            embedder = HuggingFaceEmbedder(model_name=embedding_model_name)

        self.store = DocumentStore(
            documents,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = SemanticRetriever(self.store)
        self.top_k = top_k

        # Initialize the base class (generator + judge)
        super().__init__(
            model_name=model_name,
            generation_kwargs=generation_kwargs,
            text_pipeline=text_pipeline,
            answer_judge=answer_judge,
            judge_model_name=judge_model_name,
            judge_generation_kwargs=judge_generation_kwargs,
            judge_pipeline=judge_pipeline,
        )

    def _format_evidence(self, retrieved: Sequence[Document]) -> str:
        if not retrieved:
            return "No external clinical knowledge was retrieved."
        return "\n\n".join(
            [
                f"Guideline chunk {idx}:\n{doc.text.strip()}"
                for idx, doc in enumerate(retrieved, start=1)
            ]
        )

    def _extract_section_queries(self, context: str) -> List[str]:
        """Return patient sections that should drive retrieval."""

        if not context.strip():
            return []

        queries: List[str] = []
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        for label, _key in _PATIENT_SECTIONS:
            prefix = f"{label}:"
            for line in lines:
                if line.lower().startswith(prefix.lower()):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        queries.append(value)
                    break

        if not queries:
            queries.append(context)

        return queries

    def _retrieve_evidence(self, context: str) -> List[Document]:
        """Retrieve and rank evidence across patient sections."""

        section_queries = self._extract_section_queries(context)
        if not section_queries:
            return []

        best_hits: MutableMapping[str, float] = {}
        docs_by_id: MutableMapping[str, Document] = {}

        for query in section_queries:
            hits = self.retriever.retrieve(query, top_k=self.top_k)
            for hit in hits:
                doc_id = hit.document.doc_id
                if doc_id not in best_hits or hit.score > best_hits[doc_id]:
                    best_hits[doc_id] = hit.score
                    docs_by_id[doc_id] = hit.document

        ranked = sorted(best_hits.items(), key=lambda item: item[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in ranked[: self.top_k]]
        return [docs_by_id[doc_id] for doc_id in top_ids]

    def _build_prompt(self, case: PatientCase) -> str:
        cleaned_context = clean_patient_information(case.context)
        retrieved_docs = self._retrieve_evidence(cleaned_context)
        knowledge_block = self._format_evidence(retrieved_docs)
        return (
            "You are an expert cardiology clinical assistant. You specialize in diagnosing and managing "
            "cardiovascular disease using current evidence-based guidelines.\n\n"
            "Read the patient information and identify the most likely *cardiac* diagnoses and the most "
            "appropriate *cardiology-related* procedures.\n"
            "- Prioritize cardiovascular conditions (ischemic heart disease, arrhythmias, heart failure, "
            "valvular disease, cardiomyopathies, pericardial disease, pulmonary hypertension, etc.).\n"
            "- Rank diagnoses from most to least likely based on the clinical data.\n"
            "- Focus on concise, guideline-aligned procedure recommendations (e.g., ECG, troponins, "
            "echocardiography, stress testing, cath/PCI, advanced imaging).\n"
            "- Use short, clinical phrases only. Do not add explanations or extra sections.\n\n"
            f"Patient information:\n{cleaned_context}\n\n"
            f"Retrieved clinical knowledge:\n{knowledge_block}\n\n"
            "Format (exactly):\n"
            "Diagnoses:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Procedures:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Diagnoses:\n1."
        )
    
    def _postprocess_generated(self, raw: str) -> str:
        if not raw.strip().startswith("Diagnoses:"):
            return "Diagnoses:\n1. " + raw.lstrip()
        return raw

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
    "RAGPredictor",
    "load_patient_cases",
    "summarise_predictions",
]
