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
from typing import Iterable, List, Mapping, MutableMapping, Sequence
from abc import ABC, abstractmethod

from .generation import DEFAULT_MODEL_NAME, PromptOnlyGenerator
from .evaluation import LLMAnswerJudge, AnswerJudge
from .documents import Document
from .retrieval import DocumentStore, HuggingFaceEmbedder, SemanticRetriever


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

    precision_by_k, recall_by_k, _ = _judge_precision_recall(
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
) -> tuple[dict[int, float], dict[int, float], list[bool]]:
    """LLM-judged precision/recall up to *max_k* using batched scoring when available.

    Returns precision/recall dictionaries and a verdict list indicating whether each
    predicted item (up to *max_k*) matched at least one ground-truth label.
    """

    if max_k <= 0:
        raise ValueError("max_k must be positive")

    preds = list(predicted[:max_k])
    truths = [item for item in truth if item]
    if not preds or not truths:
        zeros = {k: 0.0 for k in range(1, max_k + 1)}
        return zeros, zeros, [False for _ in preds]

    queries: list[str] = []
    answers: list[str] = []
    ground_truths: list[str] = []
    for pred in preds:
        for truth_item in truths:
            query = (
                f"Does the candidate {label} match the ground truth {label}?\n"
                "Decide whether the candidate semantically matches the ground truth, "
                "allowing for minor wording differences (e.g. synonyms, abbreviations) but not for "
                "meaningful clinical differences.\n\n"
                "Guidelines:\n"
                "- Treat them as a TRUE if they refer to the same underlying clinical concept "
                "  (e.g. 'acute decompensated heart failure' vs 'ADHF', or 'NSTEMI' vs "
                "  'non-ST elevation myocardial infarction').\n"
                "- Treat them as a FALSE if they differ in key clinical meaning, severity, "
                "  acuity, or affected structure (e.g. 'mitral regurgitation' vs 'aortic stenosis', "
                "  or 'stable angina' vs 'NSTEMI').\n"
                "- Ignore differences in word order, punctuation, or capitalization.\n\n"
            )
            queries.append(query)
            answers.append(pred)
            ground_truths.append(truth_item)

    if hasattr(judge, "is_correct_batch"):
        results = judge.is_correct_batch(queries, answers, ground_truths)
    else:
        results = [
            judge.is_correct(query=q, answer=a, ground_truth=g)
            for q, a, g in zip(queries, answers, ground_truths)
        ]

    match_matrix = [[False for _ in truths] for _ in preds]
    idx = 0
    for pred_idx in range(len(preds)):
        for truth_idx in range(len(truths)):
            match_matrix[pred_idx][truth_idx] = results[idx]
            idx += 1

    hits_by_pred: list[int] = []
    used_truths: set[int] = set()
    for pred_idx, row in enumerate(match_matrix):
        for truth_idx, is_match in enumerate(row):
            if truth_idx in used_truths:
                continue
            if is_match:
                hits_by_pred.append(pred_idx)
                used_truths.add(truth_idx)
                break

    precision_by_k: dict[int, float] = {}
    recall_by_k: dict[int, float] = {}
    for k in range(1, max_k + 1):
        considered = min(k, len(preds))
        if considered == 0:
            precision_by_k[k] = 0.0
            recall_by_k[k] = 0.0
            continue
        hits = sum(1 for pred_idx in hits_by_pred if pred_idx < k)
        precision_by_k[k] = hits / considered
        recall_by_k[k] = hits / len(truths)

    verdicts = [idx in hits_by_pred for idx in range(len(preds))]

    return precision_by_k, recall_by_k, verdicts


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
        diag_precision, diag_recall, diag_verdicts = _judge_precision_recall(
            predicted_diags,
            case.diagnoses,
            judge=self.judge,
            patient_context=case.context,
            label="diagnosis",
            max_k=5,
        )
        proc_precision, proc_recall, proc_verdicts = _judge_precision_recall(
            predicted_procs,
            case.procedures,
            judge=self.judge,
            patient_context=case.context,
            label="procedure",
            max_k=5,
        )

        return diag_precision, diag_recall, diag_verdicts, proc_precision, proc_recall, proc_verdicts

    def predict(self, case: PatientCase) -> PromptPrediction:
        prompt = self._build_prompt(case)
        generated_raw = self.generator.generate(prompt)
        generated = self._postprocess_generated(generated_raw)

        predicted_diags = _parse_ranked_items(generated, "Diagnoses")
        predicted_procs = _parse_ranked_items(generated, "Procedures")

        diag_precision, diag_recall, diag_verdicts, proc_precision, proc_recall, proc_verdicts = self._score_predictions(
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
        )

    def predict_many(self, cases: Sequence[PatientCase], batch_size: int | None = None) -> List[PromptPrediction]:
        """Predict diagnoses and procedures for multiple cases with batched LLM calls."""

        if not cases:
            return []

        prompts = [self._build_prompt(case) for case in cases]
        generations_raw = self.generator.generate_batch(prompts, batch_size=batch_size)

        predictions: List[PromptPrediction] = []
        for case, prompt, generated_raw in zip(cases, prompts, generations_raw):
            generated = self._postprocess_generated(generated_raw)

            predicted_diags = _parse_ranked_items(generated, "Diagnoses")
            predicted_procs = _parse_ranked_items(generated, "Procedures")

            diag_precision, diag_recall, diag_verdicts, proc_precision, proc_recall, proc_verdicts = self._score_predictions(
                case, predicted_diags, predicted_procs
            )

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
                    diagnoses_precision_at_k=diag_precision,
                    diagnoses_recall_at_k=diag_recall,
                    procedures_precision_at_k=proc_precision,
                    procedures_recall_at_k=proc_recall,
                    diagnoses_judge_verdicts=diag_verdicts,
                    procedures_judge_verdicts=proc_verdicts,
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

    def _build_prompt(self, case: PatientCase) -> str:
        cleaned_context = clean_patient_information(case.context)
        # RAG-specific: retrieve documents and embed them into the prompt
        retrieved_items = self.retriever.retrieve(case.context, top_k=self.top_k)
        retrieved_docs = [item.document for item in retrieved_items]
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
