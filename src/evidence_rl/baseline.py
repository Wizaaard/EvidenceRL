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

from .generation import DEFAULT_MODEL_NAME, PromptOnlyGenerator
from .evaluation import LLMAnswerJudge, AnswerJudge, JudgeVerdictDetail
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
    """Prepare judge queries for all prediction/truth pairs up to *max_k*."""

    preds = list(predicted[:max_k])
    truths = [item for item in truth if item]
    if not preds or not truths:
        return preds, truths, []

    tasks: list[tuple[str, str, str]] = []
    for pred in preds:
        for truth_item in truths:
            tasks.append(
                (
                    f"Patient encounter:\n{patient_context}\n\n"
                    f"Does the candidate {label} match the gold standard {label}?",
                    pred,
                    truth_item,
                )
            )
    return preds, truths, tasks


def _run_judge_tasks(
    judge: AnswerJudge, tasks: Sequence[tuple[str, str, str]]
) -> tuple[list[bool], list[JudgeVerdictDetail]]:
    if not tasks:
        return [], []

    queries, answers, ground_truths = zip(*tasks)
    prompts: list[str | None] = []
    prompt_builder = getattr(judge, "_build_prompt", None)
    if callable(prompt_builder):
        prompts = [prompt_builder(q, a, g) for q, a, g in tasks]
    else:
        prompts = [q for q in queries]

    results: list[bool]
    verdict_texts: list[str] = []
    raw_responses: list[str | None] = []

    if hasattr(judge, "is_correct_batch"):
        results = list(judge.is_correct_batch(list(queries), list(answers), list(ground_truths)))
        if isinstance(judge, LLMAnswerJudge):
            verdict_texts = list(getattr(judge, "last_verdicts", []) or [])
            raw_responses = list(getattr(judge, "last_outputs", []) or [])
    else:
        results = []
        for q, a, g in tasks:
            results.append(judge.is_correct(query=q, answer=a, ground_truth=g))
            if isinstance(judge, LLMAnswerJudge):
                verdict_texts.append(getattr(judge, "last_answer", "correct" if results[-1] else "incorrect"))
                raw_outputs = getattr(judge, "last_outputs", None)
                raw_responses.append(raw_outputs[-1] if raw_outputs else None)

    while len(verdict_texts) < len(results):
        verdict_texts.append("correct" if results[len(verdict_texts)] else "incorrect")
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
    if not preds or not truths:
        zeros = {k: 0.0 for k in range(1, max_k + 1)}
        return zeros, zeros, [False for _ in preds]

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


def patient_cases_to_rag_queries(cases: Iterable[PatientCase]) -> List[dict[str, str]]:
    """Convert patient cases into RAG-friendly question/answer pairs."""

    rag_cases: List[dict[str, str]] = []
    for case in cases:
        prompt = (
            "You are a clinical assistant using external clinical guidelines. Based on the "
            "patient encounter, list the top 5 most likely diagnoses followed by the top 5 "
            "most appropriate procedures. Use concise bullet points.\n\n"
            f"Patient information:\n{case.context}\n\n"
            "Format:\n"
            "Diagnoses:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Procedures:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
        )

        truth_segments: List[str] = []
        if case.diagnoses:
            truth_segments.append("Diagnoses: " + "; ".join(case.diagnoses))
        if case.procedures:
            truth_segments.append("Procedures: " + "; ".join(case.procedures))
        ground_truth = "\n".join(truth_segments) if truth_segments else "No ground truth provided."

        rag_cases.append(
            {
                "query": prompt,
                "ground_truth": ground_truth,
                "case_id": case.hadm_id,
            }
        )

    return rag_cases


class PromptingPredictor:
    """Prompt an LLM for diagnoses and procedures using textualised patient data."""

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

        return PromptPrediction(
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


class RAGPredictor:
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

    def _build_prompt(
        self, case: PatientCase, retrieved: Sequence[Document]
    ) -> str:
        knowledge_block = self._format_evidence(retrieved)
        return (
            "You are a clinical assistant. Read the patient information and use the provided clinical knowledge to "
            "propose the most likely diagnoses and procedures. List the top 5 diagnoses followed by the top 5 procedures.\n\n"
            f"Patient information:\n{case.context}\n\n"
            f"Retrieved clinical knowledge:\n{knowledge_block}\n\n"
            "Format:\n"
            "Diagnoses:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "Procedures:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
        )

    def predict(self, case: PatientCase) -> PromptPrediction:
        retrieved_docs = self._retrieve_evidence(case.context)
        prompt = self._build_prompt(case, retrieved_docs)
        generated = self.generator.generate(prompt)

        predicted_diags = _parse_ranked_items(generated, "Diagnoses")
        predicted_procs = _parse_ranked_items(generated, "Procedures")

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

        return PromptPrediction(
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
        if not cases:
            return []

        retrieved_lists = [self._retrieve_evidence(case.context) for case in cases]
        prompts = [self._build_prompt(case, retrieved) for case, retrieved in zip(cases, retrieved_lists)]
        generations = self.generator.generate_batch(prompts, batch_size=batch_size)

        predicted_diags_list: list[list[str]] = []
        predicted_procs_list: list[list[str]] = []
        generations_list = list(generations)
        for generated in generations_list:
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
    "patient_cases_to_rag_queries",
    "summarise_predictions",
]
