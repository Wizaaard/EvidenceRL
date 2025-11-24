from __future__ import annotations

import json
from pathlib import Path

from evidence_rl.baseline import (
    PromptingPredictor,
    _judge_precision_recall_at_k,
    _parse_ranked_items,
    _precision_recall_at_k,
    _textualise_patient,
    load_patient_cases,
    patient_cases_to_rag_queries,
)


def test_textualise_patient_includes_sections():
    row = {
        "note_id": "n1",
        "subject_id": "s1",
        "hadm_id": "h1",
        "chief_complaint": "Chest pain",
        "HPI": "Patient with pressure radiating to jaw",
        "ECG": "Shows ST elevations",
        "reports": "ECG machine notes|Sinus rhythm",
    }
    text = _textualise_patient(row)
    assert "note_id" in text.lower() or "note id" in text.lower()
    assert "chest pain" in text.lower()
    assert "ecg machine" in text.lower()
    assert "sinus rhythm" in text.lower()


def test_precision_recall_at_k_handles_limits():
    preds = ["A", "B", "C"]
    truth = ["B", "C", "D"]
    precision, recall = _precision_recall_at_k(preds, truth, 2)
    assert precision == 0.5
    assert recall == 0.3333333333333333


def test_parse_ranked_items_extracts_lists():
    text = """
    Diagnoses:
    1. Acute MI
    2. Heart failure
    Procedures:
    1. Cath lab
    2. Echo
    """
    diags = _parse_ranked_items(text, "Diagnoses")
    procs = _parse_ranked_items(text, "Procedures")
    assert diags == ["Acute MI", "Heart failure"]
    assert procs == ["Cath lab", "Echo"]


class StubJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.batch_calls: list[list[tuple[str, str, str]]] = []

    def is_correct(self, query: str, answer: str, ground_truth: str) -> bool:  # type: ignore[override]
        self.calls.append((query, answer, ground_truth))
        return answer.split()[0].lower() == ground_truth.split()[0].lower()

    def is_correct_batch(
        self,
        queries: list[str],
        answers: list[str],
        ground_truths: list[str],
    ) -> list[bool]:  # type: ignore[override]
        triplets = list(zip(queries, answers, ground_truths))
        self.batch_calls.append(triplets)
        return [self.is_correct(q, a, g) for q, a, g in triplets]


def test_judge_precision_recall_at_k_uses_llm_judge():
    judge = StubJudge()
    precision, recall = _judge_precision_recall_at_k(
        ["Acute myocardial infarction", "Heart failure"],
        ["Acute MI", "Dilated cardiomyopathy"],
        2,
        judge=judge,
        patient_context="hadm 1",
        label="diagnosis",
    )
    assert precision == 0.5
    assert recall == 0.5
    assert judge.calls, "judge should be consulted"


def test_prompting_predictor_end_to_end(tmp_path: Path):
    base = tmp_path
    (base / "heart_diagnoses.csv").write_text(
        "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,HPI,physical_exam,chief_complaint,invasions,X-ray,CT,Ultrasound,CATH,ECG,MRI,reports\n"
        "n1,s1,h1,DS,1,2020-01-01,2020-01-01,Sharp pain,,Angina,,,Normal,,,Normal,,Sinus rhythm\n"
    )
    (base / "heart_diagnoses_all.csv").write_text(
        "subject_id,hadm_id,seq_num,icd_code,long_title\n"
        "s1,h1,1,I21,Acute MI\n"
        "s1,h1,2,I50,Heart failure\n"
    )
    (base / "heart_procedures.csv").write_text(
        "subject_id,hadm_id,seq_num,chartdate,icd_code,long_title\n"
        "s1,h1,1,2020-01-02,37.1,Cardiac catheterization\n"
        "s1,h1,2,2020-01-03,37.2,Echocardiogram\n"
    )

    text_calls = []

    def fake_pipeline(prompt: str, **_: object):  # type: ignore[override]
        text_calls.append(prompt)
        return [
            {
                "generated_text": "Diagnoses:\n1. Acute MI\n2. Heart failure\nProcedures:\n1. Cardiac catheterization\n2. Echocardiogram",
            }
        ]

    judge = StubJudge()
    predictor = PromptingPredictor(text_pipeline=fake_pipeline, answer_judge=judge)
    cases = load_patient_cases(base)
    prediction = predictor.predict(cases[0])

    assert prediction.predicted_diagnoses[:2] == ["Acute MI", "Heart failure"]
    assert prediction.predicted_procedures[:2] == ["Cardiac catheterization", "Echocardiogram"]
    assert prediction.diagnoses_precision_at_k[2] == 1.0
    assert prediction.procedures_precision_at_k[2] == 1.0
    assert prediction.ground_truth_diagnoses == ["Acute MI", "Heart failure"]
    assert prediction.ground_truth_procedures == ["Cardiac catheterization", "Echocardiogram"]
    assert text_calls, "prompt should be sent to pipeline"
    assert judge.calls, "judge should score the predictions"

    saved = tmp_path / "baseline.json"
    saved.write_text(json.dumps(prediction.to_dict()))
    loaded = json.loads(saved.read_text())
    assert loaded["predicted_diagnoses"][0] == "Acute MI"
    assert loaded["ground_truth_diagnoses"] == ["Acute MI", "Heart failure"]
    assert loaded["ground_truth_procedures"] == [
        "Cardiac catheterization",
        "Echocardiogram",
    ]


def test_predict_many_batches_prompts(tmp_path: Path):
    base = tmp_path
    (base / "heart_diagnoses.csv").write_text(
        "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,HPI,physical_exam,chief_complaint,invasions,X-ray,CT,Ultrasound,CATH,ECG,MRI,reports\n"
        "n1,s1,h1,DS,1,2020-01-01,2020-01-01,Sharp pain,,Angina,,,Normal,,,Normal,,Sinus rhythm\n"
        "n2,s2,h2,DS,1,2020-02-01,2020-02-01,Shortness of breath,,CHF,,,Normal,,,Normal,,Sinus rhythm\n"
    )
    (base / "heart_diagnoses_all.csv").write_text(
        "subject_id,hadm_id,seq_num,icd_code,long_title\n"
        "s1,h1,1,I21,Acute MI\n"
        "s2,h2,1,I50,Heart failure\n"
    )
    (base / "heart_procedures.csv").write_text(
        "subject_id,hadm_id,seq_num,chartdate,icd_code,long_title\n"
        "s1,h1,1,2020-01-02,37.1,Cardiac catheterization\n"
        "s2,h2,1,2020-02-02,37.2,Echocardiogram\n"
    )

    pipeline_calls: list[tuple[object, object]] = []

    def batch_pipeline(prompts, batch_size=None, **_: object):  # type: ignore[override]
        pipeline_calls.append((prompts, batch_size))
        outputs = []
        for idx, _ in enumerate(prompts):
            outputs.append(
                [
                    {
                        "generated_text": (
                            f"Diagnoses:\n1. Prediction {idx}\n"
                            f"Procedures:\n1. Procedure {idx}"
                        )
                    }
                ]
            )
        return outputs

    judge = StubJudge()
    predictor = PromptingPredictor(text_pipeline=batch_pipeline, answer_judge=judge)
    cases = load_patient_cases(base)
    predictions = predictor.predict_many(cases, batch_size=2)

    assert len(predictions) == 2
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0][1] == 2
    assert judge.batch_calls, "judge should batch score predictions when available"


def test_patient_cases_to_rag_queries_includes_truth(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "heart_diagnoses.csv").write_text(
        "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime,HPI,physical_exam,chief_complaint,invasions,X-ray,CT,Ultrasound,CATH,ECG,MRI,reports\n"
        "1,10,100,DS,1,2020-01-01,2020-01-01,History,Exam,Complaint,None,None,None,None,None,None,None,None\n",
        encoding="utf-8",
    )
    (base / "heart_diagnoses_all.csv").write_text(
        "subject_id,hadm_id,seq_num,icd_code,long_title\n10,100,1,I20,Angina\n",
        encoding="utf-8",
    )
    (base / "heart_procedures.csv").write_text(
        "subject_id,hadm_id,seq_num,chartdate,icd_code,long_title\n10,100,1,2020-01-01,1234,PCI\n",
        encoding="utf-8",
    )

    cases = load_patient_cases(base)
    rag_cases = patient_cases_to_rag_queries(cases)

    assert rag_cases[0]["case_id"] == "100"
    assert "Angina" in rag_cases[0]["ground_truth"]
    assert "PCI" in rag_cases[0]["ground_truth"]
    assert "Diagnoses" in rag_cases[0]["query"]
