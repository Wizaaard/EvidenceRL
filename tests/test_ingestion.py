import sys
from pathlib import Path

from evidence_rl.ingestion import (
    chunk_guideline_text,
    export_documents_jsonl,
    load_documents_from_hf_dataset,
    load_documents_from_jsonl,
    load_pdf_knowledge_documents,
)


def test_chunk_guideline_text_builds_section_chunks():
    text = (
        "INTRODUCTION\nThis guideline describes care.\n"
        "1 INITIAL ASSESSMENT\nReview vitals and symptoms.\n"
        "1.1 LABS\nOrder troponin and BNP.\n"
    )

    chunks = chunk_guideline_text(text, source_id="guide", chunk_size=5, overlap=1)

    assert len(chunks) >= 2
    assert all(chunk.metadata["section_title"] for chunk in chunks)
    assert chunks[0].doc_id.startswith("guide::section-0")
    assert "BNP" in chunks[-1].text


def test_load_pdf_knowledge_documents_supports_text_and_jsonl_roundtrip(tmp_path: Path):
    guideline_dir = tmp_path / "Medical_Knowledge"
    guideline_dir.mkdir()
    first = guideline_dir / "cardiac.txt"
    first.write_text("BACKGROUND\nHeart failure care.\n2 MANAGEMENT\nUse ACEi.", encoding="utf-8")
    second = guideline_dir / "arrhythmia.txt"
    second.write_text("ARRHYTHMIA\nTreat AF with rate control.", encoding="utf-8")

    documents = load_pdf_knowledge_documents(guideline_dir, chunk_size=4, overlap=1)

    assert len(documents) >= 2
    assert all(doc.metadata.get("source_filename") for doc in documents)
    assert {doc.metadata["source_id"] for doc in documents} == {"cardiac", "arrhythmia"}

    jsonl_path = tmp_path / "chunks.jsonl"
    export_documents_jsonl(documents, jsonl_path)

    loaded = load_documents_from_jsonl(jsonl_path)
    assert loaded == documents


def test_load_documents_from_hf_dataset_uses_text_field_and_limits(monkeypatch):
    records = [
        {"text": "Cardiology guidance chunk A."},
        {"text": "Cardiology guidance chunk B."},
        {"text": "Cardiology guidance chunk C."},
    ]

    class _FakeDataset(list):
        pass

    def _fake_load_dataset(name, split):
        assert name == "ilyassacha/cardiologyChunks"
        assert split == "train"
        return _FakeDataset(records)

    import types

    fake_module = types.SimpleNamespace(load_dataset=_fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    docs = load_documents_from_hf_dataset(
        "ilyassacha/cardiologyChunks", split="train", text_field="text", max_records=2
    )

    assert len(docs) == 2
    assert all(doc.metadata["source_dataset"] == "ilyassacha/cardiologyChunks" for doc in docs)
    assert docs[0].doc_id.endswith("::train::0")
    assert "chunk A" in docs[0].text

