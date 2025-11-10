from evidence_rl import Document, RagRlPipeline
from evidence_rl.generation import HuggingFaceGenerator


def test_pipeline_runs():
    documents = [
        Document(doc_id="a", text="alpha beta gamma"),
        Document(doc_id="b", text="beta gamma delta"),
        Document(doc_id="c", text="delta epsilon zeta"),
    ]
    
    class DummyGenerator:
        def generate(self, query, retrieved):
            del query
            return " ".join(item.document.text for item in retrieved)

    pipeline = RagRlPipeline(documents, top_k=2, generator=DummyGenerator())
    result = pipeline.run("beta gamma usage")

    assert result.pre_evidence
    assert result.post_evidence
    assert 0.0 <= result.reward <= 1.0
    assert result.generated_answer


def test_huggingface_generator_builds_prompt():
    captured = {}

    def fake_pipeline(prompt: str, **kwargs):  # type: ignore[override]
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return [{"generated_text": prompt + " Generated."}]

    generator = HuggingFaceGenerator(text_pipeline=fake_pipeline)

    documents = [
        Document(doc_id="doc1", text="First doc."),
        Document(doc_id="doc2", text="Second doc."),
    ]
    retrieved = RagRlPipeline(documents, top_k=2, generator=generator).retriever.retrieve(
        "question", top_k=2
    )

    answer = generator.generate("question", retrieved)

    assert "Document 1" in captured["prompt"]
    assert "Document 2" in captured["prompt"]
    assert captured["kwargs"]["return_full_text"] is True
    assert captured["kwargs"]["do_sample"] is False
    assert captured["kwargs"]["max_new_tokens"] == 128
    assert answer.strip() == "Generated."
