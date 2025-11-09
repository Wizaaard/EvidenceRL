from evidence_rl import Document, RagRlPipeline


def test_pipeline_runs():
    documents = [
        Document(doc_id="a", text="alpha beta gamma"),
        Document(doc_id="b", text="beta gamma delta"),
        Document(doc_id="c", text="delta epsilon zeta"),
    ]
    pipeline = RagRlPipeline(documents, top_k=2)
    result = pipeline.run("beta gamma usage")

    assert result.pre_evidence
    assert result.post_evidence
    assert 0.0 <= result.reward <= 1.0
    assert result.generated_answer
