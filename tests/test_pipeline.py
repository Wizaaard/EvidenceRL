from evidence_rl import Document, RagRlPipeline
from evidence_rl.generation import HuggingFaceGenerator


def test_pipeline_runs():
    documents = [
        Document(
            doc_id="htn-overview",
            text="Chronic hypertension complicates roughly 1-2% of pregnancies.",
        ),
        Document(
            doc_id="htn-therapy",
            text="Labetalol is recommended as a first-line oral agent for chronic hypertension in pregnancy.",
        ),
        Document(
            doc_id="htn-dosing",
            text="Initial labetalol dosing is commonly 100 mg twice daily with titration as needed.",
        ),
        Document(
            doc_id="htn-alt",
            text="Nifedipine extended release can be used when labetalol is contraindicated.",
        ),
        Document(
            doc_id="htn-prevent",
            text="Low-dose aspirin starting in the late first trimester reduces the risk of preeclampsia.",
        ),
    ]

    class DummyGenerator:
        def generate(self, query, retrieved):
            del query
            del retrieved
            return "First-line therapy for chronic hypertension in pregnancy is labetalol."

    pipeline = RagRlPipeline(documents, top_k=3, generator=DummyGenerator())
    result = pipeline.run(
        "What is the recommended first-line therapy for chronic hypertension in pregnancy?",
        ground_truth="First-line therapy for chronic hypertension in pregnancy is labetalol.",
    )

    assert result.pre_evidence
    assert result.post_evidence
    assert -1.0 <= result.alignment_score <= 1.0
    assert -1.0 <= result.reward <= 1.0
    assert result.generated_answer
    assert result.is_correct is True
    assert result.reward == result.alignment_score


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


def test_huggingface_generator_multi_gpu_uses_device_map(monkeypatch):
    import sys
    import types

    recorded: dict[str, object] = {}

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.SimpleNamespace()

    def fake_model_loader(name: str, **kwargs):
        recorded["model_kwargs"] = kwargs

        class FakeModel:
            def to(self, device: str) -> None:
                recorded["to"] = device

        return FakeModel()

    def fake_tokenizer_loader(name: str):
        recorded["tokenizer_name"] = name
        return "TOKENIZER"

    def fake_pipeline(task: str, model=None, tokenizer=None, **kwargs):
        recorded["pipeline_task"] = task
        recorded["pipeline_kwargs"] = kwargs

        def runner(prompt: str, **unused):  # type: ignore[override]
            return [{"generated_text": prompt}]

        return runner

    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=fake_model_loader
    )
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=fake_tokenizer_loader
    )
    fake_transformers.pipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = HuggingFaceGenerator()
    generator.generate("question", [])

    assert recorded["model_kwargs"] == {"device_map": "auto"}
    assert "to" not in recorded
    assert recorded["pipeline_kwargs"] == {}


def test_huggingface_generator_single_gpu_moves_model(monkeypatch):
    import sys
    import types

    recorded: dict[str, object] = {}

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.SimpleNamespace()

    def fake_model_loader(name: str, **kwargs):
        recorded["model_kwargs"] = kwargs

        class FakeModel:
            def to(self, device: str) -> None:
                recorded["to"] = device

        return FakeModel()

    def fake_tokenizer_loader(name: str):
        recorded["tokenizer_name"] = name
        return "TOKENIZER"

    def fake_pipeline(task: str, model=None, tokenizer=None, **kwargs):
        recorded["pipeline_task"] = task
        recorded["pipeline_kwargs"] = kwargs

        def runner(prompt: str, **unused):  # type: ignore[override]
            return [{"generated_text": prompt}]

        return runner

    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=fake_model_loader
    )
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=fake_tokenizer_loader
    )
    fake_transformers.pipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = HuggingFaceGenerator()
    generator.generate("question", [])

    assert recorded["model_kwargs"] == {}
    assert recorded["to"] == "cuda:0"
    assert recorded["pipeline_kwargs"] == {"device": 0}


def test_incorrect_answer_zero_reward():
    documents = [
        Document(doc_id="guideline", text="Urinary tract infections are treated with nitrofurantoin in pregnancy."),
        Document(doc_id="alt", text="Fosfomycin is an alternative single-dose therapy for acute cystitis."),
        Document(doc_id="avoid", text="Fluoroquinolones should be avoided in pregnancy due to fetal toxicity."),
    ]

    class IncorrectGenerator:
        def generate(self, query, retrieved):
            del query, retrieved
            return "Penicillin is the best choice for acute cystitis in pregnancy."

    pipeline = RagRlPipeline(documents, top_k=2, generator=IncorrectGenerator())
    result = pipeline.run(
        "How should acute cystitis be treated during pregnancy?",
        ground_truth="Nitrofurantoin is recommended for acute cystitis in pregnancy.",
    )

    assert result.is_correct is False
    assert result.reward == 0.0
    assert -1.0 <= result.alignment_score <= 1.0
