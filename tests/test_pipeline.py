from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List

from evidence_rl import Document, RagRlPipeline
from evidence_rl.evaluation import LLMAnswerJudge
from evidence_rl.generation import HuggingFaceGenerator


SYNTHETIC_DOCUMENTS: List[Document] = [
    Document(
        doc_id="htn-overview",
        text="Chronic hypertension complicates roughly one to two percent of pregnancies.",
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
        doc_id="htn-severe",
        text="Severe hypertension with end-organ symptoms warrants intravenous therapy.",
    ),
    Document(
        doc_id="preeclampsia-ppx",
        text="Low-dose aspirin beginning in the late first trimester reduces preeclampsia risk.",
    ),
    Document(
        doc_id="preeclampsia-signs",
        text="Persistent headache, visual changes, and right upper quadrant pain suggest preeclampsia.",
    ),
    Document(
        doc_id="proteinuria",
        text="Proteinuria above 300 mg in 24 hours fulfills diagnostic criteria for preeclampsia.",
    ),
]


class ScriptedGenerator:
    def __init__(self, responses: Dict[str, str]):
        self.responses = responses

    def generate(self, query: str, retrieved: Iterable) -> str:  # type: ignore[override]
        del retrieved
        return self.responses[query]


class ScriptedJudge:
    def __init__(self, verdicts: Dict[str, bool]):
        self.verdicts = verdicts
        self.calls: List[tuple[str, str, str]] = []

    def is_correct(self, query: str, answer: str, ground_truth: str) -> bool:  # type: ignore[override]
        self.calls.append((query, answer, ground_truth))
        return self.verdicts[query]


def test_pipeline_runs_with_llm_judge_hook():
    query = "What is the recommended first-line therapy for chronic hypertension in pregnancy?"
    ground_truth = "First-line therapy for chronic hypertension in pregnancy is labetalol."

    generator = ScriptedGenerator(
        {query: "First-line therapy for chronic hypertension in pregnancy is labetalol."}
    )
    judge = ScriptedJudge({query: True})

    pipeline = RagRlPipeline(
        SYNTHETIC_DOCUMENTS,
        top_k=3,
        generator=generator,
        answer_judge=judge,
    )
    result = pipeline.run(query, ground_truth=ground_truth)

    assert result.pre_evidence
    assert result.post_evidence
    assert -1.0 <= result.alignment_score <= 1.0
    assert -1.0 <= result.reward <= 1.0
    assert result.generated_answer
    assert result.is_correct is True
    assert result.reward == result.alignment_score
    assert judge.calls == [(query, result.generated_answer, ground_truth)]


def test_pipeline_multiple_queries_handles_mixed_correctness():
    cases = [
        {
            "query": "Which medication prevents preeclampsia in high-risk pregnancies?",
            "answer": "Low-dose aspirin started in the late first trimester reduces risk.",
            "ground_truth": "Low-dose aspirin beginning in the late first trimester reduces preeclampsia risk.",
            "correct": True,
        },
        {
            "query": "What symptom profile suggests preeclampsia?",
            "answer": "Vision changes and right upper quadrant pain raise concern for preeclampsia.",
            "ground_truth": "Persistent headache, visual changes, and right upper quadrant pain suggest preeclampsia.",
            "correct": True,
        },
        {
            "query": "How is mild chronic hypertension initially managed in pregnancy?",
            "answer": "Start hydrochlorothiazide at a low dose.",
            "ground_truth": "Labetalol is recommended as a first-line oral agent for chronic hypertension in pregnancy.",
            "correct": False,
        },
    ]

    generator = ScriptedGenerator({case["query"]: case["answer"] for case in cases})
    judge = ScriptedJudge({case["query"]: case["correct"] for case in cases})

    pipeline = RagRlPipeline(
        SYNTHETIC_DOCUMENTS,
        top_k=3,
        generator=generator,
        answer_judge=judge,
    )

    rewards = []
    correctness_flags = []
    for case in cases:
        result = pipeline.run(case["query"], ground_truth=case["ground_truth"])
        rewards.append(result.reward)
        correctness_flags.append(result.is_correct)

    assert any(flag is True for flag in correctness_flags)
    assert any(flag is False for flag in correctness_flags)
    assert any(score == 0.0 for score in rewards)


def test_huggingface_generator_builds_prompt():
    captured = {}

    def fake_pipeline(prompt: str, **kwargs):  # type: ignore[override]
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return [{"generated_text": prompt + " Generated."}]

    generator = HuggingFaceGenerator(text_pipeline=fake_pipeline)

    retrieved = RagRlPipeline(
        SYNTHETIC_DOCUMENTS,
        top_k=2,
        generator=generator,
        answer_judge=ScriptedJudge({}),
    ).retriever.retrieve("question", top_k=2)

    answer = generator.generate("question", retrieved)

    assert "Document 1" in captured["prompt"]
    assert "Document 2" in captured["prompt"]
    assert captured["kwargs"]["return_full_text"] is True
    assert captured["kwargs"]["do_sample"] is False
    assert captured["kwargs"]["max_new_tokens"] == 128
    assert answer.strip() == "Generated."


def test_huggingface_generator_multi_gpu_uses_device_map(monkeypatch):
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


def test_llm_answer_judge_prompt_and_parsing():
    captured = {}

    def fake_pipeline(prompt: str, **kwargs):  # type: ignore[override]
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return [{"generated_text": prompt + " Correct."}]

    judge = LLMAnswerJudge(text_pipeline=fake_pipeline)

    verdict = judge.is_correct(
        "What medication treats chronic hypertension in pregnancy?",
        "Labetalol is preferred",
        "Labetalol is recommended as a first-line oral agent for chronic hypertension in pregnancy.",
    )

    assert "Verdict:" in captured["prompt"]
    assert "Gold standard answer" in captured["prompt"]
    assert captured["kwargs"]["return_full_text"] is True
    assert captured["kwargs"]["do_sample"] is False
    assert verdict is True


def test_llm_answer_judge_gpu_behaviour_matches_generator(monkeypatch):
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
            return [{"generated_text": "correct"}]

        return runner

    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=fake_model_loader
    )
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=fake_tokenizer_loader
    )
    fake_transformers.pipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    judge = LLMAnswerJudge()
    verdict = judge.is_correct("q", "a", "g")

    assert recorded["model_kwargs"] == {"device_map": "auto"}
    assert "to" not in recorded
    assert recorded["pipeline_kwargs"] == {}
    assert verdict is True


def test_llm_answer_judge_handles_incorrect():
    def fake_pipeline(prompt: str, **kwargs):  # type: ignore[override]
        del prompt, kwargs
        return [{"generated_text": "incorrect"}]

    judge = LLMAnswerJudge(text_pipeline=fake_pipeline)
    assert judge.is_correct("q", "a", "g") is False


def test_reward_distribution_plot(tmp_path, monkeypatch):
    class FakeFigure:
        def __init__(self) -> None:
            self.saved_paths: List[Path] = []

        def tight_layout(self) -> None:
            pass

        def savefig(self, path: Path, **kwargs) -> None:
            self.saved_paths.append(Path(path))
            Path(path).touch()

    class FakeAxis:
        def __init__(self) -> None:
            self.points: List[tuple[float, float]] = []
            self.labels: Dict[str, str] = {}

        def scatter(self, xs, ys, **kwargs):
            self.points.extend(zip(xs, ys))

        def set_title(self, title):
            self.labels["title"] = title

        def set_xlabel(self, label):
            self.labels["xlabel"] = label

        def set_ylabel(self, label):
            self.labels["ylabel"] = label

        def set_xticks(self, ticks):
            self.labels["xticks"] = str(list(ticks))

        def set_xticklabels(self, labels):
            self.labels["xticklabels"] = str(list(labels))

        def grid(self, **kwargs):
            pass

    class FakePyplot:
        def subplots(self, *args, **kwargs):  # type: ignore[override]
            return FakeFigure(), FakeAxis()

        def close(self, fig=None):  # type: ignore[override]
            pass

    fake_pyplot = FakePyplot()
    monkeypatch.setitem(sys.modules, "matplotlib", types.SimpleNamespace(pyplot=fake_pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from main import plot_distributions

    class DummyResult:
        def __init__(self, correct: bool, reward: float, alignment: float) -> None:
            self.query = ""
            self.pre_evidence = []
            self.post_evidence = []
            self.generated_answer = ""
            self.alignment_score = alignment
            self.reward = reward
            self.is_correct = correct

    results = [DummyResult(True, 0.6, 0.7), DummyResult(False, 0.0, -0.2)]

    plot_distributions(results, tmp_path)

    reward_plot = tmp_path / "reward_vs_correctness.png"
    alignment_plot = tmp_path / "alignment_vs_correctness.png"
    assert reward_plot.exists()
    assert alignment_plot.exists()


