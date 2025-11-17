"""Answer evaluation utilities backed by language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Protocol

from .generation import DEFAULT_MODEL_NAME


class AnswerJudge(Protocol):
    """Protocol describing how to score an answer against ground truth."""

    def is_correct(self, query: str, answer: str, ground_truth: str) -> bool:
        """Return ``True`` when ``answer`` sufficiently matches ``ground_truth``."""


@dataclass
class LLMAnswerJudge:
    """Leverage a Hugging Face model to grade answers via prompting.

    Parameters
    ----------
    model_name:
        Model identifier on Hugging Face Hub. Defaults to ``sshleifer/tiny-gpt2``
        so tests and examples run quickly.
    generation_kwargs:
        Extra keyword arguments forwarded to the underlying text-generation
        pipeline.
    text_pipeline:
        Optional callable with the same signature as ``transformers.pipeline``
        for ``text-generation``.  Tests can inject a lightweight stub here to
        avoid downloading weights.
    """

    model_name: str = DEFAULT_MODEL_NAME
    generation_kwargs: Mapping[str, Any] | None = None
    text_pipeline: Callable[..., List[Mapping[str, str]]] | None = None

    def __post_init__(self) -> None:
        base_kwargs: MutableMapping[str, Any] = {
            "max_new_tokens": 32,
            "return_full_text": True,
            "do_sample": False,
        }
        if self.generation_kwargs:
            base_kwargs.update(dict(self.generation_kwargs))
        self._generation_kwargs = dict(base_kwargs)

        if self.text_pipeline is None:
            self._pipeline = self._build_hf_pipeline()
        else:
            self._pipeline = self.text_pipeline
        self.last_prompt: str | None = None
        self.last_answer: str | None = None

    def _build_hf_pipeline(self) -> Callable[..., List[Mapping[str, str]]]:
        """Instantiate a GPU-aware Hugging Face text-generation pipeline."""

        try:
            import torch
        except ImportError:  # pragma: no cover - torch may be absent in tests
            torch = None  # type: ignore[assignment]

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline as hf_pipeline,
        )

        model_kwargs: MutableMapping[str, Any] = {}
        pipeline_kwargs: MutableMapping[str, Any] = {}

        multi_gpu = False
        if torch is not None and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                multi_gpu = True
                model_kwargs["device_map"] = "auto"
            else:
                pipeline_kwargs["device"] = 0

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if not multi_gpu and pipeline_kwargs.get("device") == 0:
            model.to("cuda:0")

        if not pipeline_kwargs:
            return hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

        return hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **pipeline_kwargs,
        )
        
    def _build_prompt(self, query: str, answer: str, ground_truth: str) -> str:
        return (
            "You are an impartial evaluator tasked with judging the equivalence of two answers.\n"
            "Compare the candidate answer with the ground truth and decide if they convey the same meaning.\n"
            "Respond only with 'True' if the candidate answer is semantically equivalent to the ground truth, "
            "or 'False' otherwise.\n\n"
            f"Question: {query}\n"
            f"Ground truth: {ground_truth}\n"
            f"Candidate answer: {answer}\n\n"
            "Verdict:"
        )

    @staticmethod
    def _extract_verdict(outputs: Iterable[Mapping[str, str]], prompt: str) -> str:
        for output in outputs:
            generated = output.get("generated_text") or output.get("text")
            if not generated:
                continue
            text = generated[len(prompt) :] if generated.startswith(prompt) else generated
            return text.strip().lower()
        return ""

    def is_correct(self, query: str, answer: str, ground_truth: str) -> bool:
        prompt = self._build_prompt(query, answer, ground_truth)
        self.last_prompt = prompt
        outputs = self._pipeline(prompt, **self._generation_kwargs)
        verdict = self._extract_verdict(outputs, prompt).strip().lower()
        return verdict.startswith("true")



__all__ = ["AnswerJudge", "LLMAnswerJudge"]
