"""Answer evaluation utilities backed by language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Protocol, Sequence
from tqdm.auto import tqdm

from .generation import DEFAULT_MODEL_NAME


@dataclass
class JudgeVerdictDetail:
    """Record of a single judge evaluation."""

    query: str
    answer: str
    ground_truth: str
    prompt: str | None
    verdict_text: str
    is_correct: bool
    raw_response: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "prompt": self.prompt,
            "verdict_text": self.verdict_text,
            "is_correct": self.is_correct,
            "raw_response": self.raw_response,
        }


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
        self.last_verdicts: list[str] | None = None
        self.last_outputs: list[str] | None = None

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
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                try:
                    tokenizer.pad_token_id = eos_token_id
                except Exception:  # pragma: no cover - defensive for stub tokenizers
                    pass
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if getattr(model, "config", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            
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
        
    # def _build_prompt(self, query: str, answer: str, ground_truth: str) -> str:
    #     return (
    #         "Respond only with 'True' or 'False'.\n\n"
    #         f"Question: {query}\n"
    #         f"Ground truth: {ground_truth}\n"
    #         f"Candidate answer: {answer}\n\n"
    #         "Verdict:"
    #     )

    @staticmethod
    def _extract_verdict(outputs: Iterable[Mapping[str, str]], prompt: str) -> str:
        for output in outputs:
            generated = output.get("generated_text") or output.get("text")
            if not generated:
                continue
            text = generated[len(prompt) :] if generated.startswith(prompt) else generated
            return text.strip().lower()
        return ""

    @staticmethod
    def _extract_first_output(outputs: Iterable[Mapping[str, str]], prompt: str) -> str | None:
        for output in outputs:
            generated = output.get("generated_text") or output.get("text")
            if generated:
                return generated[len(prompt) :] if generated.startswith(prompt) else generated
        return None

    def is_correct(self, prompt: str) -> bool:
        # prompt = self._build_prompt(query, answer, ground_truth)
        self.last_prompt = prompt
        outputs = self._pipeline(prompt, **self._generation_kwargs)
        verdict = self._extract_verdict(outputs, prompt)
        raw_output = self._extract_first_output(outputs, prompt)
        self.last_answer = verdict
        self.last_verdicts = [verdict]
        self.last_outputs = [raw_output] if raw_output is not None else []
        return verdict.startswith("true")

    def is_correct_batch(
        self,
        prompts: list[str | None],
        # answers: Sequence[str],
        # ground_truths: Sequence[str],
    ) -> list[bool]:
        # if not (len(queries) == len(answers) == len(ground_truths)):
        #     raise ValueError("queries, answers, and ground_truths must be the same length")

        # prompts = [self._build_prompt(q, a, g) for q, a, g in zip(queries, answers, ground_truths)]
        self.last_prompt = prompts[-1] if prompts else None
        outputs = self._pipeline(prompts, **self._generation_kwargs)

        verdicts: list[str] = []
        raw_outputs: list[str | None] = []
        for prompt, output in zip(prompts, outputs):
            prompt_outputs = output if isinstance(output, list) else [output]
            verdicts.append(self._extract_verdict(prompt_outputs, prompt))
            raw_outputs.append(self._extract_first_output(prompt_outputs, prompt))

        self.last_answer = verdicts[-1] if verdicts else None
        self.last_verdicts = verdicts
        self.last_outputs = raw_outputs
        return [verdict.startswith("true") for verdict in verdicts]


__all__ = ["LLMAnswerJudge", "JudgeVerdictDetail"]
