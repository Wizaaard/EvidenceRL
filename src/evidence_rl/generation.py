"""LLM-backed generation utilities for the evidence alignment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Protocol
from math import ceil
from .documents import RetrievedDocument
from tqdm.auto import tqdm



DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"


class EvidenceGenerator(Protocol):
    """Protocol describing the generator interface used by the pipeline."""

    def generate(self, query: str, retrieved: Iterable[RetrievedDocument]) -> str:
        """Return an answer grounded in the provided retrieved evidence."""


@dataclass
class HuggingFaceGenerator:
    """Generate answers using a Hugging Face text-generation pipeline.

    Parameters
    ----------
    model_name:
        Name of the model to load from Hugging Face Hub.  Defaults to a very
        small GPT-2 checkpoint suitable for tests and quick experiments.
    generation_kwargs:
        Additional keyword arguments forwarded to the pipeline ``__call__``
        method (e.g., ``max_new_tokens``).
    text_pipeline:
        Optional callable matching the signature of the Hugging Face
        ``pipeline`` object.  Supplying a custom callable is primarily useful
        in tests where we do not want to download actual model weights.
    """

    model_name: str = DEFAULT_MODEL_NAME
    generation_kwargs: Mapping[str, Any] | None = None
    text_pipeline: Callable[..., List[Mapping[str, str]]] | None = None

    def __post_init__(self) -> None:
        base_kwargs: MutableMapping[str, Any] = {
            # --- sampling ---
            "do_sample": True,
            "temperature": 0.8,          # 0.7–0.9 usually good
            "top_p": 0.9,                # nucleus sampling
            "top_k": 50,                 # cap the candidate set
            # --- length & formatting ---
            "max_new_tokens": 256,       # reduce if answers should be short
            "return_full_text": False,   # usually cleaner for post-processing
            # --- repetition controls ---
            "repetition_penalty": 1.15,  # 1.05–1.25; higher = stronger penalty
            "no_repeat_ngram_size": 4,   # prevents exact n-gram repeats
            "renormalize_logits": True,  # keeps logits sane after penalties
        }

        if self.generation_kwargs:
            base_kwargs.update(dict(self.generation_kwargs))
        self._generation_kwargs = dict(base_kwargs)

        if self.text_pipeline is None:
            self._pipeline = self._build_hf_pipeline()
        else:
            self._pipeline = self.text_pipeline
        self.last_prompt: str | None = None

    def _build_hf_pipeline(self) -> Callable[..., List[Mapping[str, str]]]:
        """Instantiate a Hugging Face pipeline with GPU-aware placement."""

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

    def _format_evidence(self, retrieved: Iterable[RetrievedDocument]) -> str:
        lines: List[str] = []
        for index, item in enumerate(retrieved, start=1):
            lines.append(
                f"Document {index} ({item.document.doc_id}):\n{item.document.text.strip()}"
            )
        if not lines:
            return "No documents provided."
        return "\n\n".join(lines)

    def _build_prompt(self, query: str, retrieved: Iterable[RetrievedDocument]) -> str:
        evidence_block = self._format_evidence(retrieved)
        return (
            "You are a knowledgeable assistant that must answer using the supplied documents.\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Answer:"
        )

    def generate(self, query: str, retrieved: Iterable[RetrievedDocument]) -> str:
        retrieved_list = list(retrieved)
        prompt = self._build_prompt(query, retrieved_list)
        self.last_prompt = prompt
        outputs = self._pipeline(prompt, **self._generation_kwargs)
        if not outputs:
            return ""

        generated = outputs[0].get("generated_text") or outputs[0].get("text") or ""
        trimmed = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated.strip()
        return trimmed or generated


@dataclass
class PromptOnlyGenerator:
    """Generate free-form text from a full prompt without evidence injection."""

    model_name: str = DEFAULT_MODEL_NAME
    generation_kwargs: Mapping[str, Any] | None = None
    text_pipeline: Callable[..., List[Mapping[str, str]]] | None = None

    def __post_init__(self) -> None:
        base_kwargs: MutableMapping[str, Any] = {
            # --- sampling ---
            "do_sample": True,
            "temperature": 0.8,          # 0.7–0.9 usually good
            "top_p": 0.9,                # nucleus sampling
            "top_k": 50,                 # cap the candidate set
            # --- length & formatting ---
            "max_new_tokens": 256,       # reduce if answers should be short
            "return_full_text": False,   # usually cleaner for post-processing
            # --- repetition controls ---
            "repetition_penalty": 1.15,  # 1.05–1.25; higher = stronger penalty
            "no_repeat_ngram_size": 4,   # prevents exact n-gram repeats
            "renormalize_logits": True,  # keeps logits sane after penalties
        }
        if self.generation_kwargs:
            base_kwargs.update(dict(self.generation_kwargs))
        self._generation_kwargs = dict(base_kwargs)

        if self.text_pipeline is None:
            self._pipeline = self._build_hf_pipeline()
        else:
            self._pipeline = self.text_pipeline
        self.last_prompt: str | None = None

    def _build_hf_pipeline(self) -> Callable[..., List[Mapping[str, str]]]:
        try:
            import torch
        except ImportError:  # pragma: no cover
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

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        outputs = self._pipeline(prompt, **self._generation_kwargs)
        if not outputs:
            return ""
        generated = outputs[0].get("generated_text") or outputs[0].get("text") or ""
        trimmed = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated.strip()
        return trimmed or generated

    def generate_batch(
        self,
        prompts: Iterable[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Generate text for multiple prompts, with optional progress bar."""

        prompt_list = list(prompts)
        if not prompt_list:
            return []
 
        # copy base kwargs
        kwargs = dict(self._generation_kwargs)

        # Choose an effective batch size:
        # - if user passed one, use it
        # - otherwise, fall back to length (single call = old behavior)
        effective_bs = batch_size or len(prompt_list)

        # We *won't* pass batch_size to the pipeline now, since we're chunking ourselves.
        kwargs.pop("batch_size", None)

        indices = range(0, len(prompt_list), effective_bs)
        if show_progress and tqdm is not None:
            indices = tqdm(
                indices,
                total=ceil(len(prompt_list) / effective_bs),
                desc="Generating",
            )

        results: List[str] = []

        try:
            for start in indices:
                end = start + effective_bs
                batch_prompts = prompt_list[start:end]

                # batched call
                outputs = self._pipeline(batch_prompts, **kwargs)

                # HF text-generation pipeline returns List[List[dict]]
                for prompt, output in zip(batch_prompts, outputs):
                    if not output:
                        results.append("")
                        continue

                    generated = (
                        output[0].get("generated_text")
                        or output[0].get("text")
                        or ""
                    )
                    trimmed = (
                        generated[len(prompt):].strip()
                        if generated.startswith(prompt)
                        else generated.strip()
                    )
                    results.append(trimmed or generated)

        except TypeError:
            # Fallback: pipeline doesn't support list -> run per prompt (still with progress)
            per_prompt_iter = prompt_list
            if show_progress and tqdm is not None:
                per_prompt_iter = tqdm(prompt_list, desc="Generating")

            results = [self.generate(p) for p in per_prompt_iter]

        if prompt_list:
            self.last_prompt = prompt_list[-1]

        return results


__all__ = ["EvidenceGenerator", "HuggingFaceGenerator", "PromptOnlyGenerator"]
