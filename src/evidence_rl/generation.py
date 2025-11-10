"""LLM-backed generation utilities for the evidence alignment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Protocol

from .documents import RetrievedDocument


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
            "max_new_tokens": 128,
            "return_full_text": True,
            "do_sample": False,
        }
        if self.generation_kwargs:
            base_kwargs.update(dict(self.generation_kwargs))
        self._generation_kwargs = dict(base_kwargs)

        if self.text_pipeline is None:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        else:
            self._pipeline = self.text_pipeline

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
        outputs = self._pipeline(prompt, **self._generation_kwargs)
        if not outputs:
            return ""

        generated = outputs[0].get("generated_text") or outputs[0].get("text") or ""
        trimmed = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated.strip()
        return trimmed or generated


__all__ = ["EvidenceGenerator", "HuggingFaceGenerator"]
