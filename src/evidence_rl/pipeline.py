"""High level orchestration of the evidence-based RL reward pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .alignment import EvidenceAligner
from .documents import Document, RetrievedDocument
from .evaluation import AnswerJudge, LLMAnswerJudge
from .generation import DEFAULT_MODEL_NAME, EvidenceGenerator, HuggingFaceGenerator
from .retrieval import DocumentStore, TfidfRetriever
from .reward import combined_reward


@dataclass
class RagRlResult:
    """Container capturing intermediate artifacts from the pipeline."""

    query: str
    pre_evidence: List[RetrievedDocument]
    generated_answer: str
    post_evidence: List[RetrievedDocument]
    alignment_score: float
    reward: float
    is_correct: bool | None = None
    generator_prompt: str | None = None
    ground_truth: str | None = None
    judge_answer: str | None = None

    @staticmethod
    def _serialize_evidence(items: Iterable[RetrievedDocument]) -> List[dict[str, object]]:
        serialized = []
        for item in items:
            metadata = item.document.metadata
            if metadata and "concepts" in metadata and isinstance(metadata["concepts"], set):
                metadata = dict(metadata)
                metadata["concepts"] = sorted(metadata["concepts"])
            serialized.append(
                {
                    "doc_id": item.document.doc_id,
                    "text": item.document.text,
                    "metadata": metadata,
                    "score": float(item.score),
                }
            )
        return serialized

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the result."""

        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "generator_prompt": self.generator_prompt,
            "generated_answer": self.generated_answer,
            "pre_evidence": self._serialize_evidence(self.pre_evidence),
            "post_evidence": self._serialize_evidence(self.post_evidence),
            "alignment_score": float(self.alignment_score),
            "reward": float(self.reward),
            "is_correct": self.is_correct,
            "judge_answer": self.judge_answer,
        }

    def save_json(self, path: str | Path) -> None:
        """Persist the result to ``path`` in JSON format."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


class RagRlPipeline:
    """Run retrieval, generation, evidence alignment, and reward computation."""

    def __init__(
        self,
        documents: List[Document],
        top_k: int = 3,
        generator: EvidenceGenerator | None = None,
        model_name: str | None = None,
        generator_kwargs: dict | None = None,
        answer_judge: AnswerJudge | None = None,
        judge_model_name: str | None = None,
        judge_kwargs: dict | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not documents:
            raise ValueError("Pipeline requires at least one document")
        if generator is not None and (model_name is not None or generator_kwargs):
            raise ValueError(
                "Provide either a generator instance or model configuration, not both."
            )
        if answer_judge is not None and (judge_model_name is not None or judge_kwargs):
            raise ValueError(
                "Provide either an answer judge instance or model configuration, not both."
            )

        self.store = DocumentStore(documents)
        self.retriever = TfidfRetriever(self.store)
        if generator is not None:
            self.generator = generator
        else:
            self.generator = HuggingFaceGenerator(
                model_name=model_name or DEFAULT_MODEL_NAME,
                generation_kwargs=generator_kwargs,
            )
        if answer_judge is not None:
            self.judge = answer_judge
        else:
            self.judge = LLMAnswerJudge(
                model_name=judge_model_name or DEFAULT_MODEL_NAME,
                generation_kwargs=judge_kwargs,
            )
        self.aligner = EvidenceAligner(self.store)
        self.top_k = top_k

    def run(
        self,
        query: str,
        ground_truth: str | None = None,
        save_path: str | Path | None = None,
    ) -> RagRlResult:
        """Execute the full pipeline for a single query."""

        pre_evidence = self.retriever.retrieve(query, top_k=self.top_k)
        generated_answer = self.generator.generate(query, pre_evidence)
        post_evidence = self.aligner.align(generated_answer, None, top_k=self.top_k)
        alignment_score = combined_reward(self.store, pre_evidence, post_evidence)

        is_correct: bool | None = None
        reward = alignment_score
        judge_answer: str | None = None
        if ground_truth is not None:
            is_correct = self.judge.is_correct(query, generated_answer, ground_truth)
            reward = alignment_score * (1.0 if is_correct else 0.0)
            judge_answer = getattr(self.judge, "last_answer", None)

        generator_prompt = getattr(self.generator, "last_prompt", None)

        result = RagRlResult(
            query,
            pre_evidence,
            generated_answer,
            post_evidence,
            alignment_score,
            reward,
            is_correct,
            generator_prompt=generator_prompt,
            ground_truth=ground_truth,
            judge_answer=judge_answer,
        )

        if save_path is not None:
            result.save_json(save_path)

        return result


__all__ = ["RagRlPipeline", "RagRlResult"]
