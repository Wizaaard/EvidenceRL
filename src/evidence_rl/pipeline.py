"""High level orchestration of the evidence-based RL reward pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .alignment import EvidenceAligner
from .documents import Document, RetrievedDocument
from .generation import EvidenceConcatenationGenerator
from .retrieval import DocumentStore, TfidfRetriever
from .reward import combined_reward


@dataclass
class RagRlResult:
    """Container capturing intermediate artifacts from the pipeline."""

    query: str
    pre_evidence: List[RetrievedDocument]
    generated_answer: str
    post_evidence: List[RetrievedDocument]
    reward: float


class RagRlPipeline:
    """Run retrieval, generation, evidence alignment, and reward computation."""

    def __init__(
        self,
        documents: List[Document],
        top_k: int = 3,
        generator: EvidenceConcatenationGenerator | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not documents:
            raise ValueError("Pipeline requires at least one document")

        self.store = DocumentStore(documents)
        self.retriever = TfidfRetriever(self.store)
        self.generator = generator or EvidenceConcatenationGenerator()
        self.aligner = EvidenceAligner(self.store)
        self.top_k = top_k

    def run(self, query: str) -> RagRlResult:
        """Execute the full pipeline for a single query."""

        pre_evidence = self.retriever.retrieve(query, top_k=self.top_k)
        generated_answer = self.generator.generate(query, pre_evidence)
        post_evidence = self.aligner.align(generated_answer, pre_evidence, top_k=self.top_k)
        reward = combined_reward(self.store, pre_evidence, post_evidence)
        return RagRlResult(query, pre_evidence, generated_answer, post_evidence, reward)


__all__ = ["RagRlPipeline", "RagRlResult"]
