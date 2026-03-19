"""Abstract DomainConfig protocol for EvidenceRL.

A DomainConfig bundles every task-specific knob so that the core pipeline
(reward computation, retrieval, GRPO/DPO training, metrics) stays generic.

To add a new domain:
    1. Subclass DomainConfig in a new file under domains/
    2. Implement all abstract methods
    3. Register it in domains/__init__.py
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ParsedPrediction:
    """One structured prediction from the model (domain-agnostic)."""
    name: str       # diagnosis name, answer choice, claim, etc.
    reasoning: str   # reasoning / explanation text


class DomainConfig(ABC):
    """Abstract base for domain-specific configuration.

    Subclasses must implement every @abstractmethod. The core reward functions
    call these methods instead of using hardcoded medical constants.
    """

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'medical', 'alce'."""
        ...

    # ── Context Parsing ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def section_labels(self) -> tuple[str, ...] | None:
        """Ordered labels for splitting context into sections.

        Return None if the domain's context is unstructured (e.g. a plain
        paragraph) — reward.py will treat the full context as one block.
        """
        ...

    @property
    @abstractmethod
    def anchor_labels(self) -> list[str] | None:
        """Labels of sections that form the global anchor context.

        These are always prepended to every NLI premise.
        Return None if there is no anchor/non-anchor distinction.
        """
        ...

    def parse_context(self, context: str) -> Dict[str, str]:
        """Parse raw context into {label: content} sections.

        Default implementation splits on "Label:" lines using self.section_labels.
        Override for domains with different formatting.
        """
        if not self.section_labels or not context.strip():
            return {"_full": context.strip()} if context.strip() else {}

        sections: Dict[str, str] = {}
        lines = context.split('\n')
        current_section = None
        current_content: list[str] = []

        for line in lines:
            match = None
            for label in self.section_labels:
                if line.strip().lower().startswith(label.lower() + ':'):
                    match = label
                    content_start = line.lower().find(label.lower() + ':') + len(label) + 1
                    remaining = line[content_start:].strip()
                    break

            if match:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content).strip()
                current_section = match
                current_content = [remaining] if remaining else []
            elif current_section:
                if line.strip():
                    current_content.append(line.strip())

        if current_section and current_content:
            sections[current_section] = ' '.join(current_content).strip()

        return sections

    def build_anchor_context(self, sections: Dict[str, str]) -> str:
        """Build anchor context string from parsed sections."""
        if not self.anchor_labels:
            # No anchor distinction — use full context
            return "\n".join(f"{k}: {v}" for k, v in sections.items())
        return "\n".join(
            f"{k}: {sections[k]}" for k in self.anchor_labels if k in sections
        )

    def get_non_anchor_sections(self, sections: Dict[str, str]) -> list[str]:
        """Return contents of non-anchor sections."""
        if not self.anchor_labels:
            return list(sections.values())
        return [
            content for label, content in sections.items()
            if label not in self.anchor_labels
        ]

    # ── NLI Model ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def nli_model_name(self) -> str:
        """HuggingFace model name for NLI grounding evaluation."""
        ...

    # ── Prompt Templates ──────────────────────────────────────────────────

    @abstractmethod
    def build_rag_prompt(self, context: str, evidence_texts: list[str]) -> str:
        """Build the RAG prompt (context + retrieved evidence → generation)."""
        ...

    @abstractmethod
    def build_norag_prompt(self, context: str) -> str:
        """Build the no-RAG prompt (context only → generation)."""
        ...

    # ── Output Format ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def num_predictions(self) -> int:
        """Expected number of predictions per case (e.g. 5 for medical, 1 for QA)."""
        ...

    @abstractmethod
    def parse_output(self, raw_output: str) -> list[ParsedPrediction] | None:
        """Parse model raw output into structured predictions.

        Returns None on parse failure.
        """
        ...

    @abstractmethod
    def validate_output(self, raw_output: str) -> bool:
        """Check if raw output is valid (for format reward)."""
        ...

    # ── Accuracy Reward ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def accuracy_embedder_name(self) -> str | None:
        """HuggingFace model for embedding-based accuracy.

        Return None if accuracy uses exact match or other non-embedding method.
        """
        ...

    @property
    @abstractmethod
    def accuracy_threshold(self) -> float:
        """Similarity threshold for embedding-based accuracy matching."""
        ...

    @abstractmethod
    def compute_accuracy(
        self,
        predictions: list[ParsedPrediction],
        ground_truth: list[str],
        embedder: Any = None,
    ) -> float:
        """Compute accuracy reward for a set of predictions.

        Args:
            predictions: Parsed model predictions
            ground_truth: Gold answers
            embedder: Optional embedding model (pre-loaded)

        Returns:
            Accuracy score in [0, 1]
        """
        ...

    # ── Judge Prompt ──────────────────────────────────────────────────────

    @abstractmethod
    def build_judge_prompt(self, predicted: str, ground_truths: list[str]) -> str:
        """Build the LLM judge prompt for evaluating a single prediction.

        Args:
            predicted: The predicted answer/diagnosis name
            ground_truths: List of acceptable ground truth answers

        Returns:
            Prompt string for the judge model
        """
        ...
