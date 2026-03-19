"""BarExam QA domain configuration for EvidenceRL.

Supports the reglab/barexam_qa benchmark (Stanford RegLab).
MBE (Multistate Bar Examination) multiple-choice questions with
retrieved legal passages as evidence.

Key properties:
  - Output: JSON with answer choice (A/B/C/D) + reasoning
  - Accuracy: Exact match on answer letter
  - Grounding: NLI between reasoning and retrieved/gold passages
  - NLI model: General-domain DeBERTa (not PubMedBERT)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from .base import DomainConfig, ParsedPrediction


class BarExamDomain(DomainConfig):
    """BarExam QA (MBE multiple-choice) domain."""

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "barexam"

    # ── Context Parsing ───────────────────────────────────────────────────

    @property
    def section_labels(self) -> None:
        # Legal questions are unstructured — no section parsing
        return None

    @property
    def anchor_labels(self) -> None:
        # The question + fact pattern IS the anchor; passages are NLI premises
        return None

    def parse_context(self, context: str) -> Dict[str, str]:
        if context.strip():
            return {"_full": context.strip()}
        return {}

    def build_anchor_context(self, sections: Dict[str, str]) -> str:
        return sections.get("_full", "")

    def get_non_anchor_sections(self, sections: Dict[str, str]) -> list[str]:
        return []

    # ── NLI Model ─────────────────────────────────────────────────────────

    @property
    def nli_model_name(self) -> str:
        return "cross-encoder/nli-deberta-v3-large"

    # ── Prompt Templates ──────────────────────────────────────────────────

    def build_rag_prompt(self, context: str, evidence_texts: list[str]) -> str:
        """Build RAG prompt for BarExam QA.

        Args:
            context: The full question text (preamble + question + choices).
            evidence_texts: Retrieved legal passages.
        """
        evidence_block = "\n\n".join(
            f"[Legal Authority {i+1}]:\n{t}" for i, t in enumerate(evidence_texts)
        )

        return f'''You are an expert legal analyst taking the Multistate Bar Examination (MBE). Based on the question and retrieved legal authorities, select the best answer.

For your answer, you MUST provide:
1. The answer letter (A, B, C, or D)
2. A reasoning paragraph following these "Legal Synthesis" rules:
   - Factual Application: Identify the key facts from the question and explain how they map to the legal rule or doctrine at issue.
   - Authority Anchoring: Cite specific language, holdings, or principles from the Retrieved Legal Authorities as direct support for your conclusion. Do NOT reason beyond what the authorities state.
   - Rule-to-Fact Bridge: Explicitly connect the legal rule from the authority to the specific facts given, showing why the rule applies (or does not apply) to each answer choice.
   - Avoid generic summaries: Do not just restate the rule; explain the "why" using the specific facts of the question.

IMPORTANT: Your response MUST be valid JSON in exactly this format:
{{
  "answer": "X",
  "reasoning": "Detailed legal reasoning..."
}}

Question:
{context}

Retrieved Legal Authorities:
{evidence_block}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- The "answer" field MUST be exactly one letter: A, B, C, or D
- All reasoning goes in the "reasoning" field
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    def build_norag_prompt(self, context: str) -> str:
        return f'''You are an expert legal analyst taking the Multistate Bar Examination (MBE). Based on your legal knowledge, select the best answer.

For your answer, you MUST provide:
1. The answer letter (A, B, C, or D)
2. A reasoning paragraph following these "Legal Synthesis" rules:
   - Factual Application: Identify the key facts from the question and explain how they map to the legal rule or doctrine at issue.
   - Rule Statement: State the applicable legal rule or doctrine from your knowledge.
   - Rule-to-Fact Bridge: Explicitly connect the legal rule to the specific facts given, showing why the rule applies (or does not apply) to each answer choice.
   - Avoid generic summaries: Do not just restate the rule; explain the "why" using the specific facts of the question.

IMPORTANT: Your response MUST be valid JSON in exactly this format:
{{
  "answer": "X",
  "reasoning": "Detailed legal reasoning..."
}}

Question:
{context}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- The "answer" field MUST be exactly one letter: A, B, C, or D
- All reasoning goes in the "reasoning" field
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    # ── Output Format ─────────────────────────────────────────────────────

    @property
    def num_predictions(self) -> int:
        return 1

    def parse_output(self, raw_output: str) -> list[ParsedPrediction] | None:
        text = raw_output.strip()
        if not text:
            return None

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        # Try JSON parse
        try:
            data = json.loads(text)
            answer = str(data.get("answer", "")).strip().upper()
            reasoning = data.get("reasoning", "")
            if answer in ("A", "B", "C", "D"):
                return [ParsedPrediction(name=answer, reasoning=reasoning)]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: regex extract answer letter from free text
        answer_match = re.search(
            r'\b(?:answer|correct)\s*(?:is|:)\s*\(?([A-D])\)?',
            text, re.IGNORECASE
        )
        if not answer_match:
            # Try standalone letter at start
            answer_match = re.match(r'^([A-D])\b', text)
        if answer_match:
            return [ParsedPrediction(name=answer_match.group(1).upper(), reasoning=text)]

        return None

    def validate_output(self, raw_output: str) -> bool:
        preds = self.parse_output(raw_output)
        return preds is not None and len(preds) >= 1

    # ── Accuracy Reward ───────────────────────────────────────────────────

    @property
    def accuracy_embedder_name(self) -> None:
        # MCQ uses exact match, not embedding similarity
        return None

    @property
    def accuracy_threshold(self) -> float:
        return 0.0

    def compute_accuracy(
        self,
        predictions: list[ParsedPrediction],
        ground_truth: list[str],
        embedder: Any = None,
    ) -> float:
        """Exact match on answer letter (A/B/C/D)."""
        if not predictions or not ground_truth:
            return 0.0

        predicted = predictions[0].name.strip().upper()
        correct = ground_truth[0].strip().upper()

        return 1.0 if predicted == correct else 0.0

    # ── Judge Prompt ──────────────────────────────────────────────────────

    def build_judge_prompt(self, predicted: str, ground_truths: list[str]) -> str:
        return f"""You are evaluating a legal reasoning system on a bar exam question.

PREDICTED ANSWER: "{predicted}"
CORRECT ANSWER: "{ground_truths[0]}"

TASK: Does the PREDICTED ANSWER match the CORRECT ANSWER?
Respond 'TRUE' if they are the same letter.
Respond 'FALSE' otherwise.

Verdict:"""
