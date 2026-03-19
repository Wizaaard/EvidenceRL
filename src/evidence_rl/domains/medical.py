"""Medical (cardiac) domain configuration for EvidenceRL.

Wraps all previously-hardcoded medical constants so existing code keeps
running identically when ``DomainConfig`` is not explicitly passed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .base import DomainConfig, ParsedPrediction


class MedicalDomain(DomainConfig):
    """Cardiac-diagnosis domain (MIMIC-IV-Ext)."""

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "medical"

    # ── Context Parsing ───────────────────────────────────────────────────

    @property
    def section_labels(self) -> tuple[str, ...]:
        return (
            "Chief complaint",
            "History of present illness",
            "Physical exam",
            "Invasions",
            "X-ray",
            "CT",
            "Ultrasound",
            "CATH",
            "ECG",
            "MRI",
            "ECG machine report",
        )

    @property
    def anchor_labels(self) -> list[str]:
        return ["Chief complaint", "History of present illness"]

    # ── NLI Model ─────────────────────────────────────────────────────────

    @property
    def nli_model_name(self) -> str:
        return "pritamdeka/PubMedBERT-MNLI-MedNLI"

    # ── Prompt Templates ──────────────────────────────────────────────────

    def build_rag_prompt(self, context: str, evidence_texts: list[str]) -> str:
        evidence_block = "\n\n".join(
            f"[Evidence {i+1}]:\n{t}" for i, t in enumerate(evidence_texts)
        )
        return f'''You are an expert cardiology clinical assistant. Based on the patient information and retrieved clinical evidence, provide exactly 5 cardiac diagnoses ranked from most to least likely.

For EACH diagnosis, you MUST provide:
1. The diagnosis name (concise clinical term)
2. A reasoning paragraph following these "Clinical Synthesis" rules:
   - Pathophysiological Link: Explain how the specific symptoms (e.g., dyspnea) are directly explained by the clinical findings (e.g., the mitral regurgitation seen on ultrasound).
   - Evidence Integration: Use exact values from the Physical Exam (BP, RR) and Imaging (LVEF, PA pressures) as "anchors" for your argument.
   - Guideline Alignment: Explicitly state which criteria from the Retrieved Clinical Evidence are met by this specific patient's data.

IMPORTANT: Your response MUST be valid JSON in exactly this format:
{{
  "diagnoses": [
    {{"name": "Diagnosis 1 name", "reasoning": "Detailed reasoning for diagnosis 1..."}},
    {{"name": "Diagnosis 2 name", "reasoning": "Detailed reasoning for diagnosis 2..."}},
    {{"name": "Diagnosis 3 name", "reasoning": "Detailed reasoning for diagnosis 3..."}},
    {{"name": "Diagnosis 4 name", "reasoning": "Detailed reasoning for diagnosis 4..."}},
    {{"name": "Diagnosis 5 name", "reasoning": "Detailed reasoning for diagnosis 5..."}}
  ]
}}

Patient Information:
{context}

Retrieved Clinical Evidence:
{evidence_block}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    def build_norag_prompt(self, context: str) -> str:
        return f'''You are an expert cardiology clinical assistant. Based on the patient information below, provide exactly 5 cardiac diagnoses ranked from most to least likely.

For EACH diagnosis, you MUST provide:
1. The diagnosis name (concise clinical term)
2. A reasoning paragraph following these "Clinical Synthesis" rules:
   - Pathophysiological Link: Explain how the specific symptoms (e.g., dyspnea) are directly explained by the clinical findings (e.g., the mitral regurgitation seen on ultrasound).
   - Evidence Integration: Use exact values from the Physical Exam (BP, RR) and Imaging (LVEF, PA pressures) as "anchors" for your argument.
   - Avoid generic summaries: Do not just list facts; explain the "why" behind the diagnosis using the patient's unique data.

IMPORTANT: Your response MUST be valid JSON in exactly this format:
{{
  "diagnoses": [
    {{"name": "Diagnosis 1 name", "reasoning": "Detailed reasoning for diagnosis 1..."}},
    {{"name": "Diagnosis 2 name", "reasoning": "Detailed reasoning for diagnosis 2..."}},
    {{"name": "Diagnosis 3 name", "reasoning": "Detailed reasoning for diagnosis 3..."}},
    {{"name": "Diagnosis 4 name", "reasoning": "Detailed reasoning for diagnosis 4..."}},
    {{"name": "Diagnosis 5 name", "reasoning": "Detailed reasoning for diagnosis 5..."}}
  ]
}}

Patient Information:
{context}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    # ── Output Format ─────────────────────────────────────────────────────

    @property
    def num_predictions(self) -> int:
        return 5

    def parse_output(self, raw_output: str) -> list[ParsedPrediction] | None:
        try:
            data = json.loads(raw_output.strip())
            diagnoses = data.get("diagnoses", [])
            if not isinstance(diagnoses, list):
                return None
            return [
                ParsedPrediction(
                    name=d.get("name", ""),
                    reasoning=d.get("reasoning", ""),
                )
                for d in diagnoses
                if isinstance(d, dict) and d.get("name")
            ]
        except (json.JSONDecodeError, AttributeError):
            return None

    def validate_output(self, raw_output: str) -> bool:
        preds = self.parse_output(raw_output)
        return preds is not None and len(preds) == self.num_predictions

    # ── Accuracy Reward ───────────────────────────────────────────────────

    @property
    def accuracy_embedder_name(self) -> str:
        return "FremyCompany/BioLORD-2023"

    @property
    def accuracy_threshold(self) -> float:
        return 0.80

    def compute_accuracy(
        self,
        predictions: list[ParsedPrediction],
        ground_truth: list[str],
        embedder: Any = None,
    ) -> float:
        """Embedding-based accuracy: fraction of top-3 predictions matching ground truth."""
        if embedder is None or not predictions or not ground_truth:
            return 0.0

        top_k = min(3, len(predictions))
        pred_names = [p.name for p in predictions[:top_k]]

        pred_embs = embedder.encode(pred_names)
        gt_embs = embedder.encode(ground_truth)

        import numpy as np
        pred_embs = np.array(pred_embs)
        gt_embs = np.array(gt_embs)

        matches = 0
        for pred_emb in pred_embs:
            sims = np.dot(gt_embs, pred_emb) / (
                np.linalg.norm(gt_embs, axis=1) * np.linalg.norm(pred_emb) + 1e-8
            )
            if np.max(sims) >= self.accuracy_threshold:
                matches += 1

        return matches / top_k

    # ── Judge Prompt ──────────────────────────────────────────────────────

    def build_judge_prompt(self, predicted: str, ground_truths: list[str]) -> str:
        ground_truth_str = "\n- ".join(ground_truths)
        return f"""You are evaluating the diagnosis prediction of a clinical model.

CANDIDATE ANSWER: "{predicted}"

ACCEPTED GROUND TRUTHS:
- {ground_truth_str}

TASK: Does the CANDIDATE ANSWER semantically match ANY of the ACCEPTED GROUND TRUTHS?
Respond 'TRUE' if the candidate refers to the same underlying clinical concept as any item in the list (allowing for synonyms, abbreviations, or minor wording differences).
Respond 'FALSE' if it represents a different clinical concept, severity, or anatomical location.

Verdict:"""
