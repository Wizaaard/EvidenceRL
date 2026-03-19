#!/usr/bin/env python3
"""Prompt templates and parsers for the Self-RAG pipeline.

Self-RAG generates diagnoses in 3 steps:
1. Zero-shot generation (reuses baseline prompt)
2. Self-critique: model evaluates confidence per diagnosis
3. Refinement: regenerate with retrieved evidence for uncertain diagnoses

This module provides the critique and refinement prompts plus parsing logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .documents import RetrievedDocument
from .evidence_pipeline import (
    StructuredDiagnosisOutput,
    _clean_evidence_text,
    _normalize_json_string,
)
from .vllm_generation import _strip_thinking_tokens


@dataclass
class DiagnosisCritique:
    """Self-critique result for a single diagnosis."""
    diagnosis: str
    confidence: str  # "HIGH" or "LOW"
    needs_evidence: bool
    reason: str = ""


@dataclass
class SelfRAGCritique:
    """Self-critique result for all 5 diagnoses of a patient."""
    critiques: List[DiagnosisCritique] = field(default_factory=list)
    raw_output: str = ""
    parse_success: bool = True
    parse_error: str | None = None

    @property
    def needs_retrieval(self) -> bool:
        """Whether any diagnosis needs evidence retrieval."""
        return any(c.needs_evidence for c in self.critiques)

    @property
    def num_uncertain(self) -> int:
        """Number of diagnoses marked as needing evidence."""
        return sum(1 for c in self.critiques if c.needs_evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critiques": [
                {
                    "diagnosis": c.diagnosis,
                    "confidence": c.confidence,
                    "needs_evidence": c.needs_evidence,
                    "reason": c.reason,
                }
                for c in self.critiques
            ],
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }


@dataclass
class SelfRAGMetadata:
    """Metadata from the self-RAG pipeline for analysis."""
    initial_diagnoses: List[Dict[str, str]] = field(default_factory=list)
    critique: Dict[str, Any] = field(default_factory=dict)
    retrieval_triggered: bool = False
    num_uncertain_diagnoses: int = 0
    retrieval_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_diagnoses": self.initial_diagnoses,
            "critique": self.critique,
            "retrieval_triggered": self.retrieval_triggered,
            "num_uncertain_diagnoses": self.num_uncertain_diagnoses,
            "retrieval_query": self.retrieval_query,
        }


def build_selfrag_critique_prompt(
    patient_context: str,
    initial_output: StructuredDiagnosisOutput,
) -> str:
    """Build prompt asking the model to critique its own diagnoses.

    The model evaluates each diagnosis for confidence and whether
    clinical evidence retrieval would help.
    """
    # Format initial diagnoses
    diag_lines = []
    for idx, diag in enumerate(initial_output.diagnoses, 1):
        reasoning_text = diag.reasoning if diag.reasoning else "(no reasoning provided)"
        diag_lines.append(f"{idx}. {diag.name}: {reasoning_text}")
    formatted_diagnoses = "\n".join(diag_lines)

    prompt = f'''You are an expert cardiology clinical assistant performing self-evaluation of your previous diagnoses.

You analyzed the following patient and generated the diagnoses listed below. Now critically evaluate each diagnosis.

Patient Information:
{patient_context}

Your Initial Diagnoses:
{formatted_diagnoses}

For EACH diagnosis, assess:
1. Confidence level: HIGH (strong supporting evidence in the patient data, clear pathophysiological link) or LOW (uncertain, ambiguous findings, missing key data points)
2. Whether retrieving clinical guidelines or evidence from medical literature would help confirm, refine, or correct this diagnosis

IMPORTANT: Your response MUST be valid JSON in exactly this format:
{{
  "critiques": [
    {{"diagnosis": "Diagnosis 1 name", "confidence": "HIGH or LOW", "needs_evidence": true or false, "reason": "Brief explanation of your confidence assessment"}},
    {{"diagnosis": "Diagnosis 2 name", "confidence": "HIGH or LOW", "needs_evidence": true or false, "reason": "Brief explanation"}},
    {{"diagnosis": "Diagnosis 3 name", "confidence": "HIGH or LOW", "needs_evidence": true or false, "reason": "Brief explanation"}},
    {{"diagnosis": "Diagnosis 4 name", "confidence": "HIGH or LOW", "needs_evidence": true or false, "reason": "Brief explanation"}},
    {{"diagnosis": "Diagnosis 5 name", "confidence": "HIGH or LOW", "needs_evidence": true or false, "reason": "Brief explanation"}}
  ]
}}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    return prompt


def build_selfrag_refinement_prompt(
    patient_context: str,
    initial_output: StructuredDiagnosisOutput,
    evidence: List[RetrievedDocument],
    critique: SelfRAGCritique,
) -> str:
    """Build prompt for refining diagnoses with selectively retrieved evidence.

    Includes the initial diagnoses with confidence annotations and the
    retrieved clinical evidence for uncertain diagnoses.
    """
    # Format initial diagnoses with confidence annotations.
    # Use index-based matching (critiques are generated in diagnosis order).
    # Fall back to name-based matching only when lengths are misaligned.
    diag_lines = []
    critique_map = {c.diagnosis.lower(): c for c in critique.critiques if c.diagnosis}
    for idx, diag in enumerate(initial_output.diagnoses):
        # Index-based match first, then name-based fallback
        if idx < len(critique.critiques):
            crit = critique.critiques[idx]
        else:
            crit = critique_map.get(diag.name.lower())

        if crit and crit.needs_evidence:
            confidence_tag = "LOW confidence - evidence retrieved"
        elif crit:
            confidence_tag = f"{crit.confidence} confidence"
        else:
            confidence_tag = "assessed"
        diag_lines.append(f"{idx + 1}. {diag.name} [{confidence_tag}]: {diag.reasoning}")
    formatted_diagnoses = "\n".join(diag_lines)

    # Format retrieved evidence
    evidence_lines = []
    for idx, item in enumerate(evidence, 1):
        cleaned_text = _clean_evidence_text(item.document.text)
        evidence_lines.append(f"Evidence {idx}:\n{cleaned_text}")
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No evidence retrieved."

    prompt = f'''You are an expert cardiology clinical assistant. You previously generated diagnoses for this patient and identified areas of uncertainty. Additional clinical evidence has been retrieved for the uncertain diagnoses.

Review the evidence and provide your FINAL 5 cardiac diagnoses ranked from most to least likely. You may revise, reorder, or replace diagnoses based on the new evidence.

Patient Information:
{patient_context}

Your Initial Diagnoses (with confidence assessment):
{formatted_diagnoses}

Retrieved Clinical Evidence (for uncertain diagnoses):
{evidence_text}

For EACH diagnosis, you MUST provide:
1. The diagnosis name (concise clinical term)
2. A reasoning paragraph following these "Clinical Synthesis" rules:
   - Pathophysiological Link: Explain how the specific symptoms are directly explained by the clinical findings.
   - Evidence Integration: Use exact values from the Physical Exam (BP, RR) and Imaging (LVEF, PA pressures) as "anchors" for your argument.
   - Guideline Alignment: Explicitly state which criteria from the Retrieved Clinical Evidence are met by this patient's data.
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

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    return prompt


def parse_selfrag_critique(raw_output: str) -> SelfRAGCritique:
    """Parse self-critique JSON output into structured format.

    Falls back to treating all diagnoses as uncertain if parsing fails
    (conservative fallback — effectively becomes standard RAG).
    """
    # Strip thinking tokens if present
    cleaned = _strip_thinking_tokens(raw_output.strip())

    # Try to extract JSON from the output
    # Look for the first { and last }
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start == -1 or json_end == -1:
        return SelfRAGCritique(
            raw_output=raw_output,
            parse_success=False,
            parse_error="No JSON object found in critique output",
        )

    json_str = cleaned[json_start : json_end + 1]

    # Normalize JSON (reuse existing robust normalizer)
    json_str = _normalize_json_string(json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return SelfRAGCritique(
            raw_output=raw_output,
            parse_success=False,
            parse_error=f"JSON parse error: {e}",
        )

    # Extract critiques from parsed data
    critiques_data = data.get("critiques", [])
    if not critiques_data:
        # Try alternate keys
        for key in ("critique", "evaluations", "assessments", "results"):
            critiques_data = data.get(key, [])
            if critiques_data:
                break

    if not isinstance(critiques_data, list):
        return SelfRAGCritique(
            raw_output=raw_output,
            parse_success=False,
            parse_error="'critiques' field is not a list",
        )

    critiques = []
    for item in critiques_data:
        if not isinstance(item, dict):
            continue

        # Extract diagnosis name (try multiple keys).
        # NOTE: _normalize_json_string renames "diagnosis" -> "diagnoses"
        # and "reason" -> "reasoning", so we must check the normalized
        # keys as well as the original ones.
        diagnosis = (
            item.get("diagnosis", "")
            or item.get("diagnoses", "")   # after _normalize_json_string
            or item.get("name", "")
            or item.get("diagnosis_name", "")
        )
        # "diagnoses" should be a string here (the diagnosis name for this
        # critique item), not the top-level array.  Guard against the case
        # where it accidentally holds a list.
        if isinstance(diagnosis, list):
            diagnosis = ""

        # Extract confidence
        confidence = str(item.get("confidence", "LOW")).upper()
        if confidence not in ("HIGH", "LOW"):
            confidence = "LOW"

        # Extract needs_evidence (handle various formats)
        needs_evidence = item.get("needs_evidence", True)
        if isinstance(needs_evidence, str):
            needs_evidence = needs_evidence.lower() in ("true", "yes", "1")

        # NOTE: _normalize_json_string renames "reason" -> "reasoning"
        reason = str(item.get("reason", "") or item.get("reasoning", ""))

        critiques.append(DiagnosisCritique(
            diagnosis=diagnosis,
            confidence=confidence,
            needs_evidence=needs_evidence,
            reason=reason,
        ))

    if not critiques:
        return SelfRAGCritique(
            raw_output=raw_output,
            parse_success=False,
            parse_error="No valid critiques extracted",
        )

    return SelfRAGCritique(
        critiques=critiques,
        raw_output=raw_output,
        parse_success=True,
    )


def build_retrieval_query(
    initial_output: StructuredDiagnosisOutput,
    critique: SelfRAGCritique,
) -> str:
    """Build a FAISS retrieval query from uncertain diagnoses.

    Uses the diagnosis name + reasoning for uncertain diagnoses as the query,
    which should retrieve more targeted evidence than using raw patient context.

    Matching strategy (in priority order):
    1. Index-based: critiques[i] corresponds to diagnoses[i] (most reliable,
       since the critique prompt lists diagnoses in order).
    2. Name-based: match critique.diagnosis to diag.name (fallback for
       misaligned lengths).
    3. Fallback: use all reasoning text if nothing matched.
    """
    query_parts = []

    # Primary: index-based matching (critiques are generated in diagnosis order)
    for i, diag in enumerate(initial_output.diagnoses):
        if i < len(critique.critiques):
            crit = critique.critiques[i]
        else:
            # More diagnoses than critiques — try name-based fallback
            critique_map = {c.diagnosis.lower(): c for c in critique.critiques if c.diagnosis}
            crit = critique_map.get(diag.name.lower())

        if crit and crit.needs_evidence:
            query_parts.append(f"{diag.name}. {diag.reasoning}")

    if not query_parts:
        # Fallback: use all diagnoses if somehow nothing matched
        return initial_output.get_all_reasoning_text()

    return " ".join(query_parts)


def build_fallback_critique(initial_output: StructuredDiagnosisOutput) -> SelfRAGCritique:
    """Build a fallback critique that marks ALL diagnoses as needing evidence.

    Used when critique parsing fails — conservative fallback that effectively
    makes self-RAG behave like standard RAG with diagnosis-guided retrieval.
    """
    critiques = [
        DiagnosisCritique(
            diagnosis=diag.name,
            confidence="LOW",
            needs_evidence=True,
            reason="Fallback: critique parsing failed, retrieving evidence for all diagnoses",
        )
        for diag in initial_output.diagnoses
    ]
    return SelfRAGCritique(
        critiques=critiques,
        raw_output="",
        parse_success=False,
        parse_error="Using fallback critique (all uncertain)",
    )
