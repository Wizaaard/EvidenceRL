"""Evidence-RL pipeline with structured diagnosis generation.

This module implements the core EvidenceRL workflow:
1. Pre-retrieval: Retrieve evidence documents based on patient context
2. Generation: LLM generates top-5 diagnoses with reasoning (JSON format)

Reward computation (grounding + precision) is done separately in the analysis phase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from tqdm.auto import tqdm

from .baseline import PatientCase, clean_patient_information, load_patient_cases
from .documents import Document, RetrievedDocument
from .retrieval import HuggingFaceEmbedder

# Use FAISS for fast vector search (10-500x faster than exhaustive search)
try:
    from .faiss_retrieval import FAISSDocumentStore, FAISSRetriever
    _USE_FAISS = True
except ImportError:
    # Fallback to exhaustive search if FAISS not available
    from .retrieval import DocumentStore, SemanticRetriever
    _USE_FAISS = False
    print("WARNING: FAISS not installed. Using slower exhaustive search. Install with: pip install faiss-gpu")

# Reward computation is done in the analysis phase (compute_evidencerl_metrics.py)


def _clean_evidence_text(text: str) -> str:
    """Clean XML/HTML tags and entities from retrieved evidence text.

    Many documents from PubMed contain XML metadata that pollutes the evidence.
    This function removes:
    - All XML/HTML tags (e.g., <AuthorList>...</AuthorList>)
    - Common HTML entities (e.g., &#xb7; &#xe4;)
    - Extra whitespace
    """
    import html

    # Remove all XML/HTML tags (including self-closing tags)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode HTML entities (&#xb7; -> ·, &#xe4; -> ä, &lt; -> <, etc.)
    text = html.unescape(text)

    # Remove any remaining numeric character references that weren't decoded
    text = re.sub(r'&#x?[0-9a-fA-F]+;', ' ', text)

    # Normalize whitespace (collapse multiple spaces/newlines into single space)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class CrossEncoderNLI:
    """NLI model wrapper using sentence-transformers CrossEncoder.

    Uses a pre-trained NLI model to compute entailment/neutral/contradiction
    probabilities for premise-hypothesis pairs.

    This is used for computing the grounding reward by checking whether
    reasoning sentences are entailed by the patient context + evidence.
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/PubMedBERT-MNLI-MedNLI", # Changed from cross-encoder/nli-deberta-v3-base

        batch_size: int = 8,
        force_cpu: bool = False,
        device: str = None,
    ):
        """Initialize the NLI model.

        Args:
            model_name: HuggingFace model name for NLI. Recommended models:
                - "cross-encoder/nli-deberta-v3-base" (good accuracy, reasonable speed)
                - "cross-encoder/nli-deberta-v3-small" (faster, slightly less accurate)
                - "cross-encoder/nli-MiniLM2-L6-H768" (fastest, good for prototyping)
            batch_size: Batch size for inference
            force_cpu: If True, load model on CPU (needed when GPUs are owned by vLLM)
            device: Explicit device string (e.g. "cuda:0"). Overrides force_cpu if set.
        """
        from sentence_transformers import CrossEncoder

        if device is not None:
            effective_device = device
        elif force_cpu:
            effective_device = "cpu"
        else:
            effective_device = None
        print(f"Loading NLI model: {model_name}... (device={effective_device or 'auto'})")
        self.model = CrossEncoder(model_name, device=effective_device)
        self.batch_size = batch_size

        # Map model output indices to label names
        # Most NLI models use: 0=contradiction, 1=entailment, 2=neutral
        # but some use different orderings. We'll detect this from the model config.
        self._setup_label_mapping()
        print(f"NLI model loaded. Label mapping: {self.label_mapping}")

    def _setup_label_mapping(self):
        """Set up the mapping from model output indices to label names."""
        # Try to get label mapping from model config
        try:
            config = self.model.model.config
            if hasattr(config, 'id2label'):
                # Use model's label mapping
                id2label = config.id2label
                self.label_mapping = {
                    int(k): v.lower() for k, v in id2label.items()
                }
                return
        except Exception:
            pass

        # Default mapping (most common for cross-encoder/nli models)
        self.label_mapping = {
            0: 'contradiction',
            1: 'entailment',
            2: 'neutral',
        }

    def predict(
        self,
        premise_hypothesis_pairs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """Predict NLI probabilities for premise-hypothesis pairs.

        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples

        Returns:
            List of dicts with keys 'entailment', 'neutral', 'contradiction'
            representing probabilities for each class.
        """
        import numpy as np

        if not premise_hypothesis_pairs:
            return []

        # CrossEncoder expects list of [premise, hypothesis] pairs
        pairs = [[p, h] for p, h in premise_hypothesis_pairs]

        # Get raw logits from model
        # apply_softmax=True converts logits to probabilities
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            apply_softmax=True,
        )

        # Convert to numpy array if needed
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Ensure 2D array (batch_size x num_labels)
        num_labels = len(self.label_mapping)
        if scores.ndim == 1:
            scores = scores.reshape(-1, num_labels)

        # Convert to list of dicts
        results = []
        for probs in scores:
            result = {}
            for idx, prob in enumerate(probs):
                label = self.label_mapping.get(idx, f'label_{idx}')
                result[label] = float(prob)

            # Ensure all expected keys are present
            for key in ['entailment', 'neutral', 'contradiction']:
                if key not in result:
                    result[key] = 0.0

            results.append(result)

        return results


@dataclass
class DiagnosisWithReasoning:
    """A single diagnosis with its supporting reasoning."""

    name: str
    reasoning: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "reasoning": self.reasoning}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DiagnosisWithReasoning":
        return cls(name=data.get("name", ""), reasoning=data.get("reasoning", ""))


@dataclass
class StructuredDiagnosisOutput:
    """Structured output containing 5 diagnoses with reasoning."""

    diagnoses: List[DiagnosisWithReasoning] = field(default_factory=list)
    raw_output: str = ""
    parse_success: bool = True
    parse_error: str | None = None
    all_samples: Optional[List["StructuredDiagnosisOutput"]] = None

    def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        d = {
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "raw_output": self.raw_output,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }
        if include_samples and self.all_samples is not None:
            d["all_samples"] = [s.to_dict(include_samples=False) for s in self.all_samples]
        return d

    def get_all_reasoning_text(self) -> str:
        """Concatenate all reasoning text for Strategy A retrieval."""
        return " ".join(d.reasoning for d in self.diagnoses if d.reasoning)

    def get_diagnosis_query(self, index: int, include_name: bool = True) -> str:
        """Get query text for a single diagnosis (Strategy B)."""
        if index < 0 or index >= len(self.diagnoses):
            return ""
        d = self.diagnoses[index]
        if include_name:
            return f"{d.name}. {d.reasoning}"
        return d.reasoning


@dataclass
class EvidenceRLResult:
    """Complete result from the EvidenceRL pipeline for one patient case.

    This contains only the generation outputs. Reward computation is done
    separately in the analysis phase (compute_evidencerl_metrics.py).
    """

    hadm_id: str
    subject_id: str
    patient_context: str

    # Pre-retrieval evidence
    pre_evidence: List[RetrievedDocument] = field(default_factory=list)

    # Generation output
    structured_output: StructuredDiagnosisOutput | None = None
    generation_prompt: str = ""

    # Ground truth for evaluation
    ground_truth_diagnoses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def serialize_evidence(items: Iterable[RetrievedDocument]) -> List[Dict[str, Any]]:
            serialized = []
            for item in items:
                # Clean XML/HTML tags from evidence text
                cleaned_text = _clean_evidence_text(item.document.text)
                serialized.append({
                    "doc_id": item.document.doc_id,
                    "text": cleaned_text,
                    "score": float(item.score),
                })
            return serialized

        return {
            "hadm_id": self.hadm_id,
            "subject_id": self.subject_id,
            "patient_context": self.patient_context, #self.patient_context[:1000] + "..." if len(self.patient_context) > 1000 else self.patient_context,
            "pre_evidence": serialize_evidence(self.pre_evidence),
            "structured_output": self.structured_output.to_dict() if self.structured_output else None,
            # "generation_prompt": self.generation_prompt, #self.generation_prompt[:2000] + "..." if len(self.generation_prompt) > 2000 else self.generation_prompt,
            "ground_truth_diagnoses": self.ground_truth_diagnoses,
        }

    def save_json(self, path: str | Path) -> None:
        """Save result to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _build_structured_diagnosis_prompt(
    patient_context: str,
    pre_evidence: List[RetrievedDocument],
) -> str:
    """Build prompt for structured JSON diagnosis generation."""

    # Format evidence (with XML/HTML cleanup)
    evidence_text = ""
    if pre_evidence:
        evidence_lines = []
        for idx, item in enumerate(pre_evidence, start=1):
            # Clean XML/HTML tags and entities from evidence text
            cleaned_text = _clean_evidence_text(item.document.text)
            evidence_lines.append(f"Evidence {idx}:\n{cleaned_text}")
        evidence_text = "\n\n".join(evidence_lines)
    else:
        evidence_text = "No external clinical evidence available."

    prompt = f'''You are an expert cardiology clinical assistant. Based on the patient information and retrieved clinical evidence, provide exactly 5 cardiac diagnoses ranked from most to least likely.

For EACH diagnosis, you MUST provide:
1. The diagnosis name (concise clinical term)
2. A reasoning paragraph following these "Clinical Synthesis" rules:
   - Pathophysiological Link: Explain how the specific symptoms (e.g., dyspnea) are directly explained by the clinical findings (e.g., the mitral regurgitation seen on ultrasound).
   - Evidence Integration: Use exact values from the Physical Exam (BP, RR) and Imaging (LVEF, PA pressures) as "anchors" for your argument.
   - Guideline Alignment: Explicitly state which criteria from the Retrieved Clinical Evidence are met by this specific patient's data.
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
{patient_context}

Retrieved Clinical Evidence:
{evidence_text}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    return prompt


def _build_baseline_diagnosis_prompt(patient_context: str) -> str:
    """Build prompt for zero-shot diagnosis generation without RAG.

    This baseline prompt does not include any retrieved clinical evidence,
    allowing the model to generate diagnoses based solely on its training knowledge.
    """
    prompt = f'''You are an expert cardiology clinical assistant. Based on the patient information below, provide exactly 5 cardiac diagnoses ranked from most to least likely.

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
{patient_context}

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
- Begin your response IMMEDIATELY with the opening brace {{
- Do NOT include any thinking, explanation, preamble, or commentary before the JSON
- Do NOT show your reasoning process outside the JSON - all reasoning goes in the "reasoning" fields
- Output ONLY valid JSON, nothing else
- Start your response with {{'''

    return prompt


def _normalize_json_string(json_str: str) -> str:
    """Normalize malformed JSON to make it parseable.

    Handles:
    - Mixed quote types (single, double, smart quotes)
    - Unicode characters in field names
    - Field name variations (Name/name, Reasoning/reasoning, etc.)
    - Top-level field variations (diagnoises, Diagnoses, estimates, diseases, etc.)
    - Trailing commas
    - Full-width punctuation
    - Invalid escape sequences
    - Unescaped double quotes inside string values
    - Missing closing quotes before } or ]
    - Extra whitespace
    """
    # Replace smart quotes with standard quotes
    json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
    json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")

    # Replace full-width punctuation
    json_str = json_str.replace('\uff0c', ',')  # full-width comma
    json_str = json_str.replace('\uff1a', ':')  # full-width colon

    # Strip invalid JSON escape sequences (e.g., \* \~ \# etc.)
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_str)

    # Replace single quotes with double quotes (but be careful with apostrophes in text)
    # This regex matches single quotes around field names and values
    json_str = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'(\s*):", r'"\1"\2:', json_str)

    # Fix unescaped double quotes inside JSON string values.
    # Strategy: find "key": "value with "quoted words" inside" patterns and
    # escape the inner quotes. We look for quotes that are clearly inside a value
    # (preceded by a word char and followed by a word char — i.e., quoting a term).
    json_str = _fix_unescaped_inner_quotes(json_str)

    # Fix missing closing quote before }, ] or ,}
    # Pattern: a non-quote char followed by whitespace then }, or ]
    # where we expect a closing " for a string value
    json_str = re.sub(r'([a-zA-Z0-9.!?;])\s*(\}\s*[,\]\}])', r'\1"\2', json_str)
    json_str = re.sub(r'([a-zA-Z0-9.!?;])\s*(\}\s*$)', r'\1"\2', json_str)

    # Fix missing comma between fields: }" "  -> }", "
    json_str = re.sub(r'"\s*"(name|reasoning)":', r'", "\1":', json_str)

    # Clean unicode from field names (e.g., "reasonিং" -> "reasoning")
    # Match field names and remove non-ASCII characters
    def clean_field_name(match):
        field_name = match.group(1)
        # Remove non-ASCII characters and convert to lowercase
        cleaned = ''.join(c for c in field_name if ord(c) < 128).lower()

        # Map top-level field variations to canonical "diagnoses"
        # Common variations: diagnoises, diagnoses, diagnosis, Diagnoses, estimates, diseases,
        # differential_diagnoses, cardiac_diagnoses, possible_diagnoses, etc.
        if any(variant in cleaned for variant in ['diagnos', 'diagno', 'diag']):
            cleaned = 'diagnoses'
        elif cleaned in ('estimates', 'diseases', 'conditions', 'findings', 'results',
                         'predictions', 'assessments', 'conclusions'):
            cleaned = 'diagnoses'
        elif 'differential' in cleaned:
            cleaned = 'diagnoses'
        # Map common variations for inner fields
        elif 'reason' in cleaned:
            cleaned = 'reasoning'
        elif 'name' in cleaned or 'nam' in cleaned or cleaned in ('title', 'diagnosis_name'):
            cleaned = 'name'
        elif cleaned in ('description', 'explanation', 'rationale', 'justification'):
            cleaned = 'reasoning'
        return f'"{cleaned}":'

    json_str = re.sub(r'"([^"]*?)"(\s*):', clean_field_name, json_str)

    # Remove trailing commas before } or ]
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # Remove extra newlines and normalize whitespace
    json_str = re.sub(r'\n\n+', '\n', json_str)

    return json_str


def _fix_unescaped_inner_quotes(json_str: str) -> str:
    """Fix unescaped double quotes inside JSON string values.

    LLMs often embed quoted terms like "insignificant" inside JSON strings,
    which breaks parsing. This function escapes inner quotes that are clearly
    not JSON structural delimiters.

    Strategy: Walk through the string tracking whether we're inside a JSON
    string value. When we encounter a " that appears to be quoting a word
    inside a value (not a field boundary), escape it.
    """
    result = []
    i = 0
    in_string = False
    prev_was_backslash = False

    while i < len(json_str):
        ch = json_str[i]

        if prev_was_backslash:
            result.append(ch)
            prev_was_backslash = False
            i += 1
            continue

        if ch == '\\':
            result.append(ch)
            prev_was_backslash = True
            i += 1
            continue

        if ch == '"':
            if not in_string:
                # Opening a string
                in_string = True
                result.append(ch)
            else:
                # We're inside a string and hit a ".
                # Is this the real closing quote, or an unescaped inner quote?
                # Look ahead: if followed by structural JSON chars, it's a real close.
                rest = json_str[i+1:i+20].lstrip()
                if rest and rest[0] in ':,}]\n':
                    # Real closing quote (followed by : , } ] or newline)
                    in_string = False
                    result.append(ch)
                elif rest.startswith('"') and len(rest) > 1 and rest[1:].lstrip()[:1] in ':':
                    # Pattern: "value" "next_key": — this " closes the value
                    # Actually this is: end of value, missing comma, start of next key
                    in_string = False
                    result.append(ch)
                else:
                    # Inner quote — escape it
                    result.append('\\"')
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def _parse_structured_output(
    raw_output: str,
    llm_extractor: "LLMDiagnosisExtractor | None" = None,
    skip_fallback: bool = False,
) -> StructuredDiagnosisOutput:
    """Parse LLM output into structured diagnoses with robust error handling.

    Strategy:
    1. Try JSON normalization + standard parsing
    2. If that fails and skip_fallback=False, use LLM extraction or regex fallback
    3. If skip_fallback=True, leave parse_success=False for batch LLM extraction later

    Args:
        raw_output: Raw LLM output text
        llm_extractor: Optional LLM extractor for fallback parsing
        skip_fallback: If True, don't use regex fallback - leave for batch LLM extraction
    """

    result = StructuredDiagnosisOutput(raw_output=raw_output)

    # Try to extract JSON from the output
    json_str = raw_output.strip()

    # Handle cases where JSON is wrapped in markdown code blocks
    if "```json" in json_str:
        match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
    elif "```" in json_str:
        match = re.search(r"```\s*(.*?)\s*```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1)

    # Try to find JSON object in the text
    # Match various field name patterns: diagnoses, diagnosis, diagnoises, estimates, diseases, etc.
    # Pattern explanation: find a JSON object containing a field that looks like diagnoses with an array
    json_match = re.search(
        r'\{[^{}]*["\'](?:diagnos\w*|diagnosis|estimates|diseases|conditions|findings|results)["\'][^{}]*\[.*?\]\s*\}',
        json_str, re.DOTALL | re.IGNORECASE
    )
    if json_match:
        json_str = json_match.group(0)

    # Normalize the JSON string to fix common issues (converts field names to canonical forms)
    json_str_normalized = _normalize_json_string(json_str)

    # Stage 1: Try standard JSON parsing
    try:
        data = json.loads(json_str_normalized)
        # After normalization, all variants should be "diagnoses"
        # But still check common variations as fallback
        diagnoses_data = None
        for key in ("diagnoses", "diagnosis", "estimates", "diseases", "conditions", "findings"):
            val = data.get(key)
            if isinstance(val, list) and val:
                diagnoses_data = val
                break
        if diagnoses_data is None:
            diagnoses_data = []

        for d in diagnoses_data[:5]:  # Take at most 5
            if isinstance(d, dict):
                # After normalization, fields should be "name" and "reasoning"
                # But still check variations as fallback
                name = (d.get("name") or d.get("title") or
                       d.get("diagnosis") or "").strip()
                reasoning = (d.get("reasoning") or d.get("reason") or
                           d.get("rationale") or d.get("explanation") or
                           d.get("description") or "").strip()

                result.diagnoses.append(DiagnosisWithReasoning(
                    name=name,
                    reasoning=reasoning,
                ))

        # Pad with empty diagnoses if we have fewer than 5
        while len(result.diagnoses) < 5:
            result.diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))

        # Check if we got enough complete diagnoses (at least 3 with name AND reasoning)
        complete_diagnoses = sum(1 for d in result.diagnoses if d.name and d.reasoning)
        if complete_diagnoses >= 3:
            result.parse_success = True
        else:
            # Partial parse - mark as failed so LLM extraction can retry
            result.parse_success = False
            result.parse_error = f"Only {complete_diagnoses}/5 complete diagnoses parsed"

    except json.JSONDecodeError as e:
        # Stage 2: Use LLM extraction if available (and not skipping fallback)
        result.parse_success = False
        result.parse_error = f"JSON parse error: {str(e)}"

        if skip_fallback:
            # Don't use fallback - leave for batch LLM extraction
            # Just pad with empty diagnoses
            while len(result.diagnoses) < 5:
                result.diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))
        elif llm_extractor is not None:
            print(f"JSON parsing failed, using LLM extraction...")
            result.diagnoses = llm_extractor.extract(raw_output)

            # Check if LLM extraction succeeded (at least 3 complete diagnoses)
            complete_diagnoses = sum(1 for d in result.diagnoses if d.name and d.reasoning)
            if complete_diagnoses >= 3:
                result.parse_success = True
                result.parse_error = ""
        else:
            # Fallback to regex if no LLM extractor provided
            result.diagnoses = _fallback_parse_diagnoses(raw_output)

            # Check if regex extraction got enough diagnoses
            complete_diagnoses = sum(1 for d in result.diagnoses if d.name and d.reasoning)
            if complete_diagnoses >= 3:
                result.parse_success = True
                result.parse_error = ""

    return result


def _fallback_parse_diagnoses(text: str) -> List[DiagnosisWithReasoning]:
    """Enhanced fallback parser when JSON parsing fails.

    Tries multiple regex patterns to extract diagnosis name/reasoning pairs.
    """
    diagnoses = []

    # Pattern 1: JSON-like objects with name/reasoning fields
    # Matches: {"name": "...", "reasoning": "..."}
    json_obj_pattern = r'\{[^}]*["\']?(?:name|Name)["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?(?:reasoning|Reasoning|reason)["\']?\s*:\s*["\']([^"\']+)["\']'
    json_matches = re.findall(json_obj_pattern, text, re.DOTALL | re.IGNORECASE)

    for name, reasoning in json_matches[:5]:
        name = name.strip()
        reasoning = reasoning.strip()
        if name:
            diagnoses.append(DiagnosisWithReasoning(name=name, reasoning=reasoning))

    # If we found enough diagnoses, return them
    if len(diagnoses) >= 3:
        while len(diagnoses) < 5:
            diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))
        return diagnoses[:5]

    # Pattern 2: Key-value pairs on separate lines
    # Matches: "name": "Heart Failure"  followed by  "reasoning": "..."
    kv_pattern = r'["\']?(?:name|Name)["\']?\s*:\s*["\']([^"\']+)["\'][^{}\n]*["\']?(?:reasoning|Reasoning|reason)["\']?\s*:\s*["\']([^"\']+)["\']'
    kv_matches = re.findall(kv_pattern, text, re.DOTALL | re.IGNORECASE)

    for name, reasoning in kv_matches[:5]:
        name = name.strip()
        reasoning = reasoning.strip()
        if name and name not in [d.name for d in diagnoses]:
            diagnoses.append(DiagnosisWithReasoning(name=name, reasoning=reasoning))

    # If we found enough diagnoses, return them
    if len(diagnoses) >= 3:
        while len(diagnoses) < 5:
            diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))
        return diagnoses[:5]

    # Pattern 3: Numbered list format
    # Matches: 1. Diagnosis Name\n  Reasoning text...
    numbered_pattern = r'(?:^|\n)\s*(\d+)[.):\s]+([^\n]+?)(?:\n|$)(.*?)(?=(?:\n\s*\d+[.):\s])|$)'
    numbered_matches = re.findall(numbered_pattern, text, re.DOTALL | re.MULTILINE)

    for match in numbered_matches[:5]:
        name = match[1].strip()
        reasoning = match[2].strip() if len(match) > 2 else ""
        if name and name not in [d.name for d in diagnoses]:
            diagnoses.append(DiagnosisWithReasoning(name=name, reasoning=reasoning))

    # Pad to 5 diagnoses
    while len(diagnoses) < 5:
        diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))

    return diagnoses[:5]


class LLMDiagnosisExtractor:
    """Use an LLM to extract diagnoses from malformed output."""

    def __init__(
        self,
        model_name: str,
        generation_kwargs: Mapping[str, Any] | None = None,
        text_pipeline: Callable | None = None,
    ) -> None:
        self.model_name = model_name
        self._generation_kwargs = self._build_generation_kwargs(generation_kwargs)
        # Reuse passed pipeline if provided, otherwise build our own
        if text_pipeline is not None:
            self._pipeline = text_pipeline
        else:
            self._pipeline = self._build_hf_pipeline()

    def _build_generation_kwargs(self, custom_kwargs: Mapping[str, Any] | None) -> Dict[str, Any]:
        base_kwargs: MutableMapping[str, Any] = {
            "do_sample": False,
            "temperature": 0.1,
            "max_new_tokens": 1024,
            "max_length": None,  # Disable to avoid conflict with max_new_tokens
            "return_full_text": False,
        }
        if custom_kwargs:
            base_kwargs.update(dict(custom_kwargs))
        return dict(base_kwargs)

    def _build_hf_pipeline(self) -> Callable:
        try:
            import torch
        except ImportError:
            torch = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

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
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if hasattr(model, "config") and model.config is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if not multi_gpu and pipeline_kwargs.get("device") == 0:
            model.to("cuda:0")

        return hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **pipeline_kwargs,
        )

    def extract(self, raw_output: str) -> List[DiagnosisWithReasoning]:
        """Use LLM to extract diagnoses from malformed output."""

        prompt = self._build_extraction_prompt(raw_output)

        try:
            outputs = self._pipeline(prompt, **self._generation_kwargs)
            if not outputs:
                return self._empty_diagnoses()

            generated = outputs[0].get("generated_text") or outputs[0].get("text") or ""
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()

            # Parse the LLM's extraction attempt
            return self._parse_llm_extraction(generated)

        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return self._empty_diagnoses()
    
    def extract_batch(
        self,
        raw_outputs: List[str],
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> List[List[DiagnosisWithReasoning]]:
        """Batch extraction for malformed outputs."""
        if not raw_outputs:
            return []

        prompts = [self._build_extraction_prompt(x) for x in raw_outputs]
        results: List[List[DiagnosisWithReasoning]] = []

        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                iterator,
                desc="Extracting diagnoses (LLM)",
                total=(len(prompts) + batch_size - 1) // batch_size,
            )

        for start in iterator:
            batch_prompts = prompts[start : start + batch_size]

            try:
                outputs = self._pipeline(batch_prompts, **self._generation_kwargs)

                # HF returns: List[List[Dict]] or List[Dict] depending on pipeline/settings
                for prompt, output in zip(batch_prompts, outputs):
                    if not output:
                        results.append(self._empty_diagnoses())
                        continue

                    out_item = output[0] if isinstance(output, list) else output
                    generated = out_item.get("generated_text") or out_item.get("text") or ""
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()

                    try:
                        results.append(self._parse_llm_extraction(generated))
                    except Exception:
                        results.append(self._empty_diagnoses())

            except Exception as e:
                # Fallback: degrade gracefully to sequential for this chunk
                print(f"Batch LLM extraction failed, falling back to sequential: {e}")
                for prompt in batch_prompts:
                    try:
                        out = self._pipeline(prompt, **self._generation_kwargs)
                        out_item = out[0] if out else {}
                        generated = out_item.get("generated_text") or out_item.get("text") or ""
                        if generated.startswith(prompt):
                            generated = generated[len(prompt):].strip()
                        results.append(self._parse_llm_extraction(generated))
                    except Exception:
                        results.append(self._empty_diagnoses())

        return results

    def _build_extraction_prompt(self, raw_output: str) -> str:
        """Build a prompt asking the LLM to extract structured diagnoses."""
        return f"""Extract medical diagnoses from the following text. For each diagnosis, provide:
1. The diagnosis name
2. The reasoning/evidence supporting it

Output ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "diagnoses": [
    {{"name": "Diagnosis 1", "reasoning": "Reasoning for diagnosis 1"}},
    {{"name": "Diagnosis 2", "reasoning": "Reasoning for diagnosis 2"}},
    {{"name": "Diagnosis 3", "reasoning": "Reasoning for diagnosis 3"}},
    {{"name": "Diagnosis 4", "reasoning": "Reasoning for diagnosis 4"}},
    {{"name": "Diagnosis 5", "reasoning": "Reasoning for diagnosis 5"}}
  ]
}}

Text to extract from:
{raw_output}

JSON output:"""

    def _parse_llm_extraction(self, generated: str) -> List[DiagnosisWithReasoning]:
        """Parse the LLM's extraction output."""
        # Try to extract JSON
        json_str = generated.strip()

        # Remove markdown code blocks if present
        if "```json" in json_str:
            match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
        elif "```" in json_str:
            match = re.search(r"```\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)

        # Normalize and parse
        json_str = _normalize_json_string(json_str)

        try:
            data = json.loads(json_str)
            diagnoses_data = data.get("diagnoses") or data.get("diagnosis") or []

            diagnoses = []
            for d in diagnoses_data[:5]:
                if isinstance(d, dict):
                    name = (d.get("name") or d.get("Name") or "").strip()
                    reasoning = (d.get("reasoning") or d.get("Reasoning") or "").strip()
                    diagnoses.append(DiagnosisWithReasoning(name=name, reasoning=reasoning))

            # Pad to 5
            while len(diagnoses) < 5:
                diagnoses.append(DiagnosisWithReasoning(name="", reasoning=""))

            return diagnoses[:5]

        except json.JSONDecodeError:
            # If LLM's output is also malformed, try regex as last resort
            diagnoses = _fallback_parse_diagnoses(generated)
            return diagnoses if diagnoses else self._empty_diagnoses()

    def _empty_diagnoses(self) -> List[DiagnosisWithReasoning]:
        """Return empty diagnoses list."""
        return [DiagnosisWithReasoning(name="", reasoning="") for _ in range(5)]


class StructuredDiagnosisGenerator:
    """Generator that produces structured JSON diagnosis output."""

    def __init__(
        self,
        model_name: str,
        generation_kwargs: Mapping[str, Any] | None = None,
        text_pipeline: Callable | None = None,
        use_llm_extraction: bool = True,
        baseline_mode: bool = False,
    ) -> None:
        self.model_name = model_name
        self.baseline_mode = baseline_mode  # If True, skip RAG and use zero-shot prompt
        self._generation_kwargs = self._build_generation_kwargs(generation_kwargs)

        if text_pipeline is None:
            self._pipeline = self._build_hf_pipeline()
        else:
            self._pipeline = text_pipeline

        self.last_prompt: str | None = None

        # Initialize LLM extractor for parsing failures (reuses generator's pipeline)
        self.llm_extractor: LLMDiagnosisExtractor | None = None
        if use_llm_extraction:
            try:
                print("Initializing LLM diagnosis extractor (reusing generator pipeline)...")
                self.llm_extractor = LLMDiagnosisExtractor(
                    model_name=model_name,
                    generation_kwargs={"do_sample": False, "temperature": 0.1, "max_new_tokens": 1024},
                    text_pipeline=self._pipeline,  # Reuse the generator's pipeline
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM extractor: {e}")
                self.llm_extractor = None

    def _build_generation_kwargs(self, custom_kwargs: Mapping[str, Any] | None) -> Dict[str, Any]:
        base_kwargs: MutableMapping[str, Any] = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 1024,  # Need more tokens for detailed reasoning
            "max_length": None,  # Disable to avoid conflict with max_new_tokens
            "return_full_text": False,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
        }
        if custom_kwargs:
            base_kwargs.update(dict(custom_kwargs))
        return dict(base_kwargs)

    def _build_hf_pipeline(self) -> Callable:
        try:
            import torch
        except ImportError:
            torch = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

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
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if hasattr(model, "config") and model.config is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if not multi_gpu and pipeline_kwargs.get("device") == 0:
            model.to("cuda:0")

        return hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **pipeline_kwargs,
        )

    def generate(
        self,
        patient_context: str,
        pre_evidence: List[RetrievedDocument],
    ) -> StructuredDiagnosisOutput:
        """Generate structured diagnosis output."""

        if self.baseline_mode:
            prompt = _build_baseline_diagnosis_prompt(patient_context)
        else:
            prompt = _build_structured_diagnosis_prompt(patient_context, pre_evidence)
        self.last_prompt = prompt

        outputs = self._pipeline(prompt, **self._generation_kwargs)
        if not outputs:
            return StructuredDiagnosisOutput(
                raw_output="",
                parse_success=False,
                parse_error="No output from model",
            )

        generated = outputs[0].get("generated_text") or outputs[0].get("text") or ""
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        return _parse_structured_output(generated, llm_extractor=self.llm_extractor)

    def generate_batch(
        self,
        contexts_and_evidence: List[tuple[str, List[RetrievedDocument]]],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[StructuredDiagnosisOutput]:
        """Generate structured outputs for multiple cases.

        Uses batched LLM extraction for any outputs that fail JSON parsing,
        rather than processing them sequentially.
        """

        if self.baseline_mode:
            prompts = [
                _build_baseline_diagnosis_prompt(ctx)
                for ctx, evidence in contexts_and_evidence
            ]
        else:
            prompts = [
                _build_structured_diagnosis_prompt(ctx, evidence)
                for ctx, evidence in contexts_and_evidence
            ]

        if prompts:
            self.last_prompt = prompts[-1]

        results: List[StructuredDiagnosisOutput] = []

        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating diagnoses", total=(len(prompts) + batch_size - 1) // batch_size)

        for start in iterator:
            batch_prompts = prompts[start:start + batch_size]

            try:
                outputs = self._pipeline(batch_prompts, **self._generation_kwargs)

                for prompt, output in zip(batch_prompts, outputs):
                    if not output:
                        results.append(StructuredDiagnosisOutput(
                            raw_output="",
                            parse_success=False,
                            parse_error="No output from model",
                        ))
                        continue

                    out_item = output[0] if isinstance(output, list) else output
                    generated = out_item.get("generated_text") or out_item.get("text") or ""
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()

                    # Parse without LLM extraction first, skip regex fallback (we'll batch failed ones later)
                    results.append(_parse_structured_output(generated, llm_extractor=None, skip_fallback=True))

            except Exception as e:
                # Fallback to single generation
                for prompt in batch_prompts:
                    try:
                        outputs = self._pipeline(prompt, **self._generation_kwargs)
                        out_item = outputs[0] if outputs else {}
                        generated = out_item.get("generated_text") or out_item.get("text") or ""
                        if generated.startswith(prompt):
                            generated = generated[len(prompt):].strip()
                        # Parse without LLM extraction first, skip regex fallback
                        results.append(_parse_structured_output(generated, llm_extractor=None, skip_fallback=True))
                    except Exception:
                        results.append(StructuredDiagnosisOutput(
                            raw_output="",
                            parse_success=False,
                            parse_error=str(e),
                        ))

        # Batch LLM extraction for all failed parses
        if self.llm_extractor is not None:
            failed_indices = [
                i for i, r in enumerate(results)
                if not r.parse_success and r.raw_output
            ]

            if failed_indices:
                failed_raw_outputs = [results[i].raw_output for i in failed_indices]

                if show_progress:
                    print(f"Batched LLM extraction for {len(failed_indices)} failed parses...")

                extracted_diagnoses = self.llm_extractor.extract_batch(
                    failed_raw_outputs,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )

                # Update results with extracted diagnoses
                for idx, diagnoses in zip(failed_indices, extracted_diagnoses):
                    results[idx].diagnoses = diagnoses
                    # Mark as successful if we got at least 3 complete diagnoses (name AND reasoning)
                    complete_diagnoses = sum(1 for d in diagnoses if d.name and d.reasoning)
                    if complete_diagnoses >= 3:
                        results[idx].parse_success = True
                        results[idx].parse_error = None
                    else:
                        results[idx].parse_error = f"LLM extraction: only {complete_diagnoses}/5 complete diagnoses"

        return results


class EvidenceRLPipeline:
    """
    End-to-end pipeline for EvidenceRL:
    patient query -> pre-retrieval -> generation

    Reward computation is done separately in the analysis phase.
    """

    def __init__(
        self,
        documents: Sequence[Document] | None = None,
        model_name: str = "",
        embedding_model_name: str | None = None,
        top_k_pre: int = 3,
        generation_kwargs: Mapping[str, Any] | None = None,
        chunk_size: int = 320,
        chunk_overlap: int = 64,
        embedding_batch_size: int = 32,
        use_llm_extraction: bool = True,
        faiss_index_dir: str | None = None,
        baseline_mode: bool = False,
    ) -> None:
        """Initialize the EvidenceRL pipeline.

        Args:
            documents: Knowledge documents to index. Can be None if faiss_index_dir is provided
                      or baseline_mode is True.
            model_name: HuggingFace model path for generation.
            embedding_model_name: Model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2).
            top_k_pre: Number of documents to retrieve per query.
            generation_kwargs: Additional kwargs for the generator.
            chunk_size: Chunk size in tokens for document splitting.
            chunk_overlap: Overlap between chunks in tokens.
            embedding_batch_size: Batch size for embedding computation.
            use_llm_extraction: Whether to use LLM for fallback parsing.
            faiss_index_dir: Path to pre-built FAISS index. If provided, documents are ignored
                            and the index is loaded from disk (much faster startup).
            baseline_mode: If True, skip retrieval and use zero-shot generation (no RAG).
        """
        self.baseline_mode = baseline_mode

        # Skip document loading and retriever setup for baseline mode
        if baseline_mode:
            print("Baseline mode enabled: skipping document retrieval (zero-shot generation)")
            self.embedder = None
            self.store = None
            self.retriever = None
            self.top_k_pre = 0
            self.generator = StructuredDiagnosisGenerator(
                model_name=model_name,
                generation_kwargs=generation_kwargs,
                use_llm_extraction=use_llm_extraction,
                baseline_mode=True,
            )
            return
        # Set up embedder
        self.embedder = HuggingFaceEmbedder(
            model_name=embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2",
            batch_size=embedding_batch_size,
        )

        # Set up document store and retriever
        if faiss_index_dir and _USE_FAISS:
            # Load pre-built FAISS index (fast path)
            if FAISSDocumentStore.index_exists(faiss_index_dir):
                print(f"Loading pre-built FAISS index from {faiss_index_dir}...")
                self.store = FAISSDocumentStore.load(
                    directory=faiss_index_dir,
                    embedder=self.embedder,
                    use_gpu=False,
                )
                self.retriever = FAISSRetriever(self.store)
            else:
                raise FileNotFoundError(
                    f"FAISS index not found at {faiss_index_dir}. "
                    f"Build it first with: python scripts/build_faiss_index.py --output-dir {faiss_index_dir}"
                )
        elif _USE_FAISS:
            # Build FAISS index from documents (slow path)
            if documents is None:
                raise ValueError("documents must be provided when faiss_index_dir is not set")

            # Chunk documents first (same as DocumentStore does)
            def chunk_text(text: str, chunk_sz: int, overlap: int) -> List[str]:
                tokens = text.split()
                if not tokens:
                    return []
                chunks_list = []
                start = 0
                while start < len(tokens):
                    end = min(len(tokens), start + chunk_sz)
                    chunks_list.append(" ".join(tokens[start:end]))
                    if end == len(tokens):
                        break
                    start = end - overlap
                return chunks_list

            chunked_docs = []
            for doc in documents:
                chunks = chunk_text(doc.text, chunk_size, chunk_overlap) or [doc.text]
                for idx, chunk in enumerate(chunks):
                    metadata = dict(doc.metadata) if doc.metadata else {}
                    metadata.setdefault("source_doc_id", doc.doc_id)
                    metadata["chunk_id"] = idx
                    chunk_doc_id = doc.doc_id if len(chunks) == 1 else f"{doc.doc_id}::chunk-{idx}"
                    chunked_docs.append(
                        Document(
                            doc_id=chunk_doc_id,
                            text=chunk,
                            metadata=metadata,
                        )
                    )

            self.store = FAISSDocumentStore(
                documents=chunked_docs,
                embedder=self.embedder,
                index_type="HNSW",  # Fastest option
                use_gpu=False,  # HNSW works best on CPU
            )
            self.retriever = FAISSRetriever(self.store)
        else:
            # Fallback to exhaustive search
            if documents is None:
                raise ValueError("documents must be provided when FAISS is not available")
            self.store = DocumentStore(
                documents,
                embedder=self.embedder,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.retriever = SemanticRetriever(self.store)

        # Set up generator
        self.generator = StructuredDiagnosisGenerator(
            model_name=model_name,
            generation_kwargs=generation_kwargs,
            use_llm_extraction=use_llm_extraction,
        )

        self.top_k_pre = top_k_pre

    def run_single(
        self,
        case: PatientCase,
    ) -> EvidenceRLResult:
        """Run the full pipeline for a single patient case."""

        # Clean patient context
        cleaned_context = clean_patient_information(case.context)

        # Step 1: Pre-retrieval (skip in baseline mode)
        if self.baseline_mode:
            pre_evidence = []
        else:
            pre_evidence = self.retriever.retrieve(cleaned_context, top_k=self.top_k_pre)

        # Step 2: Generation
        structured_output = self.generator.generate(cleaned_context, pre_evidence)

        return EvidenceRLResult(
            hadm_id=case.hadm_id,
            subject_id=case.subject_id,
            patient_context=cleaned_context,
            pre_evidence=pre_evidence,
            structured_output=structured_output,
            generation_prompt=self.generator.last_prompt or "",
            ground_truth_diagnoses=list(case.diagnoses),
        )

    def run_batch(
        self,
        cases: Sequence[PatientCase],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[EvidenceRLResult]:
        """Run the pipeline for multiple patient cases with batched generation."""

        results: List[EvidenceRLResult] = []

        # Step 1: Pre-retrieval for all cases (skip in baseline mode)
        pre_evidence_list: List[List[RetrievedDocument]] = []
        cleaned_contexts: List[str] = []

        if self.baseline_mode:
            # Baseline mode: skip retrieval, use empty evidence
            iter_cases = tqdm(cases, desc="Preparing contexts") if show_progress else cases
            for case in iter_cases:
                cleaned = clean_patient_information(case.context)
                cleaned_contexts.append(cleaned)
                pre_evidence_list.append([])  # Empty evidence for baseline
        else:
            iter_cases = tqdm(cases, desc="Pre-retrieval") if show_progress else cases
            for case in iter_cases:
                cleaned = clean_patient_information(case.context)
                cleaned_contexts.append(cleaned)
                pre_evidence = self.retriever.retrieve(cleaned, top_k=self.top_k_pre)
                pre_evidence_list.append(pre_evidence)

        # Step 2: Batch generation
        contexts_and_evidence = list(zip(cleaned_contexts, pre_evidence_list))
        structured_outputs = self.generator.generate_batch(
            contexts_and_evidence,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Step 3: Create results (no reward computation - done in analysis phase)
        for case, cleaned_context, pre_evidence, structured_output in zip(
            cases, cleaned_contexts, pre_evidence_list, structured_outputs
        ):
            result = EvidenceRLResult(
                hadm_id=case.hadm_id,
                subject_id=case.subject_id,
                patient_context=cleaned_context,
                pre_evidence=pre_evidence,
                structured_output=structured_output,
                generation_prompt=self.generator.last_prompt or "",
                ground_truth_diagnoses=list(case.diagnoses),
            )
            results.append(result)

        return results


def summarize_results(results: Sequence[EvidenceRLResult]) -> Dict[str, Any]:
    """Compute summary statistics for a batch of results.

    Note: Reward statistics are computed in the analysis phase.
    """

    if not results:
        return {}

    n = len(results)

    # Parse success rate
    parse_successes = sum(1 for r in results if r.structured_output and r.structured_output.parse_success)

    # Average diagnoses with non-empty reasoning
    avg_diagnoses_with_reasoning = sum(
        sum(1 for d in r.structured_output.diagnoses if d.reasoning.strip())
        for r in results if r.structured_output
    ) / n

    return {
        "num_cases": n,
        "parse_success_rate": parse_successes / n,
        "avg_diagnoses_with_reasoning": avg_diagnoses_with_reasoning,
    }


__all__ = [
    "DiagnosisWithReasoning",
    "StructuredDiagnosisOutput",
    "EvidenceRLResult",
    "StructuredDiagnosisGenerator",
    "EvidenceRLPipeline",
    "summarize_results",
]
