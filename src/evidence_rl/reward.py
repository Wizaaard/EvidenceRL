"""Reward computation for evidence grounding and diagnosis precision.

New reward design (Focus-Then-Verify):
- R_grounding: Uses NLI with section-focused verification to measure if diagnosis reasoning
               is entailed by (grounded in) patient context + evidence
- R_precision: Measures if diagnosis matches ground truth (using LLM judge)
- R_combined: Weighted combination of grounding and precision

Focus-Then-Verify Architecture (solves context window and dilution problems):
- Stage 1: For each reasoning sentence, select the most relevant patient section
           using bi-encoder (embedding similarity)
- Stage 2: Construct "anchored pairs" for NLI verification:
           Premise = [Best Patient Section] + [One Evidence Document]
           Hypothesis = Reasoning sentence
- Aggregation: MAX across evidence docs per sentence, then MEAN across sentences
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence, List, Optional, Dict, TYPE_CHECKING

import numpy as np

from .documents import RetrievedDocument

if TYPE_CHECKING:
    from .domains.base import DomainConfig


# Patient sections that can be parsed from cleaned patient context
# (kept for backward compatibility — new code should use DomainConfig.section_labels)
_PATIENT_SECTION_LABELS = (
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

# Default medical anchor labels (backward compat)
_DEFAULT_ANCHOR_LABELS = ["Chief complaint", "History of present illness"]


class AnswerJudge(Protocol):
    """Protocol for answer judging."""

    def is_correct_batch(self, prompts: list[str]) -> list[bool]:
        """Judge multiple prompts in batch."""
        ...


class NLIModel(Protocol):
    """Protocol for NLI (Natural Language Inference) model.

    The model should predict entailment/neutral/contradiction for premise-hypothesis pairs.
    """

    def predict(self, premise_hypothesis_pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Predict NLI scores for premise-hypothesis pairs.

        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples

        Returns:
            List of dicts with keys 'entailment', 'neutral', 'contradiction'
            representing probabilities for each class.
        """
        ...


class TextEmbedder(Protocol):
    """Protocol for text embedding (bi-encoder for section selection)."""

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        ...


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex.

    Handles common sentence-ending punctuation while avoiding splits on
    abbreviations like "e.g.", "i.e.", "Dr.", etc.
    """
    if not text.strip():
        return []

    # Simple sentence splitting - split on period/question/exclamation followed by space or end
    # Avoid splitting on common abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    # If no sentence boundaries found, return the whole text as one sentence
    if not sentences:
        sentences = [text.strip()]

    return sentences


def _parse_patient_sections(patient_context: str) -> Dict[str, str]:
    """Parse cleaned patient context into individual sections.

    The patient_context is formatted as:
        "Section Name: content...\nAnother Section: content..."

    Returns:
        Dictionary mapping section label -> section content
    """
    sections = {}

    if not patient_context.strip():
        return sections

    # Build pattern to find section headers
    # Each section starts with "Label:" at the beginning of a line or after newline
    section_pattern = r'^(' + '|'.join(re.escape(label) for label in _PATIENT_SECTION_LABELS) + r'):\s*'

    # Split by section headers
    lines = patient_context.split('\n')
    current_section = None
    current_content = []

    for line in lines:
        # Check if this line starts a new section
        match = None
        for label in _PATIENT_SECTION_LABELS:
            if line.strip().lower().startswith(label.lower() + ':'):
                match = label
                # Extract content after the colon
                content_start = line.lower().find(label.lower() + ':') + len(label) + 1
                remaining = line[content_start:].strip()
                break

        if match:
            # Save previous section
            if current_section and current_content:
                sections[current_section] = ' '.join(current_content).strip()

            # Start new section
            current_section = match
            current_content = [remaining] if remaining else []
        elif current_section:
            # Continue current section
            if line.strip():
                current_content.append(line.strip())

    # Save last section
    if current_section and current_content:
        sections[current_section] = ' '.join(current_content).strip()

    return sections


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _select_relevant_section(
    sentence: str,
    sections: Dict[str, str],
    embedder: TextEmbedder,
) -> tuple[str, str]:
    """Select the most relevant patient section for a sentence using embedding similarity.

    Args:
        sentence: The reasoning sentence to match
        sections: Dictionary of section_label -> section_content
        embedder: Text embedder for computing similarities

    Returns:
        Tuple of (section_label, section_content) for the most relevant section.
        If no sections available, returns ("", "").
    """
    if not sections:
        return ("", "")

    # Get section labels and contents
    section_labels = list(sections.keys())
    section_contents = list(sections.values())

    # Encode all texts: [sentence] + all section contents
    all_texts = [sentence] + section_contents
    embeddings = embedder.encode(all_texts)

    # Convert to numpy arrays
    sentence_emb = np.array(embeddings[0])
    section_embs = [np.array(emb) for emb in embeddings[1:]]

    # Compute similarity with each section
    similarities = [_cosine_similarity(sentence_emb, sec_emb) for sec_emb in section_embs]

    # Find best matching section
    best_idx = int(np.argmax(similarities))
    best_label = section_labels[best_idx]
    best_content = section_contents[best_idx]

    return (best_label, best_content)
#
def _split_into_atomic_facts(text: str) -> List[str]:
    # Split by sentence-ending punctuation
    initial_split = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    
    atomic_facts = []
    for s in initial_split:
        # Split on commas/conjunctions if the sentence is long (likely a list of signs)
        if len(s.split()) > 8:
            # Patterns: ", and", ";", "accompanied by", "alongside"
            sub_parts = re.split(r', and|;|\balongside\b|\bcompounded by\b|\bfurthermore\b', s, flags=re.IGNORECASE)
            atomic_facts.extend([p.strip() for p in sub_parts if len(p.strip()) > 4])
        else:
            atomic_facts.append(s.strip())
    return atomic_facts

def _select_top_k_sections(sentence: str, sections: Dict[str, str], embedder, k: int = 2) -> List[str]:
    """Retrieves top K most relevant section contents using embedding similarity."""
    if not sections: return []

    labels = list(sections.keys())
    contents = list(sections.values())

    # 1. Convert output to numpy arrays immediately
    query_emb = np.array(embedder.encode([sentence])) # Shape: (1, D)
    section_embs = np.array(embedder.encode(contents)) # Shape: (N, D)

    # 2. Compute cosine similarity safely
    # We use query_emb[0] to get a 1D vector for the dot product
    dot_product = np.dot(section_embs, query_emb[0])

    norm_q = np.linalg.norm(query_emb[0])
    norm_s = np.linalg.norm(section_embs, axis=1)

    # Avoid division by zero
    similarities = dot_product / (norm_s * norm_q + 1e-8)

    # 3. Get indices of top k scores
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return [contents[i] for i in top_k_indices]


def _select_top_k_sections_cached(
    sentence: str,
    section_contents: List[str],
    section_embs: np.ndarray,
    embedder: TextEmbedder,
    k: int = 2,
) -> List[str]:
    """Retrieves top K sections using pre-computed section embeddings (cached).

    Args:
        sentence: The reasoning sentence to match
        section_contents: List of section content strings
        section_embs: Pre-computed section embeddings, shape (N, D)
        embedder: Text embedder (only used to encode the sentence)
        k: Number of top sections to return

    Returns:
        List of top-k section contents
    """
    if len(section_contents) == 0:
        return []

    # Only encode the query sentence (sections are cached)
    query_emb = np.array(embedder.encode([sentence]))[0]  # Shape: (D,)

    # Compute cosine similarity
    dot_product = np.dot(section_embs, query_emb)
    norm_q = np.linalg.norm(query_emb)
    norm_s = np.linalg.norm(section_embs, axis=1)

    # Avoid division by zero
    similarities = dot_product / (norm_s * norm_q + 1e-8)

    # Get indices of top k scores
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return [section_contents[i] for i in top_k_indices]


class PatientContextCache:
    """Cache for patient context data to avoid redundant computation across diagnoses.

    This cache stores:
    - Parsed patient sections
    - Section embeddings (computed once)
    - Anchor context string

    Use this when processing multiple diagnoses for the same patient.
    """

    def __init__(
        self,
        patient_context: str,
        embedder: TextEmbedder | None = None,
        domain: DomainConfig | None = None,
    ):
        """Initialize cache for a patient context.

        Args:
            patient_context: The patient's clinical presentation (or question for ALCE)
            embedder: Text embedder for computing section embeddings.
                      If None, section selection will fall back to using full context.
            domain: Optional DomainConfig for domain-specific context parsing.
                    If None, uses legacy medical parsing (backward compatible).
        """
        self.patient_context = patient_context

        if domain is not None:
            # Domain-aware parsing
            self.sections = domain.parse_context(patient_context)
            self.anchor_context = domain.build_anchor_context(self.sections)
            self.non_anchor_sections = domain.get_non_anchor_sections(self.sections)
        else:
            # Legacy medical parsing (backward compatible)
            self.sections = _parse_patient_sections(patient_context)
            anchor_labels = _DEFAULT_ANCHOR_LABELS
            self.anchor_context = "\n".join([
                f"{k}: {self.sections[k]}"
                for k in anchor_labels
                if k in self.sections
            ])
            self.non_anchor_sections = [
                content for label, content in self.sections.items()
                if label not in anchor_labels
            ]

        # Cache all section contents and embeddings (kept for compatibility)
        self.section_contents = list(self.sections.values())
        self.section_embs: Optional[np.ndarray] = None

        if embedder and self.section_contents:
            # Pre-compute section embeddings
            self.section_embs = np.array(embedder.encode(self.section_contents))

def reward_grounding_cached(
    diagnosis_reasoning: str,
    cache: PatientContextCache,
    evidence_docs: Sequence[RetrievedDocument],
    nli_model: NLIModel,
    embedder: TextEmbedder | None = None,
    max_premise_length: int = 1200,  # Keeps total tokens within NLI model's 512-token limit
    sentence_level: bool = True,
) -> tuple[float, float]:
    """Compute grounding reward using cached patient context data.

    This is an optimized version of reward_grounding that uses pre-computed
    non-anchor sections from PatientContextCache. Each section and evidence
    document is paired individually with anchor context for NLI.

    Args:
        diagnosis_reasoning: The reasoning text for a diagnosis
        cache: Pre-computed PatientContextCache for this patient
        evidence_docs: Retrieved evidence documents
        nli_model: NLI model for entailment verification
        embedder: Unused (kept for API compatibility)
        max_premise_length: Maximum character length for premise
        sentence_level: If True, split diagnosis_reasoning into sentences and
            score each sentence independently. grounding_max and grounding_avg
            are then computed as the mean of per-sentence max-abs scores
            (i.e., each sentence must be individually grounded). If False
            (default), the full reasoning is treated as one hypothesis.

    Returns:
        Tuple of (grounding_max, grounding_avg), each in [-1, 1].
    """
    if not diagnosis_reasoning.strip():
        return (0.0, 0.0)

    evidence_texts = [doc.document.text.strip() for doc in evidence_docs if doc.document.text.strip()]

    # Build list of premises (shared across all hypotheses)
    premises = []
    for section_content in cache.non_anchor_sections:
        premise = f"{cache.anchor_context}\n\n{section_content}".strip()
        if len(premise) > max_premise_length:
            premise = premise[:max_premise_length]
        premises.append(premise)
    for evidence_text in evidence_texts:
        premise = f"{cache.anchor_context}\n\n{evidence_text}".strip()
        if len(premise) > max_premise_length:
            premise = premise[:max_premise_length]
        premises.append(premise)
    if not premises:
        if cache.anchor_context.strip():
            premises.append(cache.anchor_context)
        else:
            return (0.0, 0.0)

    # Determine hypotheses
    if sentence_level:
        hypotheses = _split_into_sentences(diagnosis_reasoning)
        if not hypotheses:
            hypotheses = [diagnosis_reasoning]
    else:
        hypotheses = [diagnosis_reasoning]

    # Build all (premise, hypothesis) pairs in one flat list; track sentence boundaries
    all_pairs = []
    sentence_pair_counts = []  # number of pairs per sentence
    for hyp in hypotheses:
        count = 0
        for premise in premises:
            all_pairs.append((premise, hyp))
            count += 1
        sentence_pair_counts.append(count)

    # Single batched NLI call
    predictions = nli_model.predict(all_pairs)
    all_scores = [p.get('entailment', 0.0) - p.get('contradiction', 0.0) for p in predictions]

    # Aggregate: per-sentence max-abs, then mean across sentences
    per_sentence_scores = []
    idx = 0
    for count in sentence_pair_counts:
        sentence_scores = all_scores[idx: idx + count]
        per_sentence_scores.append(max(sentence_scores, key=abs))
        idx += count

    grounding_max = max(per_sentence_scores, key=abs)  # strongest sentence (sign preserved)
    grounding_avg = sum(per_sentence_scores) / len(per_sentence_scores)  # mean across sentences

    return (grounding_max, grounding_avg)


def reward_grounding(
    diagnosis_reasoning: str,
    patient_context: str,
    evidence_docs: Sequence[RetrievedDocument],
    nli_model: NLIModel,
    embedder: TextEmbedder | None = None,
    max_premise_length: int = 1200,  # Keeps total tokens within NLI model's 512-token limit
    sentence_level: bool = True,
    domain: DomainConfig | None = None,
) -> tuple[float, float]:
    """Compute grounding reward by checking NLI against each source individually.

    Each patient section and each evidence document is paired separately with the
    anchor context to form a short, focused premise. The NLI model checks each
    premise against the hypothesis (diagnosis reasoning) independently.

    Premise construction (one per source):
        anchor_context  +  ONE patient section   →  hypothesis
        anchor_context  +  ONE evidence document  →  hypothesis

    Scoring (sentence_level=True, default):
        - Split diagnosis_reasoning into sentences
        - For each sentence: score = max(entailment-contradiction) across all premises
        - grounding_max = grounding_avg = mean of per-sentence scores

    Scoring (sentence_level=False):
        - For each pair: Score = P(Entailment) - P(Contradiction)
        - grounding_max: Score with largest absolute value (sign preserved).
        - grounding_avg: Mean across all pairs.

    Args:
        diagnosis_reasoning: The reasoning text for a diagnosis
        patient_context: The patient's clinical presentation (cleaned, with sections)
        evidence_docs: Retrieved evidence documents (clinical guidelines)
        nli_model: Cross-encoder NLI model for entailment verification
        embedder: Unused (kept for API compatibility)
        max_premise_length: Maximum character length for premise
        sentence_level: If True (default), split reasoning into sentences and score
            each independently. Provides finer-grained grounding signal.

    Returns:
        Tuple of (grounding_max, grounding_avg), each in [-1, 1].
    """
    if not diagnosis_reasoning.strip():
        return (0.0, 0.0)

    if domain is not None:
        sections = domain.parse_context(patient_context)
        anchor_context = domain.build_anchor_context(sections)
        non_anchor_sections = domain.get_non_anchor_sections(sections)
    else:
        sections = _parse_patient_sections(patient_context)
        anchor_labels = _DEFAULT_ANCHOR_LABELS
        anchor_context = "\n".join([f"{k}: {sections[k]}" for k in anchor_labels if k in sections])
        non_anchor_sections = [
            content for label, content in sections.items()
            if label not in anchor_labels
        ]

    evidence_texts = [doc.document.text.strip() for doc in evidence_docs if doc.document.text.strip()]

    # Build premises list
    premises = []
    for section_content in non_anchor_sections:
        premise = f"{anchor_context}\n\n{section_content}".strip()
        if len(premise) > max_premise_length:
            premise = premise[:max_premise_length]
        premises.append(premise)
    for evidence_text in evidence_texts:
        premise = f"{anchor_context}\n\n{evidence_text}".strip()
        if len(premise) > max_premise_length:
            premise = premise[:max_premise_length]
        premises.append(premise)
    if not premises:
        if anchor_context.strip():
            premises.append(anchor_context)
        else:
            return (0.0, 0.0)

    # Determine hypotheses
    if sentence_level:
        hypotheses = _split_into_sentences(diagnosis_reasoning)
        if not hypotheses:
            hypotheses = [diagnosis_reasoning]
    else:
        hypotheses = [diagnosis_reasoning]

    # Build all pairs in one flat list; track per-sentence counts
    all_pairs = []
    sentence_pair_counts = []
    for hyp in hypotheses:
        count = 0
        for premise in premises:
            all_pairs.append((premise, hyp))
            count += 1
        sentence_pair_counts.append(count)

    # Single batched NLI call
    predictions = nli_model.predict(all_pairs)
    all_scores = [p.get('entailment', 0.0) - p.get('contradiction', 0.0) for p in predictions]

    if sentence_level:
        # Per-sentence max-abs, then mean across sentences
        per_sentence_scores = []
        idx = 0
        for count in sentence_pair_counts:
            sentence_scores = all_scores[idx: idx + count]
            per_sentence_scores.append(max(sentence_scores, key=abs))
            idx += count
        grounding_max = max(per_sentence_scores, key=abs)  # strongest sentence (sign preserved)
        grounding_avg = sum(per_sentence_scores) / len(per_sentence_scores)  # mean across sentences
    else:
        grounding_max = max(all_scores, key=abs)
        grounding_avg = sum(all_scores) / len(all_scores)

    return (grounding_max, grounding_avg)


def reward_precision(
    diagnosis_name: str,
    ground_truth_diagnoses: List[str],
    judge: AnswerJudge,
    patient_context: str,
    domain: DomainConfig | None = None,
) -> float:
    """Compute precision reward: whether prediction matches ground truth using LLM judge.

    Args:
        diagnosis_name: The predicted diagnosis/answer name
        ground_truth_diagnoses: List of ground truth answers
        judge: LLM judge for semantic matching
        patient_context: Context for judge prompt
        domain: Optional DomainConfig for domain-specific judge prompts.
                If None, uses legacy medical judge prompt.

    Returns:
        Precision score: 1.0 if correct, 0.0 if incorrect
    """
    if not diagnosis_name.strip() or not ground_truth_diagnoses:
        return 0.0

    if domain is not None:
        prompt = domain.build_judge_prompt(diagnosis_name, ground_truth_diagnoses)
    else:
        # Legacy medical judge prompt
        ground_truth_str = "\n- ".join(ground_truth_diagnoses)
        prompt = f"""You are evaluating the diagnosis prediction of a clinical model.

CANDIDATE ANSWER: "{diagnosis_name}"

ACCEPTED GROUND TRUTHS:
- {ground_truth_str}

TASK: Does the CANDIDATE ANSWER semantically match ANY of the ACCEPTED GROUND TRUTHS?
Respond 'TRUE' if the candidate refers to the same underlying clinical concept as any item in the list (allowing for synonyms, abbreviations, or minor wording differences).
Respond 'FALSE' if it represents a different clinical concept, severity, or anatomical location.

Verdict:"""

    # Use judge to determine if correct
    try:
        results = judge.is_correct_batch([prompt])
        is_correct = results[0] if results else False
        return 1.0 if is_correct else 0.0
    except Exception:
        # If judge fails, return 0.0
        return 0.0


def combined_reward(
    diagnosis_name: str,
    diagnosis_reasoning: str,
    patient_context: str,
    evidence_docs: Sequence[RetrievedDocument],
    ground_truth_diagnoses: List[str],
    nli_model: NLIModel,
    judge: AnswerJudge,
    embedder: TextEmbedder | None = None,
    weight_grounding: float = 0.5,
    domain: DomainConfig | None = None,
) -> tuple[float, float, float, float]:
    """Compute combined reward from grounding and precision.

    Args:
        diagnosis_name: Predicted diagnosis name
        diagnosis_reasoning: Reasoning for the diagnosis
        patient_context: Patient clinical presentation
        evidence_docs: Retrieved clinical guidelines
        ground_truth_diagnoses: Ground truth diagnosis names
        nli_model: NLI model for entailment-based grounding evaluation
        judge: LLM judge for precision evaluation
        embedder: Unused (kept for API compatibility)
        weight_grounding: Weight for grounding component (default 0.5)

    Returns:
        Tuple of (combined_reward, grounding_max, grounding_avg, precision_reward)
        - grounding_max: max by absolute value (sign preserved) — strongest signal
        - grounding_avg: mean across all source pairs — overall consistency
        - combined_reward uses grounding_max for the combined score
    """
    if not 0.0 <= weight_grounding <= 1.0:
        raise ValueError("weight_grounding must be within [0, 1]")

    # Compute grounding reward
    r_grounding_max, r_grounding_avg = reward_grounding(
        diagnosis_reasoning=diagnosis_reasoning,
        patient_context=patient_context,
        evidence_docs=evidence_docs,
        nli_model=nli_model,
        embedder=embedder,
        domain=domain,
    )

    # Compute precision reward
    r_precision = reward_precision(
        diagnosis_name=diagnosis_name,
        ground_truth_diagnoses=ground_truth_diagnoses,
        judge=judge,
        patient_context=patient_context,
        domain=domain,
    )

    # Combined reward uses grounding_max
    r_combined = weight_grounding * r_grounding_max + (1.0 - weight_grounding) * r_precision

    return r_combined, r_grounding_max, r_grounding_avg, r_precision


def combined_reward_batch(
    diagnoses: List[Dict[str, str]],
    patient_context: str,
    evidence_docs: Sequence[RetrievedDocument],
    ground_truth_diagnoses: List[str],
    nli_model: NLIModel,
    judge: AnswerJudge,
    embedder: TextEmbedder | None = None,
    weight_grounding: float = 0.5,
    max_premise_length: int = 1200,  # Keeps total tokens within NLI model's 512-token limit
    domain: DomainConfig | None = None,
) -> List[tuple[float, float, float, float]]:
    """Compute combined rewards for multiple diagnoses with batched inference.

    This optimized function processes all diagnoses for a patient in batch:
    1. Caches patient context (anchor + non-anchor sections)
    2. For each diagnosis, creates individual NLI pairs: anchor+section, anchor+evidence
    3. Runs single batch NLI prediction across all diagnoses
    4. Collects all judge prompts and runs single batch prediction
    5. Distributes results back to each diagnosis

    Args:
        diagnoses: List of {"name": str, "reasoning": str} dicts
        patient_context: Patient clinical presentation
        evidence_docs: Retrieved clinical guidelines
        ground_truth_diagnoses: Ground truth diagnosis names
        nli_model: NLI model for grounding evaluation
        judge: LLM judge for precision evaluation
        embedder: Unused (kept for API compatibility)
        weight_grounding: Weight for grounding component (default 0.5)
        max_premise_length: Maximum character length for NLI premise

    Returns:
        List of (combined_reward, grounding_max, grounding_avg, precision_reward) tuples,
        one per diagnosis in the same order as input.
    """
    if not 0.0 <= weight_grounding <= 1.0:
        raise ValueError("weight_grounding must be within [0, 1]")

    if not diagnoses:
        return []

    # === Phase 1: Create patient context cache ===
    cache = PatientContextCache(patient_context, embedder, domain=domain)

    evidence_texts = [doc.document.text.strip() for doc in evidence_docs if doc.document.text.strip()]

    # === Phase 2: Collect all NLI pairs across diagnoses ===
    # Each diagnosis gets individual pairs: anchor+section and anchor+evidence
    all_nli_pairs = []
    diagnosis_nli_ranges = []  # Track which pairs belong to which diagnosis

    for diag_idx, diag in enumerate(diagnoses):
        reasoning = diag.get("reasoning", "").strip()
        start_idx = len(all_nli_pairs)

        if reasoning:
            hypothesis = reasoning

            # Pair with each non-anchor patient section
            for section_content in cache.non_anchor_sections:
                premise = f"{cache.anchor_context}\n\n{section_content}".strip()
                if len(premise) > max_premise_length:
                    premise = premise[:max_premise_length]
                all_nli_pairs.append((premise, hypothesis))

            # Pair with each evidence document
            for evidence_text in evidence_texts:
                premise = f"{cache.anchor_context}\n\n{evidence_text}".strip()
                if len(premise) > max_premise_length:
                    premise = premise[:max_premise_length]
                all_nli_pairs.append((premise, hypothesis))

            # Fallback: anchor alone if no sections and no evidence
            if not cache.non_anchor_sections and not evidence_texts:
                if cache.anchor_context.strip():
                    all_nli_pairs.append((cache.anchor_context, hypothesis))

        end_idx = len(all_nli_pairs)
        num_sources = len(cache.non_anchor_sections) + len(evidence_texts)
        diagnosis_nli_ranges.append((start_idx, end_idx, max(num_sources, 1)))

    # === Phase 3: Batch NLI prediction ===
    if all_nli_pairs:
        nli_predictions = nli_model.predict(all_nli_pairs)
        nli_scores = [p.get('entailment', 0.0) - p.get('contradiction', 0.0) for p in nli_predictions]
    else:
        nli_scores = []

    # === Phase 4: Collect all judge prompts ===
    judge_prompts = []
    ground_truth_str = "\n- ".join(ground_truth_diagnoses) if ground_truth_diagnoses else ""

    for diag in diagnoses:
        name = diag.get("name", "").strip()
        if name and ground_truth_diagnoses:
            if domain is not None:
                prompt = domain.build_judge_prompt(name, ground_truth_diagnoses)
            else:
                # Legacy medical judge prompt
                prompt = f"""You are evaluating the diagnosis prediction of a clinical model.

CANDIDATE ANSWER: "{name}"

ACCEPTED GROUND TRUTHS:
- {ground_truth_str}

TASK: Does the CANDIDATE ANSWER semantically match ANY of the ACCEPTED GROUND TRUTHS?
Respond 'TRUE' if the candidate refers to the same underlying clinical concept as any item in the list (allowing for synonyms, abbreviations, or minor wording differences).
Respond 'FALSE' if it represents a different clinical concept, severity, or anatomical location.

Verdict:"""
            judge_prompts.append(prompt)
        else:
            judge_prompts.append(None)  # Placeholder for empty diagnoses

    # === Phase 5: Batch judge prediction ===
    valid_prompts = [p for p in judge_prompts if p is not None]
    if valid_prompts:
        try:
            judge_results = judge.is_correct_batch(valid_prompts)
        except Exception:
            judge_results = [False] * len(valid_prompts)
    else:
        judge_results = []

    # Map judge results back to diagnoses
    judge_iter = iter(judge_results)
    precision_scores = []
    for prompt in judge_prompts:
        if prompt is not None:
            precision_scores.append(1.0 if next(judge_iter) else 0.0)
        else:
            precision_scores.append(0.0)

    # === Phase 6: Compute grounding scores from NLI results ===
    grounding_max_scores = []
    grounding_avg_scores = []
    for start_idx, end_idx, num_sources in diagnosis_nli_ranges:
        if start_idx == end_idx:
            # No NLI pairs (empty reasoning)
            grounding_max_scores.append(0.0)
            grounding_avg_scores.append(0.0)
        else:
            diag_scores = nli_scores[start_idx:end_idx]
            # Max by absolute value (sign preserved) — strongest signal
            grounding_max_scores.append(max(diag_scores, key=abs))
            # Mean across all sources — overall consistency
            grounding_avg_scores.append(sum(diag_scores) / len(diag_scores))

    # === Phase 7: Compute combined rewards ===
    results = []
    for i in range(len(diagnoses)):
        r_grounding_max = grounding_max_scores[i]
        r_grounding_avg = grounding_avg_scores[i]
        r_precision = precision_scores[i]
        # Combined reward uses grounding_max
        r_combined = weight_grounding * r_grounding_max + (1.0 - weight_grounding) * r_precision
        results.append((r_combined, r_grounding_max, r_grounding_avg, r_precision))

    return results


def compute_reward_at_k(
    per_diagnosis_rewards: List[float],
    max_k: int = 5,
) -> dict[int, float]:
    """Compute reward@k from per-diagnosis rewards.

    Similar to precision@k and recall@k, computes the average reward
    for the top-k diagnoses.

    Args:
        per_diagnosis_rewards: List of reward values (one per diagnosis, in order)
        max_k: Maximum k value to compute (default: 5)

    Returns:
        Dictionary mapping k -> average reward for top-k diagnoses

    Example:
        >>> rewards = [0.8, 0.6, 0.4, 0.2, 0.1]
        >>> compute_reward_at_k(rewards, max_k=3)
        {1: 0.8, 2: 0.7, 3: 0.6}
    """
    if not per_diagnosis_rewards:
        return {k: 0.0 for k in range(1, max_k + 1)}

    reward_at_k = {}
    for k in range(1, max_k + 1):
        # Take top-k diagnoses (or all if fewer than k)
        top_k_rewards = per_diagnosis_rewards[:k]
        if top_k_rewards:
            reward_at_k[k] = sum(top_k_rewards) / len(top_k_rewards)
        else:
            reward_at_k[k] = 0.0

    return reward_at_k


__all__ = [
    "reward_grounding",
    "reward_grounding_cached",
    "reward_precision",
    "combined_reward",
    "combined_reward_batch",
    "compute_reward_at_k",
    "PatientContextCache",
    "AnswerJudge",
    "NLIModel",
    "TextEmbedder",
]
