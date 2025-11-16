"""Core data structures used across the EvidenceRL package.

This module provides lightweight containers for representing text documents and
retrieval results.  Using dataclasses keeps the code explicit while providing a
clear contract for other modules that operate on documents and retrieval hits.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class Document:
    """Container for a text document in the corpus.

    Attributes
    ----------
    doc_id:
        Unique identifier for the document.  Identifiers are opaque strings but
        typically correspond to a filename or database key.
    text:
        The textual content of the document.
    metadata:
        Optional arbitrary metadata that may be useful for inspection or
        downstream processing.
    """

    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RetrievedDocument:
    """Represents a document returned by the retriever.

    In addition to the :class:`Document`, we store the similarity score that was
    used to rank the document.  Keeping the score makes downstream analysis and
    debugging easier.
    """

    document: Document
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be a numeric value")


CARDIAC_ICD_PREFIXES: Set[str] = {
    "I20",
    "I21",
    "I22",
    "I23",
    "I24",
    "I25",
    "I30",
    "I31",
    "I32",
    "I33",
    "I34",
    "I35",
    "I36",
    "I37",
    "I38",
    "I39",
    "I40",
    "I41",
    "I42",
    "I43",
    "I44",
    "I45",
    "I46",
    "I47",
    "I48",
    "I49",
    "I50",
    "410",
    "411",
    "412",
    "413",
    "414",
    "420",
    "421",
    "422",
    "423",
    "424",
    "425",
    "426",
    "427",
    "428",
}


def _load_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items() if key}


def load_cardiac_icd_documents(
    data_path: str | Path,
    icd_prefixes: Sequence[str] | None = None,
) -> List[Document]:
    """Create documents from the MIMIC-IV-Ext cardiac ICD extracts.

    The loader groups diagnosis rows by their ICD prefix (e.g., ``I20`` or
    ``410``) and produces one :class:`Document` per prefix.  The resulting text
    concatenates representative long titles so the retriever has real clinical
    content to embed.  Concept tags capture the ICD prefix and a normalized
    "cardiac" marker so the expertise-aware filter remains effective.

    Args:
        data_path: Directory containing the cardiac CSV files.
        icd_prefixes: Optional whitelist of ICD prefixes to include.  Defaults
            to the cardiac-focused prefixes described in the prompt.
    """

    base_path = Path(data_path)
    diag_path = base_path / "heart_diagnoses_all_true.csv"
    truncated_path = base_path / "heart_diagnoses_all.csv"

    if not diag_path.exists() and not truncated_path.exists():
        raise FileNotFoundError(
            "Expected cardiac diagnosis CSVs (heart_diagnoses_all_true.csv or heart_diagnoses_all.csv) "
            f"under {base_path}"
        )

    prefixes = {prefix.upper() for prefix in (icd_prefixes or CARDIAC_ICD_PREFIXES)}

    def allowed_prefix(code: str) -> str | None:
        normalized = code.strip().upper()
        if not normalized:
            return None
        truncated = normalized[:3]
        if truncated in prefixes:
            return truncated
        return None

    records: dict[str, dict[str, Any]] = {}

    source_path = diag_path if diag_path.exists() else truncated_path
    for row in _load_rows(source_path):
        prefix = allowed_prefix(row.get("icd_code", ""))
        if prefix is None:
            continue
        long_title = row.get("long_title") or ""
        entry = records.setdefault(
            prefix,
            {"codes": set(), "titles": set(), "count": 0},
        )
        entry["codes"].add(row.get("icd_code", prefix))
        if long_title:
            entry["titles"].add(long_title)
        entry["count"] += 1

    documents: List[Document] = []
    for prefix, info in sorted(records.items()):
        titles = sorted(info["titles"]) or [f"ICD {prefix} cardiac diagnosis"]
        examples = "; ".join(titles[:6])
        text = (
            f"ICD prefix {prefix} appears in the cardiac cohort (n={info['count']}). "
            f"Representative diagnoses include: {examples}."
        )
        metadata = {
            "concepts": {"cardiac", prefix.lower(), f"icd {prefix.lower()}"},
            "icd_codes": sorted(info["codes"]),
            "example_titles": titles[:6],
        }
        documents.append(Document(doc_id=f"icd-{prefix.lower()}", text=text, metadata=metadata))

    if not documents:
        raise ValueError(
            "No cardiac ICD prefixes matched the provided data. Ensure the dataset contains the expected codes."
        )

    return documents


__all__ = ["Document", "RetrievedDocument", "CARDIAC_ICD_PREFIXES", "load_cardiac_icd_documents"]
